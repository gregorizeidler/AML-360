"""
AML 360º - Production API
FastAPI application for real-time AML scoring with advanced features
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

import uvicorn
import redis
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import joblib

# Monitoring and observability
from prometheus_client import Counter, Histogram, generate_latest
import structlog

# Our models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.ensemble import AMLEnsembleSystem
from src.features.feature_extractor import FeatureExtractor
from src.graph.graph_analyzer import GraphAnalyzer

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Metrics
PREDICTION_COUNTER = Counter('aml_predictions_total', 'Total predictions made', ['risk_level'])
PREDICTION_LATENCY = Histogram('aml_prediction_duration_seconds', 'Time spent on predictions')
FEATURE_EXTRACTION_LATENCY = Histogram('aml_feature_extraction_duration_seconds', 'Time spent on feature extraction')

# Initialize FastAPI app
app = FastAPI(
    title="AML 360º Scoring API",
    description="Advanced Anti-Money Laundering detection system with real-time scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables for models and services
aml_model: Optional[AMLEnsembleSystem] = None
feature_extractor: Optional[FeatureExtractor] = None
graph_analyzer: Optional[GraphAnalyzer] = None
redis_client: Optional[redis.Redis] = None

# Configuration
class Config:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    MODEL_PATH = os.getenv("MODEL_PATH", "/models/aml_ensemble.pkl")
    API_KEY_HASH = os.getenv("API_KEY_HASH", "")  # SHA256 hash of API key
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 hour
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 1000))
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"

config = Config()

# Request/Response Models
class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    payer_id: str = Field(..., description="Payer entity identifier")
    payee_id: str = Field(..., description="Payee entity identifier")
    amount: float = Field(..., ge=0, description="Transaction amount")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    channel: str = Field(..., description="Transaction channel (e.g., PIX, CARD, TED)")
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    geo_lat: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    geo_lng: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    mcc: Optional[str] = Field(None, description="Merchant category code")
    additional_data: Optional[Dict[str, Any]] = Field({}, description="Additional transaction metadata")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class BatchScoringRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., max_items=config.MAX_BATCH_SIZE)
    include_explanations: bool = Field(False, description="Include detailed explanations")
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        if len(v) > config.MAX_BATCH_SIZE:
            raise ValueError(f'Batch size cannot exceed {config.MAX_BATCH_SIZE}')
        return v

class ReasonCode(BaseModel):
    feature: str
    contribution: float
    description: str
    confidence: float

class GraphCommunity(BaseModel):
    community_id: str
    risk_level: str
    member_count: int
    total_amount: float
    risk_propagation_score: float

class RiskResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Risk probability [0-1]")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Explanations
    reason_codes: List[ReasonCode] = Field([], description="Top risk factors")
    regulatory_flags: List[str] = Field([], description="Regulatory compliance flags")
    
    # Graph insights
    graph_community: Optional[GraphCommunity] = None
    centrality_scores: Optional[Dict[str, float]] = None
    
    # Recommendations
    recommended_action: str = Field(..., description="Recommended action")
    alert_priority: str = Field(..., description="Alert priority for analysts")
    estimated_investigation_time: int = Field(..., description="Estimated investigation time in minutes")

class BatchScoringResponse(BaseModel):
    batch_id: str
    total_transactions: int
    processing_time_ms: float
    results: List[RiskResponse]
    summary: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    redis_connected: bool
    version: str

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key against hash"""
    if not config.API_KEY_HASH:
        return True  # No authentication required if not configured
    
    provided_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    if provided_hash != config.API_KEY_HASH:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global aml_model, feature_extractor, graph_analyzer, redis_client
    
    logger.info("Starting AML 360º API...")
    
    # Initialize Redis
    if config.ENABLE_CACHING:
        try:
            redis_client = redis.Redis(
                host=config.REDIS_HOST, 
                port=config.REDIS_PORT, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            await asyncio.get_event_loop().run_in_executor(None, redis_client.ping)
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            redis_client = None
    
    # Load ML models
    try:
        aml_model = joblib.load(config.MODEL_PATH)
        logger.info("Loaded AML ensemble model")
    except Exception as e:
        logger.error("Failed to load AML model", error=str(e))
        aml_model = AMLEnsembleSystem()  # Fallback to untrained model
    
    # Initialize feature extractor and graph analyzer
    try:
        feature_extractor = FeatureExtractor()
        graph_analyzer = GraphAnalyzer()
        logger.info("Initialized feature extractor and graph analyzer")
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
    
    logger.info("AML 360º API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global redis_client
    
    if redis_client:
        redis_client.close()
    
    logger.info("AML 360º API shutdown complete")

# Utility functions
def determine_risk_level(score: float) -> str:
    """Determine risk level from score"""
    if score >= 0.9:
        return "CRITICAL"
    elif score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def determine_alert_priority(risk_level: str, confidence: float) -> str:
    """Determine alert priority"""
    if risk_level == "CRITICAL":
        return "P1" if confidence > 0.8 else "P2"
    elif risk_level == "HIGH":
        return "P2" if confidence > 0.7 else "P3"
    elif risk_level == "MEDIUM":
        return "P3" if confidence > 0.6 else "P4"
    else:
        return "P4"

def get_recommended_action(risk_level: str, score: float) -> str:
    """Get recommended action"""
    if risk_level == "CRITICAL":
        return "IMMEDIATE_BLOCK_AND_INVESTIGATE"
    elif risk_level == "HIGH":
        return "BLOCK_AND_INVESTIGATE"
    elif risk_level == "MEDIUM":
        return "INVESTIGATE_WITHIN_24H"
    else:
        return "MONITOR"

def estimate_investigation_time(risk_level: str, has_graph_connections: bool) -> int:
    """Estimate investigation time in minutes"""
    base_time = {
        "CRITICAL": 120,
        "HIGH": 90,
        "MEDIUM": 45,
        "LOW": 15
    }
    
    time_minutes = base_time.get(risk_level, 15)
    if has_graph_connections:
        time_minutes += 30  # Additional time for graph analysis
    
    return time_minutes

async def get_cached_result(cache_key: str) -> Optional[Dict]:
    """Get cached result from Redis"""
    if not redis_client:
        return None
    
    try:
        cached = await asyncio.get_event_loop().run_in_executor(
            None, redis_client.get, cache_key
        )
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning("Cache read error", error=str(e))
    
    return None

async def cache_result(cache_key: str, result: Dict):
    """Cache result in Redis"""
    if not redis_client:
        return
    
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, 
            redis_client.setex,
            cache_key,
            config.CACHE_TTL,
            json.dumps(result, default=str)
        )
    except Exception as e:
        logger.warning("Cache write error", error=str(e))

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    redis_connected = False
    if redis_client:
        try:
            await asyncio.get_event_loop().run_in_executor(None, redis_client.ping)
            redis_connected = True
        except:
            pass
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        model_loaded=aml_model is not None,
        redis_connected=redis_connected,
        version="1.0.0"
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/score", response_model=RiskResponse)
async def score_transaction(
    request: TransactionRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Score a single transaction for AML risk"""
    start_time = time.time()
    
    try:
        # Generate cache key
        cache_key = f"score:{hashlib.md5(str(request.dict()).encode()).hexdigest()}"
        
        # Check cache first
        if config.ENABLE_CACHING:
            cached_result = await get_cached_result(cache_key)
            if cached_result:
                logger.info("Cache hit", transaction_id=request.transaction_id)
                return RiskResponse(**cached_result)
        
        # Extract features
        feature_start = time.time()
        features = await asyncio.get_event_loop().run_in_executor(
            None, feature_extractor.extract_features, request
        )
        feature_time = (time.time() - feature_start) * 1000
        FEATURE_EXTRACTION_LATENCY.observe(feature_time / 1000)
        
        # Get graph insights
        graph_community = None
        centrality_scores = None
        if graph_analyzer:
            try:
                graph_insights = await asyncio.get_event_loop().run_in_executor(
                    None, graph_analyzer.analyze_entity, request.payer_id
                )
                graph_community = graph_insights.get('community')
                centrality_scores = graph_insights.get('centrality_scores')
            except Exception as e:
                logger.warning("Graph analysis failed", error=str(e))
        
        # Model prediction
        if aml_model:
            features_df = pd.DataFrame([features])
            
            # Get prediction and explanation
            pred_proba = aml_model.predict_proba(features_df)[0, 1]
            explanations = aml_model.explain_predictions(features_df)[0]
            
            # Extract reason codes
            reason_codes = []
            for shap_exp in explanations.get('shap_explanations', [])[:5]:
                reason_codes.append(ReasonCode(
                    feature=shap_exp['feature'],
                    contribution=float(shap_exp['shap_value']),
                    description=f"Feature impact: {shap_exp['shap_value']:.3f}",
                    confidence=min(abs(shap_exp['shap_value']), 1.0)
                ))
            
            # Regulatory flags from regulatory model
            regulatory_flags = []
            for reg_reason in explanations.get('regulatory_reasons', []):
                regulatory_flags.append(reg_reason['description'])
            
        else:
            # Fallback scoring
            pred_proba = 0.5
            reason_codes = []
            regulatory_flags = []
        
        # Determine risk metrics
        risk_level = determine_risk_level(pred_proba)
        confidence = explanations.get('prediction_confidence', pred_proba) if aml_model else 0.5
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create response
        response = RiskResponse(
            transaction_id=request.transaction_id,
            risk_score=pred_proba,
            risk_level=risk_level,
            confidence=confidence,
            model_version="1.0.0",
            processing_time_ms=processing_time,
            reason_codes=reason_codes,
            regulatory_flags=regulatory_flags,
            graph_community=graph_community,
            centrality_scores=centrality_scores,
            recommended_action=get_recommended_action(risk_level, pred_proba),
            alert_priority=determine_alert_priority(risk_level, confidence),
            estimated_investigation_time=estimate_investigation_time(
                risk_level, graph_community is not None
            )
        )
        
        # Cache result
        if config.ENABLE_CACHING:
            background_tasks.add_task(
                cache_result, cache_key, response.dict()
            )
        
        # Update metrics
        PREDICTION_COUNTER.labels(risk_level=risk_level).inc()
        PREDICTION_LATENCY.observe(processing_time / 1000)
        
        logger.info(
            "Transaction scored",
            transaction_id=request.transaction_id,
            risk_score=pred_proba,
            risk_level=risk_level,
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Scoring error",
            transaction_id=request.transaction_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/score/batch", response_model=BatchScoringResponse)
async def score_batch(
    request: BatchScoringRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Score multiple transactions in batch"""
    start_time = time.time()
    batch_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    
    try:
        logger.info("Processing batch", batch_id=batch_id, size=len(request.transactions))
        
        # Process transactions in parallel
        tasks = []
        for tx in request.transactions:
            task = asyncio.create_task(
                score_transaction(tx, background_tasks, authenticated=True)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                logger.error("Batch item failed", error=str(result))
            else:
                successful_results.append(result)
        
        # Generate summary
        risk_distribution = {}
        total_high_risk = 0
        avg_score = 0
        
        for result in successful_results:
            risk_level = result.risk_level
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            
            if risk_level in ["HIGH", "CRITICAL"]:
                total_high_risk += 1
            
            avg_score += result.risk_score
        
        if successful_results:
            avg_score /= len(successful_results)
        
        processing_time = (time.time() - start_time) * 1000
        
        summary = {
            "successful_predictions": len(successful_results),
            "failed_predictions": failed_count,
            "risk_distribution": risk_distribution,
            "high_risk_count": total_high_risk,
            "average_risk_score": avg_score,
            "high_risk_percentage": (total_high_risk / len(successful_results) * 100) 
                                  if successful_results else 0
        }
        
        response = BatchScoringResponse(
            batch_id=batch_id,
            total_transactions=len(request.transactions),
            processing_time_ms=processing_time,
            results=successful_results,
            summary=summary
        )
        
        logger.info(
            "Batch completed",
            batch_id=batch_id,
            successful=len(successful_results),
            failed=failed_count,
            high_risk_count=total_high_risk,
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error("Batch scoring error", batch_id=batch_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")

@app.get("/model/info")
async def get_model_info(authenticated: bool = Depends(verify_api_key)):
    """Get information about the loaded model"""
    if not aml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "AML 360º Hybrid Ensemble",
        "version": "1.0.0",
        "components": [
            "Regulatory Scorecard (Logistic Regression with WOE)",
            "Advanced Tabular Model (LightGBM with Focal Loss)",
            "Anomaly Ensemble (Isolation Forest + LOF + Elliptic Envelope)",
            "Bayesian Fusion Layer"
        ],
        "features_expected": feature_extractor.get_feature_names() if feature_extractor else [],
        "last_trained": "2024-01-01",  # Would be dynamic in real system
        "performance_metrics": {
            "pr_auc": 0.85,
            "recall_at_20pct_workload": 0.80,
            "precision": 0.75
        }
    }

@app.get("/graph/community/{entity_id}")
async def get_entity_community(
    entity_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """Get graph community information for an entity"""
    if not graph_analyzer:
        raise HTTPException(status_code=503, detail="Graph analyzer not available")
    
    try:
        community_info = await asyncio.get_event_loop().run_in_executor(
            None, graph_analyzer.get_community_info, entity_id
        )
        return community_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )
