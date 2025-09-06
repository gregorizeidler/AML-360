# Technical Architecture - AML 360º

## 1. Data Architecture & Layers (BigQuery Focus)

### Data Sources
- **Transactional**: PIX, cards, TED/DOC, boletos
- **KYC/KYB**: Onboarding data, entity information
- **External**: PEP lists, sanctions, chargebacks, complaints
- **Behavioral**: Device/IP data, geolocation, open-source intel

### Data Vault / Lakehouse Structure

#### Bronze Layer (Raw Ingestion)
```sql
-- Raw transaction stream
CREATE TABLE bronze.transactions (
  id STRING,
  timestamp TIMESTAMP,
  payer_id STRING,
  payee_id STRING,
  amount NUMERIC,
  channel STRING,
  device_id STRING,
  ip_address STRING,
  geo_lat FLOAT64,
  geo_lng FLOAT64,
  mcc STRING,
  status STRING,
  _ingestion_time TIMESTAMP
) PARTITION BY DATE(timestamp);
```

#### Silver Layer (Cleaned & Resolved)
- Entity resolution (CPF/CNPJ, keys, IBAN)
- SCD Type 2 for historical tracking
- Data quality validation and cleansing

#### Gold Layer (Feature Store)
- Time-windowed aggregations (1h/24h/7d/30d)
- Entity-level features (user_id, counterparty_id, merchant_id)
- Graph-derived features (community, centrality)
- Sequential pattern features

## 2. Advanced Feature Engineering

### Statistical Methods

#### Extreme Value Theory (EVT)
```python
from scipy import stats
import numpy as np

def evt_threshold(amounts, percentile=95):
    """Calculate EVT threshold using peaks-over-threshold"""
    threshold = np.percentile(amounts, percentile)
    excesses = amounts[amounts > threshold] - threshold
    
    # Fit Generalized Pareto Distribution
    xi, _, sigma = stats.genpareto.fit(excesses)
    return threshold, xi, sigma

def evt_score(amount, threshold, xi, sigma):
    """Calculate EVT-based anomaly score"""
    if amount <= threshold:
        return 0
    excess = amount - threshold
    return 1 - stats.genpareto.cdf(excess, xi, scale=sigma)
```

#### GARCH Volatility Modeling
```python
from arch import arch_model

def garch_volatility_score(amounts_series, window=30):
    """Calculate GARCH-based volatility score"""
    returns = np.log(amounts_series / amounts_series.shift(1)).dropna()
    
    # Fit GARCH(1,1) model
    model = arch_model(returns, vol='Garch', p=1, q=1)
    fitted_model = model.fit(disp='off')
    
    # Get conditional volatility
    volatility = fitted_model.conditional_volatility
    return volatility.iloc[-1]  # Most recent volatility
```

#### Weight of Evidence (WOE) & Information Value (IV)
```python
def calculate_woe_iv(df, feature, target):
    """Calculate WOE and IV for feature selection"""
    crosstab = pd.crosstab(df[feature], df[target])
    crosstab['Total'] = crosstab.sum(axis=1)
    crosstab['Dist_Good'] = crosstab[0] / crosstab[0].sum()
    crosstab['Dist_Bad'] = crosstab[1] / crosstab[1].sum()
    crosstab['WOE'] = np.log(crosstab['Dist_Good'] / crosstab['Dist_Bad'])
    crosstab['IV'] = (crosstab['Dist_Good'] - crosstab['Dist_Bad']) * crosstab['WOE']
    
    return crosstab['WOE'].to_dict(), crosstab['IV'].sum()
```

### Burstiness Analysis
```python
def burstiness_metrics(timestamps):
    """Calculate burstiness using Fano factor and Clark-Evans index"""
    # Convert to interevent times
    intervals = np.diff(timestamps)
    
    # Fano factor (variance-to-mean ratio)
    fano_factor = np.var(intervals) / np.mean(intervals)
    
    # Clark-Evans index for spatial/temporal clustering
    mean_interval = np.mean(intervals)
    expected_nearest = 1 / (2 * len(intervals))
    clark_evans = mean_interval / expected_nearest
    
    return fano_factor, clark_evans
```

## 3. Hybrid Modeling Architecture

### A) Regulatory Baseline (Explainable)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class RegulatoryScorecardModel:
    def __init__(self):
        self.woe_encoders = {}
        self.scaler = StandardScaler()
        self.logit_model = LogisticRegression(
            penalty='l1', 
            solver='liblinear',
            class_weight='balanced'
        )
    
    def fit(self, X, y):
        # Apply WOE encoding
        X_woe = self.apply_woe_encoding(X)
        
        # Fit logistic regression
        X_scaled = self.scaler.fit_transform(X_woe)
        self.logit_model.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X):
        X_woe = self.apply_woe_encoding(X)
        X_scaled = self.scaler.transform(X_woe)
        return self.logit_model.predict_proba(X_scaled)
    
    def get_reason_codes(self, X):
        """Generate explainable reason codes"""
        coefficients = self.logit_model.coef_[0]
        feature_importance = np.abs(coefficients)
        top_features = np.argsort(feature_importance)[-5:]
        
        return [f"High risk in {feature}" for feature in top_features]
```

### B) Supervised Tabular (High Performance)
```python
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

class HighPerformanceTabularModel:
    def __init__(self):
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'class_weight': 'balanced'
        }
        self.model = None
        self.calibrator = None
    
    def focal_loss_objective(self, y_pred, y_true, alpha=0.25, gamma=2.0):
        """Custom focal loss for imbalanced data"""
        y_true = y_true.get_label()
        p = 1 / (1 + np.exp(-y_pred))
        
        # Focal loss calculation
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        pt = p * y_true + (1 - p) * (1 - y_true)
        
        grad = alpha_t * (1 - pt) ** gamma * ((gamma * pt * np.log(pt + 1e-8)) - pt + 1)
        hess = alpha_t * (1 - pt) ** gamma * (gamma * (2 * pt - 1) * np.log(pt + 1e-8) + gamma * (1 - pt) + 2 * pt - 1)
        
        return grad, hess
    
    def fit(self, X, y, X_val=None, y_val=None):
        train_data = lgb.Dataset(X, label=y)
        
        if X_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [valid_data]
        else:
            valid_sets = None
        
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100)]
        )
        
        # Calibrate probabilities
        y_pred_uncalibrated = self.model.predict(X, num_iteration=self.model.best_iteration)
        self.calibrator = CalibratedClassifierCV(cv=3, method='isotonic')
        self.calibrator.fit(y_pred_uncalibrated.reshape(-1, 1), y)
        
        return self
    
    def predict_proba(self, X):
        y_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        return self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
```

### C) Graph Neural Network Architecture
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphSAGE

class AML_GraphModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=64, num_heads=8):
        super(AML_GraphModel, self).__init__()
        
        # Node embedding layers
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Graph Attention Networks for relationship modeling
        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.3)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.3)
        
        # GraphSAGE for scalable neighborhood aggregation
        self.sage = GraphSAGE(hidden_dim, hidden_dim, num_layers=2, dropout=0.3)
        
        # Risk propagation mechanism
        self.risk_propagator = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layers
        self.node_classifier = nn.Linear(hidden_dim, 2)  # Node risk classification
        self.edge_classifier = nn.Linear(hidden_dim * 2, 2)  # Edge risk classification
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Node feature embedding
        h = F.relu(self.node_embedding(x))
        
        # Multi-head attention for relationship modeling
        h = F.relu(self.gat1(h, edge_index))
        h = F.dropout(h, training=self.training)
        h = self.gat2(h, edge_index)
        
        # Neighborhood aggregation with GraphSAGE
        h_sage = self.sage(h, edge_index)
        
        # Combine GAT and SAGE representations
        h_combined = h + h_sage
        
        # Risk propagation through temporal dynamics
        h_prop, _ = self.risk_propagator(h_combined.unsqueeze(0))
        h_final = h_prop.squeeze(0)
        
        # Node classification (account risk)
        node_logits = self.node_classifier(h_final)
        
        # Edge classification (transaction risk)
        edge_embeddings = torch.cat([h_final[edge_index[0]], h_final[edge_index[1]]], dim=1)
        edge_logits = self.edge_classifier(edge_embeddings)
        
        return node_logits, edge_logits, h_final

class TemporalGraphModel(nn.Module):
    """Temporal Graph Attention Network for dynamic pattern detection"""
    def __init__(self, node_features, time_features, hidden_dim=64):
        super(TemporalGraphModel, self).__init__()
        
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.time_encoder = nn.Linear(time_features, hidden_dim)
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # LSTM for sequential pattern modeling
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3)  # Smurfing, Layering, Integration
        )
    
    def forward(self, node_features, time_features, temporal_edges):
        # Encode node and temporal features
        h_nodes = self.node_encoder(node_features)
        h_time = self.time_encoder(time_features)
        
        # Temporal attention over transaction sequences
        h_temporal, _ = self.temporal_attention(h_time, h_time, h_time)
        
        # Sequential pattern detection
        h_seq, _ = self.lstm(h_temporal)
        
        # Pattern classification
        pattern_logits = self.pattern_classifier(h_seq)
        
        return pattern_logits
```

### D) Bayesian Fusion Architecture
```python
import pymc3 as pm
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BayesianEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, cost_matrix=None):
        self.models = models  # List of trained models
        self.cost_matrix = cost_matrix or {'c_fn': 10, 'c_fp': 1, 'c_rev': 0.1}
        self.fusion_weights = None
        
    def fit(self, X, y):
        """Learn optimal fusion weights using Bayesian approach"""
        # Get predictions from all models
        predictions = np.column_stack([
            model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') 
            else model.predict(X) for model in self.models
        ])
        
        # Bayesian model for weight learning
        with pm.Model() as fusion_model:
            # Priors for fusion weights
            weights = pm.Dirichlet('weights', a=np.ones(len(self.models)))
            
            # Linear combination of model predictions
            combined_logits = pm.math.dot(predictions, weights)
            
            # Likelihood
            likelihood = pm.Bernoulli('likelihood', logit_p=combined_logits, observed=y)
            
            # MCMC sampling
            trace = pm.sample(2000, tune=1000, chains=2)
            
        self.fusion_weights = trace['weights'].mean(axis=0)
        return self
    
    def predict_proba(self, X):
        """Generate calibrated probability predictions"""
        predictions = np.column_stack([
            model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') 
            else model.predict(X) for model in self.models
        ])
        
        # Weighted combination
        combined_scores = np.dot(predictions, self.fusion_weights)
        
        # Convert to probabilities
        probs = 1 / (1 + np.exp(-combined_scores))
        return np.column_stack([1 - probs, probs])
    
    def optimal_threshold(self, y_true, y_scores, workload_capacity=0.2):
        """Calculate cost-sensitive optimal threshold"""
        thresholds = np.percentile(y_scores, np.arange(0, 100, 1))
        min_cost = float('inf')
        optimal_thresh = 0.5
        
        for thresh in thresholds:
            predictions = (y_scores >= thresh).astype(int)
            
            # Calculate confusion matrix elements
            fn = np.sum((y_true == 1) & (predictions == 0))
            fp = np.sum((y_true == 0) & (predictions == 1))
            workload = np.mean(predictions)
            
            # Skip if workload exceeds capacity
            if workload > workload_capacity:
                continue
            
            # Calculate total cost
            total_cost = (self.cost_matrix['c_fn'] * fn + 
                         self.cost_matrix['c_fp'] * fp + 
                         self.cost_matrix['c_rev'] * workload * len(y_true))
            
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_thresh = thresh
                
        return optimal_thresh
```

## 4. Production Architecture

### Real-time Scoring API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import redis
import json

app = FastAPI(title="AML 360° Scoring API")

# Redis for feature caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class TransactionRequest(BaseModel):
    transaction_id: str
    payer_id: str
    payee_id: str
    amount: float
    timestamp: str
    channel: str
    device_id: str
    
class RiskResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: str
    reason_codes: list
    graph_community: str
    recommended_action: str

@app.post("/score", response_model=RiskResponse)
async def score_transaction(request: TransactionRequest):
    try:
        # Extract features in parallel
        features = await asyncio.gather(
            extract_transactional_features(request),
            extract_graph_features(request.payer_id, request.payee_id),
            extract_sequential_features(request.payer_id),
            extract_behavioral_features(request.device_id)
        )
        
        # Combine features
        feature_vector = np.concatenate(features)
        
        # Model ensemble scoring
        scores = {
            'tabular': tabular_model.predict_proba([feature_vector])[0][1],
            'graph': graph_model.predict(feature_vector),
            'sequential': sequential_model.predict(feature_vector),
            'anomaly': anomaly_model.score_samples([feature_vector])[0]
        }
        
        # Bayesian fusion
        final_score = bayesian_ensemble.predict_proba([feature_vector])[0][1]
        
        # Risk level determination
        risk_level = determine_risk_level(final_score)
        
        # Generate explanations
        reason_codes = generate_reason_codes(feature_vector, scores)
        
        # Cache results
        redis_client.setex(
            f"score:{request.transaction_id}", 
            3600, 
            json.dumps({'score': final_score, 'timestamp': request.timestamp})
        )
        
        return RiskResponse(
            transaction_id=request.transaction_id,
            risk_score=final_score,
            risk_level=risk_level,
            reason_codes=reason_codes,
            graph_community=get_graph_community(request.payer_id),
            recommended_action=get_recommended_action(final_score, risk_level)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def determine_risk_level(score):
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"
```

This technical architecture provides the foundation for implementing the complete AML 360º system with enterprise-grade performance and regulatory compliance.
