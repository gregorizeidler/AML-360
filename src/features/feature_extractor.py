"""
AML 360º - Feature Extraction
Advanced statistical feature engineering for AML detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Google Cloud BigQuery (optional for local development)
try:
    from google.cloud import bigquery
    HAS_BIGQUERY = True
except ImportError:
    bigquery = None
    HAS_BIGQUERY = False

# Statistical libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """
    Advanced feature extractor with statistical methods
    """
    
    def __init__(self, bq_client: Optional[any] = None):
        if HAS_BIGQUERY and bq_client is None:
            try:
                self.bq_client = bigquery.Client()
            except Exception:
                self.bq_client = None
        else:
            self.bq_client = bq_client
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def extract_features(self, transaction_request) -> Dict[str, float]:
        """
        Extract comprehensive features for a transaction
        """
        features = {}
        
        # Basic transaction features
        features.update(self._extract_basic_features(transaction_request))
        
        # Temporal features
        features.update(self._extract_temporal_features(transaction_request))
        
        # Statistical features (EVT, GARCH, etc.)
        features.update(self._extract_statistical_features(transaction_request))
        
        # Behavioral features
        features.update(self._extract_behavioral_features(transaction_request))
        
        # Network features
        features.update(self._extract_network_features(transaction_request))
        
        return features
    
    def _extract_basic_features(self, tx) -> Dict[str, float]:
        """Extract basic transaction features"""
        return {
            'amount': float(tx.amount),
            'amount_log': np.log1p(tx.amount),
            'hour_of_day': tx.timestamp.hour,
            'day_of_week': tx.timestamp.weekday(),
            'is_weekend': float(tx.timestamp.weekday() >= 5),
            'is_night': float(tx.timestamp.hour >= 22 or tx.timestamp.hour <= 6),
            'channel_pix': float(tx.channel == 'PIX'),
            'channel_card': float(tx.channel == 'CARD'),
            'channel_ted': float(tx.channel == 'TED'),
            'has_geo': float(tx.geo_lat is not None and tx.geo_lng is not None),
        }
    
    def _extract_temporal_features(self, tx) -> Dict[str, float]:
        """Extract temporal window features from BigQuery"""
        
        query = f"""
        SELECT
            COUNT(*) as tx_count_24h,
            COALESCE(SUM(amount), 0) as total_amount_24h,
            COUNT(DISTINCT payee_id) as unique_payees_24h,
            STDDEV(amount) as amount_std_24h,
            AVG(amount) as amount_mean_24h
        FROM `aml.transactions`
        WHERE payer_id = '{tx.payer_id}'
          AND timestamp >= TIMESTAMP_SUB('{tx.timestamp}', INTERVAL 24 HOUR)
          AND timestamp < '{tx.timestamp}'
        """
        
        try:
            result = list(self.bq_client.query(query))
            if result:
                row = result[0]
                return {
                    'tx_count_24h': float(row.tx_count_24h or 0),
                    'total_amount_24h': float(row.total_amount_24h or 0),
                    'unique_payees_24h': float(row.unique_payees_24h or 0),
                    'amount_std_24h': float(row.amount_std_24h or 0),
                    'amount_mean_24h': float(row.amount_mean_24h or 0),
                    'amount_vs_mean_ratio': tx.amount / max(row.amount_mean_24h or 1, 1),
                }
        except Exception as e:
            print(f"Error extracting temporal features: {e}")
            
        return {
            'tx_count_24h': 0,
            'total_amount_24h': 0,
            'unique_payees_24h': 0,
            'amount_std_24h': 0,
            'amount_mean_24h': 0,
            'amount_vs_mean_ratio': 1,
        }
    
    def _extract_statistical_features(self, tx) -> Dict[str, float]:
        """Extract EVT and other statistical features"""
        
        # Get historical amounts for EVT
        query = f"""
        SELECT amount
        FROM `aml.transactions`
        WHERE payer_id = '{tx.payer_id}'
          AND timestamp >= TIMESTAMP_SUB('{tx.timestamp}', INTERVAL 30 DAY)
          AND timestamp < '{tx.timestamp}'
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        
        features = {}
        
        try:
            amounts = [row.amount for row in self.bq_client.query(query)]
            if len(amounts) >= 10:
                # EVT features
                p95_threshold = np.percentile(amounts, 95)
                features['evt_p95_threshold'] = p95_threshold
                features['amount_above_evt'] = float(tx.amount > p95_threshold)
                
                if tx.amount > p95_threshold:
                    excess = tx.amount - p95_threshold
                    excesses = [a - p95_threshold for a in amounts if a > p95_threshold]
                    if excesses:
                        features['evt_excess_zscore'] = (excess - np.mean(excesses)) / (np.std(excesses) + 1e-8)
                else:
                    features['evt_excess_zscore'] = 0
                
                # Burstiness (Fano factor approximation)
                if len(amounts) >= 5:
                    fano_factor = np.var(amounts) / (np.mean(amounts) + 1e-8)
                    features['fano_factor'] = min(fano_factor, 10)  # Cap extreme values
                else:
                    features['fano_factor'] = 1
                    
                # Distribution features
                features['amount_percentile'] = stats.percentileofscore(amounts, tx.amount) / 100
                features['amount_zscore'] = (tx.amount - np.mean(amounts)) / (np.std(amounts) + 1e-8)
                
        except Exception as e:
            print(f"Error extracting statistical features: {e}")
            
        # Default values if query failed
        features.setdefault('evt_p95_threshold', tx.amount)
        features.setdefault('amount_above_evt', 0)
        features.setdefault('evt_excess_zscore', 0)
        features.setdefault('fano_factor', 1)
        features.setdefault('amount_percentile', 0.5)
        features.setdefault('amount_zscore', 0)
        
        return features
    
    def _extract_behavioral_features(self, tx) -> Dict[str, float]:
        """Extract behavioral pattern features"""
        
        query = f"""
        WITH recent_tx AS (
            SELECT 
                amount,
                timestamp,
                payee_id,
                channel,
                LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                LAG(amount) OVER (ORDER BY timestamp) as prev_amount
            FROM `aml.transactions`
            WHERE payer_id = '{tx.payer_id}'
              AND timestamp >= TIMESTAMP_SUB('{tx.timestamp}', INTERVAL 7 DAY)
              AND timestamp < '{tx.timestamp}'
            ORDER BY timestamp DESC
            LIMIT 50
        )
        SELECT
            COUNT(*) as recent_tx_count,
            -- Round amount patterns (structuring indicator)
            COUNTIF(MOD(CAST(amount AS INT64), 1000) = 0) as round_1000_count,
            COUNTIF(MOD(CAST(amount AS INT64), 500) = 0) as round_500_count,
            -- Rapid sequences
            COUNTIF(TIMESTAMP_DIFF(timestamp, prev_timestamp, MINUTE) < 30) as rapid_sequence_count,
            -- Similar amounts (smurfing indicator)
            COUNTIF(ABS(amount - prev_amount) < 100 AND amount > 5000) as similar_amount_count,
            -- Channel diversity
            COUNT(DISTINCT channel) as channel_diversity
        FROM recent_tx
        """
        
        try:
            result = list(self.bq_client.query(query))
            if result:
                row = result[0]
                total_tx = max(row.recent_tx_count or 1, 1)
                
                return {
                    'recent_tx_count': float(row.recent_tx_count or 0),
                    'round_amount_ratio': (row.round_1000_count + row.round_500_count) / total_tx,
                    'rapid_sequence_ratio': (row.rapid_sequence_count or 0) / total_tx,
                    'similar_amount_ratio': (row.similar_amount_count or 0) / total_tx,
                    'channel_diversity': float(row.channel_diversity or 1),
                    'amount_is_round_1000': float(tx.amount % 1000 == 0),
                    'amount_is_round_500': float(tx.amount % 500 == 0),
                }
        except Exception as e:
            print(f"Error extracting behavioral features: {e}")
            
        return {
            'recent_tx_count': 0,
            'round_amount_ratio': 0,
            'rapid_sequence_ratio': 0,
            'similar_amount_ratio': 0,
            'channel_diversity': 1,
            'amount_is_round_1000': float(tx.amount % 1000 == 0),
            'amount_is_round_500': float(tx.amount % 500 == 0),
        }
    
    def _extract_network_features(self, tx) -> Dict[str, float]:
        """Extract graph/network features"""
        
        # Simplified network features query
        query = f"""
        WITH payer_network AS (
            SELECT payee_id, COUNT(*) as tx_count, SUM(amount) as total_amount
            FROM `aml.transactions`
            WHERE payer_id = '{tx.payer_id}'
              AND timestamp >= TIMESTAMP_SUB('{tx.timestamp}', INTERVAL 30 DAY)
              AND timestamp < '{tx.timestamp}'
            GROUP BY payee_id
        ),
        payee_network AS (
            SELECT payer_id, COUNT(*) as tx_count, SUM(amount) as total_amount  
            FROM `aml.transactions`
            WHERE payee_id = '{tx.payee_id}'
              AND timestamp >= TIMESTAMP_SUB('{tx.timestamp}', INTERVAL 30 DAY)
              AND timestamp < '{tx.timestamp}'
            GROUP BY payer_id
        )
        SELECT
            (SELECT COUNT(*) FROM payer_network) as payer_out_degree,
            (SELECT COUNT(*) FROM payee_network) as payee_in_degree,
            (SELECT COALESCE(MAX(tx_count), 0) FROM payer_network) as max_tx_to_counterparty,
            -- Check if this is a repeated transaction
            (SELECT COUNT(*) FROM `aml.transactions` 
             WHERE payer_id = '{tx.payer_id}' AND payee_id = '{tx.payee_id}'
               AND timestamp < '{tx.timestamp}') as previous_tx_count
        """
        
        try:
            result = list(self.bq_client.query(query))
            if result:
                row = result[0]
                return {
                    'payer_out_degree': float(row.payer_out_degree or 0),
                    'payee_in_degree': float(row.payee_in_degree or 0),
                    'max_tx_to_counterparty': float(row.max_tx_to_counterparty or 0),
                    'is_repeated_counterparty': float((row.previous_tx_count or 0) > 0),
                    'previous_tx_count': float(row.previous_tx_count or 0),
                    'network_centrality_proxy': np.log1p(
                        (row.payer_out_degree or 0) * (row.payee_in_degree or 0)
                    )
                }
        except Exception as e:
            print(f"Error extracting network features: {e}")
            
        return {
            'payer_out_degree': 0,
            'payee_in_degree': 0,
            'max_tx_to_counterparty': 0,
            'is_repeated_counterparty': 0,
            'previous_tx_count': 0,
            'network_centrality_proxy': 0,
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return [
            # Basic features
            'amount', 'amount_log', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
            'channel_pix', 'channel_card', 'channel_ted', 'has_geo',
            
            # Temporal features
            'tx_count_24h', 'total_amount_24h', 'unique_payees_24h', 'amount_std_24h', 
            'amount_mean_24h', 'amount_vs_mean_ratio',
            
            # Statistical features
            'evt_p95_threshold', 'amount_above_evt', 'evt_excess_zscore', 'fano_factor',
            'amount_percentile', 'amount_zscore',
            
            # Behavioral features
            'recent_tx_count', 'round_amount_ratio', 'rapid_sequence_ratio', 
            'similar_amount_ratio', 'channel_diversity', 'amount_is_round_1000', 
            'amount_is_round_500',
            
            # Network features
            'payer_out_degree', 'payee_in_degree', 'max_tx_to_counterparty',
            'is_repeated_counterparty', 'previous_tx_count', 'network_centrality_proxy'
        ]
    
    async def extract_features_batch(self, transaction_requests: List) -> List[Dict[str, float]]:
        """Extract features for multiple transactions in parallel"""
        
        loop = asyncio.get_event_loop()
        tasks = []
        
        for tx in transaction_requests:
            task = loop.run_in_executor(self.executor, self.extract_features, tx)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return results
