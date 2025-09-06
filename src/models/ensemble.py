"""
AML 360º - Advanced Ensemble Models
Bayesian fusion of multiple specialized models for AML detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Advanced ML models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Bayesian inference
# import pymc3 as pm  # Disabled temporarily
# import theano.tensor as tt  # Disabled temporarily

# Anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Explainability
import shap

class RegulatoryScorecardModel(BaseEstimator, ClassifierMixin):
    """
    Regulatory compliant logistic regression model with WOE encoding
    """
    
    def __init__(self, c_reg=0.01, max_features=20):
        self.c_reg = c_reg
        self.max_features = max_features
        self.woe_encoders = {}
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=c_reg, 
            penalty='l1', 
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        )
        self.feature_names = None
        self.reason_codes = {}
        
    def _calculate_woe_iv(self, X: pd.Series, y: pd.Series) -> Tuple[Dict, float]:
        """Calculate Weight of Evidence and Information Value"""
        # Create bins for continuous variables
        if X.dtype in ['float64', 'int64']:
            X_binned = pd.qcut(X, q=5, duplicates='drop')
        else:
            X_binned = X
            
        crosstab = pd.crosstab(X_binned, y)
        
        if crosstab.shape[1] < 2:
            return {}, 0
            
        crosstab['Total'] = crosstab.sum(axis=1)
        total_good = crosstab[0].sum()
        total_bad = crosstab[1].sum()
        
        if total_good == 0 or total_bad == 0:
            return {}, 0
            
        crosstab['Good_Rate'] = crosstab[0] / total_good
        crosstab['Bad_Rate'] = crosstab[1] / total_bad
        
        # Avoid division by zero
        crosstab['Good_Rate'] = np.maximum(crosstab['Good_Rate'], 1e-6)
        crosstab['Bad_Rate'] = np.maximum(crosstab['Bad_Rate'], 1e-6)
        
        crosstab['WOE'] = np.log(crosstab['Good_Rate'] / crosstab['Bad_Rate'])
        crosstab['IV'] = (crosstab['Good_Rate'] - crosstab['Bad_Rate']) * crosstab['WOE']
        
        woe_dict = crosstab['WOE'].to_dict()
        iv = crosstab['IV'].sum()
        
        return woe_dict, iv
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the regulatory scorecard model"""
        self.feature_names = X.columns.tolist()
        
        # Calculate WOE and IV for all features
        woe_features = {}
        iv_scores = {}
        
        for col in X.columns:
            woe_dict, iv = self._calculate_woe_iv(X[col], y)
            if iv > 0.02:  # Minimum IV threshold
                woe_features[col] = woe_dict
                iv_scores[col] = iv
        
        # Select top features by IV
        top_features = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        selected_features = [feat[0] for feat in top_features]
        
        # Apply WOE encoding
        X_woe = pd.DataFrame(index=X.index)
        for col in selected_features:
            if X[col].dtype in ['float64', 'int64']:
                X_binned = pd.qcut(X[col], q=5, duplicates='drop')
            else:
                X_binned = X[col]
            
            X_woe[col] = X_binned.map(woe_features[col]).fillna(0)
        
        # Standardize and fit logistic regression
        X_scaled = self.scaler.fit_transform(X_woe)
        self.model.fit(X_scaled, y)
        
        # Store WOE encoders and selected features
        self.woe_encoders = {col: woe_features[col] for col in selected_features}
        self.selected_features = selected_features
        self.iv_scores = dict(top_features)
        
        # Generate reason codes
        coefficients = self.model.coef_[0]
        for i, feature in enumerate(selected_features):
            self.reason_codes[feature] = f"Risk factor: {feature} (coef: {coefficients[i]:.3f})"
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply WOE transformation to features"""
        X_woe = pd.DataFrame(index=X.index)
        
        for col in self.selected_features:
            if col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X_binned = pd.qcut(X[col], q=5, duplicates='drop')
                else:
                    X_binned = X[col]
                    
                X_woe[col] = X_binned.map(self.woe_encoders[col]).fillna(0)
            else:
                X_woe[col] = 0
        
        return self.scaler.transform(X_woe)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with WOE transformation"""
        X_transformed = self.transform(X)
        return self.model.predict_proba(X_transformed)
    
    def get_reason_codes(self, X: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Generate reason codes for predictions"""
        X_transformed = self.transform(X)
        coefficients = self.model.coef_[0]
        
        reason_codes = []
        for i in range(X_transformed.shape[0]):
            feature_contributions = X_transformed[i] * coefficients
            top_features_idx = np.argsort(np.abs(feature_contributions))[-top_n:]
            
            reasons = []
            for idx in reversed(top_features_idx):
                feature_name = self.selected_features[idx]
                contribution = feature_contributions[idx]
                reasons.append({
                    'feature': feature_name,
                    'contribution': contribution,
                    'description': self.reason_codes[feature_name]
                })
            
            reason_codes.append(reasons)
        
        return reason_codes

class AdvancedTabularModel(BaseEstimator, ClassifierMixin):
    """
    High-performance gradient boosting ensemble with focal loss
    """
    
    def __init__(self, model_type='lightgbm', use_focal_loss=True, alpha=0.25, gamma=2.0):
        self.model_type = model_type
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.model = None
        self.calibrator = None
        self.feature_importance = None
        
    def focal_loss_lgb(self, y_pred, y_true):
        """Focal loss for LightGBM"""
        y_true = y_true.get_label()
        p = 1 / (1 + np.exp(-y_pred))
        
        # Focal loss components
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        pt = p * y_true + (1 - p) * (1 - y_true)
        
        # Gradients and Hessians
        grad = alpha_t * (1 - pt) ** self.gamma * (
            (self.gamma * pt * np.log(pt + 1e-8)) - pt + 1
        )
        hess = alpha_t * (1 - pt) ** self.gamma * (
            self.gamma * (2 * pt - 1) * np.log(pt + 1e-8) + 
            self.gamma * (1 - pt) + 2 * pt - 1
        )
        
        return grad, hess
    
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        """Fit the advanced tabular model"""
        
        if self.model_type == 'lightgbm':
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            
            train_data = lgb.Dataset(X, label=y)
            
            if X_val is not None and y_val is not None:
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets = [train_data, valid_data]
                valid_names = ['train', 'valid']
            else:
                valid_sets = [train_data]
                valid_names = ['train']
            
            if self.use_focal_loss:
                self.model = lgb.train(
                    params,
                    train_data,
                    fobj=self.focal_loss_lgb,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
            else:
                self.model = lgb.train(
                    params,
                    train_data,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
                
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=42,
                scale_pos_weight=len(y) / (2 * np.sum(y))
            )
            
            if X_val is not None:
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=100,
                    verbose=False
                )
            else:
                self.model.fit(X, y)
                
        elif self.model_type == 'catboost':
            self.model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=False,
                auto_class_weights='Balanced'
            )
            
            self.model.fit(X, y, eval_set=(X_val, y_val) if X_val is not None else None)
        
        # Calibrate probabilities
        if self.model_type == 'lightgbm':
            y_pred_uncalibrated = self.model.predict(X, num_iteration=self.model.best_iteration)
        else:
            y_pred_uncalibrated = self.model.predict_proba(X)[:, 1]
            
        self.calibrator = CalibratedClassifierCV(cv=3, method='isotonic')
        self.calibrator.fit(y_pred_uncalibrated.reshape(-1, 1), y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importance'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importance()))
        elif hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities"""
        if self.model_type == 'lightgbm':
            y_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        else:
            y_pred = self.model.predict_proba(X)[:, 1]
            
        calibrated_probs = self.calibrator.predict_proba(y_pred.reshape(-1, 1))
        return calibrated_probs

class AnomalyEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of anomaly detection models
    """
    
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.models = {}
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit ensemble of anomaly detectors"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.models['isolation_forest'].fit(X_scaled)
        
        # Local Outlier Factor
        self.models['lof'] = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )
        self.models['lof'].fit(X_scaled)
        
        # Robust Covariance (Elliptic Envelope)
        self.models['elliptic_envelope'] = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        self.models['elliptic_envelope'].fit(X_scaled)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomaly probabilities"""
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores from all models
        scores = {}
        scores['isolation_forest'] = self.models['isolation_forest'].score_samples(X_scaled)
        scores['lof'] = self.models['lof'].score_samples(X_scaled)
        scores['elliptic_envelope'] = self.models['elliptic_envelope'].score_samples(X_scaled)
        
        # Convert to probabilities (higher score = more normal)
        probs = {}
        for name, score in scores.items():
            # Normalize to [0, 1] and invert (1 = anomaly)
            normalized_score = (score - score.min()) / (score.max() - score.min() + 1e-8)
            probs[name] = 1 - normalized_score
        
        # Ensemble average
        ensemble_prob = np.mean([probs[name] for name in probs], axis=0)
        
        return np.column_stack([1 - ensemble_prob, ensemble_prob])

class SequentialPatternModel(nn.Module):
    """
    LSTM-based sequential pattern detection for structuring
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3):
        super(SequentialPatternModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size * 2, num_heads=8, batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

class BayesianEnsemble(BaseEstimator, ClassifierMixin):
    """
    Bayesian fusion of multiple models with uncertainty quantification
    """
    
    def __init__(self, models: List[BaseEstimator], cost_matrix=None):
        self.models = models
        self.cost_matrix = cost_matrix or {'c_fn': 10, 'c_fp': 1, 'c_rev': 0.1}
        self.fusion_weights = None
        self.optimal_threshold = 0.5
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn Bayesian fusion weights"""
        
        # Get predictions from all models
        predictions = []
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.decision_function(X)
                # Normalize to probabilities
                pred = 1 / (1 + np.exp(-pred))
            predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        # Bayesian model for weight learning
        with pm.Model() as fusion_model:
            # Priors for fusion weights (Dirichlet distribution)
            weights = pm.Dirichlet('weights', a=np.ones(len(self.models)))
            
            # Linear combination of model predictions
            combined_logits = pm.math.dot(predictions, weights)
            
            # Likelihood
            likelihood = pm.Bernoulli('likelihood', p=combined_logits, observed=y)
            
            # MCMC sampling
            trace = pm.sample(2000, tune=1000, chains=2, return_inferencedata=False)
        
        # Extract learned weights
        self.fusion_weights = np.mean(trace['weights'], axis=0)
        
        # Calculate optimal threshold
        combined_scores = np.dot(predictions, self.fusion_weights)
        self.optimal_threshold = self._calculate_optimal_threshold(y, combined_scores)
        
        return self
    
    def _calculate_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                   workload_capacity: float = 0.2) -> float:
        """Calculate cost-sensitive optimal threshold"""
        
        thresholds = np.percentile(y_scores, np.linspace(0, 99, 100))
        min_cost = float('inf')
        optimal_thresh = 0.5
        
        for thresh in thresholds:
            predictions = (y_scores >= thresh).astype(int)
            
            # Calculate confusion matrix
            tp = np.sum((y_true == 1) & (predictions == 1))
            fn = np.sum((y_true == 1) & (predictions == 0))
            fp = np.sum((y_true == 0) & (predictions == 1))
            tn = np.sum((y_true == 0) & (predictions == 0))
            
            workload = np.mean(predictions)
            
            # Skip if workload exceeds capacity
            if workload > workload_capacity:
                continue
            
            # Calculate total cost
            total_cost = (
                self.cost_matrix['c_fn'] * fn +
                self.cost_matrix['c_fp'] * fp +
                self.cost_matrix['c_rev'] * workload * len(y_true)
            )
            
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_thresh = thresh
        
        return optimal_thresh
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with Bayesian ensemble"""
        
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.decision_function(X)
                pred = 1 / (1 + np.exp(-pred))
            predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        # Weighted combination
        combined_scores = np.dot(predictions, self.fusion_weights)
        
        return np.column_stack([1 - combined_scores, combined_scores])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions using optimal threshold"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.optimal_threshold).astype(int)

class AMLEnsembleSystem:
    """
    Complete AML ensemble system orchestrator
    """
    
    def __init__(self):
        self.regulatory_model = RegulatoryScorecardModel()
        self.tabular_model = AdvancedTabularModel(model_type='lightgbm')
        self.anomaly_model = AnomalyEnsemble()
        self.bayesian_ensemble = None
        self.explainer = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Fit the complete ensemble system"""
        
        print("Training regulatory scorecard...")
        self.regulatory_model.fit(X_train, y_train)
        
        print("Training tabular model...")
        self.tabular_model.fit(X_train, y_train, X_val, y_val)
        
        print("Training anomaly ensemble...")
        self.anomaly_model.fit(X_train)
        
        print("Training Bayesian fusion...")
        models = [self.regulatory_model, self.tabular_model, self.anomaly_model]
        self.bayesian_ensemble = BayesianEnsemble(models)
        self.bayesian_ensemble.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        print("Initializing explainer...")
        self.explainer = shap.TreeExplainer(self.tabular_model.model)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using Bayesian ensemble"""
        return self.bayesian_ensemble.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions"""
        return self.bayesian_ensemble.predict(X)
    
    def explain_predictions(self, X: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Generate explanations for predictions"""
        
        # SHAP explanations
        if hasattr(self.tabular_model.model, 'predict'):
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Take positive class
        else:
            shap_values = np.zeros((X.shape[0], X.shape[1]))
        
        # Regulatory reason codes
        reason_codes = self.regulatory_model.get_reason_codes(X, top_n)
        
        explanations = []
        for i in range(X.shape[0]):
            # Top SHAP features
            feature_impacts = []
            shap_row = shap_values[i]
            top_shap_idx = np.argsort(np.abs(shap_row))[-top_n:]
            
            for idx in reversed(top_shap_idx):
                feature_impacts.append({
                    'feature': X.columns[idx],
                    'shap_value': shap_row[idx],
                    'feature_value': X.iloc[i, idx]
                })
            
            explanations.append({
                'shap_explanations': feature_impacts,
                'regulatory_reasons': reason_codes[i] if i < len(reason_codes) else [],
                'prediction_confidence': self.predict_proba(X.iloc[[i]])[:, 1][0]
            })
        
        return explanations
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Comprehensive model evaluation"""
        
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        ap_score = average_precision_score(y_test, y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        # Workload metrics
        workload = np.mean(y_pred)
        recall_at_workload = np.mean(y_test[y_pred == 1]) if np.sum(y_pred) > 0 else 0
        
        return {
            'auc': auc_score,
            'pr_auc': ap_score,
            'workload': workload,
            'recall_at_workload': recall_at_workload,
            'precision': np.mean(y_test[y_pred == 1]) if np.sum(y_pred) > 0 else 0,
            'optimal_threshold': self.bayesian_ensemble.optimal_threshold
        }
    
    def save_models(self, filepath: str):
        """Save the complete ensemble"""
        models = {
            'regulatory_model': self.regulatory_model,
            'tabular_model': self.tabular_model,
            'anomaly_model': self.anomaly_model,
            'bayesian_ensemble': self.bayesian_ensemble
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
    
    def load_models(self, filepath: str):
        """Load the complete ensemble"""
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
        
        self.regulatory_model = models['regulatory_model']
        self.tabular_model = models['tabular_model']
        self.anomaly_model = models['anomaly_model']
        self.bayesian_ensemble = models['bayesian_ensemble']
        
        # Reinitialize explainer
        if hasattr(self.tabular_model, 'model'):
            self.explainer = shap.TreeExplainer(self.tabular_model.model)

if __name__ == "__main__":
    # Example usage
    print("AML 360º Ensemble System - Ready for Training")
    
    # Initialize system
    aml_system = AMLEnsembleSystem()
    
    print("System initialized. Use aml_system.fit(X_train, y_train) to start training.")
    print("Features expected: behavioral, graph, sequential, and entity features from BigQuery")
