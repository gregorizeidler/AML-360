"""
AML 360º - Causal Inference Framework
Advanced causal analysis for understanding AML risk factors and treatment effects
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

class CausalInferenceAML:
    """
    Causal inference framework for AML risk factor analysis
    """
    
    def __init__(self):
        self.propensity_model = None
        self.outcome_model = None
        self.scaler = StandardScaler()
        
    def estimate_propensity_scores(self, 
                                 X: pd.DataFrame, 
                                 treatment: pd.Series,
                                 method: str = 'logistic') -> np.ndarray:
        """
        Estimate propensity scores e(x) = P(T=1|X=x)
        
        Formula: e(x) = P(T=1|X=x) = 1 / (1 + exp(-β'x))  [logistic]
        """
        
        if method == 'logistic':
            self.propensity_model = LogisticRegression(
                penalty='l2', 
                class_weight='balanced',
                random_state=42
            )
        elif method == 'random_forest':
            self.propensity_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        X_scaled = self.scaler.fit_transform(X)
        self.propensity_model.fit(X_scaled, treatment)
        
        propensity_scores = self.propensity_model.predict_proba(X_scaled)[:, 1]
        
        # Avoid extreme propensity scores (trim at 0.01 and 0.99)
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
        
        return propensity_scores
    
    def calculate_ate_ipw(self, 
                         treatment: np.ndarray, 
                         outcome: np.ndarray,
                         propensity_scores: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """
        Average Treatment Effect using Inverse Probability Weighting
        
        Formula: τ̂_IPW = (1/n) Σ[T_i*Y_i/e(X_i) - (1-T_i)*Y_i/(1-e(X_i))]
        """
        
        # IPW weights
        weights = treatment / propensity_scores + (1 - treatment) / (1 - propensity_scores)
        
        # Weighted outcomes
        weighted_treated = (treatment * outcome) / propensity_scores
        weighted_control = ((1 - treatment) * outcome) / (1 - propensity_scores)
        
        # ATE estimate
        ate = np.mean(weighted_treated) - np.mean(weighted_control)
        
        # Variance estimation (Horvitz-Thompson)
        var_treated = np.var(weighted_treated) / np.sum(treatment)
        var_control = np.var(weighted_control) / np.sum(1 - treatment)
        ate_se = np.sqrt(var_treated + var_control)
        
        # 95% Confidence interval
        ci_lower = ate - 1.96 * ate_se
        ci_upper = ate + 1.96 * ate_se
        
        return ate, ate_se, (ci_lower, ci_upper)
    
    def calculate_ate_doubly_robust(self, 
                                  X: pd.DataFrame,
                                  treatment: np.ndarray, 
                                  outcome: np.ndarray,
                                  propensity_scores: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """
        Doubly Robust Estimator for Average Treatment Effect
        
        Formula: τ̂_DR = (1/n) Σ[μ̂_1(X_i) - μ̂_0(X_i) + T_i(Y_i - μ̂_1(X_i))/e(X_i) - (1-T_i)(Y_i - μ̂_0(X_i))/(1-e(X_i))]
        """
        
        X_scaled = self.scaler.transform(X)
        
        # Fit outcome models for treated and control groups
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        # Outcome model for treated group
        if np.sum(treated_mask) > 0:
            outcome_model_treated = RandomForestRegressor(n_estimators=100, random_state=42)
            outcome_model_treated.fit(X_scaled[treated_mask], outcome[treated_mask])
            mu1_pred = outcome_model_treated.predict(X_scaled)
        else:
            mu1_pred = np.zeros(len(X))
            
        # Outcome model for control group  
        if np.sum(control_mask) > 0:
            outcome_model_control = RandomForestRegressor(n_estimators=100, random_state=42)
            outcome_model_control.fit(X_scaled[control_mask], outcome[control_mask])
            mu0_pred = outcome_model_control.predict(X_scaled)
        else:
            mu0_pred = np.zeros(len(X))
        
        # Doubly robust estimator
        dr_treated = treatment * (outcome - mu1_pred) / propensity_scores
        dr_control = (1 - treatment) * (outcome - mu0_pred) / (1 - propensity_scores)
        
        ate_dr = np.mean(mu1_pred - mu0_pred + dr_treated - dr_control)
        
        # Variance estimation (simplified)
        influence_function = (
            mu1_pred - mu0_pred + dr_treated - dr_control - ate_dr
        )
        ate_se = np.std(influence_function) / np.sqrt(len(influence_function))
        
        # 95% Confidence interval
        ci_lower = ate_dr - 1.96 * ate_se
        ci_upper = ate_dr + 1.96 * ate_se
        
        return ate_dr, ate_se, (ci_lower, ci_upper)
    
    def test_ignorability_assumption(self, 
                                   X: pd.DataFrame,
                                   treatment: np.ndarray,
                                   propensity_scores: np.ndarray) -> Dict[str, float]:
        """
        Test the ignorability assumption (no unmeasured confounders)
        Using covariate balance tests
        """
        
        results = {}
        
        # Standardized mean differences for each covariate
        smd_values = []
        
        for col in X.columns:
            treated_mean = X.loc[treatment == 1, col].mean()
            control_mean = X.loc[treatment == 0, col].mean()
            
            treated_var = X.loc[treatment == 1, col].var()
            control_var = X.loc[treatment == 0, col].var()
            
            pooled_std = np.sqrt((treated_var + control_var) / 2)
            
            if pooled_std > 0:
                smd = (treated_mean - control_mean) / pooled_std
            else:
                smd = 0
                
            smd_values.append(abs(smd))
        
        results['max_smd'] = max(smd_values)
        results['mean_smd'] = np.mean(smd_values)
        results['n_imbalanced'] = sum(1 for smd in smd_values if smd > 0.1)
        
        # Propensity score overlap
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]
        
        results['ps_overlap_min'] = max(min(treated_ps), min(control_ps))
        results['ps_overlap_max'] = min(max(treated_ps), max(control_ps))
        results['ps_overlap_range'] = results['ps_overlap_max'] - results['ps_overlap_min']
        
        # Two-sample KS test for propensity score distributions
        ks_stat, ks_p = stats.ks_2samp(treated_ps, control_ps)
        results['ps_ks_statistic'] = ks_stat
        results['ps_ks_pvalue'] = ks_p
        
        return results
    
    def sensitivity_analysis_rosenbaum(self, 
                                     treatment: np.ndarray,
                                     outcome: np.ndarray,
                                     propensity_scores: np.ndarray,
                                     gamma_range: List[float] = [1.0, 1.2, 1.5, 2.0, 2.5]) -> Dict[str, List[float]]:
        """
        Rosenbaum bounds for sensitivity analysis
        
        Tests how results change under different levels of unobserved confounding
        """
        
        results = {'gamma': gamma_range, 'p_values_upper': [], 'p_values_lower': []}
        
        # Sort by propensity scores and create matched pairs (simplified)
        sorted_indices = np.argsort(propensity_scores)
        
        # Simple matching (nearest neighbor)
        treated_indices = sorted_indices[treatment[sorted_indices] == 1]
        control_indices = sorted_indices[treatment[sorted_indices] == 0]
        
        n_matches = min(len(treated_indices), len(control_indices))
        
        if n_matches > 10:  # Need sufficient matches
            matched_treated = outcome[treated_indices[:n_matches]]
            matched_control = outcome[control_indices[:n_matches]]
            differences = matched_treated - matched_control
            
            for gamma in gamma_range:
                # Rosenbaum bounds calculation (simplified)
                # In practice, this would use more sophisticated methods
                
                # Upper bound (assuming confounding increases treatment probability)
                prob_upper = gamma / (1 + gamma)
                
                # Lower bound (assuming confounding decreases treatment probability)  
                prob_lower = 1 / (1 + gamma)
                
                # Approximate p-values using Wilcoxon signed-rank test
                if np.std(differences) > 0:
                    # Upper bound p-value
                    stat_upper, p_upper = stats.wilcoxon(differences, 
                                                       alternative='greater')
                    # Adjust for sensitivity parameter
                    p_upper = p_upper * (gamma / (gamma + 1))
                    
                    # Lower bound p-value
                    stat_lower, p_lower = stats.wilcoxon(differences, 
                                                       alternative='greater')
                    p_lower = p_lower * ((gamma + 1) / gamma)
                else:
                    p_upper = p_lower = 0.5
                
                results['p_values_upper'].append(p_upper)
                results['p_values_lower'].append(p_lower)
        else:
            # Not enough matches for sensitivity analysis
            results['p_values_upper'] = [np.nan] * len(gamma_range)
            results['p_values_lower'] = [np.nan] * len(gamma_range)
        
        return results
    
    def instrumental_variables_analysis(self, 
                                      X: pd.DataFrame,
                                      treatment: np.ndarray,
                                      outcome: np.ndarray,
                                      instrument: np.ndarray) -> Dict[str, float]:
        """
        Instrumental Variables (2SLS) estimation
        
        Two-Stage Least Squares:
        Stage 1: T = α₀ + α₁Z + α₂X + ε₁  (First stage)
        Stage 2: Y = β₀ + β₁T̂ + β₂X + ε₂  (Second stage)
        """
        
        results = {}
        
        # Test instrument relevance (First-stage F-test)
        X_with_instrument = np.column_stack([X, instrument])
        first_stage = LinearRegression()
        first_stage.fit(X_with_instrument, treatment)
        
        # F-statistic for instrument relevance
        treatment_pred = first_stage.predict(X_with_instrument)
        residuals_first = treatment - treatment_pred
        
        tss_first = np.sum((treatment - np.mean(treatment))**2)
        rss_first = np.sum(residuals_first**2)
        f_stat = ((tss_first - rss_first) / 1) / (rss_first / (len(treatment) - X.shape[1] - 2))
        
        results['first_stage_f_statistic'] = f_stat
        results['instrument_relevance'] = f_stat > 10  # Rule of thumb: F > 10
        
        # Test instrument exogeneity (correlation with outcome residuals)
        # This is not directly testable, but we can check reduced form
        reduced_form = LinearRegression()
        reduced_form.fit(X_with_instrument, outcome)
        outcome_pred_reduced = reduced_form.predict(X_with_instrument)
        
        # Coefficient on instrument in reduced form
        instrument_coef_reduced = reduced_form.coef_[-1]
        results['reduced_form_instrument_coef'] = instrument_coef_reduced
        
        # 2SLS estimate
        # Second stage: use predicted treatment from first stage
        second_stage = LinearRegression()
        X_second_stage = np.column_stack([X, treatment_pred])
        second_stage.fit(X_second_stage, outcome)
        
        # IV estimate (coefficient on predicted treatment)
        iv_estimate = second_stage.coef_[-1]
        results['iv_estimate'] = iv_estimate
        
        # Standard errors (simplified - in practice would use robust SEs)
        outcome_pred_second = second_stage.predict(X_second_stage)
        residuals_second = outcome - outcome_pred_second
        mse_second = np.mean(residuals_second**2)
        
        # Approximate standard error
        iv_se = np.sqrt(mse_second / len(outcome))
        results['iv_standard_error'] = iv_se
        
        # 95% Confidence interval
        results['iv_ci_lower'] = iv_estimate - 1.96 * iv_se
        results['iv_ci_upper'] = iv_estimate + 1.96 * iv_se
        
        return results
    
    def granger_causality_test(self, 
                             variable_x: np.ndarray, 
                             variable_y: np.ndarray,
                             max_lags: int = 5) -> Dict[str, Union[float, bool]]:
        """
        Granger causality test: Does X Granger-cause Y?
        
        H₀: Past values of X do not help predict Y beyond past values of Y alone
        H₁: Past values of X help predict Y
        """
        
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Prepare data (requires 2D array with [y, x] columns)
        data = np.column_stack([variable_y, variable_x])
        
        try:
            # Granger causality test
            gc_results = grangercausalitytests(data, max_lags, verbose=False)
            
            # Extract results for optimal lag
            best_lag = 1
            best_p_value = 1.0
            
            for lag in range(1, max_lags + 1):
                if lag in gc_results:
                    p_val = gc_results[lag][0]['ssr_ftest'][1]  # F-test p-value
                    if p_val < best_p_value:
                        best_p_value = p_val
                        best_lag = lag
            
            results = {
                'optimal_lag': best_lag,
                'p_value': best_p_value,
                'granger_causes': best_p_value < 0.05,
                'f_statistic': gc_results[best_lag][0]['ssr_ftest'][0] if best_lag in gc_results else np.nan
            }
            
        except Exception as e:
            results = {
                'optimal_lag': np.nan,
                'p_value': np.nan,
                'granger_causes': False,
                'f_statistic': np.nan,
                'error': str(e)
            }
        
        return results
    
    def mediation_analysis(self, 
                          X: pd.DataFrame,
                          treatment: np.ndarray,
                          mediator: np.ndarray,
                          outcome: np.ndarray) -> Dict[str, float]:
        """
        Mediation analysis: T → M → Y
        
        Total Effect: c = c' + ab
        Direct Effect: c'
        Indirect Effect: ab
        """
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Path a: Treatment → Mediator
        mediator_model = LinearRegression()
        X_with_treatment = np.column_stack([X_scaled, treatment])
        mediator_model.fit(X_with_treatment, mediator)
        coef_a = mediator_model.coef_[-1]  # Coefficient on treatment
        
        # Path b: Mediator → Outcome (controlling for treatment)
        outcome_model = LinearRegression()
        X_with_treatment_mediator = np.column_stack([X_scaled, treatment, mediator])
        outcome_model.fit(X_with_treatment_mediator, outcome)
        coef_b = outcome_model.coef_[-1]  # Coefficient on mediator
        coef_c_prime = outcome_model.coef_[-2]  # Direct effect of treatment
        
        # Path c: Total effect (Treatment → Outcome)
        total_effect_model = LinearRegression()
        total_effect_model.fit(X_with_treatment, outcome)
        coef_c = total_effect_model.coef_[-1]  # Total effect
        
        # Mediation effects
        indirect_effect = coef_a * coef_b  # ab
        direct_effect = coef_c_prime       # c'
        total_effect = coef_c              # c
        
        # Proportion mediated
        prop_mediated = indirect_effect / total_effect if total_effect != 0 else 0
        
        results = {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'proportion_mediated': prop_mediated,
            'path_a_coefficient': coef_a,
            'path_b_coefficient': coef_b,
            'path_c_coefficient': coef_c,
            'path_c_prime_coefficient': coef_c_prime
        }
        
        return results
    
    def plot_causal_diagnostics(self, 
                              X: pd.DataFrame,
                              treatment: np.ndarray,
                              outcome: np.ndarray,
                              propensity_scores: np.ndarray) -> plt.Figure:
        """
        Generate diagnostic plots for causal analysis
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Causal Inference Diagnostic Plots', fontsize=16, fontweight='bold')
        
        # 1. Propensity score distribution
        axes[0, 0].hist(propensity_scores[treatment == 1], alpha=0.7, bins=30, 
                       label='Treated', density=True)
        axes[0, 0].hist(propensity_scores[treatment == 0], alpha=0.7, bins=30, 
                       label='Control', density=True)
        axes[0, 0].set_xlabel('Propensity Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Propensity Score Overlap')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Covariate balance (before matching)
        smd_values = []
        feature_names = []
        
        for col in X.columns:
            treated_mean = X.loc[treatment == 1, col].mean()
            control_mean = X.loc[treatment == 0, col].mean()
            
            treated_var = X.loc[treatment == 1, col].var()
            control_var = X.loc[treatment == 0, col].var()
            
            pooled_std = np.sqrt((treated_var + control_var) / 2)
            
            if pooled_std > 0:
                smd = abs((treated_mean - control_mean) / pooled_std)
            else:
                smd = 0
                
            smd_values.append(smd)
            feature_names.append(col)
        
        y_pos = np.arange(len(feature_names))
        axes[0, 1].barh(y_pos, smd_values)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(feature_names)
        axes[0, 1].axvline(x=0.1, color='red', linestyle='--', label='SMD = 0.1')
        axes[0, 1].set_xlabel('Standardized Mean Difference')
        axes[0, 1].set_title('Covariate Balance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Outcome vs Propensity Score
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        axes[0, 2].scatter(propensity_scores[treated_mask], outcome[treated_mask], 
                          alpha=0.6, label='Treated', s=20)
        axes[0, 2].scatter(propensity_scores[control_mask], outcome[control_mask], 
                          alpha=0.6, label='Control', s=20)
        axes[0, 2].set_xlabel('Propensity Score')
        axes[0, 2].set_ylabel('Outcome')
        axes[0, 2].set_title('Outcome vs Propensity Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Propensity score bins analysis
        n_bins = 5
        bin_edges = np.percentile(propensity_scores, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        treated_means = []
        control_means = []
        
        for i in range(n_bins):
            mask = (propensity_scores >= bin_edges[i]) & (propensity_scores < bin_edges[i + 1])
            if i == n_bins - 1:  # Include upper bound in last bin
                mask = (propensity_scores >= bin_edges[i]) & (propensity_scores <= bin_edges[i + 1])
            
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_centers.append(bin_center)
            
            treated_in_bin = outcome[(mask) & (treatment == 1)]
            control_in_bin = outcome[(mask) & (treatment == 0)]
            
            treated_means.append(np.mean(treated_in_bin) if len(treated_in_bin) > 0 else np.nan)
            control_means.append(np.mean(control_in_bin) if len(control_in_bin) > 0 else np.nan)
        
        axes[1, 0].plot(bin_centers, treated_means, 'o-', label='Treated', linewidth=2)
        axes[1, 0].plot(bin_centers, control_means, 's-', label='Control', linewidth=2)
        axes[1, 0].set_xlabel('Propensity Score Bin Center')
        axes[1, 0].set_ylabel('Mean Outcome')
        axes[1, 0].set_title('Outcome by Propensity Score Bins')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Residual plots
        # Simple residual analysis
        y_pred_treated = np.mean(outcome[treated_mask]) * np.ones(sum(treated_mask))
        y_pred_control = np.mean(outcome[control_mask]) * np.ones(sum(control_mask))
        
        residuals_treated = outcome[treated_mask] - y_pred_treated
        residuals_control = outcome[control_mask] - y_pred_control
        
        axes[1, 1].scatter(propensity_scores[treated_mask], residuals_treated, 
                          alpha=0.6, label='Treated')
        axes[1, 1].scatter(propensity_scores[control_mask], residuals_control, 
                          alpha=0.6, label='Control')
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Propensity Score')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Treatment effect heterogeneity
        # Estimate treatment effects in different propensity score quantiles
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        ps_quantiles = np.quantile(propensity_scores, quantiles)
        treatment_effects = []
        
        for i in range(len(quantiles) - 1):
            mask = (propensity_scores >= ps_quantiles[i]) & (propensity_scores < ps_quantiles[i + 1])
            
            treated_outcome = np.mean(outcome[mask & (treatment == 1)]) if np.sum(mask & (treatment == 1)) > 0 else np.nan
            control_outcome = np.mean(outcome[mask & (treatment == 0)]) if np.sum(mask & (treatment == 0)) > 0 else np.nan
            
            te = treated_outcome - control_outcome
            treatment_effects.append(te)
        
        quantile_centers = [(quantiles[i] + quantiles[i+1])/2 for i in range(len(quantiles)-1)]
        axes[1, 2].bar(quantile_centers, treatment_effects, width=0.2, alpha=0.7)
        axes[1, 2].axhline(y=0, color='red', linestyle='--')
        axes[1, 2].set_xlabel('Propensity Score Quantile')
        axes[1, 2].set_ylabel('Treatment Effect')
        axes[1, 2].set_title('Treatment Effect Heterogeneity')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    print("🔬 AML 360° Causal Inference Framework")
    print("Advanced causal analysis for AML risk factors")
    
    # Initialize causal inference framework
    causal_analyzer = CausalInferenceAML()
    
    print("✅ Causal inference framework ready for AML analysis!")
    print("📊 Available methods:")
    print("  • Propensity Score Estimation")
    print("  • Average Treatment Effect (IPW & Doubly Robust)")
    print("  • Sensitivity Analysis (Rosenbaum Bounds)")
    print("  • Instrumental Variables")
    print("  • Granger Causality")
    print("  • Mediation Analysis")
