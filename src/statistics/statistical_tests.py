"""
AML 360º - Advanced Statistical Testing Framework
Rigorous statistical validation and hypothesis testing for AML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from scipy import stats
from scipy.special import gammaln, digamma
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical libraries
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.diagnostic import het_breuschpagan
from arch import arch_model
from arch.unitroot import ADF, KPSS

# For bootstrap and permutation tests
from sklearn.utils import resample
from scipy.stats import kstest, anderson, jarque_bera
import pingouin as pg

@dataclass
class StatisticalTestResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    critical_values: Optional[Dict[str, float]] = None
    conclusion: str = ""
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Optional[Dict[str, Any]] = None

class AdvancedStatisticalTests:
    """
    Comprehensive statistical testing suite for AML model validation
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.test_results = []
    
    def test_population_stability_index(self, 
                                      expected_dist: np.ndarray, 
                                      actual_dist: np.ndarray) -> StatisticalTestResult:
        """
        Population Stability Index (PSI) test with statistical significance
        
        PSI = Σ[(Actual_i - Expected_i) × ln(Actual_i / Expected_i)]
        """
        
        # Avoid division by zero
        expected_dist = np.maximum(expected_dist, 1e-8)
        actual_dist = np.maximum(actual_dist, 1e-8)
        
        # Calculate PSI
        psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
        
        # Bootstrap confidence interval for PSI
        n_bootstrap = 1000
        psi_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample
            indices = np.random.choice(len(actual_dist), size=len(actual_dist), replace=True)
            actual_boot = actual_dist[indices]
            actual_boot = actual_boot / actual_boot.sum()  # Renormalize
            
            psi_boot = np.sum((actual_boot - expected_dist) * np.log(actual_boot / expected_dist))
            psi_bootstrap.append(psi_boot)
        
        psi_bootstrap = np.array(psi_bootstrap)
        ci_lower = np.percentile(psi_bootstrap, 2.5)
        ci_upper = np.percentile(psi_bootstrap, 97.5)
        
        # Interpretation
        if psi < 0.1:
            conclusion = "No significant population change (PSI < 0.1)"
        elif psi < 0.25:
            conclusion = "Minor population change detected (0.1 ≤ PSI < 0.25)"
        else:
            conclusion = "Major population shift detected (PSI ≥ 0.25)"
        
        # Approximate p-value using bootstrap distribution
        p_value = np.mean(np.abs(psi_bootstrap) >= np.abs(psi))
        
        return StatisticalTestResult(
            test_name="Population Stability Index",
            statistic=psi,
            p_value=p_value,
            conclusion=conclusion,
            confidence_interval=(ci_lower, ci_upper),
            critical_values={"minor_change": 0.1, "major_change": 0.25}
        )
    
    def test_kolmogorov_smirnov_drift(self, 
                                    baseline: np.ndarray, 
                                    current: np.ndarray) -> StatisticalTestResult:
        """
        Two-sample Kolmogorov-Smirnov test for distribution drift
        
        D_n,m = sup_x |F_1,n(x) - F_2,m(x)|
        """
        
        statistic, p_value = stats.ks_2samp(baseline, current)
        
        # Effect size (Cohen's d equivalent for distributions)
        effect_size = np.abs(np.mean(baseline) - np.mean(current)) / np.sqrt(
            (np.var(baseline) + np.var(current)) / 2
        )
        
        conclusion = (
            f"Significant distribution drift detected (p={p_value:.4f})" 
            if p_value < self.alpha 
            else f"No significant distribution drift (p={p_value:.4f})"
        )
        
        return StatisticalTestResult(
            test_name="Kolmogorov-Smirnov Drift Test",
            statistic=statistic,
            p_value=p_value,
            conclusion=conclusion,
            effect_size=effect_size,
            critical_values={"alpha": self.alpha}
        )
    
    def test_jensen_shannon_divergence(self, 
                                     P: np.ndarray, 
                                     Q: np.ndarray) -> StatisticalTestResult:
        """
        Jensen-Shannon Divergence with statistical significance testing
        
        JSD(P||Q) = 1/2 * KL(P||M) + 1/2 * KL(Q||M), where M = (P+Q)/2
        """
        
        # Normalize to probabilities
        P = P / P.sum()
        Q = Q / Q.sum()
        M = (P + Q) / 2
        
        # Avoid log(0)
        P = np.maximum(P, 1e-10)
        Q = np.maximum(Q, 1e-10)
        M = np.maximum(M, 1e-10)
        
        # Calculate JSD
        kl_pm = np.sum(P * np.log(P / M))
        kl_qm = np.sum(Q * np.log(Q / M))
        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        
        # Bootstrap for confidence interval
        n_bootstrap = 1000
        jsd_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            p_boot = resample(P, n_samples=len(P))
            q_boot = resample(Q, n_samples=len(Q))
            
            p_boot = p_boot / p_boot.sum()
            q_boot = q_boot / q_boot.sum()
            m_boot = (p_boot + q_boot) / 2
            
            kl_pm_boot = np.sum(p_boot * np.log(p_boot / m_boot))
            kl_qm_boot = np.sum(q_boot * np.log(q_boot / m_boot))
            jsd_boot = 0.5 * kl_pm_boot + 0.5 * kl_qm_boot
            
            jsd_bootstrap.append(jsd_boot)
        
        jsd_bootstrap = np.array(jsd_bootstrap)
        ci_lower = np.percentile(jsd_bootstrap, 2.5)
        ci_upper = np.percentile(jsd_bootstrap, 97.5)
        
        # Approximate p-value (JSD > 0.1 considered significant drift)
        p_value = np.mean(jsd_bootstrap > 0.1)
        
        conclusion = (
            f"Significant distributional difference (JSD={jsd:.4f})" 
            if jsd > 0.1 
            else f"Distributions are similar (JSD={jsd:.4f})"
        )
        
        return StatisticalTestResult(
            test_name="Jensen-Shannon Divergence",
            statistic=jsd,
            p_value=p_value,
            conclusion=conclusion,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def test_garch_effects(self, returns: np.ndarray) -> StatisticalTestResult:
        """
        Test for GARCH effects using Lagrange Multiplier test
        """
        
        # Fit GARCH(1,1) model
        try:
            garch_model = arch_model(returns, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            
            # Extract results
            llf = garch_fit.loglikelihood
            params = garch_fit.params
            
            # LM test for ARCH effects
            residuals = garch_fit.resid / garch_fit.conditional_volatility
            residuals_squared = residuals ** 2
            
            # Regression of squared residuals on lagged squared residuals
            from sklearn.linear_model import LinearRegression
            X = residuals_squared[:-1].reshape(-1, 1)
            y = residuals_squared[1:]
            
            reg = LinearRegression().fit(X, y)
            r_squared = reg.score(X, y)
            
            # LM statistic
            n = len(y)
            lm_stat = n * r_squared
            p_value = 1 - stats.chi2.cdf(lm_stat, df=1)
            
            conclusion = (
                f"GARCH effects detected (p={p_value:.4f})" 
                if p_value < self.alpha 
                else f"No significant GARCH effects (p={p_value:.4f})"
            )
            
            return StatisticalTestResult(
                test_name="GARCH Effects Test",
                statistic=lm_stat,
                p_value=p_value,
                conclusion=conclusion,
                additional_info={
                    "omega": params.get('omega', np.nan),
                    "alpha": params.get('alpha[1]', np.nan),
                    "beta": params.get('beta[1]', np.nan),
                    "log_likelihood": llf
                }
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name="GARCH Effects Test",
                statistic=np.nan,
                p_value=np.nan,
                conclusion=f"GARCH test failed: {str(e)}"
            )
    
    def test_extreme_value_theory(self, 
                                data: np.ndarray, 
                                threshold_percentile: float = 95) -> StatisticalTestResult:
        """
        Test for GPD fit in extreme value analysis
        """
        
        threshold = np.percentile(data, threshold_percentile)
        excesses = data[data > threshold] - threshold
        
        if len(excesses) < 10:
            return StatisticalTestResult(
                test_name="Extreme Value Theory Test",
                statistic=np.nan,
                p_value=np.nan,
                conclusion="Insufficient extreme values for EVT analysis"
            )
        
        try:
            # Fit GPD
            params = stats.genpareto.fit(excesses)
            xi, _, sigma = params
            
            # Goodness of fit test (Anderson-Darling)
            ad_stat, ad_crit, ad_sig = stats.anderson(excesses, dist='genpareto')
            
            # KS test for GPD
            ks_stat, ks_p = stats.kstest(excesses, 
                                       lambda x: stats.genpareto.cdf(x, *params))
            
            conclusion = (
                f"GPD provides good fit (KS p={ks_p:.4f}, ξ={xi:.3f})" 
                if ks_p > self.alpha 
                else f"GPD fit questionable (KS p={ks_p:.4f})"
            )
            
            return StatisticalTestResult(
                test_name="Extreme Value Theory (GPD Fit)",
                statistic=ks_stat,
                p_value=ks_p,
                conclusion=conclusion,
                additional_info={
                    "shape_parameter_xi": xi,
                    "scale_parameter_sigma": sigma,
                    "threshold": threshold,
                    "n_excesses": len(excesses),
                    "anderson_darling_stat": ad_stat
                }
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name="Extreme Value Theory Test",
                statistic=np.nan,
                p_value=np.nan,
                conclusion=f"EVT test failed: {str(e)}"
            )
    
    def test_burstiness_significance(self, 
                                   event_times: np.ndarray) -> StatisticalTestResult:
        """
        Test statistical significance of burstiness using Fano factor
        """
        
        # Calculate inter-event intervals
        intervals = np.diff(sorted(event_times))
        
        # Fano factor
        fano_factor = np.var(intervals) / np.mean(intervals)
        
        # Under Poisson null hypothesis, Fano factor should be ~1
        # Test against Poisson using overdispersion test
        n = len(intervals)
        
        # Asymptotic test: (n-1) * Fano_factor ~ Chi-square(n-1) under Poisson
        test_statistic = (n - 1) * fano_factor
        p_value = 2 * min(
            stats.chi2.cdf(test_statistic, df=n-1),
            1 - stats.chi2.cdf(test_statistic, df=n-1)
        )
        
        if fano_factor > 1.5:
            conclusion = f"Significant burstiness detected (Fano={fano_factor:.3f})"
        elif fano_factor < 0.67:
            conclusion = f"Significant regularity detected (Fano={fano_factor:.3f})"
        else:
            conclusion = f"Process appears Poisson-like (Fano={fano_factor:.3f})"
        
        # Clark-Evans index
        if len(event_times) > 1:
            # Nearest neighbor distances
            sorted_times = np.sort(event_times)
            nn_distances = np.diff(sorted_times)
            mean_nn = np.mean(nn_distances)
            
            # Expected under CSR
            density = len(event_times) / (event_times.max() - event_times.min())
            expected_nn = 1 / (2 * density)
            
            clark_evans = mean_nn / expected_nn if expected_nn > 0 else np.nan
        else:
            clark_evans = np.nan
        
        return StatisticalTestResult(
            test_name="Burstiness Significance Test",
            statistic=fano_factor,
            p_value=p_value,
            conclusion=conclusion,
            additional_info={
                "clark_evans_index": clark_evans,
                "mean_interval": np.mean(intervals),
                "var_interval": np.var(intervals),
                "n_intervals": n
            }
        )
    
    def test_model_stability_mcnemar(self, 
                                   y_true: np.ndarray, 
                                   y_pred1: np.ndarray, 
                                   y_pred2: np.ndarray) -> StatisticalTestResult:
        """
        McNemar test for comparing two model predictions
        """
        
        # Create contingency table
        both_correct = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
        model1_correct = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
        model2_correct = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
        both_incorrect = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
        
        # McNemar test focuses on discordant pairs
        b = model1_correct
        c = model2_correct
        
        if b + c < 10:
            # Exact binomial test for small samples
            p_value = 2 * stats.binom.cdf(min(b, c), b + c, 0.5)
            statistic = min(b, c)
        else:
            # Chi-square approximation
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        conclusion = (
            f"Significant difference between models (p={p_value:.4f})" 
            if p_value < self.alpha 
            else f"No significant difference between models (p={p_value:.4f})"
        )
        
        return StatisticalTestResult(
            test_name="McNemar Test (Model Comparison)",
            statistic=statistic,
            p_value=p_value,
            conclusion=conclusion,
            additional_info={
                "both_correct": both_correct,
                "model1_only_correct": model1_correct,
                "model2_only_correct": model2_correct,
                "both_incorrect": both_incorrect,
                "discordant_pairs": b + c
            }
        )
    
    def test_feature_importance_stability(self, 
                                        importance_matrix: np.ndarray) -> StatisticalTestResult:
        """
        Test stability of feature importance across CV folds
        Uses coefficient of variation and Friedman test
        """
        
        n_features, n_folds = importance_matrix.shape
        
        # Coefficient of variation for each feature
        cv_scores = np.std(importance_matrix, axis=1) / np.mean(importance_matrix, axis=1)
        mean_cv = np.mean(cv_scores)
        
        # Friedman test for consistent ranking across folds
        friedman_stat, friedman_p = stats.friedmanchisquare(*importance_matrix)
        
        # Kendall's W (coefficient of concordance)
        n = n_folds
        k = n_features
        
        # Rank features within each fold
        ranks = np.apply_along_axis(stats.rankdata, axis=0, arr=-importance_matrix)
        rank_sums = np.sum(ranks, axis=1)
        
        # Kendall's W
        mean_rank = (n + 1) / 2
        s = np.sum((rank_sums - n * mean_rank) ** 2)
        w = 12 * s / (n ** 2 * (k ** 3 - k))
        
        conclusion = (
            f"Feature importance is {'stable' if mean_cv < 0.5 else 'unstable'} "
            f"(mean CV = {mean_cv:.3f}, Kendall's W = {w:.3f})"
        )
        
        return StatisticalTestResult(
            test_name="Feature Importance Stability",
            statistic=w,
            p_value=friedman_p,
            conclusion=conclusion,
            additional_info={
                "mean_coefficient_variation": mean_cv,
                "friedman_statistic": friedman_stat,
                "kendalls_w": w,
                "most_stable_feature": np.argmin(cv_scores),
                "least_stable_feature": np.argmax(cv_scores)
            }
        )
    
    def comprehensive_model_validation(self, 
                                     y_true: np.ndarray,
                                     y_pred_proba: np.ndarray,
                                     feature_data: np.ndarray,
                                     baseline_predictions: Optional[np.ndarray] = None) -> Dict[str, StatisticalTestResult]:
        """
        Comprehensive statistical validation suite
        """
        
        results = {}
        
        # 1. Calibration test (Hosmer-Lemeshow)
        results['calibration'] = self._test_calibration_hosmer_lemeshow(y_true, y_pred_proba)
        
        # 2. Discrimination test (DeLong test for AUC)
        if baseline_predictions is not None:
            results['discrimination'] = self._test_auc_comparison_delong(
                y_true, y_pred_proba, baseline_predictions
            )
        
        # 3. Residual analysis
        results['residuals'] = self._test_residual_patterns(y_true, y_pred_proba)
        
        # 4. Feature multicollinearity
        results['multicollinearity'] = self._test_multicollinearity_vif(feature_data)
        
        # 5. Homoscedasticity
        results['homoscedasticity'] = self._test_homoscedasticity(y_true, y_pred_proba)
        
        return results
    
    def _test_calibration_hosmer_lemeshow(self, 
                                        y_true: np.ndarray, 
                                        y_pred_proba: np.ndarray,
                                        n_bins: int = 10) -> StatisticalTestResult:
        """Hosmer-Lemeshow goodness-of-fit test"""
        
        # Create bins based on predicted probabilities
        bin_edges = np.percentile(y_pred_proba, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] = 1.0  # Ensure last bin includes 1.0
        
        observed = []
        expected = []
        total = []
        
        for i in range(n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if i == n_bins - 1:  # Include upper bound in last bin
                mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
            
            if np.sum(mask) == 0:
                continue
                
            obs = np.sum(y_true[mask])
            exp = np.sum(y_pred_proba[mask])
            tot = np.sum(mask)
            
            observed.append(obs)
            expected.append(exp)
            total.append(tot)
        
        observed = np.array(observed)
        expected = np.array(expected)
        total = np.array(total)
        
        # Chi-square statistic
        chi2_stat = np.sum((observed - expected) ** 2 / (expected * (1 - expected / total)))
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=len(observed) - 2)
        
        conclusion = (
            f"Model is well-calibrated (p={p_value:.4f})" 
            if p_value > self.alpha 
            else f"Model calibration is poor (p={p_value:.4f})"
        )
        
        return StatisticalTestResult(
            test_name="Hosmer-Lemeshow Calibration Test",
            statistic=chi2_stat,
            p_value=p_value,
            conclusion=conclusion
        )
    
    def _test_auc_comparison_delong(self, 
                                  y_true: np.ndarray,
                                  y_pred1: np.ndarray, 
                                  y_pred2: np.ndarray) -> StatisticalTestResult:
        """DeLong test for comparing two AUC scores"""
        
        from sklearn.metrics import roc_auc_score
        
        auc1 = roc_auc_score(y_true, y_pred1)
        auc2 = roc_auc_score(y_true, y_pred2)
        
        # Simplified DeLong test implementation
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)
        
        # This is a simplified version - full DeLong test is more complex
        se_diff = np.sqrt((auc1 * (1 - auc1)) / n1 + (auc2 * (1 - auc2)) / n0)
        z_stat = (auc1 - auc2) / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        conclusion = (
            f"Significant AUC difference (AUC1={auc1:.3f} vs AUC2={auc2:.3f}, p={p_value:.4f})" 
            if p_value < self.alpha 
            else f"No significant AUC difference (p={p_value:.4f})"
        )
        
        return StatisticalTestResult(
            test_name="DeLong AUC Comparison",
            statistic=z_stat,
            p_value=p_value,
            conclusion=conclusion,
            additional_info={"auc1": auc1, "auc2": auc2}
        )
    
    def _test_residual_patterns(self, 
                              y_true: np.ndarray, 
                              y_pred_proba: np.ndarray) -> StatisticalTestResult:
        """Test for patterns in residuals"""
        
        # Pearson residuals for logistic regression
        residuals = (y_true - y_pred_proba) / np.sqrt(y_pred_proba * (1 - y_pred_proba))
        
        # Runs test for randomness
        median_resid = np.median(residuals)
        runs, n1, n2 = 0, 0, 0
        
        above_median = residuals > median_resid
        for i in range(1, len(above_median)):
            if above_median[i] != above_median[i-1]:
                runs += 1
        
        n1 = np.sum(above_median)
        n2 = len(above_median) - n1
        
        # Expected runs and variance under null hypothesis
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        z_stat = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        conclusion = (
            f"Residuals show significant patterns (p={p_value:.4f})" 
            if p_value < self.alpha 
            else f"Residuals appear random (p={p_value:.4f})"
        )
        
        return StatisticalTestResult(
            test_name="Residual Randomness Test",
            statistic=z_stat,
            p_value=p_value,
            conclusion=conclusion,
            additional_info={
                "observed_runs": runs,
                "expected_runs": expected_runs,
                "mean_residual": np.mean(residuals),
                "std_residual": np.std(residuals)
            }
        )
    
    def _test_multicollinearity_vif(self, X: np.ndarray) -> StatisticalTestResult:
        """Variance Inflation Factor test for multicollinearity"""
        
        from sklearn.linear_model import LinearRegression
        
        n_features = X.shape[1]
        vif_scores = []
        
        for i in range(n_features):
            # Regress feature i on all other features
            y = X[:, i]
            X_others = np.delete(X, i, axis=1)
            
            try:
                reg = LinearRegression().fit(X_others, y)
                r_squared = reg.score(X_others, y)
                vif = 1 / (1 - r_squared) if r_squared < 0.999 else np.inf
                vif_scores.append(vif)
            except:
                vif_scores.append(np.nan)
        
        vif_scores = np.array(vif_scores)
        max_vif = np.nanmax(vif_scores)
        mean_vif = np.nanmean(vif_scores)
        
        # VIF > 5 indicates multicollinearity, VIF > 10 indicates severe multicollinearity
        if max_vif > 10:
            conclusion = f"Severe multicollinearity detected (max VIF = {max_vif:.2f})"
        elif max_vif > 5:
            conclusion = f"Multicollinearity detected (max VIF = {max_vif:.2f})"
        else:
            conclusion = f"Low multicollinearity (max VIF = {max_vif:.2f})"
        
        return StatisticalTestResult(
            test_name="Variance Inflation Factor",
            statistic=max_vif,
            p_value=np.nan,  # VIF doesn't have a p-value
            conclusion=conclusion,
            additional_info={
                "mean_vif": mean_vif,
                "vif_scores": vif_scores.tolist(),
                "n_high_vif_features": np.sum(vif_scores > 5)
            }
        )
    
    def _test_homoscedasticity(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray) -> StatisticalTestResult:
        """Breusch-Pagan test for homoscedasticity"""
        
        residuals = y_true - y_pred
        
        # Simple Breusch-Pagan test
        # Regress squared residuals on predicted values
        from sklearn.linear_model import LinearRegression
        
        reg = LinearRegression().fit(y_pred.reshape(-1, 1), residuals ** 2)
        r_squared = reg.score(y_pred.reshape(-1, 1), residuals ** 2)
        
        # BP statistic
        n = len(residuals)
        bp_stat = n * r_squared
        p_value = 1 - stats.chi2.cdf(bp_stat, df=1)
        
        conclusion = (
            f"Heteroscedasticity detected (p={p_value:.4f})" 
            if p_value < self.alpha 
            else f"Homoscedasticity confirmed (p={p_value:.4f})"
        )
        
        return StatisticalTestResult(
            test_name="Breusch-Pagan Homoscedasticity Test",
            statistic=bp_stat,
            p_value=p_value,
            conclusion=conclusion
        )
    
    def generate_statistical_report(self, results: List[StatisticalTestResult]) -> str:
        """Generate comprehensive statistical validation report"""
        
        report = ["=" * 80]
        report.append("STATISTICAL VALIDATION REPORT - AML 360°")
        report.append("=" * 80)
        report.append(f"Significance Level: α = {self.alpha}")
        report.append(f"Total Tests Performed: {len(results)}")
        report.append("")
        
        significant_tests = [r for r in results if r.p_value is not None and r.p_value < self.alpha]
        report.append(f"Significant Results: {len(significant_tests)}")
        report.append("")
        
        for i, result in enumerate(results, 1):
            report.append(f"{i}. {result.test_name}")
            report.append("-" * 50)
            report.append(f"   Statistic: {result.statistic:.4f}")
            if result.p_value is not None:
                report.append(f"   P-value: {result.p_value:.6f}")
            if result.effect_size is not None:
                report.append(f"   Effect Size: {result.effect_size:.4f}")
            if result.confidence_interval is not None:
                ci = result.confidence_interval
                report.append(f"   95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
            report.append(f"   Conclusion: {result.conclusion}")
            if result.additional_info:
                report.append("   Additional Information:")
                for key, value in result.additional_info.items():
                    if isinstance(value, (int, float)):
                        report.append(f"     {key}: {value:.4f}")
                    else:
                        report.append(f"     {key}: {value}")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF STATISTICAL REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    print("🔬 AML 360° Statistical Testing Framework")
    print("Advanced statistical validation for AML models")
    
    # Initialize tester
    tester = AdvancedStatisticalTests(alpha=0.05)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    baseline_dist = np.random.dirichlet(np.ones(10), size=1)[0]
    current_dist = baseline_dist + np.random.normal(0, 0.05, 10)
    current_dist = np.abs(current_dist)
    current_dist = current_dist / current_dist.sum()
    
    # Run PSI test
    psi_result = tester.test_population_stability_index(baseline_dist, current_dist)
    print(f"PSI Test Result: {psi_result.conclusion}")
    
    print("\n✅ Statistical testing framework ready for AML model validation!")
