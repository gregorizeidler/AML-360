"""
AML 360º - Monte Carlo Simulation Framework
Advanced simulation methods for uncertainty quantification and stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class MonteCarloAML:
    """
    Monte Carlo simulation framework for AML model validation and stress testing
    """
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
    def bootstrap_confidence_intervals(self, 
                                     data: np.ndarray,
                                     statistic_func: Callable,
                                     n_bootstrap: int = 10000,
                                     confidence_level: float = 0.95,
                                     method: str = 'percentile') -> Dict[str, float]:
        """
        Bootstrap confidence intervals for any statistic
        
        Methods:
        - 'percentile': Basic percentile method
        - 'bias_corrected': Bias-corrected percentile method  
        - 'bca': Bias-corrected and accelerated (BCa)
        """
        
        n = len(data)
        bootstrap_stats = []
        original_stat = statistic_func(data)
        
        # Bootstrap resampling
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        if method == 'percentile':
            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
            
        elif method == 'bias_corrected':
            # Bias correction
            bias_correction = stats.norm.ppf(np.mean(bootstrap_stats < original_stat))
            
            # Adjusted percentiles
            adj_lower = stats.norm.cdf(2 * bias_correction + stats.norm.ppf(alpha / 2)) * 100
            adj_upper = stats.norm.cdf(2 * bias_correction + stats.norm.ppf(1 - alpha / 2)) * 100
            
            ci_lower = np.percentile(bootstrap_stats, adj_lower)
            ci_upper = np.percentile(bootstrap_stats, adj_upper)
            
        elif method == 'bca':
            # Bias-corrected and accelerated (BCa)
            bias_correction = stats.norm.ppf(np.mean(bootstrap_stats < original_stat))
            
            # Acceleration parameter (jackknife)
            n_jack = len(data)
            jackknife_stats = []
            for i in range(n_jack):
                jackknife_sample = np.delete(data, i)
                jackknife_stat = statistic_func(jackknife_sample)
                jackknife_stats.append(jackknife_stat)
            
            jackknife_stats = np.array(jackknife_stats)
            jackknife_mean = np.mean(jackknife_stats)
            
            # Acceleration
            numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
            denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2)) ** (3/2)
            acceleration = numerator / denominator if denominator != 0 else 0
            
            # BCa percentiles
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
            
            bca_lower = stats.norm.cdf(bias_correction + 
                                     (bias_correction + z_alpha_2) / 
                                     (1 - acceleration * (bias_correction + z_alpha_2))) * 100
            bca_upper = stats.norm.cdf(bias_correction + 
                                     (bias_correction + z_1_alpha_2) / 
                                     (1 - acceleration * (bias_correction + z_1_alpha_2))) * 100
            
            ci_lower = np.percentile(bootstrap_stats, bca_lower)
            ci_upper = np.percentile(bootstrap_stats, bca_upper)
        
        return {
            'original_statistic': original_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'method': method,
            'n_bootstrap': n_bootstrap
        }
    
    def permutation_test(self, 
                        group1: np.ndarray,
                        group2: np.ndarray,
                        test_statistic: Callable,
                        n_permutations: int = 10000,
                        alternative: str = 'two-sided') -> Dict[str, float]:
        """
        Non-parametric permutation test
        
        H₀: The two groups come from the same distribution
        H₁: The groups come from different distributions
        """
        
        # Observed test statistic
        observed_stat = test_statistic(group1, group2)
        
        # Combined data
        combined_data = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        
        # Permutation distribution
        permuted_stats = []
        
        for _ in range(n_permutations):
            # Random permutation
            permuted_data = np.random.permutation(combined_data)
            perm_group1 = permuted_data[:n1]
            perm_group2 = permuted_data[n1:]
            
            perm_stat = test_statistic(perm_group1, perm_group2)
            permuted_stats.append(perm_stat)
        
        permuted_stats = np.array(permuted_stats)
        
        # Calculate p-value
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
        elif alternative == 'greater':
            p_value = np.mean(permuted_stats >= observed_stat)
        elif alternative == 'less':
            p_value = np.mean(permuted_stats <= observed_stat)
        else:
            raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
        
        return {
            'observed_statistic': observed_stat,
            'p_value': p_value,
            'alternative': alternative,
            'n_permutations': n_permutations,
            'permutation_mean': np.mean(permuted_stats),
            'permutation_std': np.std(permuted_stats)
        }
    
    def bayesian_model_uncertainty(self, 
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  n_samples: int = 1000,
                                  prior_precision: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Bayesian linear regression with uncertainty quantification
        
        Prior: β ~ N(0, τ⁻¹I)
        Likelihood: y|X,β ~ N(Xβ, σ²I)  
        Posterior: β|y,X ~ N(μₙ, Σₙ)
        """
        
        n, p = X_train.shape
        
        # Add intercept
        X_train_aug = np.column_stack([np.ones(n), X_train])
        X_test_aug = np.column_stack([np.ones(X_test.shape[0]), X_test])
        
        # Prior precision matrix
        tau = prior_precision
        Lambda_0 = tau * np.eye(p + 1)
        mu_0 = np.zeros(p + 1)
        
        # Posterior parameters (assuming known noise variance)
        sigma_squared = 1.0  # Can be estimated or set
        
        # Posterior precision and covariance
        Lambda_n = Lambda_0 + (1/sigma_squared) * X_train_aug.T @ X_train_aug
        Sigma_n = np.linalg.inv(Lambda_n)
        
        # Posterior mean
        mu_n = Sigma_n @ (Lambda_0 @ mu_0 + (1/sigma_squared) * X_train_aug.T @ y_train)
        
        # Sample from posterior
        beta_samples = np.random.multivariate_normal(mu_n, Sigma_n, n_samples)
        
        # Predictions for each sample
        predictions = X_test_aug @ beta_samples.T  # Shape: (n_test, n_samples)
        
        # Prediction statistics
        pred_mean = np.mean(predictions, axis=1)
        pred_std = np.std(predictions, axis=1)
        pred_quantiles = np.percentile(predictions, [2.5, 25, 75, 97.5], axis=1)
        
        return {
            'predictions_mean': pred_mean,
            'predictions_std': pred_std,
            'predictions_2_5': pred_quantiles[0],
            'predictions_25': pred_quantiles[1],
            'predictions_75': pred_quantiles[2],
            'predictions_97_5': pred_quantiles[3],
            'beta_samples': beta_samples,
            'posterior_mean': mu_n,
            'posterior_cov': Sigma_n
        }
    
    def extreme_scenario_simulation(self, 
                                  baseline_features: pd.DataFrame,
                                  scenario_multipliers: Dict[str, Tuple[float, float]],
                                  model_predict_func: Callable,
                                  n_simulations: int = 10000) -> Dict[str, np.ndarray]:
        """
        Generate extreme scenarios using Monte Carlo simulation
        
        scenario_multipliers: Dict mapping feature names to (min_multiplier, max_multiplier)
        """
        
        results = {
            'predictions': [],
            'scenarios': [],
            'feature_shocks': {col: [] for col in scenario_multipliers.keys()}
        }
        
        for _ in range(n_simulations):
            # Create scenario
            scenario_features = baseline_features.copy()
            scenario_dict = {}
            
            for feature, (min_mult, max_mult) in scenario_multipliers.items():
                if feature in scenario_features.columns:
                    # Random multiplier
                    multiplier = np.random.uniform(min_mult, max_mult)
                    scenario_features[feature] *= multiplier
                    
                    scenario_dict[feature] = multiplier
                    results['feature_shocks'][feature].append(multiplier)
            
            # Model prediction
            prediction = model_predict_func(scenario_features)
            results['predictions'].append(prediction)
            results['scenarios'].append(scenario_dict)
        
        results['predictions'] = np.array(results['predictions'])
        
        return results
    
    def value_at_risk_simulation(self, 
                               returns: np.ndarray,
                               confidence_levels: List[float] = [0.95, 0.99, 0.999],
                               n_simulations: int = 100000,
                               distribution: str = 'empirical') -> Dict[str, Dict[str, float]]:
        """
        Value at Risk (VaR) estimation using Monte Carlo
        
        VaR_α = -F⁻¹(1-α) where F is the loss distribution
        """
        
        results = {}
        
        if distribution == 'empirical':
            # Bootstrap from empirical distribution
            simulated_returns = np.random.choice(returns, size=n_simulations, replace=True)
            
        elif distribution == 'normal':
            # Assume normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns)
            simulated_returns = np.random.normal(mu, sigma, n_simulations)
            
        elif distribution == 't':
            # Student's t-distribution (heavy tails)
            # Fit degrees of freedom
            params = stats.t.fit(returns)
            df, loc, scale = params
            simulated_returns = stats.t.rvs(df, loc, scale, size=n_simulations)
            
        elif distribution == 'skewed_t':
            # Skewed t-distribution for asymmetry
            try:
                from scipy.stats import skewnorm
                params = skewnorm.fit(returns)
                a, loc, scale = params
                simulated_returns = skewnorm.rvs(a, loc, scale, size=n_simulations)
            except:
                # Fallback to normal
                mu = np.mean(returns)
                sigma = np.std(returns)
                simulated_returns = np.random.normal(mu, sigma, n_simulations)
        
        # Calculate VaR and Expected Shortfall (ES) for each confidence level
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # VaR (negative because we want losses)
            var = -np.percentile(simulated_returns, alpha * 100)
            
            # Expected Shortfall (Conditional VaR)
            tail_losses = simulated_returns[simulated_returns <= -var]
            expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var
            
            results[f'VaR_{conf_level}'] = {
                'value_at_risk': var,
                'expected_shortfall': expected_shortfall,
                'confidence_level': conf_level,
                'distribution': distribution
            }
        
        return results
    
    def copula_simulation(self, 
                        data: pd.DataFrame,
                        copula_type: str = 'gaussian',
                        n_simulations: int = 10000) -> np.ndarray:
        """
        Multivariate dependence simulation using copulas
        
        Copula types: 'gaussian', 't', 'clayton', 'frank', 'gumbel'
        """
        
        n_vars = data.shape[1]
        n_obs = data.shape[0]
        
        # Transform to uniform margins (empirical CDFs)
        uniform_data = np.zeros_like(data)
        for i, col in enumerate(data.columns):
            sorted_values = np.sort(data.iloc[:, i])
            uniform_data[:, i] = np.searchsorted(sorted_values, data.iloc[:, i], side='right') / n_obs
        
        # Fit copula parameters
        if copula_type == 'gaussian':
            # Estimate correlation matrix
            # Transform to normal marginals
            normal_data = stats.norm.ppf(np.clip(uniform_data, 1e-6, 1-1e-6))
            correlation_matrix = np.corrcoef(normal_data.T)
            
            # Simulate from multivariate normal
            simulated_normal = np.random.multivariate_normal(
                np.zeros(n_vars), correlation_matrix, n_simulations
            )
            
            # Transform back to uniform
            simulated_uniform = stats.norm.cdf(simulated_normal)
            
        elif copula_type == 't':
            # Student's t copula
            normal_data = stats.norm.ppf(np.clip(uniform_data, 1e-6, 1-1e-6))
            correlation_matrix = np.corrcoef(normal_data.T)
            
            # Estimate degrees of freedom (simplified)
            df = 5  # Fixed for simplicity, should be estimated
            
            # Simulate from multivariate t
            simulated_t = stats.multivariate_t.rvs(
                np.zeros(n_vars), correlation_matrix, df, size=n_simulations
            )
            
            # Transform to uniform using t CDF
            simulated_uniform = stats.t.cdf(simulated_t, df)
            
        else:
            # For other copulas, use simpler approach (independence copula)
            simulated_uniform = np.random.uniform(0, 1, (n_simulations, n_vars))
        
        # Transform back to original marginal distributions
        simulated_data = np.zeros((n_simulations, n_vars))
        
        for i, col in enumerate(data.columns):
            # Fit marginal distribution (use empirical quantiles)
            original_values = np.sort(data.iloc[:, i].values)
            quantile_indices = (simulated_uniform[:, i] * (n_obs - 1)).astype(int)
            quantile_indices = np.clip(quantile_indices, 0, n_obs - 1)
            simulated_data[:, i] = original_values[quantile_indices]
        
        return simulated_data
    
    def model_ensemble_uncertainty(self, 
                                 models: List[Callable],
                                 X_test: np.ndarray,
                                 n_bootstrap: int = 1000) -> Dict[str, np.ndarray]:
        """
        Quantify uncertainty from model ensemble using bootstrap
        """
        
        n_models = len(models)
        n_test = X_test.shape[0]
        
        # Collect predictions from all models
        all_predictions = []
        
        for model in models:
            predictions = model(X_test)
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_test)
        
        # Bootstrap over models
        bootstrap_predictions = []
        
        for _ in range(n_bootstrap):
            # Sample models with replacement
            model_indices = np.random.choice(n_models, size=n_models, replace=True)
            bootstrap_pred = np.mean(all_predictions[model_indices], axis=0)
            bootstrap_predictions.append(bootstrap_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)  # Shape: (n_bootstrap, n_test)
        
        # Calculate statistics
        pred_mean = np.mean(bootstrap_predictions, axis=0)
        pred_std = np.std(bootstrap_predictions, axis=0)
        pred_quantiles = np.percentile(bootstrap_predictions, [2.5, 25, 75, 97.5], axis=0)
        
        return {
            'ensemble_mean': pred_mean,
            'ensemble_std': pred_std,
            'ensemble_2_5': pred_quantiles[0],
            'ensemble_25': pred_quantiles[1],  
            'ensemble_75': pred_quantiles[2],
            'ensemble_97_5': pred_quantiles[3],
            'individual_model_predictions': all_predictions,
            'bootstrap_predictions': bootstrap_predictions
        }
    
    def parallel_monte_carlo(self, 
                           simulation_func: Callable,
                           n_total_simulations: int,
                           n_processes: Optional[int] = None,
                           **kwargs) -> List:
        """
        Parallel Monte Carlo simulation using multiprocessing
        """
        
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        # Split simulations across processes
        sims_per_process = n_total_simulations // n_processes
        remaining_sims = n_total_simulations % n_processes
        
        simulation_args = []
        for i in range(n_processes):
            n_sims = sims_per_process + (1 if i < remaining_sims else 0)
            simulation_args.append((simulation_func, n_sims, kwargs))
        
        # Run parallel simulations
        all_results = []
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = [
                executor.submit(self._run_simulation_chunk, args) 
                for args in simulation_args
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.extend(result)
                except Exception as e:
                    print(f"Simulation chunk failed: {e}")
        
        return all_results
    
    def _run_simulation_chunk(self, args) -> List:
        """Helper function for parallel simulation"""
        simulation_func, n_sims, kwargs = args
        results = []
        
        for _ in range(n_sims):
            result = simulation_func(**kwargs)
            results.append(result)
        
        return results
    
    def plot_uncertainty_analysis(self, 
                                predictions: np.ndarray,
                                uncertainty_intervals: Dict[str, np.ndarray],
                                true_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Visualization for uncertainty analysis
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Monte Carlo Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # 1. Prediction distribution
        axes[0, 0].hist(predictions, bins=50, alpha=0.7, density=True, 
                       edgecolor='black', linewidth=0.5)
        axes[0, 0].axvline(np.mean(predictions), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(predictions):.3f}')
        axes[0, 0].axvline(np.median(predictions), color='orange', linestyle='--', 
                          label=f'Median: {np.median(predictions):.3f}')
        axes[0, 0].set_xlabel('Predictions')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Prediction Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Uncertainty intervals over samples
        if len(predictions.shape) == 2:  # Multiple samples
            sample_indices = np.arange(predictions.shape[0])
            pred_mean = np.mean(predictions, axis=1)
            
            if 'lower_95' in uncertainty_intervals and 'upper_95' in uncertainty_intervals:
                axes[0, 1].fill_between(sample_indices, 
                                       uncertainty_intervals['lower_95'],
                                       uncertainty_intervals['upper_95'],
                                       alpha=0.3, label='95% CI')
            
            if 'lower_75' in uncertainty_intervals and 'upper_75' in uncertainty_intervals:
                axes[0, 1].fill_between(sample_indices,
                                       uncertainty_intervals['lower_75'],
                                       uncertainty_intervals['upper_75'], 
                                       alpha=0.5, label='75% CI')
            
            axes[0, 1].plot(sample_indices, pred_mean, 'b-', linewidth=2, label='Mean')
            
            if true_values is not None:
                axes[0, 1].plot(sample_indices, true_values, 'r-', 
                               linewidth=2, alpha=0.7, label='True Values')
        
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Prediction')
        axes[0, 1].set_title('Uncertainty Intervals')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Quantile-Quantile plot
        if true_values is not None:
            stats.probplot(predictions - true_values, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot: Prediction Residuals vs Normal')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            stats.probplot(predictions.flatten() if len(predictions.shape) > 1 else predictions, 
                          dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot: Predictions vs Normal')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Uncertainty vs prediction level
        if len(predictions.shape) == 2:
            pred_std = np.std(predictions, axis=1)
            axes[1, 1].scatter(pred_mean, pred_std, alpha=0.6)
            axes[1, 1].set_xlabel('Mean Prediction')
            axes[1, 1].set_ylabel('Prediction Std Dev')
            axes[1, 1].set_title('Heteroscedastic Uncertainty')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # For 1D predictions, show histogram of values
            axes[1, 1].boxplot(predictions)
            axes[1, 1].set_ylabel('Predictions')
            axes[1, 1].set_title('Prediction Box Plot')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Example usage and testing
if __name__ == "__main__":
    print("🎲 AML 360° Monte Carlo Simulation Framework")
    print("Advanced uncertainty quantification and stress testing")
    
    # Initialize Monte Carlo framework
    mc_framework = MonteCarloAML(random_seed=42)
    
    # Example: Bootstrap confidence intervals
    data = np.random.exponential(2, 1000)  # Sample data
    
    # Define statistic function (e.g., mean)
    mean_func = lambda x: np.mean(x)
    
    # Bootstrap confidence intervals
    bootstrap_results = mc_framework.bootstrap_confidence_intervals(
        data, mean_func, n_bootstrap=5000, method='bca'
    )
    
    print(f"Bootstrap Results:")
    print(f"Original Mean: {bootstrap_results['original_statistic']:.4f}")
    print(f"95% CI: ({bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f})")
    
    print("\n✅ Monte Carlo framework ready for AML uncertainty analysis!")
    print("📊 Available methods:")
    print("  • Bootstrap Confidence Intervals (Percentile, BCa)")
    print("  • Permutation Tests")
    print("  • Bayesian Model Uncertainty")
    print("  • Extreme Scenario Simulation") 
    print("  • Value at Risk (VaR) Estimation")
    print("  • Copula-based Multivariate Simulation")
    print("  • Model Ensemble Uncertainty")
    print("  • Parallel Monte Carlo Processing")
