# Mathematical Foundations - AML 360º
## Statistical Methods and Formal Mathematical Framework

### 1. Extreme Value Theory (EVT) - Formal Treatment

#### 1.1 Generalized Pareto Distribution

For excesses over threshold $u$, we model using GPD:

$$F_u(x) = P(X - u \leq x | X > u) = 1 - \left(1 + \xi \frac{x}{\sigma}\right)^{-1/\xi}$$

Where:
- $\xi$ = shape parameter (tail index)
- $\sigma$ = scale parameter  
- $x \geq 0$ when $\xi \geq 0$, and $0 \leq x \leq -\sigma/\xi$ when $\xi < 0$

#### 1.2 Threshold Selection via Mean Excess Function

$$e(u) = E[X - u | X > u] = \frac{\sigma + \xi u}{1 - \xi}$$

For GPD, the mean excess function is linear in $u$.

#### 1.3 Tail Risk Measure

$$\text{VaR}_p = u + \frac{\sigma}{\xi}\left[\left(\frac{n}{k}\frac{1-p}{1}\right)^{-\xi} - 1\right]$$

Where $n$ = total observations, $k$ = exceedances above threshold $u$.

### 2. GARCH Models for Volatility Clustering

#### 2.1 GARCH(1,1) Specification

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

With constraints: $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$, $\alpha + \beta < 1$

#### 2.2 Log-Likelihood Function

$$\mathcal{L}(\theta) = -\frac{1}{2}\sum_{t=1}^T \left[\log(2\pi) + \log(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2}\right]$$

#### 2.3 Unconditional Variance

$$E[\epsilon_t^2] = \frac{\omega}{1 - \alpha - \beta}$$

### 3. Weight of Evidence (WOE) and Information Value (IV)

#### 3.1 Weight of Evidence Formula

$$\text{WOE}_i = \ln\left(\frac{\text{Distribution of Goods}_i}{\text{Distribution of Bads}_i}\right) = \ln\left(\frac{G_i/G_T}{B_i/B_T}\right)$$

Where:
- $G_i$ = number of goods in bin $i$
- $B_i$ = number of bads in bin $i$  
- $G_T$ = total goods
- $B_T$ = total bads

#### 3.2 Information Value

$$\text{IV} = \sum_{i=1}^n (\text{Distribution of Goods}_i - \text{Distribution of Bads}_i) \times \text{WOE}_i$$

$$\text{IV} = \sum_{i=1}^n \left(\frac{G_i}{G_T} - \frac{B_i}{B_T}\right) \times \ln\left(\frac{G_i/G_T}{B_i/B_T}\right)$$

#### 3.3 IV Interpretation Scale
- $\text{IV} < 0.02$: Not predictive
- $0.02 \leq \text{IV} < 0.1$: Weak predictive power  
- $0.1 \leq \text{IV} < 0.3$: Medium predictive power
- $\text{IV} \geq 0.3$: Strong predictive power (check for overfitting)

### 4. Burstiness and Temporal Clustering

#### 4.1 Fano Factor (Index of Dispersion)

$$F = \frac{\text{Var}(N)}{\text{E}[N]} = \frac{\sigma^2}{\mu}$$

For Poisson process: $F = 1$
- $F > 1$: Overdispersed (bursty)
- $F < 1$: Underdispersed (regular)

#### 4.2 Clark-Evans Aggregation Index

$$R = \frac{\bar{r}_A}{\bar{r}_E} = \frac{\bar{r}_A}{1/(2\sqrt{\rho})}$$

Where:
- $\bar{r}_A$ = observed mean nearest neighbor distance
- $\bar{r}_E$ = expected mean distance under CSR
- $\rho$ = density of points

### 5. Graph Theory Centrality Measures

#### 5.1 Degree Centrality

$$C_D(v) = \frac{\deg(v)}{n-1}$$

#### 5.2 Betweenness Centrality

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Where $\sigma_{st}$ = total number of shortest paths from $s$ to $t$, $\sigma_{st}(v)$ = paths passing through $v$.

#### 5.3 PageRank

$$\text{PR}(v) = \frac{1-d}{N} + d \sum_{u \in M(v)} \frac{\text{PR}(u)}{L(u)}$$

Where:
- $d$ = damping factor (typically 0.85)
- $M(v)$ = set of pages linking to $v$
- $L(u)$ = number of outbound links from $u$

#### 5.4 Eigenvector Centrality

$$Ax = \lambda x$$

Where $A$ is adjacency matrix, centrality is the eigenvector corresponding to largest eigenvalue $\lambda$.

### 6. Bayesian Framework for Model Fusion

#### 6.1 Posterior Model Probabilities

Given models $M_1, M_2, ..., M_K$ and data $D$:

$$P(M_k|D) = \frac{P(D|M_k)P(M_k)}{\sum_{j=1}^K P(D|M_j)P(M_j)}$$

#### 6.2 Bayesian Model Averaging

$$P(\text{SAR}|x, D) = \sum_{k=1}^K P(\text{SAR}|x, M_k, D) \cdot P(M_k|D)$$

#### 6.3 Weight Learning via Dirichlet Prior

$$\boldsymbol{w} \sim \text{Dirichlet}(\boldsymbol{\alpha})$$
$$p(\boldsymbol{w}) = \frac{\Gamma(\sum_{i=1}^K \alpha_i)}{\prod_{i=1}^K \Gamma(\alpha_i)} \prod_{i=1}^K w_i^{\alpha_i - 1}$$

### 7. Cost-Sensitive Learning Framework

#### 7.1 Expected Cost Minimization

$$J(\tau) = c_{\text{FN}} \cdot \text{FN}(\tau) + c_{\text{FP}} \cdot \text{FP}(\tau) + c_{\text{rev}} \cdot \text{Workload}(\tau)$$

#### 7.2 Optimal Threshold

$$\tau^* = \arg\min_{\tau} J(\tau)$$

#### 7.3 ROC-based Threshold Selection

$$\tau^* = \arg\max_{\tau} \left[\text{TPR}(\tau) - \frac{c_{\text{FP}}}{c_{\text{FN}}} \cdot \text{FPR}(\tau)\right]$$

### 8. Information Theory Measures

#### 8.1 Entropy

$$H(X) = -\sum_{i=1}^n p(x_i) \log_2 p(x_i)$$

#### 8.2 Mutual Information

$$I(X;Y) = \sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x)p(y)} = H(X) - H(X|Y)$$

#### 8.3 Kullback-Leibler Divergence

$$D_{\text{KL}}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

### 9. Model Stability and Drift Detection

#### 9.1 Population Stability Index (PSI)

$$\text{PSI} = \sum_{i=1}^n (\text{Actual}_i - \text{Expected}_i) \times \ln\left(\frac{\text{Actual}_i}{\text{Expected}_i}\right)$$

Interpretation:
- $\text{PSI} < 0.1$: No significant population change
- $0.1 \leq \text{PSI} < 0.25$: Some minor population change  
- $\text{PSI} \geq 0.25$: Major population shift

#### 9.2 Kolmogorov-Smirnov Test Statistic

$$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$

Where $F_{1,n}$ and $F_{2,m}$ are empirical distribution functions.

#### 9.3 Jensen-Shannon Divergence

$$\text{JSD}(P||Q) = \frac{1}{2}D_{\text{KL}}(P||M) + \frac{1}{2}D_{\text{KL}}(Q||M)$$

Where $M = \frac{1}{2}(P + Q)$.

### 10. Sequential Pattern Detection

#### 10.1 Hidden Markov Model for Transaction Sequences

State transition probability:
$$a_{ij} = P(S_t = j | S_{t-1} = i)$$

Emission probability:
$$b_j(o) = P(O_t = o | S_t = j)$$

#### 10.2 Viterbi Algorithm for Optimal Path

$$\delta_t(j) = \max_{1 \leq i \leq N} \delta_{t-1}(i) a_{ij} b_j(o_t)$$

### 11. Focal Loss for Imbalanced Learning

#### 11.1 Focal Loss Function

$$\text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

Where:
- $\alpha_t$ = weighting factor for class imbalance
- $\gamma$ = focusing parameter to down-weight easy examples
- $p_t$ = model's estimated probability for ground truth class

#### 11.2 Gradient and Hessian for LightGBM

$$\frac{\partial \text{FL}}{\partial z} = \alpha_t (1-p_t)^\gamma [\gamma p_t \log(p_t) + p_t - 1]$$

$$\frac{\partial^2 \text{FL}}{\partial z^2} = \alpha_t (1-p_t)^\gamma p_t [\gamma(1-p_t)(\log(p_t) + 2p_t - 1) + 1 - p_t]$$

### 12. Graph Neural Network Mathematics

#### 12.1 Graph Convolution

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

Where:
- $\tilde{A} = A + I$ (adjacency matrix with self-loops)
- $\tilde{D}$ = diagonal degree matrix of $\tilde{A}$

#### 12.2 Graph Attention Mechanism

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}h_i||\mathbf{W}h_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}h_i||\mathbf{W}h_k]))}$$

#### 12.3 Message Passing

$$h_v^{(l+1)} = \text{UPDATE}^{(l)}\left(h_v^{(l)}, \text{AGGREGATE}^{(l)}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

### 13. Uncertainty Quantification

#### 13.1 Conformal Prediction

For prediction set $C(x)$ with confidence level $1-\alpha$:

$$P(Y \in C(X)) \geq 1 - \alpha$$

#### 13.2 Bootstrap Confidence Intervals

For parameter $\theta$, the $(1-\alpha)$ confidence interval:

$$[\hat{\theta}_{\alpha/2}^*, \hat{\theta}_{1-\alpha/2}^*]$$

Where $\hat{\theta}_{\alpha/2}^*$ is the $\alpha/2$ quantile of bootstrap distribution.

### 14. Survival Analysis for Time-to-Detection

#### 14.1 Hazard Function

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t | T \geq t)}{\Delta t}$$

#### 14.2 Cox Proportional Hazards Model

$$h(t|x) = h_0(t) \exp(\beta^T x)$$

Where $h_0(t)$ is the baseline hazard function.

### 15. Causal Inference Framework

#### 15.1 Propensity Score

$$e(x) = P(T = 1 | X = x)$$

#### 15.2 Average Treatment Effect

$$\text{ATE} = E[Y_1 - Y_0] = E[Y | T = 1] - E[Y | T = 0]$$

#### 15.3 Doubly Robust Estimator

$$\hat{\tau}_{\text{DR}} = \frac{1}{n} \sum_{i=1}^n \left[\frac{T_i Y_i}{\hat{e}(X_i)} - \frac{(1-T_i)Y_i}{1-\hat{e}(X_i)} + \left(\frac{T_i - \hat{e}(X_i)}{\hat{e}(X_i)(1-\hat{e}(X_i))}\right)\hat{\mu}(X_i)\right]$$

This mathematical foundation ensures the AML 360º system is built on rigorous statistical principles with full theoretical justification for each component.
