-- AML 360º Feature Engineering - BigQuery SQL
-- Advanced statistical features with temporal windows

-- =====================================================
-- 1. BEHAVIORAL FEATURES (24h/7d/30d windows)
-- =====================================================

CREATE OR REPLACE TABLE `aml.features_behavioral` AS
WITH tx_windows AS (
  SELECT 
    payer_id,
    payee_id,
    amount,
    timestamp,
    channel,
    device_id,
    ip_address,
    geo_lat,
    geo_lng,
    mcc,
    -- 24h window
    COUNT(*) OVER(
      PARTITION BY payer_id 
      ORDER BY UNIX_SECONDS(timestamp) 
      RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    ) AS tx_count_24h,
    
    SUM(amount) OVER(
      PARTITION BY payer_id 
      ORDER BY UNIX_SECONDS(timestamp) 
      RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    ) AS total_amount_24h,
    
    -- 7d window  
    COUNT(*) OVER(
      PARTITION BY payer_id 
      ORDER BY UNIX_SECONDS(timestamp) 
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS tx_count_7d,
    
    -- 30d window
    COUNT(*) OVER(
      PARTITION BY payer_id 
      ORDER BY UNIX_SECONDS(timestamp) 
      RANGE BETWEEN 2592000 PRECEDING AND CURRENT ROW
    ) AS tx_count_30d,
    
    -- Velocity features
    COUNT(DISTINCT payee_id) OVER(
      PARTITION BY payer_id 
      ORDER BY UNIX_SECONDS(timestamp) 
      RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    ) AS unique_payees_24h,
    
    COUNT(DISTINCT channel) OVER(
      PARTITION BY payer_id 
      ORDER BY UNIX_SECONDS(timestamp) 
      RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    ) AS unique_channels_24h
    
  FROM `aml.transactions`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
),

-- Extreme Value Theory (EVT) features
evt_features AS (
  SELECT 
    payer_id,
    -- P95 threshold for EVT
    APPROX_QUANTILES(amount, 100)[OFFSET(95)] AS evt_threshold_p95,
    -- Count of extreme values (above P95)
    COUNTIF(amount > APPROX_QUANTILES(amount, 100)[OFFSET(95)]) AS extreme_count,
    -- Tail index estimation
    STDDEV(amount) / AVG(amount) AS tail_index_proxy
  FROM `aml.transactions`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY payer_id
),

-- Burstiness analysis using Fano factor
burstiness AS (
  SELECT 
    payer_id,
    -- Inter-event times
    AVG(UNIX_SECONDS(timestamp) - LAG(UNIX_SECONDS(timestamp)) OVER(PARTITION BY payer_id ORDER BY timestamp)) AS mean_interval,
    STDDEV(UNIX_SECONDS(timestamp) - LAG(UNIX_SECONDS(timestamp)) OVER(PARTITION BY payer_id ORDER BY timestamp)) AS std_interval,
    -- Fano factor (variance-to-mean ratio)
    SAFE_DIVIDE(
      POWER(STDDEV(UNIX_SECONDS(timestamp) - LAG(UNIX_SECONDS(timestamp)) OVER(PARTITION BY payer_id ORDER BY timestamp)), 2),
      AVG(UNIX_SECONDS(timestamp) - LAG(UNIX_SECONDS(timestamp)) OVER(PARTITION BY payer_id ORDER BY timestamp))
    ) AS fano_factor
  FROM `aml.transactions`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY payer_id
),

-- Geographic entropy and risk
geo_features AS (
  SELECT 
    payer_id,
    -- Geographic entropy
    -SUM(geo_prob * SAFE.LOG(geo_prob)) AS geo_entropy,
    -- Max distance between transactions
    MAX(geo_distance_km) AS max_geo_distance,
    -- Number of unique locations
    COUNT(DISTINCT geo_grid) AS unique_locations
  FROM (
    SELECT 
      payer_id,
      CONCAT(CAST(ROUND(geo_lat, 2) AS STRING), '_', CAST(ROUND(geo_lng, 2) AS STRING)) AS geo_grid,
      ST_DISTANCE(ST_GEOGPOINT(geo_lng, geo_lat), 
                  ST_GEOGPOINT(LAG(geo_lng) OVER(PARTITION BY payer_id ORDER BY timestamp), 
                               LAG(geo_lat) OVER(PARTITION BY payer_id ORDER BY timestamp))) / 1000 AS geo_distance_km,
      COUNT(*) OVER(PARTITION BY payer_id, CONCAT(CAST(ROUND(geo_lat, 2) AS STRING), '_', CAST(ROUND(geo_lng, 2) AS STRING))) / 
      COUNT(*) OVER(PARTITION BY payer_id) AS geo_prob
    FROM `aml.transactions`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
      AND geo_lat IS NOT NULL AND geo_lng IS NOT NULL
  )
  GROUP BY payer_id
),

-- Network centrality features (simplified)
network_features AS (
  SELECT 
    payer_id,
    -- Degree centrality (connections)
    COUNT(DISTINCT payee_id) AS out_degree,
    -- Betweenness proxy (transactions through this node)
    SUM(amount) / COUNT(*) AS avg_tx_amount,
    -- PageRank proxy (weighted by transaction amounts)
    SUM(amount * LOG(1 + COUNT(*) OVER(PARTITION BY payee_id))) AS pagerank_proxy
  FROM `aml.transactions`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY payer_id
)

SELECT 
  tw.*,
  evt.evt_threshold_p95,
  evt.extreme_count,
  evt.tail_index_proxy,
  b.fano_factor,
  b.mean_interval,
  geo.geo_entropy,
  geo.max_geo_distance,
  geo.unique_locations,
  nf.out_degree,
  nf.avg_tx_amount,
  nf.pagerank_proxy,
  
  -- Risk ratios and derived features
  SAFE_DIVIDE(tw.tx_count_24h, tw.tx_count_7d) AS velocity_ratio_24h_7d,
  SAFE_DIVIDE(tw.total_amount_24h, evt.evt_threshold_p95) AS amount_evt_ratio,
  CASE 
    WHEN b.fano_factor > 2 THEN 'BURSTY'
    WHEN b.fano_factor < 0.5 THEN 'REGULAR' 
    ELSE 'NORMAL'
  END AS burstiness_pattern,
  
  -- Anomaly flags
  CASE WHEN tw.tx_count_24h > PERCENTILE_CONT(tw.tx_count_24h, 0.95) OVER() THEN 1 ELSE 0 END AS high_velocity_flag,
  CASE WHEN amount > evt.evt_threshold_p95 THEN 1 ELSE 0 END AS extreme_amount_flag,
  CASE WHEN geo.unique_locations > 10 THEN 1 ELSE 0 END AS geo_dispersion_flag

FROM tx_windows tw
LEFT JOIN evt_features evt ON tw.payer_id = evt.payer_id
LEFT JOIN burstiness b ON tw.payer_id = b.payer_id  
LEFT JOIN geo_features geo ON tw.payer_id = geo.payer_id
LEFT JOIN network_features nf ON tw.payer_id = nf.payer_id;

-- =====================================================
-- 2. GRAPH EDGE CREATION FOR GNN
-- =====================================================

CREATE OR REPLACE TABLE `aml.graph_edges` AS
WITH transaction_edges AS (
  SELECT 
    payer_id AS source_node,
    payee_id AS target_node,
    timestamp,
    amount,
    channel,
    -- Edge weights based on transaction patterns
    COUNT(*) OVER(PARTITION BY payer_id, payee_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS edge_weight,
    SUM(amount) OVER(PARTITION BY payer_id, payee_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_amount,
    -- Risk propagation signals
    CASE 
      WHEN amount > PERCENTILE_CONT(amount, 0.95) OVER(PARTITION BY payer_id) THEN 1 
      ELSE 0 
    END AS high_amount_signal,
    CASE 
      WHEN TIMESTAMP_DIFF(timestamp, LAG(timestamp) OVER(PARTITION BY payer_id, payee_id ORDER BY timestamp), SECOND) < 300 THEN 1
      ELSE 0
    END AS rapid_sequence_signal
  FROM `aml.transactions`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
),

-- Community detection preparation (Louvain algorithm input)
community_prep AS (
  SELECT 
    source_node,
    target_node,
    SUM(amount) AS total_amount,
    COUNT(*) AS transaction_count,
    AVG(amount) AS avg_amount,
    MAX(edge_weight) AS max_edge_weight,
    -- Jaccard similarity for community detection
    COUNT(DISTINCT 
      CASE WHEN EXISTS(
        SELECT 1 FROM transaction_edges te2 
        WHERE te2.source_node = te.target_node 
        AND te2.target_node IN (
          SELECT DISTINCT target_node FROM transaction_edges te3 WHERE te3.source_node = te.source_node
        )
      ) THEN te.target_node END
    ) / NULLIF(COUNT(DISTINCT te.target_node), 0) AS jaccard_similarity
  FROM transaction_edges te
  GROUP BY source_node, target_node
  HAVING COUNT(*) >= 2  -- Filter weak connections
)

SELECT 
  source_node,
  target_node,
  total_amount,
  transaction_count,
  avg_amount,
  max_edge_weight,
  jaccard_similarity,
  -- Risk scores for edge classification
  CASE 
    WHEN total_amount > PERCENTILE_CONT(total_amount, 0.9) OVER() 
         AND transaction_count > PERCENTILE_CONT(transaction_count, 0.9) OVER() THEN 'HIGH_RISK'
    WHEN avg_amount > PERCENTILE_CONT(avg_amount, 0.8) OVER() THEN 'MEDIUM_RISK'
    ELSE 'LOW_RISK'
  END AS edge_risk_category,
  
  -- Temporal pattern flags
  CASE WHEN transaction_count > 50 AND max_edge_weight > 100 THEN 1 ELSE 0 END AS potential_structuring,
  CASE WHEN jaccard_similarity > 0.3 THEN 1 ELSE 0 END AS community_bridge
  
FROM community_prep;

-- =====================================================
-- 3. SEQUENTIAL PATTERNS FOR STRUCTURING DETECTION
-- =====================================================

CREATE OR REPLACE TABLE `aml.sequence_features` AS
WITH transaction_sequences AS (
  SELECT 
    payer_id,
    timestamp,
    amount,
    payee_id,
    channel,
    -- Sequence analysis
    ROW_NUMBER() OVER(PARTITION BY payer_id ORDER BY timestamp) AS seq_position,
    LAG(amount, 1) OVER(PARTITION BY payer_id ORDER BY timestamp) AS prev_amount_1,
    LAG(amount, 2) OVER(PARTITION BY payer_id ORDER BY timestamp) AS prev_amount_2,
    LAG(amount, 3) OVER(PARTITION BY payer_id ORDER BY timestamp) AS prev_amount_3,
    
    -- Time gaps
    TIMESTAMP_DIFF(timestamp, LAG(timestamp) OVER(PARTITION BY payer_id ORDER BY timestamp), MINUTE) AS time_gap_minutes,
    
    -- Round amount patterns (structuring indicator)
    CASE 
      WHEN MOD(CAST(amount AS INT64), 1000) = 0 THEN 'ROUND_1000'
      WHEN MOD(CAST(amount AS INT64), 500) = 0 THEN 'ROUND_500'
      WHEN MOD(CAST(amount AS INT64), 100) = 0 THEN 'ROUND_100'
      ELSE 'NOT_ROUND'
    END AS amount_pattern,
    
    -- Payee diversity in sequence
    COUNT(DISTINCT payee_id) OVER(
      PARTITION BY payer_id 
      ORDER BY timestamp 
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS payee_diversity_5tx
  FROM `aml.transactions`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
),

-- Smurfing pattern detection
smurfing_patterns AS (
  SELECT 
    payer_id,
    timestamp,
    amount,
    -- Detect consistent just-under-threshold amounts
    CASE 
      WHEN amount BETWEEN 9800 AND 9999 
           AND COUNT(*) OVER(
             PARTITION BY payer_id 
             ORDER BY timestamp 
             ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
           ) >= 5 THEN 1
      ELSE 0
    END AS potential_smurfing,
    
    -- Velocity spikes
    CASE 
      WHEN time_gap_minutes < 30 
           AND amount > AVG(amount) OVER(PARTITION BY payer_id) * 1.5 THEN 1
      ELSE 0
    END AS velocity_spike,
    
    -- Layering detection (complex transaction chains)
    CASE 
      WHEN payee_diversity_5tx >= 4 
           AND time_gap_minutes < 60 THEN 1
      ELSE 0
    END AS potential_layering
  FROM transaction_sequences
),

-- Run-length encoding for pattern analysis
run_length_stats AS (
  SELECT 
    payer_id,
    amount_pattern,
    COUNT(*) AS pattern_count,
    AVG(time_gap_minutes) AS avg_time_gap,
    -- Detect runs of same pattern (structuring indicator)
    MAX(run_length) AS max_run_length
  FROM (
    SELECT 
      payer_id,
      amount_pattern,
      time_gap_minutes,
      ROW_NUMBER() OVER(PARTITION BY payer_id ORDER BY timestamp) - 
      ROW_NUMBER() OVER(PARTITION BY payer_id, amount_pattern ORDER BY timestamp) AS run_group,
      COUNT(*) OVER(PARTITION BY payer_id, amount_pattern, 
                    ROW_NUMBER() OVER(PARTITION BY payer_id ORDER BY timestamp) - 
                    ROW_NUMBER() OVER(PARTITION BY payer_id, amount_pattern ORDER BY timestamp)) AS run_length
    FROM transaction_sequences
  )
  GROUP BY payer_id, amount_pattern
)

SELECT 
  ts.payer_id,
  ts.timestamp,
  ts.amount,
  ts.seq_position,
  ts.time_gap_minutes,
  ts.amount_pattern,
  ts.payee_diversity_5tx,
  sp.potential_smurfing,
  sp.velocity_spike,
  sp.potential_layering,
  rls.max_run_length,
  rls.pattern_count,
  
  -- Advanced sequence features
  CASE WHEN ts.amount < 10000 AND sp.potential_smurfing = 1 AND rls.max_run_length >= 3 THEN 'STRUCTURING_PATTERN'
       WHEN sp.velocity_spike = 1 AND sp.potential_layering = 1 THEN 'LAYERING_PATTERN'
       WHEN rls.pattern_count > 10 AND ts.amount_pattern != 'NOT_ROUND' THEN 'ROUND_AMOUNT_PATTERN'
       ELSE 'NORMAL'
  END AS sequence_risk_pattern,
  
  -- Statistical measures
  (ts.amount - AVG(ts.amount) OVER(PARTITION BY ts.payer_id)) / 
  NULLIF(STDDEV(ts.amount) OVER(PARTITION BY ts.payer_id), 0) AS amount_zscore

FROM transaction_sequences ts
LEFT JOIN smurfing_patterns sp ON ts.payer_id = sp.payer_id AND ts.timestamp = sp.timestamp
LEFT JOIN run_length_stats rls ON ts.payer_id = rls.payer_id AND ts.amount_pattern = rls.amount_pattern;

-- =====================================================
-- 4. ENTITY RISK PROFILES & WOE ENCODING
-- =====================================================

CREATE OR REPLACE FUNCTION `aml.calculate_woe`(good_count FLOAT64, bad_count FLOAT64, total_good FLOAT64, total_bad FLOAT64)
RETURNS FLOAT64
LANGUAGE js AS """
  if (good_count === 0 || bad_count === 0 || total_good === 0 || total_bad === 0) {
    return 0;
  }
  var good_rate = good_count / total_good;
  var bad_rate = bad_count / total_bad;
  return Math.log(good_rate / bad_rate);
""";

CREATE OR REPLACE TABLE `aml.entity_woe_features` AS
WITH entity_stats AS (
  SELECT 
    payer_id,
    COUNT(*) as total_transactions,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    STDDEV(amount) as std_amount,
    COUNT(DISTINCT payee_id) as unique_payees,
    COUNT(DISTINCT channel) as unique_channels,
    COUNT(DISTINCT DATE(timestamp)) as active_days,
    
    -- Risk indicators
    COUNTIF(amount > 10000) as high_amount_count,
    COUNTIF(EXTRACT(HOUR FROM timestamp) BETWEEN 22 AND 6) as night_tx_count,
    COUNTIF(channel = 'ATM') as atm_tx_count,
    
    -- Labels for WOE calculation (mock - replace with actual SAR flags)
    CASE WHEN RAND() < 0.05 THEN 1 ELSE 0 END as is_sar  -- Mock SAR flag
  FROM `aml.transactions`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  GROUP BY payer_id
),

-- Calculate WOE for categorical features
woe_calculations AS (
  SELECT 
    -- Amount bins
    CASE 
      WHEN avg_amount < 100 THEN 'LOW_AMOUNT'
      WHEN avg_amount < 1000 THEN 'MED_AMOUNT'
      WHEN avg_amount < 10000 THEN 'HIGH_AMOUNT'
      ELSE 'VERY_HIGH_AMOUNT'
    END as amount_bin,
    
    -- Transaction volume bins  
    CASE 
      WHEN total_transactions < 10 THEN 'LOW_VOLUME'
      WHEN total_transactions < 50 THEN 'MED_VOLUME'
      WHEN total_transactions < 200 THEN 'HIGH_VOLUME'
      ELSE 'VERY_HIGH_VOLUME'
    END as volume_bin,
    
    -- Payee diversity bins
    CASE 
      WHEN unique_payees < 5 THEN 'LOW_DIVERSITY'
      WHEN unique_payees < 20 THEN 'MED_DIVERSITY'
      ELSE 'HIGH_DIVERSITY'
    END as diversity_bin,
    
    SUM(CASE WHEN is_sar = 0 THEN 1 ELSE 0 END) as good_count,
    SUM(CASE WHEN is_sar = 1 THEN 1 ELSE 0 END) as bad_count,
    COUNT(*) as total_count
    
  FROM entity_stats
  GROUP BY 1, 2, 3
),

total_counts AS (
  SELECT 
    SUM(CASE WHEN is_sar = 0 THEN 1 ELSE 0 END) as total_good,
    SUM(CASE WHEN is_sar = 1 THEN 1 ELSE 0 END) as total_bad
  FROM entity_stats
)

SELECT 
  es.*,
  wc_amt.woe_amount,
  wc_vol.woe_volume, 
  wc_div.woe_diversity,
  
  -- Information Value calculation
  (wc_amt.good_count/tc.total_good - wc_amt.bad_count/tc.total_bad) * wc_amt.woe_amount as iv_amount,
  (wc_vol.good_count/tc.total_good - wc_vol.bad_count/tc.total_bad) * wc_vol.woe_volume as iv_volume,
  (wc_div.good_count/tc.total_good - wc_div.bad_count/tc.total_bad) * wc_div.woe_diversity as iv_diversity

FROM entity_stats es
CROSS JOIN total_counts tc
LEFT JOIN (
  SELECT amount_bin, `aml.calculate_woe`(good_count, bad_count, tc.total_good, tc.total_bad) as woe_amount
  FROM woe_calculations, total_counts tc
) wc_amt ON (
  CASE 
    WHEN es.avg_amount < 100 THEN 'LOW_AMOUNT'
    WHEN es.avg_amount < 1000 THEN 'MED_AMOUNT' 
    WHEN es.avg_amount < 10000 THEN 'HIGH_AMOUNT'
    ELSE 'VERY_HIGH_AMOUNT'
  END = wc_amt.amount_bin
)
LEFT JOIN (
  SELECT volume_bin, `aml.calculate_woe`(good_count, bad_count, tc.total_good, tc.total_bad) as woe_volume
  FROM woe_calculations, total_counts tc
) wc_vol ON (
  CASE 
    WHEN es.total_transactions < 10 THEN 'LOW_VOLUME'
    WHEN es.total_transactions < 50 THEN 'MED_VOLUME'
    WHEN es.total_transactions < 200 THEN 'HIGH_VOLUME'
    ELSE 'VERY_HIGH_VOLUME'
  END = wc_vol.volume_bin
)
LEFT JOIN (
  SELECT diversity_bin, `aml.calculate_woe`(good_count, bad_count, tc.total_good, tc.total_bad) as woe_diversity
  FROM woe_calculations, total_counts tc  
) wc_div ON (
  CASE 
    WHEN es.unique_payees < 5 THEN 'LOW_DIVERSITY'
    WHEN es.unique_payees < 20 THEN 'MED_DIVERSITY'
    ELSE 'HIGH_DIVERSITY'
  END = wc_div.diversity_bin
);

-- =====================================================
-- 5. MASTER FEATURE TABLE FOR ML MODELS
-- =====================================================

CREATE OR REPLACE TABLE `aml.ml_features` AS
SELECT 
  bf.payer_id,
  bf.timestamp,
  bf.amount,
  bf.payee_id,
  bf.channel,
  
  -- Behavioral features
  bf.tx_count_24h,
  bf.tx_count_7d, 
  bf.tx_count_30d,
  bf.total_amount_24h,
  bf.unique_payees_24h,
  bf.unique_channels_24h,
  bf.velocity_ratio_24h_7d,
  bf.amount_evt_ratio,
  bf.fano_factor,
  bf.geo_entropy,
  bf.max_geo_distance,
  bf.out_degree,
  bf.pagerank_proxy,
  bf.high_velocity_flag,
  bf.extreme_amount_flag,
  bf.geo_dispersion_flag,
  
  -- Graph features
  ge.edge_risk_category,
  ge.potential_structuring as graph_structuring_flag,
  ge.community_bridge,
  
  -- Sequential features
  sf.potential_smurfing,
  sf.velocity_spike,
  sf.potential_layering,
  sf.max_run_length,
  sf.amount_zscore,
  CASE WHEN sf.sequence_risk_pattern != 'NORMAL' THEN 1 ELSE 0 END as sequence_anomaly_flag,
  
  -- Entity WOE features
  ewf.woe_amount,
  ewf.woe_volume,
  ewf.woe_diversity,
  ewf.iv_amount,
  ewf.iv_volume,
  ewf.iv_diversity,
  ewf.is_sar as target_label,
  
  -- Risk aggregation
  (
    COALESCE(bf.high_velocity_flag, 0) + 
    COALESCE(bf.extreme_amount_flag, 0) + 
    COALESCE(bf.geo_dispersion_flag, 0) +
    CASE WHEN ge.edge_risk_category = 'HIGH_RISK' THEN 1 ELSE 0 END +
    COALESCE(sf.potential_smurfing, 0) +
    COALESCE(sf.potential_layering, 0) +
    CASE WHEN sf.sequence_risk_pattern != 'NORMAL' THEN 1 ELSE 0 END
  ) as composite_risk_score

FROM `aml.features_behavioral` bf
LEFT JOIN `aml.graph_edges` ge ON bf.payer_id = ge.source_node AND bf.payee_id = ge.target_node
LEFT JOIN `aml.sequence_features` sf ON bf.payer_id = sf.payer_id AND bf.timestamp = sf.timestamp
LEFT JOIN `aml.entity_woe_features` ewf ON bf.payer_id = ewf.payer_id

WHERE bf.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY);

-- Create indexes for performance
CREATE INDEX idx_ml_features_payer ON `aml.ml_features`(payer_id);
CREATE INDEX idx_ml_features_timestamp ON `aml.ml_features`(timestamp);
CREATE INDEX idx_ml_features_composite_risk ON `aml.ml_features`(composite_risk_score DESC);
