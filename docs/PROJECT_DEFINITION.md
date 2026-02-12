# 📋 ML Project Definition Document
## Credit Card Fraud Detection System

**Document Version:** 1.0
**Date:** February 2025
**Author:** ML Strategy Team
**Status:** ✅ Approved for Development

---

## 1. EXECUTIVE SUMMARY

### Problem Statement
Our organization is experiencing significant financial losses due to credit card fraud, with an estimated annual impact of $2M in fraudulent transactions. The current rule-based detection system flags 5% of all transactions for manual review, but 60% of these flagged transactions turn out to be legitimate (false positives), creating operational inefficiency and customer friction. Additionally, an unknown but substantial portion of actual fraud passes through undetected.

### Proposed Solution
Implement a machine learning-based fraud detection system using ensemble methods (Logistic Regression, Random Forest, XGBoost) trained on historical transaction data. The system will leverage multiple techniques to handle the extreme class imbalance (99.83% normal vs 0.17% fraud): SMOTE oversampling, random undersampling, and cost-sensitive learning with class weights.

### Expected Impact

| Impact Area | Current State | Target State | Annual Value |
|-------------|---------------|--------------|--------------|
| **Primary:** Fraud Loss Reduction | $2M loss | $1.2M loss (40% ↓) | **$800K saved** |
| **Secondary:** Manual Review Reduction | 50,000 reviews (60% FP) | 25,000 reviews (30% FP) | **$250K saved** |
| **Tertiary:** Customer Experience | 3% false decline rate | 2% false decline rate | **Brand value** |
| **Total Quantifiable Savings** | - | - | **$1.05M/year** |

### ROI Analysis
- **Investment:** $200K (6 months development + infrastructure)
- **Annual Savings:** $1.05M
- **ROI:** 5.25:1 (first year), 10.5:1 (second year onwards)
- **Payback Period:** 2.3 months

### Timeline
- **MVP (Minimum Viable Model):** 4 weeks
- **Production-Ready:** 8 weeks
- **Full Deployment:** 10 weeks

---

## 2. BUSINESS OBJECTIVES (Prioritized)

### 🎯 Level 6 - CRITICAL (Must Achieve)

**Objective 2.1: Reduce Direct Fraud Losses**
| Attribute | Value |
|-----------|-------|
| **Current State** | $2M annual fraud loss |
| **Target State** | $1.2M annual loss (40% reduction) |
| **Minimum Acceptable** | $1.4M annual loss (30% reduction) |
| **Metric** | Total fraud dollars detected before authorization |
| **Measurement** | Monthly fraud chargebacks + detected fraud value |
| **Impact** | Direct P&L impact, board-level visibility |
| **Owner** | CFO |

**Objective 2.2: Maintain Transaction Approval Speed**
| Attribute | Value |
|-----------|-------|
| **Current State** | < 50ms average decision time |
| **Target State** | < 100ms average decision time |
| **Minimum Acceptable** | < 150ms (p99) |
| **Metric** | Transaction authorization latency |
| **Measurement** | Real-time monitoring, APM tools |
| **Impact** | Customer experience, checkout abandonment |
| **Owner** | VP Engineering |

---

### 🎯 Level 5 - HIGH PRIORITY

**Objective 2.3: Reduce False Positive Rate**
| Attribute | Value |
|-----------|-------|
| **Current State** | 60% false positive rate (30,000 wasted reviews/year) |
| **Target State** | 30% false positive rate (15,000 reviews saved) |
| **Minimum Acceptable** | 40% false positive rate |
| **Metric** | Precision of fraud predictions |
| **Measurement** | Weekly precision calculation on flagged transactions |
| **Impact** | Fraud team productivity, $250K operational cost |
| **Owner** | Fraud Operations Manager |

**Objective 2.4: Improve Fraud Catch Rate**
| Attribute | Value |
|-----------|-------|
| **Current State** | Unknown (estimated 60-70% catch rate) |
| **Target State** | 85% catch rate |
| **Minimum Acceptable** | 75% catch rate |
| **Metric** | Recall of fraud predictions |
| **Measurement** | Monthly analysis of chargebacks vs predictions |
| **Impact** | Direct fraud loss reduction |
| **Owner** | Chief Risk Officer |

---

### 🎯 Level 4 - IMPORTANT

**Objective 2.5: Reduce Customer Friction**
| Attribute | Value |
|-----------|-------|
| **Current State** | 3% false decline rate (30,000 customers/year) |
| **Target State** | 2% false decline rate |
| **Minimum Acceptable** | 2.5% false decline rate |
| **Metric** | Legitimate transactions incorrectly blocked |
| **Measurement** | Customer complaints + manual override analysis |
| **Impact** | Customer satisfaction, NPS, churn prevention |
| **Owner** | VP Customer Experience |

---

### 🎯 Level 3 - MODERATE

**Objective 2.6: Enable Regulatory Compliance**
| Attribute | Value |
|-----------|-------|
| **Current State** | Manual audit trail, limited explainability |
| **Target State** | Automated audit logs, explainable decisions |
| **Minimum Acceptable** | Basic decision rationale available |
| **Metric** | Audit compliance score |
| **Measurement** | Quarterly compliance review |
| **Impact** | Regulatory risk, potential fines |
| **Owner** | Chief Compliance Officer |

---

## 3. TECHNICAL METRICS MAPPING

### Primary Metrics Table

| Business KPI | ML Metric | Baseline | MVP | Target | Stretch | Measurement Method |
|--------------|-----------|----------|-----|--------|---------|-------------------|
| Fraud $ caught | **Recall** | ~65% | 75% | 85% | 90% | TP / (TP + FN) on test set |
| Review efficiency | **Precision** | 15% | 50% | 60% | 70% | TP / (TP + FP) on test set |
| Overall quality | **F1-Score** | ~25% | 65% | 70% | 80% | 2 × (P × R) / (P + R) |
| Ranking quality | **PR-AUC** | ~0.55 | 0.75 | 0.80 | 0.85 | Area under PR curve |
| Real-time capability | **Latency** | N/A | <100ms | <50ms | <20ms | p99 inference time |

### Why These Metrics (NOT Accuracy)

```
⚠️ WARNING: DO NOT USE ACCURACY FOR THIS PROBLEM

With 99.83% normal transactions:
- A model that predicts EVERYTHING as "normal" achieves 99.83% accuracy
- But catches 0% of fraud = completely useless
- This is the #1 mistake in imbalanced classification

✅ USE INSTEAD:
- Recall: "Of all actual frauds, how many did we catch?"
- Precision: "Of all transactions we flagged, how many were actual fraud?"
- F1-Score: Balances both (harmonic mean prevents gaming)
- PR-AUC: Overall ranking quality, robust to imbalance
```

### Metric Priority for Fraud Detection

```
                    ┌─────────────────────────────────────┐
                    │     FRAUD DETECTION PRIORITY        │
                    └─────────────────────────────────────┘

    RECALL (Catch Rate)          PRECISION (Efficiency)
    ─────────────────            ──────────────────────
    Missing fraud = $$$          False alarm = wasted time

    For financial institutions:
    Cost of missed fraud ($500 avg) >> Cost of review ($10)

    Therefore: Recall > Precision (but both matter)

    Optimal threshold: Maximize F1-Score
    Business adjustment: May lower threshold to increase recall
```

### Confusion Matrix Business Translation

```
                        PREDICTED
                    Normal    Fraud
              ┌─────────┬─────────┐
    Normal    │   TN    │   FP    │  ← False Positive: Annoyed customer,
ACTUAL        │ (good)  │ (waste) │    wasted review time ($10/review)
              ├─────────┼─────────┤
    Fraud     │   FN    │   TP    │  ← False Negative: FRAUD PASSES THROUGH
              │ (BAD!)  │ (good)  │    Direct $ loss ($500 avg)
              └─────────┴─────────┘

    Business Impact Matrix:
    - TN: No cost (normal transaction approved correctly)
    - TP: Fraud caught (saves $500 avg)
    - FP: Wasted review (costs $10) + Customer friction
    - FN: Missed fraud (costs $500 avg) ← WORST OUTCOME

    Cost Ratio: FN is 50x worse than FP
```

---

## 4. CONSTRAINTS & ASSUMPTIONS

### Technical Constraints

#### ✅ Verified Constraints

| Constraint | Value | Verification | Implication |
|------------|-------|--------------|-------------|
| Dataset Size | 284,807 transactions | Confirmed | Sufficient for tree-based models, marginal for deep learning |
| Fraud Rate | 0.17% (492 cases) | Confirmed | Extreme imbalance, requires special handling |
| Features | 30 (V1-V28 PCA + Time + Amount) | Confirmed | No raw features available, limited feature engineering |
| Latency Requirement | < 100ms | Business requirement | Rules out complex ensemble stacking, favors single optimized model |
| Memory Limit | 8GB RAM | Infrastructure | XGBoost/RF feasible, limits batch sizes |

#### ⚠️ Assumptions to Validate

| Assumption | Risk if Wrong | Validation Method | Validation Timeline |
|------------|---------------|-------------------|---------------------|
| PCA features (V1-V28) contain fraud signal | Model won't learn patterns | Feature importance analysis | Week 2 |
| Historical patterns predict future fraud | Model drift, poor production performance | Temporal validation split | Week 3 |
| Fraud types are homogeneous | Single model inadequate | Cluster analysis of fraud cases | Week 2 |
| No seasonal patterns | Degraded performance in some periods | Time-series analysis of fraud rate | Week 2 |
| Data is representative of production | Deployment failures | Compare with recent production sample | Week 4 |

### Business Constraints

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| Budget: $200K total | Limits team size, cloud spend | Prioritize high-impact features, use spot instances |
| Timeline: 8 weeks to production | No time for exotic approaches | Focus on proven methods (RF, XGBoost), skip deep learning |
| Maintenance: 0.5 FTE allocated | Limited monitoring capacity | Automate retraining pipeline, simple alerting |
| Stakeholder availability: 5 hrs/week | Delayed feedback cycles | Weekly sync meetings, async documentation |

### Regulatory Constraints

| Regulation | Requirement | Implementation |
|------------|-------------|----------------|
| **PCI-DSS** | Secure handling of cardholder data | Data encrypted at rest and in transit, access logging |
| **GDPR (if EU customers)** | Right to explanation for automated decisions | SHAP values for decision explanation, human override available |
| **Fair Lending Laws** | No discrimination on protected attributes | Bias testing across demographics (if data available) |
| **SOX Compliance** | Audit trail for financial decisions | Logging all predictions with timestamps, model versions |

---

## 5. RISK ASSESSMENT & MITIGATION

### Risk Heat Map

```
                    IMPACT
           Low      Medium     High
         ┌────────┬────────┬────────┐
    High │   5    │   4    │  1,2   │  ← Address immediately
         ├────────┼────────┼────────┤
PROB Med │   6    │   3    │        │  ← Monitor closely
         ├────────┼────────┼────────┤
    Low  │        │        │        │  ← Accept
         └────────┴────────┴────────┘

Legend:
1 = Data Quality Issues
2 = Class Imbalance Mishandled
3 = Data Leakage
4 = Model Drift
5 = Latency Issues
6 = Interpretability Concerns
```

---

### 🔴 HIGH RISKS (Immediate Attention Required)

#### RISK 1: Data Quality Issues

| Attribute | Details |
|-----------|---------|
| **Description** | Missing values, outliers, duplicates, or inconsistent data formats that compromise model training |
| **Probability** | 70% (most ML projects encounter this) |
| **Impact** | 🔥🔥🔥 PROJECT FAILURE - Garbage in = garbage out |
| **Root Cause** | Data pipeline issues, collection errors, ETL bugs |

**Mitigation Plan:**
```
Week 1-2: Comprehensive Data Quality Audit
├── Check 1: Missing values per column (threshold: < 5%)
├── Check 2: Duplicate transactions (threshold: 0)
├── Check 3: Outlier analysis (Amount, Time distributions)
├── Check 4: Class label consistency
├── Check 5: Feature value ranges (V1-V28 should be standardized)
└── Check 6: Temporal consistency (no future data leakage)

Deliverable: Data Quality Report with pass/fail status
```

**Go/No-Go Criteria:**
- ✅ PROCEED: < 5% missing, 0 duplicates, distributions as expected
- 🛑 STOP: > 20% data unusable → Fix data pipeline first (add 2-4 weeks)
- ↻ PIVOT: 5-20% issues → Imputation strategy + reduced confidence in results

**Owner:** Data Engineer
**Status:** 🟡 Pending (Week 1-2)

---

#### RISK 2: Class Imbalance Mishandled

| Attribute | Details |
|-----------|---------|
| **Description** | With 99.83% normal transactions, naive models predict everything as "normal" and achieve 99.83% "accuracy" while catching 0% fraud |
| **Probability** | 90% if not explicitly addressed |
| **Impact** | 🔥🔥🔥 USELESS MODEL - High accuracy, zero business value |
| **Root Cause** | Default ML algorithms optimize for accuracy |

**Mitigation Plan:**
```
Strategy 1: SMOTE (Synthetic Minority Over-sampling)
├── Generate synthetic fraud samples
├── Sampling ratio: 0.5 (fraud becomes 33% of training data)
└── Validation: Compare with original class distribution

Strategy 2: Random Undersampling
├── Reduce normal transactions to match fraud count
├── Trade-off: Lose information from normal transactions
└── Use when: Training speed is critical

Strategy 3: Class Weights (Recommended for production)
├── class_weight='balanced' in sklearn
├── scale_pos_weight in XGBoost (auto-calculated)
└── Advantage: No data manipulation, works on full dataset

Evaluation Protocol:
├── NEVER use accuracy as primary metric
├── Primary: F1-Score (≥ 65% MVP, ≥ 70% target)
├── Secondary: PR-AUC (≥ 0.75 MVP, ≥ 0.80 target)
└── Sanity check: Recall > 0 (model actually detects fraud)
```

**Go/No-Go Criteria:**
- ✅ PROCEED: F1-Score ≥ 65% with any sampling strategy
- 🛑 STOP: F1-Score < 50% after trying all strategies → Problem may not be solvable with this data
- ↻ PIVOT: One strategy works, others don't → Commit to working strategy

**Owner:** ML Engineer
**Status:** 🟢 Strategy Defined (implemented in codebase)

---

#### RISK 3: Data Leakage

| Attribute | Details |
|-----------|---------|
| **Description** | Information from the future or test set leaks into training, causing inflated validation metrics that don't hold in production |
| **Probability** | 40% (common mistake, especially with time-series data) |
| **Impact** | 🔥🔥 PRODUCTION FAILURE - Model performs 10-20% worse than expected |
| **Root Cause** | Improper train/test split, feature engineering on full dataset |

**Common Leakage Sources in Fraud Detection:**
```
❌ WRONG: Random train/test split
   - Future transactions in training predict past transactions in test
   - Leaks temporal patterns

❌ WRONG: Fit scaler on full dataset, then split
   - Test set statistics influence training
   - Artificially good normalization

❌ WRONG: Use aggregated features (e.g., "user's average transaction")
   - Aggregates include future transactions
   - Model "knows" future behavior

✅ CORRECT: Temporal split (train on old, test on new)
✅ CORRECT: Fit preprocessors ONLY on training data
✅ CORRECT: Feature engineering AFTER split, per-set
```

**Mitigation Plan:**
```
Protocol 1: Strict Temporal Validation
├── Sort transactions by time
├── Train: First 80% of transactions (older)
├── Test: Last 20% of transactions (newer)
└── Never shuffle before split

Protocol 2: Pipeline Discipline
├── All preprocessing in sklearn Pipeline
├── fit_transform ONLY on train
├── transform ONLY on test
└── Code review checklist for leakage

Protocol 3: Sanity Checks
├── If validation >> training performance → Suspect leakage
├── If simple model matches complex model → Suspect leakage
├── Compare production metrics to validation within 2 weeks
└── If production < 80% of validation → Investigate immediately
```

**Go/No-Go Criteria:**
- ✅ PROCEED: Validation and temporal test performance within 5%
- 🛑 STOP: Validation performance implausibly high (>95% F1) → Definitely leakage
- ↻ PIVOT: Production performance significantly lower → Audit feature engineering

**Owner:** ML Engineer
**Status:** 🟢 Protocol Established

---

### 🟡 MEDIUM RISKS (Monitor Closely)

#### RISK 4: Model Drift

| Attribute | Details |
|-----------|---------|
| **Description** | Fraud patterns evolve over time; fraudsters adapt to detection methods |
| **Probability** | 100% (WILL happen, only question is when) |
| **Impact** | 🔥 GRADUAL DEGRADATION - Performance decays 5-10% per quarter if not addressed |
| **Timeline** | Typically noticeable within 3-6 months |

**Monitoring Plan:**
```
Daily Metrics (Automated Alerts):
├── Prediction volume (sudden changes = distribution shift)
├── Average fraud probability (drift indicator)
├── Latency percentiles (performance degradation)
└── Alert threshold: > 2 standard deviations from baseline

Weekly Metrics (Dashboard Review):
├── Precision, Recall, F1-Score (requires labeled data)
├── Confusion matrix changes
├── Feature distribution comparisons
└── Review meeting: Every Monday, 30 minutes

Monthly Actions:
├── Full model performance audit
├── Compare to baseline established at deployment
├── Trigger retraining if F1 drops > 5%
└── Document: Model Performance Log
```

**Retraining Strategy:**
```
Automatic Retraining Triggers:
├── F1-Score drops > 5% from baseline
├── Precision OR Recall drops > 10%
├── New fraud pattern identified by fraud team
└── Quarterly scheduled retrain (regardless of metrics)

Retraining Process:
├── Collect last 6 months of labeled data
├── Retrain with same hyperparameters
├── A/B test new model vs current (10% traffic)
├── If new model better → Gradual rollout (25% → 50% → 100%)
└── Keep previous model as fallback
```

**Owner:** ML Engineer + Fraud Team
**Status:** 🟡 Plan Defined (implement post-deployment)

---

#### RISK 5: Latency Issues

| Attribute | Details |
|-----------|---------|
| **Description** | Model inference too slow for real-time transaction approval |
| **Probability** | 30% |
| **Impact** | 🔥 CANNOT DEPLOY - Transaction approval requires < 100ms |

**Mitigation Plan:**
```
Phase 1: Benchmark Early (Week 3)
├── Measure inference time on sample data
├── Test on production-equivalent hardware
├── Target: p99 < 100ms for single prediction
└── Document: Latency Benchmark Report

Phase 2: Optimization (if needed)
├── Model compression (reduce trees, depth)
├── Feature selection (remove low-importance features)
├── Quantization (float32 → float16)
├── Caching (precompute static features)
└── Hardware upgrade (GPU inference, if budget allows)

Phase 3: Architecture Alternatives
├── Two-stage model (fast filter → detailed analysis)
├── Async processing for non-blocking use cases
├── Batch inference for historical analysis
└── Hybrid: Rules for obvious cases, ML for uncertain
```

**Go/No-Go Criteria:**
- ✅ PROCEED: p99 < 100ms with acceptable F1-Score
- 🛑 STOP: Cannot achieve < 200ms even with simplest model → Architecture redesign needed
- ↻ PIVOT: Latency OK but accuracy suffers → Accept accuracy trade-off OR async processing

**Owner:** ML Engineer + DevOps
**Status:** 🟢 Benchmark Scheduled (Week 3)

---

### 🟢 LOW RISKS (Accept and Monitor)

#### RISK 6: Interpretability Concerns

| Attribute | Details |
|-----------|---------|
| **Description** | Stakeholders or regulators require explanation of why transactions are flagged |
| **Probability** | 50% (depends on regulatory environment) |
| **Impact** | 🔥 ADOPTION RESISTANCE - Fraud team doesn't trust "black box" |

**Mitigation Plan:**
```
Level 1: Feature Importance (Default)
├── Global importance: Which features matter most overall?
├── Use: model.feature_importances_ for tree models
├── Visualization: Bar chart of top 20 features
└── Audience: Technical stakeholders, model documentation

Level 2: SHAP Values (If requested)
├── Local importance: Why was THIS transaction flagged?
├── Install: pip install shap
├── Output: "V14 contributed +0.3, Amount contributed +0.2..."
├── Visualization: Force plots, waterfall charts
└── Audience: Fraud analysts, compliance, customer disputes

Level 3: Rule Extraction (If required by regulation)
├── Convert complex model to approximate decision rules
├── Tools: sklearn decision tree as surrogate
├── Trade-off: Reduced accuracy for full transparency
└── Audience: Regulators, legal team
```

**Owner:** ML Engineer
**Status:** 🟢 SHAP Implementation Available

---

## 6. SUCCESS CRITERIA (Tiered)

### Tier 1: Minimum Viable Model (MVP)
**Gate: Week 4 | Must achieve ALL to proceed**

| Criterion | Threshold | Measurement | Status |
|-----------|-----------|-------------|--------|
| Precision | ≥ 50% | Test set evaluation | 🟡 Pending |
| Recall | ≥ 75% | Test set evaluation | 🟡 Pending |
| F1-Score | ≥ 65% | Test set evaluation | 🟡 Pending |
| Inference Latency | < 100ms (p99) | Benchmark on prod hardware | 🟡 Pending |
| No Data Leakage | Temporal validation within 5% of random | Audit | 🟡 Pending |
| No Demographic Bias | Equal error rates (if data available) | Fairness audit | 🟡 Pending |

**If NOT achieved:**
- Re-evaluate data quality and feature engineering
- Consider alternative algorithms
- Extend timeline by 2 weeks for remediation
- If still failing → Escalate to stakeholders for scope discussion

---

### Tier 2: Production-Ready Model
**Gate: Week 6 | Should achieve for launch**

| Criterion | Threshold | Measurement | Status |
|-----------|-----------|-------------|--------|
| Precision | ≥ 60% | Test set evaluation | 🟡 Pending |
| Recall | ≥ 85% | Test set evaluation | 🟡 Pending |
| F1-Score | ≥ 70% | Test set evaluation | 🟡 Pending |
| PR-AUC | ≥ 0.80 | Test set evaluation | 🟡 Pending |
| ROI Projection | ≥ 3:1 | Business case validation | 🟡 Pending |
| Stakeholder Approval | > 80% confidence | Sign-off meeting | 🟡 Pending |
| Documentation | Complete | Checklist review | 🟡 Pending |

**If NOT achieved:**
- Launch with MVP thresholds + enhanced monitoring
- Plan iteration cycle within 4 weeks of launch
- Document gaps and improvement roadmap

---

### Tier 3: Stretch Goals
**Timeline: Post-launch | Nice to have**

| Goal | Target | Business Value |
|------|--------|----------------|
| Precision | ≥ 70% | 20,000 fewer wasted reviews/year |
| Recall | ≥ 90% | Additional $200K fraud prevented |
| F1-Score | ≥ 80% | Best-in-class performance |
| Automated Retraining | CI/CD pipeline | Reduced maintenance burden |
| Real-time Dashboard | Grafana/similar | Operational visibility |
| A/B Testing Framework | Feature flags | Continuous improvement |

---

## 7. DECISION FRAMEWORK

### Go/No-Go Gates

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        GATE 1: After EDA (Week 2)                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  ✅ PROCEED IF:                                                                ║
║     □ Data quality acceptable (< 10% missing values)                           ║
║     □ Clear separation visible between fraud/normal in feature distributions   ║
║     □ No critical data pipeline issues identified                              ║
║     □ Feature correlations with target exist (top features |r| > 0.1)         ║
║                                                                                 ║
║  🛑 STOP IF:                                                                   ║
║     □ > 20% data unusable or corrupted                                         ║
║     □ No visible difference between fraud/normal distributions                 ║
║     □ Fundamental data collection issues (wrong time period, etc.)            ║
║     □ Data cannot be obtained in production (feature unavailable at decision) ║
║                                                                                 ║
║  ↻ PIVOT IF:                                                                   ║
║     □ Some features useful, others not → Feature selection focus              ║
║     □ Certain fraud types detectable, others not → Segment approach           ║
║     □ External data needed → Pause modeling, acquire data                     ║
║                                                                                 ║
║  DECISION OWNER: Project Lead + Data Engineer                                  ║
║  DOCUMENTATION: EDA Report with go/no-go recommendation                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║                    GATE 2: After Baseline Models (Week 4)                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  ✅ PROCEED IF:                                                                ║
║     □ At least one model achieves F1-Score ≥ 65%                              ║
║     □ Recall ≥ 75% (catching most fraud)                                      ║
║     □ No obvious data leakage detected (temporal val within 5% of random)     ║
║     □ Feature importance shows reasonable patterns                             ║
║                                                                                 ║
║  🛑 STOP IF:                                                                   ║
║     □ F1-Score < 50% despite trying all sampling strategies                   ║
║     □ Model performance no better than random baseline                         ║
║     □ Severe data leakage discovered (inflated metrics)                       ║
║     □ Fundamental problem framing issue identified                            ║
║                                                                                 ║
║  ↻ PIVOT IF:                                                                   ║
║     □ Good performance on random split, poor on temporal → Drift issue        ║
║     □ One model type clearly superior → Focus resources                       ║
║     □ Certain transaction types work, others don't → Segment models           ║
║                                                                                 ║
║  DECISION OWNER: Project Lead + ML Engineer                                    ║
║  DOCUMENTATION: Baseline Model Report with metrics comparison                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║                     GATE 3: After Optimization (Week 6)                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  ✅ PROCEED IF:                                                                ║
║     □ F1-Score ≥ 70%, PR-AUC ≥ 0.80                                           ║
║     □ Inference latency < 100ms (p99)                                         ║
║     □ All audit checks passed (no leakage, acceptable bias)                   ║
║     □ Model behavior reasonable (SHAP explanations make sense)                ║
║                                                                                 ║
║  🛑 STOP IF:                                                                   ║
║     □ Cannot meet latency AND performance simultaneously                      ║
║     □ Significant bias detected across demographics                           ║
║     □ Cost of deployment exceeds projected savings                            ║
║     □ Regulatory compliance cannot be achieved                                ║
║                                                                                 ║
║  ↻ PIVOT IF:                                                                   ║
║     □ Performance good but slow → Model compression, simpler model            ║
║     □ Fast but less accurate → Hybrid rule + ML approach                      ║
║     □ Stakeholder trust issues → Add human-in-the-loop for edge cases        ║
║                                                                                 ║
║  DECISION OWNER: Project Lead + Stakeholders                                   ║
║  DOCUMENTATION: Optimization Report with production readiness assessment       ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║                     GATE 4: Staging Deployment (Week 8)                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  ✅ LAUNCH IF:                                                                 ║
║     □ Shadow mode testing shows ≥ 30% improvement vs current system           ║
║     □ Stakeholder approval obtained (≥ 80% confidence)                        ║
║     □ Monitoring infrastructure operational                                    ║
║     □ Rollback plan documented and tested                                     ║
║     □ On-call rotation established                                            ║
║                                                                                 ║
║  🛑 STOP IF:                                                                   ║
║     □ Shadow testing shows degradation vs current system                      ║
║     □ Production environment reveals critical issues                          ║
║     □ Regulatory approval not obtained                                        ║
║     □ Stakeholders withdraw support                                           ║
║                                                                                 ║
║  ↻ PIVOT IF:                                                                   ║
║     □ Adoption concerns → Phased rollout (10% → 25% → 50% → 100%)            ║
║     □ Trust issues → Human review for high-stakes decisions                   ║
║     □ Edge cases problematic → Rules for known patterns, ML for rest         ║
║                                                                                 ║
║  DECISION OWNER: Project Lead + VP Engineering + CFO                           ║
║  DOCUMENTATION: Launch Checklist + Runbook                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 8. PROJECT TIMELINE

### Gantt Chart Overview

```
Week    1    2    3    4    5    6    7    8    9    10
        ├────┼────┼────┼────┼────┼────┼────┼────┼────┤
        │▓▓▓▓▓▓▓▓▓│    │    │    │    │    │    │    │ Phase 1: EDA & Data Quality
        │    │    │▓▓▓▓▓▓▓▓▓│    │    │    │    │    │ Phase 2: Baseline Models
        │    │    │    │    │▓▓▓▓▓▓▓▓▓│    │    │    │ Phase 3: Optimization
        │    │    │    │    │    │    │▓▓▓▓▓▓▓▓▓│    │ Phase 4: Staging & Testing
        │    │    │    │    │    │    │    │    │▓▓▓▓│ Phase 5: Production Launch
        │    │    │    │    │    │    │    │    │    │
        │  G1│    │  G2│    │  G3│    │  G4│    │ GO │ Gates
        └────┴────┴────┴────┴────┴────┴────┴────┴────┘

Legend: ▓ = Active work, G = Go/No-Go Gate
```

### Detailed Phase Breakdown

| Week | Phase | Key Activities | Deliverables | Gate |
|------|-------|----------------|--------------|------|
| 1 | EDA | Data loading, quality checks, missing value analysis | Data Quality Report | - |
| 2 | EDA | Feature analysis, correlation study, imbalance assessment | EDA Notebook, Visualizations | **G1** |
| 3 | Baseline | Train LR, RF, XGBoost with SMOTE/Undersampling/Weights | Baseline metrics | - |
| 4 | Baseline | Model comparison, leakage audit, temporal validation | Baseline Model Report | **G2** |
| 5 | Optimization | Hyperparameter tuning, threshold optimization | Tuned model | - |
| 6 | Optimization | Fairness audit, interpretability (SHAP), final evaluation | Optimization Report | **G3** |
| 7 | Staging | Deploy to staging, shadow mode testing, monitoring setup | Staging deployment | - |
| 8 | Staging | A/B testing, stakeholder demo, documentation finalization | Launch Checklist | **G4** |
| 9-10 | Launch | Production deployment, monitoring, initial support | Live system | - |

### Critical Path

```
Data Quality ──────► Must pass before modeling begins
       │
       ▼
Baseline Model ────► Must achieve MVP metrics
       │
       ▼
Optimization ──────► Must meet latency requirements
       │
       ▼
Staging Tests ─────► Must show improvement over current system
       │
       ▼
Production Launch
```

### Dependencies & Risks to Timeline

| Dependency | Risk | Mitigation | Impact if Delayed |
|------------|------|------------|-------------------|
| Data access | Medium | Early data pipeline validation | +1-2 weeks |
| Stakeholder availability | Medium | Schedule reviews in advance | +1 week per review |
| Infrastructure provisioning | Low | Use existing cloud resources | +1 week |
| Regulatory approval | High | Start compliance review Week 4 | +2-4 weeks |

---

## 9. TEAM & RESPONSIBILITIES

### RACI Matrix

| Activity | Project Lead | ML Engineer | Data Engineer | Fraud Team | Compliance |
|----------|:------------:|:-----------:|:-------------:|:----------:|:----------:|
| Project planning | **A** | C | C | I | I |
| Data quality audit | A | C | **R** | C | I |
| EDA & visualization | A | **R** | C | C | I |
| Model development | A | **R** | I | C | I |
| Hyperparameter tuning | I | **R** | I | I | I |
| Fairness & bias audit | A | **R** | I | C | **C** |
| Interpretability (SHAP) | I | **R** | I | C | C |
| Staging deployment | A | R | **R** | I | I |
| A/B testing | A | **R** | C | **C** | I |
| Monitoring setup | A | C | **R** | I | I |
| Documentation | A | **R** | C | C | C |
| Stakeholder communication | **R** | C | I | C | C |
| Go/No-Go decisions | **A** | C | C | C | C |

**Legend:** R = Responsible, A = Accountable, C = Consulted, I = Informed

### Team Allocation

| Role | Allocation | Weekly Hours | Key Responsibilities |
|------|------------|--------------|---------------------|
| Project Lead | 50% | 20 hrs | Strategy, stakeholder management, decisions |
| ML Engineer | 100% | 40 hrs | Model development, optimization, deployment |
| Data Engineer | 50% | 20 hrs | Data pipeline, quality, infrastructure |
| Fraud Team Lead | 25% | 10 hrs | Domain expertise, validation, feedback |
| Compliance Officer | 10% | 4 hrs | Regulatory review, approval |

### Escalation Path

```
Level 1: Technical Issues
ML Engineer → Project Lead
Resolution: Within 24 hours

Level 2: Resource/Timeline Issues
Project Lead → VP Engineering
Resolution: Within 48 hours

Level 3: Business/Strategic Issues
VP Engineering → CFO/CRO
Resolution: Within 1 week
```

---

## 10. MONITORING & MAINTENANCE PLAN

### Production Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION MODEL HEALTH                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  📊 REAL-TIME METRICS (Last 24 hours)                                │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐          │
│  │ Predictions │   Latency   │  Fraud Rate │   Alerts    │          │
│  │   45,230    │   23ms p50  │    0.18%    │     2       │          │
│  │   ▲ 5%      │   67ms p99  │   ▬ stable  │   ⚠ review  │          │
│  └─────────────┴─────────────┴─────────────┴─────────────┘          │
│                                                                       │
│  📈 WEEKLY PERFORMANCE                                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐          │
│  │  Precision  │   Recall    │  F1-Score   │   PR-AUC    │          │
│  │    62.3%    │   83.7%     │   71.4%     │   0.812     │          │
│  │  ▲ +1.2%    │  ▼ -0.8%    │  ▲ +0.3%    │  ▬ stable   │          │
│  └─────────────┴─────────────┴─────────────┴─────────────┘          │
│                                                                       │
│  🎯 VS BASELINE (Deployment: Jan 2026)                               │
│  Precision: +312% │ Recall: +24% │ F1: +186%                         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Monitoring Schedule

| Frequency | Metrics | Owner | Action Threshold |
|-----------|---------|-------|------------------|
| **Real-time** | Prediction count, latency, error rate | Automated | Error rate > 1% → Page on-call |
| **Hourly** | Fraud probability distribution | Automated | Mean shift > 2σ → Alert |
| **Daily** | Prediction volume trends | Data Engineer | > 20% change → Investigate |
| **Weekly** | Precision, Recall, F1, PR-AUC | ML Engineer | F1 drop > 3% → Review |
| **Monthly** | Full performance audit, drift analysis | ML Engineer + Fraud Team | F1 drop > 5% → Retrain |
| **Quarterly** | Model refresh, feature review | Full Team | Scheduled retrain |

### Alert Definitions

```python
# Alert Configuration
ALERTS = {
    "latency_high": {
        "condition": "p99_latency > 100ms for 5 minutes",
        "severity": "HIGH",
        "action": "Page on-call, consider traffic shedding"
    },
    "error_rate_high": {
        "condition": "error_rate > 1% for 10 minutes",
        "severity": "CRITICAL",
        "action": "Page on-call, activate fallback rules"
    },
    "prediction_drift": {
        "condition": "mean_fraud_prob change > 50% vs yesterday",
        "severity": "MEDIUM",
        "action": "Investigate data pipeline, alert ML Engineer"
    },
    "performance_degradation": {
        "condition": "weekly_f1 < 0.65",
        "severity": "HIGH",
        "action": "Trigger model review, consider retraining"
    }
}
```

### Maintenance Schedule

| Cadence | Activity | Owner | Duration |
|---------|----------|-------|----------|
| Weekly | Performance review meeting | ML Engineer + Fraud Team | 30 min |
| Monthly | Model drift analysis | ML Engineer | 4 hrs |
| Monthly | Retraining evaluation | ML Engineer | 8 hrs |
| Quarterly | Feature engineering review | Full Team | 1 day |
| Quarterly | Architecture review | ML Engineer + DevOps | 4 hrs |
| Annually | Full model rebuild | Full Team | 2-4 weeks |

### Retraining Protocol

```
AUTOMATIC RETRAINING TRIGGER:
├── Condition: F1-Score drops > 5% from baseline for 2 consecutive weeks
├── OR: Precision OR Recall drops > 10%
├── OR: Quarterly scheduled retrain
│
RETRAINING PROCESS:
├── 1. Collect last 6 months of labeled transaction data
├── 2. Run full training pipeline (same hyperparameters)
├── 3. Evaluate on holdout set from most recent month
├── 4. If new_model_f1 > current_model_f1:
│       └── Deploy to shadow mode (10% traffic) for 1 week
│       └── If shadow performance good → Gradual rollout
├── 5. If new_model_f1 ≤ current_model_f1:
│       └── Investigate (data quality? fraud pattern change?)
│       └── Consider hyperparameter re-tuning
│       └── Escalate if no improvement after 2 attempts
│
ROLLBACK PLAN:
├── Keep previous model version deployed in parallel
├── Feature flag to switch traffic instantly
├── Rollback decision: Production F1 < Shadow F1 by > 5%
└── Rollback execution: < 5 minutes via feature flag
```

---

## 11. APPENDIX

### A. Glossary of Terms

| Term | Definition | Business Context |
|------|------------|------------------|
| **Precision** | TP / (TP + FP) | "Of transactions we flag, what % are actually fraud?" |
| **Recall** | TP / (TP + FN) | "Of all actual frauds, what % do we catch?" |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean, balances precision and recall |
| **PR-AUC** | Area under Precision-Recall curve | Overall model quality, robust to imbalance |
| **ROC-AUC** | Area under ROC curve | Less useful for imbalanced data (can be misleading) |
| **SMOTE** | Synthetic Minority Over-sampling | Creates artificial fraud examples for training |
| **Class Weights** | Penalty multiplier for classes | Makes model care more about minority class |
| **Data Leakage** | Future info in training data | Causes inflated metrics, production failure |
| **Model Drift** | Performance decay over time | Fraudsters adapt, patterns change |
| **SHAP Values** | Feature contribution scores | Explains why a specific prediction was made |

### B. Technical Specifications

**Model Configuration (XGBoost - Recommended):**
```python
{
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": "auto",  # Calculated from class ratio
    "eval_metric": "logloss",
    "random_state": 42
}
```

**Prediction Threshold Strategy:**
```python
# Threshold tuning based on business cost
# Cost of False Negative (missed fraud): ~$500
# Cost of False Positive (wasted review): ~$10
# Cost ratio: 50:1

# Optimal threshold typically around 0.3-0.4 for fraud detection
# (lower than default 0.5 to catch more fraud at expense of more reviews)

thresholds = {
    "auto_block": 0.80,      # High confidence → block immediately
    "manual_review": 0.40,   # Medium confidence → human review
    "monitor": 0.20,         # Low confidence → log for analysis
    "approve": 0.00          # Below threshold → approve transaction
}
```

### C. Dataset Specifications

| Attribute | Value |
|-----------|-------|
| Source | Kaggle Credit Card Fraud Detection Dataset |
| Total Records | 284,807 transactions |
| Time Period | September 2013 (2 days) |
| Features | 30 (V1-V28 PCA components + Time + Amount) |
| Target | Class (0 = Normal, 1 = Fraud) |
| Class Distribution | 99.83% Normal, 0.17% Fraud |
| Missing Values | 0 |
| File Size | ~144 MB |

### D. References

1. **Dataset**: https://www.kaggle.com/mlg-ulb/creditcardfraud
2. **SMOTE Paper**: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
3. **XGBoost Paper**: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016)
4. **SHAP Paper**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)
5. **Imbalanced Learning**: He & Garcia, "Learning from Imbalanced Data" (2009)

### E. Stakeholder Sign-Off

| Stakeholder | Role | Approval Status | Date |
|-------------|------|-----------------|------|
| [Name] | CFO | ⬜ Pending | - |
| [Name] | VP Engineering | ⬜ Pending | - |
| [Name] | Fraud Operations Manager | ⬜ Pending | - |
| [Name] | Chief Risk Officer | ⬜ Pending | - |
| [Name] | Chief Compliance Officer | ⬜ Pending | - |

---

**Document Control:**
- Version: 1.0
- Created: January 2026
- Last Updated: January 2026
- Next Review: After Gate 1 (Week 2)
- Owner: Project Lead
