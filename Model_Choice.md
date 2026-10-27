# CrediLens AI — Model Selection & Underwriting Architecture Report

## 1. Executive Summary

In credit risk assessment and automated underwriting, predicting borrower default is not merely an exercise in maximizing raw accuracy. Because credit default is an inherently **imbalanced** classification task (approx. 78.2% non-default vs. 21.8% default) with **asymmetric business costs**, the modeling strategy must balance default detection sensitivity (Recall) against credit approval rates (Precision), while delivering reliable, well-calibrated probabilities for risk-based loan pricing.

After rigorous benchmarking across 5-fold Stratified Cross-Validation and an independent holdout test set (6,482 loans), the selected production model is:

> **Champion Model:** **Calibrated XGBoost Classifier (`CalibratedClassifierCV` via Sigmoid Scaling)**  
> **Optimal Operating Threshold:** **`0.3638`** (F1-optimal) / **`0.5000`** (Conservative Low-False-Positive)  
> **Key Performance:** **PR-AUC (Average Precision) = 0.9132**, **ROC-AUC = 0.9534**, **Brier Score = 0.0517**

```mermaid
flowchart TD
    A["Raw Loan Data (32,581 Records)"] --> B["Data Validation & Outlier Removal (32,409 Clean Records)"]
    B --> C["Stratified Split 80/20 & Imbalance Weight scale_pos_weight = 3.573"]
    C --> D["ColumnTransformer Preprocessing (Imputation & One-Hot Encoding)"]
    D --> E["Candidate Model Benchmarking (5-Fold Stratified CV)"]
    
    subgraph Candidate_Models["Candidate Models"]
        E1["Baseline Logistic Regression"]
        E2["LightGBM Classifier"]
        E3["XGBoost Classifier"]
    end
    
    E --> E1
    E --> E2
    E --> E3
    
    E3 -->|Winner: Highest AP & Precision| F["Hyperparameter Tuned XGBoost"]
    F --> G["Probability Calibration (CalibratedClassifierCV Sigmoid)"]
    G --> H["Threshold Optimization (F1-Max & Target Recall)"]
    H --> I["Champion Model & Threshold Artifacts Saved"]
    I --> J["SHAP TreeExplainer (Global & Local Interpretability)"]
    I --> K["FastAPI Model Serving & Web UI"]
```

---

## 2. Business Problem & Evaluation Criteria

### 2.1 The Asymmetric Cost of Credit Risk Errors
In retail credit lending, prediction errors carry drastically different consequences:
- **False Negative (Type II Error):** The model predicts an applicant is *Low Risk / Approved (0)*, but the borrower defaults (*Actual 1*).  
  **Cost:** Loss of loan principal, accrued interest, and collection/legal recovery expenses. This is the **most financially damaging error**.
- **False Positive (Type I Error):** The model predicts an applicant is *High Risk / Default (1)*, but the borrower would have paid back in full (*Actual 0*).  
  **Cost:** Lost interest margin, customer friction, and reduced loan origination volume.

### 2.2 Why Standard Accuracy Fails
A trivial model predicting that all applicants are safe would achieve **78.2% accuracy** while catching **0% of defaults**, resulting in catastrophic credit portfolio losses. Consequently, candidate models were evaluated on:
1. **Average Precision (PR-AUC):** Area under the Precision-Recall Curve. Evaluates how well the model separates defaulters from non-defaulters across all possible probability thresholds without being skewed by the large majority of non-defaulters.
2. **ROC-AUC:** Measures ranking discrimination power between default and non-default.
3. **F1-Score:** Harmonic mean of Precision and Recall.
4. **Brier Score:** Mean squared error between predicted probabilities and binary outcomes. Lower is better; measures the trustworthiness of default probabilities for Expected Loss calculation ($EL = PD \times LGD \times EAD$).

---

## 3. Candidate Models & Architecture Comparison

We evaluated three model architectures spanning linear baselines, leaf-wise gradient boosting, and depth-wise gradient boosting:

| Feature / Dimension | Logistic Regression (Baseline) | LightGBM | XGBoost (Champion) |
| :--- | :--- | :--- | :--- |
| **Model Type** | Generalized Linear Model | Histogram-based Gradient Boosting | Extreme Gradient Boosting |
| **Tree Growth Policy** | N/A (Linear) | Leaf-wise (best-first) | Depth-wise (level-wise) |
| **Imbalance Treatment** | `class_weight="balanced"` | `scale_pos_weight=3.573` | `scale_pos_weight=3.573` |
| **Non-Linear Interactions** | Requires manual feature crosses | Automatic, high-order | Automatic, regularized |
| **Outlier Robustness** | Sensitive (requires scaling) | Robust | Robust with shrinkage & $\gamma$ |
| **Regularization** | $L_2$ Ridge penalty | `reg_alpha`, `reg_lambda`, `min_child_weight` | Depth limit, $L_1$, $L_2$, `min_child_weight`, `gamma` |
| **SHAP Interpretability** | Standard coefficients | TreeExplainer | TreeExplainer (native C++ speed) |

---

## 4. Benchmark Performance Comparison

### 4.1 5-Fold Stratified Cross-Validation (Train Split: 25,927 loans)
All models were evaluated across 5 stratified folds to ensure out-of-fold generalization:

| Model | Average Precision (PR-AUC) | ROC-AUC | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline) | 0.7192 | 0.8702 | 0.6438 | 0.5504 | 0.7755 |
| **LightGBM** | 0.8941 | 0.9438 | 0.7890 | 0.7844 | 0.7938 |
| **XGBoost (Champion)** | **0.9026** | **0.9481** | **0.8059** | **0.8120** | **0.8000** |

### 4.2 Independent Test Set Benchmark (Holdout: 6,482 loans)
Evaluated on the completely unseen holdout set:

| Model Architecture | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (AP) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5000 | 80.59% | 53.89% | 78.21% | 0.6381 | 0.8716 | 0.7082 | 0.1394 |
| **LightGBM** | 0.5000 | 91.22% | 78.78% | 81.95% | 0.8033 | 0.9508 | 0.9067 | 0.0708 |
| **Raw XGBoost** | 0.5000 | 92.21% | 82.17% | 82.23% | 0.8220 | 0.9528 | 0.9117 | 0.0633 |
| **Calibrated XGBoost (Optimal Thr)** | **0.3638** | **92.59%** | **84.38%** | **81.17%** | **0.8275** | **0.9534** | **0.9132** | **0.0517** |
| **Calibrated XGBoost (Default Thr)** | 0.5000 | 93.69% | 91.39% | 78.56% | 0.8449 | 0.9534 | 0.9132 | **0.0517** |

---

## 5. Why XGBoost Was Chosen Over Other Candidates

### 5.1 Superior Precision-Recall Performance (PR-AUC: 0.9132 vs. 0.9067 & 0.7082)
The baseline Logistic Regression model achieves only 0.7082 PR-AUC and suffers from a low precision of 53.89% (nearly half of all flagged applicants were safe borrowers turned away). Both tree ensembles significantly outclassed linear modeling by capturing non-linear interactions between borrower income and debt commitment.

Comparing XGBoost against LightGBM:
- **Precision Advantage:** In the 1,000-sample bootstrap validation at target recall $80\%$, XGBoost maintained a statistically significant precision advantage of **$+1.8\%$** over LightGBM ($95\%\text{ CI: } [+0.4\%, +3.2\%]$).
- **Lower Error Rate:** On the test set, XGBoost produced 60 fewer False Positives and 4 fewer False Negatives than LightGBM.

### 5.2 Regularization & Overfitting Resistance
LightGBM's leaf-wise tree growth can aggressively specialize on rare feature combinations in financial data. XGBoost's depth-wise constraint (`max_depth=5`) combined with explicit gamma regularization (`gamma=2.68`), subsample row-sampling (`0.705`), and column-sampling (`0.985`) produced tighter generalizability between cross-validation and test scores (CV AP 0.9026 vs. Test AP 0.9117).

---

## 6. The Necessity of Probability Calibration

Standard gradient boosted trees output scores that rank order well, but their raw probability outputs are frequently **distorted**. Boosting minimizes logistic loss by driving margins higher, which pushes probability estimates away from the center toward 0 and 1.

To resolve this, we applied **Platt Sigmoid Calibration** via 5-fold cross-validation (`CalibratedClassifierCV(method="sigmoid", cv=5)`):

| Metric | Raw XGBoost | Calibrated XGBoost | Improvement |
| :--- | :---: | :---: | :---: |
| **Brier Score** (lower is better) | 0.0633 | **0.0517** | **18.3% error reduction** |
| **Expected Calibration Error** | Distorted tail scores | Aligned with empirical frequency | High reliability |
| **PR-AUC / ROC-AUC** | 0.9117 / 0.9528 | 0.9132 / 0.9534 | Monotonicity preserved |

> **Underwriting Impact:** A calibrated probability of $20\%$ means that out of 100 borrowers with this score, approximately 20 will default. This allows the credit risk team to accurately calculate risk-adjusted capital pricing and provisions.

---

## 7. Decision Threshold Optimization

The default threshold of $0.50$ is rarely optimal for asymmetric credit portfolios. We analyzed the complete Precision-Recall frontier on training probabilities to determine the optimal operating threshold:

```
Threshold = 0.3638 (F1-Optimal Operating Point)
────────────────────────────────────────────────────────
• Precision: 84.38% (When model flags Default, it is right 84.4% of the time)
• Recall:    81.17% (Catches 81.2% of all actual defaulters)
• F1-Score:  0.8275 (Optimal trade-off)
• False Positives: 213 out of 5,064 non-defaulters (4.2% error rate)
• False Negatives: 267 out of 1,418 defaulters
```

```
Threshold = 0.5000 (Conservative / Volume-Maximizing Point)
────────────────────────────────────────────────────────
• Precision: 91.39% (Extremely high confidence in flagged defaults)
• Recall:    78.56% (Catches 78.6% of defaults)
• F1-Score:  0.8449
• False Positives: Only 105 out of 5,064 non-defaulters (2.1% error rate)
```

Depending on macroeconomic cycles, the underwriting policy can dynamically toggle between:
- **0.3638 (Risk-Averse / Tight Credit Cycle):** Maximizes default interception.
- **0.5000 (Growth / Expansion Cycle):** Minimizes friction on safe applicants with 91.4% precision.

---

## 8. Explainability via SHAP (SHapley Additive exPlanations)

To comply with Fair Lending regulations (e.g., adverse action notice requirements) and ensure auditability, the pipeline integrates **TreeSHAP** on the champion model.

### 8.1 Global Feature Importance Ranking
The top features driving credit risk portfolio-wide:

1. **`person_income` (Mean $|SHAP| = 1.127$):** Annual gross income is the strongest baseline risk anchor. Lower income dramatically elevates default likelihood.
2. **`loan_percent_income` (Mean $|SHAP| = 0.835$):** The ratio of requested loan amount to annual income. When this ratio exceeds $0.25$–$0.30$, risk scores escalate rapidly regardless of credit history.
3. **`loan_int_rate` (Mean $|SHAP| = 0.602$):** Market-priced interest rate reflects the risk premium. Higher rates compound repayment burdens.
4. **`person_home_ownership_RENT` (Mean $|SHAP| = 0.435$):** Renters exhibit higher default probability than homeowners (`OWN` or `MORTGAGE`).
5. **`loan_grade` (Grades D, E, F):** Subprime loan grades directly push applicants past the threshold.

---

## 9. Generated Visual Artifacts

All supporting diagrams have been generated and saved into the project artifacts directory:

| Diagram | File Path | Key Takeaway |
| :--- | :--- | :--- |
| **Model Comparison PR Curves** | [`artifacts/visuals/model_comparison_pr_curve.png`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/visuals/model_comparison_pr_curve.png) | Demonstrates XGBoost's dominant PR curve over LightGBM and Logistic Regression across all recall thresholds. |
| **Calibration Reliability Curve** | [`artifacts/visuals/calibration_curve.png`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/visuals/calibration_curve.png) | Shows calibrated XGBoost hugging the ideal diagonal line, whereas raw XGBoost displays probability under-dispersion. |
| **Confusion Matrix** | [`artifacts/visuals/confusion_matrix.png`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/visuals/confusion_matrix.png) | Visualizes the 4,851 TN, 1,151 TP, 213 FP, and 267 FN distribution at threshold 0.364. |
| **Precision-Recall Curve** | [`artifacts/visuals/precision_recall_curve.png`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/visuals/precision_recall_curve.png) | Marks the exact optimal operating point (P=84.4%, R=81.2%) against the baseline. |
| **ROC Curve** | [`artifacts/visuals/roc_curve.png`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/visuals/roc_curve.png) | Illustrates the 0.953 AUC discrimination curve. |
| **SHAP Summary Beeswarm** | [`artifacts/visuals/shap_summary_plot.png`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/visuals/shap_summary_plot.png) | Displays feature value distributions and positive/negative directional forces on default risk. |
| **SHAP Feature Importance** | [`artifacts/visuals/shap_importance_bar.png`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/visuals/shap_importance_bar.png) | Bar chart ranking global importance by mean absolute SHAP value. |
| **SHAP Local Waterfall** | [`artifacts/visuals/shap_waterfall_sample.png`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/visuals/shap_waterfall_sample.png) | Step-by-step risk breakdown showing why sample applicant #5 received their risk score. |

---

## 10. Conclusion & Deployment Recommendation

The **Calibrated XGBoost Classifier** represents the optimal trade-off between statistical performance, probability reliability, and regulatory explainability:
1. **Performance:** Outperformed all candidates in Cross-Validation and Test sets (PR-AUC: 0.9132, ROC-AUC: 0.9534).
2. **Reliability:** Sigmoid calibration achieved a low Brier Score of 0.0517, making default probabilities directly usable for loan pricing and expected loss modeling.
3. **Auditability:** Integrated TreeSHAP allows instant generation of adverse action explanations for loan applicants.
4. **Serving Architecture:** Exported as a serialized pipeline in [`artifacts/models/credit_risk_model.pkl`](file:///C:/Users/sumit/OneDrive/Desktop/Code_PlayGround/End-To-End-ML-Projects/CrediLens-AI/artifacts/models/credit_risk_model.pkl) and served in real-time via the FastAPI backend (`app/api/app.py`).
