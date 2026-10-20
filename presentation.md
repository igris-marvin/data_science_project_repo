# SLIDE-BY-SLIDE MODEL DEFENSE OUTLINE

SLIDE 1: Problem Framing & Metric Defense — Why Macro-Averaged Recall?
- Core Objective: Classify South Africa’s 144 WSAs into High, Medium, and Low Risk categories 1–3 months in advance.
- Defense of Primary Metric (recall_macro):
  - Asymmetric Failure Cost: False Negatives cause severe crises; False Positives only trigger inspections.
  - Equity: Standard Accuracy favors metros; Macro Recall ensures rural municipalities are weighted equally.

SLIDE 2: Data Pipeline & Leakage Prevention Architecture
- Scale-Invariant Feature Engineering (WSAFeatureEngineer):
  - Population sizes vary drastically (Johannesburg 5.8M vs Laingsburg 10k).
  - Engineered ratio features (Water_Access_Ratio, Urban_Rural_Access_Gap, Unserved_Population) provide scale-invariant equity indicators.
- Leakage-Safe Encapsulation (ImbPipeline):
  - Preprocessing (imputation, scaling, transformation, SMOTE) executed strictly inside CV folds.
  - Held-out test set (29 WSAs) remained fully isolated until final evaluation.

SLIDE 3: Substantive Model Selection & Overfitting Diagnosis
- Neural Networks vs Tree Ensembles:
  - Random Forest: severe overfitting, 25.87% generalization gap (train 96.30% vs val 70.43%).
  - Deep Tabular Net (MLPClassifier): highest pre-tuning CV score (0.8377 Macro Recall).
- Defense of tanh Activations:
  - Bounded activations handled extreme demographic scale outliers better than unbounded relu.

SLIDE 5: Empirical Defense via Ablation Study & McNemar Testing
- Controlled Ablation (Q6.1):
  - Removing Feature Engineering: High-Risk Recall dropped 50% (100% → 50%).
  - Removing Tanh/Weight Decay: Recall halved again (to 50%).
- McNemar’s Test (Q6.2):
  - Result: b=3, c=1 → p=0.6250 vs Logistic Regression.
  - Defense: Statistical limits due to small test size (N=29), but practical gain is large: +43.33% High-Risk detection (83.33% vs 40.0%) with zero High-to-Low errors.

Presentation Tip:
After Slide 5, demo the Streamlit app (app.py) live with Case 1 (Medium Risk Baseline) and Case 4 (Boundary Validation Error) to prove reliability.

---

