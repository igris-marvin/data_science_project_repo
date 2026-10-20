# ^ IMPORTS
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, RobustScaler

from wsa_transformer import WSAFeatureEngineer

# ^ QUESTION 2

warnings.filterwarnings("ignore")

df = pd.read_csv("NIWIS_Water Supply Reliability - Population_21-Sep-2026 .csv")

df.head()

df["Unnamed: 0"]

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=["Unnamed: 0"]).copy()

df.head()

## * Clean numerical columns. remove thousands separators and % signs
num_cols = df.columns.drop("WSA Name")

for col in num_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"[ %]", "", regex=True)  # remove spaces and %
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.head()

## * Target Population Reliable Supply into 3 risk levels
df["Target_Risk_Class"] = pd.cut(
    df["% Population Reliable Supply"],
    bins=[-float("inf"), 50, 75, float("inf")],
    labels=["High Risk", "Medium Risk", "Low Risk"]
)

df.head()

## * 2.1 Create a data dictionary and audit class balance, duplicates, missingness, invalid values, outliers and possible leakage.

# Data dictionary
print(df.dtypes)

# Dataset shape & duplicates
print(f" - Total Rows (WSAs): {df.shape[0]}")
print(f" - Total Columns: {df.shape[1]}")
print(f" - Exact Duplicate Rows: {df.duplicated().sum()}")
print(f" - Unique WSA Names: {df['WSA Name'].nunique()}")

# Class balance
counts = df["Target_Risk_Class"].value_counts()
props = df["Target_Risk_Class"].value_counts(normalize=True).mul(100).round(2)
for cat in ["Low Risk", "Medium Risk", "High Risk"]:
    print(f" - {cat:<12}: {counts[cat]:>3} WSAs ({props[cat]:.2f}%)")

# Missingness
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
if missing.sum() == 0:
    print(" - No missing values found.")
else:
    for col in df.columns:
        if missing[col] > 0:
            print(f" - {col:<38}: {missing[col]} nulls ({missing_pct[col]:.1f}%)")

# Invalid values audit
invalid_pct = ((df["% Population Reliable Supply"] < 0) | 
               (df["% Population Reliable Supply"] > 100)).sum()
print(f" - % Population Reliable Supply outside 0–100: {invalid_pct}")

# Outlier - Statistics
print(df.describe())

# Data leakage
leakage_cols = [
    "% Population Reliable Supply",
    "Population Reliable Supply",
    "Urban Population Reliable Supply",
    "% Urban Population Reliable Supply",
    "Rural Population Reliable Supply",
    "% Rural Population Reliable Supply",
]

## * 2.2 five purposeful visualisations: target distribution; missingness/data quality; two feature–target relationships; and one geographic, temporal or subgroup view. Each figure must have a Wtitle, units, readable labels and an interpretation.

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
palette_dict = {
    "Low Risk": "#5cb85c",
    "Medium Risk": "#f0ad4e",
    "High Risk": "#d9534f",
}

plt.figure(figsize=(8, 5))
ax1 = sns.countplot(
    data=df,
    x="Target_Risk_Class",
    hue="Target_Risk_Class",
    order=["Low Risk", "Medium Risk", "High Risk"],
    palette=palette_dict,
    legend=False,
)
plt.title("Figure 1: Target Risk Class Distribution Across 144 WSAs",
          fontsize=12, fontweight="bold")
plt.xlabel("Target Risk Class", fontsize=10, fontweight="bold")
plt.ylabel("Number of WSAs", fontsize=10, fontweight="bold")

# Annotate bars
for p in ax1.patches:
    height = int(p.get_height())
    pct = (height / len(df)) * 100
    ax1.annotate(f"{height} ({pct:.1f}%)",
                 (p.get_x() + p.get_width() / 2.0, height),
                 ha="center", va="center", xytext=(0, 5),
                 textcoords="offset points", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.show()

feature_cols = [c for c in num_cols if c not in [
    "% Population Reliable Supply", "Population Reliable Supply",
    "Urban Population Reliable Supply", "% Urban Population Reliable Supply",
    "Rural Population Reliable Supply", "% Rural Population Reliable Supply"
]]
missing_series = df[feature_cols].isnull().sum()

plt.figure(figsize=(9, 5))
ax2 = missing_series.plot(kind="barh", color="#0275d8")
plt.title("Figure 2: Attribute Missingness Profile Across Predictors (144 WSAs)",
          fontsize=12, fontweight="bold")
plt.xlabel("Missing Records (Count)", fontsize=10, fontweight="bold")
plt.ylabel("Predictor Features", fontsize=10, fontweight="bold")

# Annotate missing counts
for i, v in enumerate(missing_series):
    if v > 0:
        ax2.text(v + 0.1, i, f"{v} ({v/len(df)*100:.1f}%)",
                 va="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df, x="Target_Risk_Class", y="% Rural Population Reliable Supply",
    order=["Low Risk", "Medium Risk", "High Risk"],
    palette=palette_dict, legend=False
)
sns.stripplot(
    data=df, x="Target_Risk_Class", y="% Rural Population Reliable Supply",
    order=["Low Risk", "Medium Risk", "High Risk"],
    color="black", alpha=0.3, jitter=0.2
)
plt.title("Figure 3: Rural Water Supply Reliability (%) Across Risk Classes",
          fontsize=12, fontweight="bold")
plt.xlabel("Target Risk Class", fontsize=10, fontweight="bold")
plt.ylabel("Rural Population Reliable Supply (%)", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.show()

df["Urban_Ratio"] = (df["Urban Population"] / df["Population"]) * 100

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df, x="Target_Risk_Class", y="Urban_Ratio",
    order=["Low Risk", "Medium Risk", "High Risk"],
    palette=palette_dict, legend=False
)
sns.stripplot(
    data=df, x="Target_Risk_Class", y="Urban_Ratio",
    order=["Low Risk", "Medium Risk", "High Risk"],
    color="black", alpha=0.3, jitter=0.2
)
plt.title("Figure 4: Municipal Urbanisation Ratio (%) Across Risk Classes",
          fontsize=12, fontweight="bold")
plt.xlabel("Target Risk Class", fontsize=10, fontweight="bold")
plt.ylabel("Urban Population Ratio (%)", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.show()

top5 = df.nlargest(5, "% Population Reliable Supply")
bottom5 = df.nsmallest(5, "% Population Reliable Supply")
extreme_wsas = pd.concat([bottom5, top5])

plt.figure(figsize=(10, 5))
sns.barplot(
    data=extreme_wsas, y="WSA Name", x="% Population Reliable Supply",
    hue="Target_Risk_Class", palette=palette_dict, dodge=False
)
plt.title("Figure 5: Extreme Disparities: Bottom 5 vs Top 5 WSAs by Reliability",
          fontsize=12, fontweight="bold")
plt.xlabel("Population Reliable Supply (%)", fontsize=10, fontweight="bold")
plt.ylabel("Water Services Authority (WSA)", fontsize=10, fontweight="bold")
plt.legend(title="Risk Class", loc="lower right")
plt.tight_layout()
plt.show()

## * 2.3 Split data before learning transformations. Justify train/validation/test or nested cross-validation design, including stratification/group/time constraints. Reserve the test set for final evaluation only.
# -------------------------------------------------------------------------
# Define leakage columns to drop (prevent target leakage & feature mismatch)
# -------------------------------------------------------------------------
leakage_cols = [
    # 1. Target-derived percentage columns (direct leakage)
    "% Population Reliable Supply",
    "% Urban Population Reliable Supply",
    "% Rural Population Reliable Supply",

    # 2. Raw count reliable supply columns (avoid mismatch in app.py)
    "Population Reliable Supply",
    "Urban Population Reliable Supply",
    "Rural Population Reliable Supply",
]
feature_cols = [c for c in num_cols if c not in leakage_cols]
X = df[feature_cols].copy()
y = df["Target_Risk_Class"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_props = y_train.value_counts(normalize=True).mul(100).round(2)
test_props = y_test.value_counts(normalize=True).mul(100).round(2)
split_df = pd.DataFrame({
    "Training Set (%)": train_props,
    "Test Set (%)": test_props
})
print(split_df)

## * 2.4 Implement cleaning, encoding, scaling, imputation, feature engineering and imbalance treatment inside a reproducible pipeline. Explain choices and prevent leakage. 

wsa_pipeline = ImbPipeline([
    ("feature_engineering", WSAFeatureEngineer()),
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", RobustScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

# 5-Fold Stratified Cross-Validation on Training Set
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = {
    "accuracy": "accuracy",
    "recall_macro": "recall_macro",
    "precision_macro": "precision_macro",
    "f1_macro": "f1_macro",
}
cv_results = cross_validate(
    wsa_pipeline, X_train, y_train,
    cv=cv_stratified, scoring=scoring_metrics
)

# Fit Pipeline on Full Training Set and Evaluate on Held-Out Test Set
wsa_pipeline.fit(X_train, y_train)
y_pred_test = wsa_pipeline.predict(X_test)

# 29 WSAs - Unseen Data
print(classification_report(
    y_test, y_pred_test,
    target_names=["High Risk", "Medium Risk", "Low Risk"]
))

# ^ QUESTION 3

## * 3.1 Build a transparent naïve baseline (for example, majority/stratified DummyClassifier) and explain what it establishes.  

# Naïve Baseline Pipeline (Most Frequent Class)
naive_baseline = ImbPipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("dummy", DummyClassifier(strategy="most_frequent")),
])

# Evaluation of the Naïve Baseline using 5-Fold Stratified Cross-Validation on Training Set
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring_metrics = {
    "accuracy": "accuracy",
    "recall_macro": "recall_macro",
    "precision_macro": "precision_macro",
    "f1_macro": "f1_macro",
}

cv_results = cross_validate(
    naive_baseline,
    X_train,
    y_train,
    cv=cv_stratified,
    scoring=scoring_metrics,
)

# Fit on Training Set and Evaluation on Held-Out Test Set (29 WSAs)
naive_baseline.fit(X_train, y_train)
y_pred_naive = naive_baseline.predict(X_test)

# Fold Cross Validation Performance
print(
    " - Mean Macro Recall    :"
    f" {cv_results['test_recall_macro'].mean():.4f} (+/-"
    f" {cv_results['test_recall_macro'].std():.4f})"
)
print(f" - Mean Accuracy        : {cv_results['test_accuracy'].mean():.4f}")
print(
    " - Mean Macro Precision :"
    f" {cv_results['test_precision_macro'].mean():.4f}"
)
print(f" - Mean Macro F1-Score  : {cv_results['test_f1_macro'].mean():.4f}")

print("\n2. HELD-OUT TEST SET EVALUATION (29 Test WSAs):")
print(
    classification_report(
        y_test,
        y_pred_naive,
        labels=["High Risk", "Medium Risk", "Low Risk"],
        target_names=["High Risk", "Medium Risk", "Low Risk"],
        zero_division=0,
    )
)

## * 3.2 Train at least four substantive models on identical data splits: at least two traditional models and at least two neural-learning models. The naïve baseline does not count among the four. 

# Pipelines for Substantive Models
substantive_models = {
    'Logistic Regression (Traditional 1)': ImbPipeline([
        ('fe', WSAFeatureEngineer()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    'Random Forest (Traditional 2)': ImbPipeline([
        ('fe', WSAFeatureEngineer()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ]),
    'MLP Neural Net (Neural 1: 2-Layer ReLU)': ImbPipeline([
        ('fe', WSAFeatureEngineer()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                              max_iter=1000, random_state=42)),
    ]),
    'Deep Tabular Neural Net (Neural 2: 3-Layer Tanh)': ImbPipeline([
        ('fe', WSAFeatureEngineer()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh',
                              alpha=0.01, max_iter=1000, random_state=42)),
    ]),
}

# 5-Fold Stratified Cross-Validation
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']

results = []
for name, pipeline in substantive_models.items():
    cv_res = cross_validate(pipeline, X_train, y_train,
                            cv=cv_stratified, scoring=scoring_metrics)
    results.append({
        'Model Name': name,
        'Mean Macro Recall (SD)': f"{cv_res['test_recall_macro'].mean():.4f} (± {cv_res['test_recall_macro'].std():.4f})",
        'Mean Accuracy': f"{cv_res['test_accuracy'].mean():.4f}",
        'Mean Macro Precision': f"{cv_res['test_precision_macro'].mean():.4f}",
        'Mean Macro F1-Score': f"{cv_res['test_f1_macro'].mean():.4f}",
    })

# Summary Table
df_results = pd.DataFrame(results)

# Comparative Performance Of Substantive Models
print(df_results.to_string(index=False))

# Naïve Baseline (Most Frequent Class)
naive_baseline = ImbPipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("dummy", DummyClassifier(strategy="most_frequent")),
])

# Models dictionary (Naïve + 4 Substantive Models)
models = {
    "Naïve Baseline": naive_baseline,

    "Logistic Regression": ImbPipeline([
        ("fe", WSAFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", RobustScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),

    "Random Forest": ImbPipeline([
        ("fe", WSAFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", RobustScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ]),

    "MLP Neural Net (2-Layer)": ImbPipeline([
        ("fe", WSAFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", RobustScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)),
    ]),

    "Deep Tabular Net (3-Layer)": ImbPipeline([
        ("fe", WSAFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", RobustScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32, 16),
                              activation="tanh", alpha=0.01,
                              max_iter=1000, random_state=42)),
    ]),
}

# Fitting all 5 model pipelines on X_train, y_train
print("Training all 5 models on X_train...")
for name, model_pipe in models.items():
    model_pipe.fit(X_train, y_train)
print("[SUCCESS] All 5 models are now trained and ready for Question 4 evaluation!")

# ^ QUESTION 4

## * 4.1 For every substantive model, report the same held-out metrics. Include accuracy and macro-F1; add  precision, recall, weighted-F1 and ROC-AUC as appropriate. For imbalance, include PR-AUC or  justify its omission.  

# Preparation of One-Hot Binarizer for Multiclass ROC-AUC, PR-AUC
lb = LabelBinarizer()
lb.fit(y_train)
y_test_bin = lb.transform(y_test)

heldout_results = []

# evaluate on X_test, y_test through trained pipelines
for name, model_pipe in models.items():
  # Predict discrete risk classes
  y_pred = model_pipe.predict(X_test)

  # Predict class probability distributions
  if hasattr(model_pipe, 'predict_proba'):
    y_proba = model_pipe.predict_proba(X_test)
  else:
    y_proba = None

  # Calculate Core Technical Metrics
  acc = accuracy_score(y_test, y_pred)
  prec_macro = precision_score(
      y_test, y_pred, average='macro', zero_division=0
  )
  rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
  f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
  f1_wt = f1_score(y_test, y_pred, average='weighted', zero_division=0)

  # Calculate Ranking &amp; Imbalance Metrics (ROC-AUC &amp; PR-AUC)
  if y_proba is not None:
    roc_auc = roc_auc_score(
        y_test_bin, y_proba, multi_class='ovr', average='macro'
    )
    pr_auc = average_precision_score(y_test_bin, y_proba, average='macro')
  else:
    roc_auc = np.nan
    pr_auc = np.nan

  heldout_results.append({
      'Model Name': name,
      'Accuracy': f'{acc:.4f}',
      'Macro Precision': f'{prec_macro:.4f}',
      'Macro Recall': f'{rec_macro:.4f}',
      'Macro F1': f'{f1_mac:.4f}',
      'Weighted F1': f'{f1_wt:.4f}',
      'ROC-AUC (Macro OvR)': f'{roc_auc:.4f}',
      'PR-AUC (Macro)': f'{pr_auc:.4f}',
  })

# Print Format Table
df_heldout = pd.DataFrame(heldout_results)

# Held Out Test Set Performance Reporting 29 WSA
print(df_heldout.to_string(index=False))

## * 4.2 Provide a classification report and confusion matrix for every model; provide ROC curves on one common figure (one-vs-rest macro/micro curves for multiclass). 

# Per-Model Detailed Classification Reports & Confusion Matrices
class_labels = ['High Risk', 'Medium Risk', 'Low Risk']

for name, model_pipe in models.items():
    y_pred = model_pipe.predict(X_test)
    print(f'\n>>> {name.upper()} CLASSIFICATION REPORT <<<')
    print(classification_report(
        y_test, y_pred,
        labels=class_labels,
        target_names=class_labels,
        zero_division=0
    ))
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    print(f'Confusion Matrix ({name}):\n{cm}\n' + '-' * 60)

# Generate Composite Visualization (5 Confusion Heatmaps + ROC Curves)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

# Confusion Matrices plots for all 5 models
for idx, (name, model_pipe) in enumerate(models.items()):
    y_pred = model_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=class_labels, yticklabels=class_labels,
        ax=axes[idx], annot_kws={'size': 14, 'weight': 'bold'}
    )
    axes[idx].set_title(f'{name}', fontsize=13, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Predicted Risk Class', fontsize=10)
    axes[idx].set_ylabel('Actual Risk Class', fontsize=10)

# Plot Combined Multi-Class ROC Curves in the 6th panel
axes[5].set_title('Multi-Class ROC Curves (Macro OvR)',
                  fontsize=13, fontweight='bold', pad=10)
axes[5].plot([0, 1], [0, 1], 'k--', label='Chance Baseline (AUC = 0.50)')

lb = LabelBinarizer()
lb.fit(y_train)
y_test_bin = lb.transform(y_test)

colors = ['gray', 'blue', 'orange', 'green', 'purple']

for idx, (name, model_pipe) in enumerate(models.items()):
    if hasattr(model_pipe, 'predict_proba'):
        y_proba = model_pipe.predict_proba(X_test)

        # Compute Macro OvR ROC Curve
        fpr_list, tpr_list = [], []
        for c_idx in range(len(class_labels)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, c_idx], y_proba[:, c_idx])
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        all_fpr = np.unique(np.concatenate(fpr_list))
        mean_tpr = np.zeros_like(all_fpr)
        for c_idx in range(len(class_labels)):
            mean_tpr += np.interp(all_fpr, fpr_list[c_idx], tpr_list[c_idx])
        mean_tpr /= len(class_labels)

        roc_auc = auc(all_fpr, mean_tpr)
        axes[5].plot(
            all_fpr, mean_tpr,
            label=f'{name} (AUC = {roc_auc:.2f})',
            color=colors[idx], linewidth=2
        )

axes[5].set_xlabel('False Positive Rate', fontsize=10)
axes[5].set_ylabel('True Positive Rate', fontsize=10)
axes[5].legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.show()

# Overfitting Diagnosis and Generalization Gap Analysis

## * 4.3 Compare all models with a clear table and bar chart. Show cross-validation means and variability or  confidence intervals where feasible. Select the provisional champion using validation—not test— performance. 

# Evaluation of Train, CV Validation, and Test Scores Across All Models
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gap_results = []

for name, model_pipe in models.items():
    # Fit pipeline on training set
    model_pipe.fit(X_train, y_train)

    # Training performance
    y_pred_train = model_pipe.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_rec = recall_score(y_train, y_pred_train, average='macro')

    # Cross-validation scores
    cv_scores = cross_validate(
        model_pipe, X_train, y_train,
        cv=cv_stratified,
        scoring=['accuracy', 'recall_macro'],
        return_train_score=True
    )
    cv_train_acc = cv_scores['train_accuracy'].mean()
    cv_val_acc = cv_scores['test_accuracy'].mean()
    cv_val_rec = cv_scores['test_recall_macro'].mean()

    # Held-out test performance
    y_pred_test = model_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rec = recall_score(y_test, y_pred_test, average='macro')

    # Generalization gap
    gen_gap_acc = cv_val_acc - cv_train_acc

    gap_results.append({
        'Model Name': name,
        'CV Train Acc': f'{cv_train_acc:.4f}',
        'CV Val Acc': f'{cv_val_acc:.4f}',
        'Test Acc': f'{test_acc:.4f}',
        'CV Val Recall': f'{cv_val_rec:.4f}',
        'Test Recall': f'{test_rec:.4f}',
        'Generalization Gap (Val - Train)': f'{gen_gap_acc:.4f}',
    })

# Print comparative generalization table
df_gap = pd.DataFrame(gap_results)

# Overfitting Diagnosis Generalization Gap Report
print(df_gap.to_string(index=False))

# Generate Comparative Bar Chart Plot
plt.style.use('seaborn-v0_8-whitegrid')

model_names = [
    'Naïve Baseline', 'Logistic Reg.', 'Random Forest',
    'MLP Neural Net', 'Deep Tabular Net'
]
train_acc_list = [0.4870, 0.8065, 0.9630, 0.9978, 1.0000]
val_acc_list = [0.4870, 0.7391, 0.7043, 0.8261, 0.8522]
test_acc_list = [0.4828, 0.7931, 0.6207, 0.8621, 0.8276]

x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, train_acc_list, width,
                label='CV Train Accuracy', color='#2b5c8f')
rects2 = ax.bar(x, val_acc_list, width,
                label='CV Validation Accuracy', color='#d95f02')
rects3 = ax.bar(x + width, test_acc_list, width,
                label='Held-Out Test Accuracy', color='#7570b3')

ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
ax.set_title('Question 4.3: Overfitting Diagnosis & Generalization Gap Analysis',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim(0, 1.15)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()

## * 4.4 Test for overfitting using train-versus-validation performance and learning curves; for neural models  also show training/validation loss and metric curves. Diagnose gaps and discuss early stopping,  dropout, weight decay, pruning or other controls used.

# Encode Target Classes for Neural Network Early-Stopping Compatibility
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Pipelines with Internal Early Stopping & Loss Tracking
mlp_2layer = ImbPipeline([
    ('fe', WSAFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    )),
])

mlp_3layer = ImbPipeline([
    ('fe', WSAFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='tanh',
        alpha=0.01,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    )),
])

rf_pipe = ImbPipeline([
    ('fe', WSAFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )),
])

# Neural pipelines on encoded training labels Fitting
mlp_2layer.fit(X_train, y_train_encoded)
mlp_3layer.fit(X_train, y_train_encoded)

clf2 = mlp_2layer.named_steps['clf']
clf3 = mlp_3layer.named_steps['clf']

print(f'MLP 2-Layer (ReLU) converged in {len(clf2.loss_curve_)} epochs '
      f'(Final Loss: {clf2.loss_curve_[-1]:.4f})')
print(f'Deep Net 3-Layer (Tanh) converged in {len(clf3.loss_curve_)} epochs '
      f'(Final Loss: {clf3.loss_curve_[-1]:.4f})')

# Generate Composite Plot (Panel 1: Loss & Val Acc | Panel 2: Learning Curves)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: Neural Network Training Loss & Internal Validation Accuracy
ax1 = axes[0]
ax1_val = ax1.twinx()

p1 = ax1.plot(clf2.loss_curve_, color='#1f77b4', linewidth=2.5,
              label='MLP 2-Layer Train Loss')
p2 = ax1.plot(clf3.loss_curve_, color='#9467bd', linewidth=2.5,
              label='Deep Net 3-Layer Train Loss')

p3 = ax1_val.plot(clf2.validation_scores_, color='#1f77b4', linestyle='--',
                  linewidth=2, label='MLP 2-Layer Val Acc')
p4 = ax1_val.plot(clf3.validation_scores_, color='#9467bd', linestyle='--',
                  linewidth=2, label='Deep Net 3-Layer Val Acc')

ax1.set_xlabel('Epoch / Iteration', fontsize=11, fontweight='bold')
ax1.set_ylabel('Training Loss (Cross-Entropy)', fontsize=11, fontweight='bold')
ax1_val.set_ylabel('Internal Validation Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Neural Network Epoch Loss & Validation Curves',
              fontsize=12, fontweight='bold', pad=12)

lines = p1 + p2 + p3 + p4
ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=9)

# Panel 2: Learning Curves Across Dataset Sample Sizes (Macro Recall)
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes = np.linspace(0.2, 1.0, 5)

train_sizes_abs, train_scores_m3, val_scores_m3 = learning_curve(
    mlp_3layer, X_train, y_train_encoded,
    cv=cv_stratified, scoring='recall_macro',
    train_sizes=train_sizes
)
_, train_scores_rf, val_scores_rf = learning_curve(
    rf_pipe, X_train, y_train_encoded,
    cv=cv_stratified, scoring='recall_macro',
    train_sizes=train_sizes
)

train_mean_m3 = np.mean(train_scores_m3, axis=1)
val_mean_m3 = np.mean(val_scores_m3, axis=1)
train_mean_rf = np.mean(train_scores_rf, axis=1)
val_mean_rf = np.mean(val_scores_rf, axis=1)

ax2 = axes[1]
ax2.plot(train_sizes_abs, train_mean_m3, 'o-', color='#9467bd',
         label='Deep Net (Train Macro Recall)', linewidth=2)
ax2.plot(train_sizes_abs, val_mean_m3, 's--', color='#9467bd',
         label='Deep Net (Val Macro Recall)', linewidth=2)
ax2.plot(train_sizes_abs, train_mean_rf, 'o-', color='#d62728',
         label='Random Forest (Train Macro Recall)', linewidth=2)
ax2.plot(train_sizes_abs, val_mean_rf, 's--', color='#d62728',
         label='Random Forest (Val Macro Recall)', linewidth=2)

ax2.set_xlabel('Training Sample Size (WSAs)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Macro Recall', fontsize=11, fontweight='bold')
ax2.set_title('Learning Curves: Dataset Size vs Macro Recall',
              fontsize=12, fontweight='bold', pad=12)
ax2.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.show()

# ^ QUESTION 5

## * 5.1 Optimize only the strongest validation model using a documented method such as grid search, random search, Bayesian optimisation or Optuna. Define the search space and scoring rule.

# Base Pipeline for Provisional Champion
champion_pipeline = ImbPipeline([
    ('fe', WSAFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', MLPClassifier(random_state=42, max_iter=1000)),
])

# Grid Search Space
param_grid = {
    'clf__hidden_layer_sizes': [(64, 32, 16), (128, 64, 32), (32, 16, 8)],
    'clf__activation': ['tanh', 'relu'],
    'clf__alpha': [0.0001, 0.001, 0.01, 0.1],
    'clf__learning_rate_init': [0.001, 0.01],
}

# 5-Fold Stratified Cross-Validation Configuration
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Instantiate GridSearchCV Targeting Macro Recall
grid_search = GridSearchCV(
    estimator=champion_pipeline,
    param_grid=param_grid,
    scoring='recall_macro',
    cv=cv_stratified,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

# Grid Search Configuration For Provisional Champion Deep Net
print(' - Target Estimator : Deep Tabular Neural Network (MLPClassifier)')
print(' - Search Strategy  : 5-Fold Stratified GridSearchCV')
print(' - Scoring Rule     : Macro-Averaged Recall (recall_macro)')
print(' - Parameter Space  : 48 Configurations (240 Total Fits)')

## * 5.2  Use cross-validation or a strict validation set; do not tune on the held-out test set. Compare pre- and post-optimisation results and report the chosen hyperparameters.

# Pipeline and Grid Search
champion_pipeline = ImbPipeline([
    ('fe', WSAFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', MLPClassifier(random_state=42, max_iter=1000)),
])

param_grid = {
    'clf__hidden_layer_sizes': [(64, 32, 16), (128, 64, 32), (32, 16, 8)],
    'clf__activation': ['tanh', 'relu'],
    'clf__alpha': [0.0001, 0.001, 0.01, 0.1],
    'clf__learning_rate_init': [0.001, 0.01],
}

cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=champion_pipeline,
    param_grid=param_grid,
    scoring='recall_macro',
    cv=cv_stratified,
    n_jobs=-1,
)

# Fit Grid Search on Training Set
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluation of Optimized Champion on Held-Out Test Set
y_pred_post = best_model.predict(X_test)
y_proba_post = best_model.predict_proba(X_test)

lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
roc_auc_post = roc_auc_score(
    y_test_bin, y_proba_post, multi_class='ovr', average='macro'
)

print('Best Hyperparameters Found:')
for k, v in grid_search.best_params_.items():
  print(f' - {k}: {v}')

print(f'\nBest Pre-Tuning CV Macro Recall  : 0.8377')
print(f'Best Post-Tuning CV Macro Recall : {grid_search.best_score_:.4f}')

print('\nHeld-Out Test Evaluation of Optimized Champion:')
print(
    classification_report(
        y_test,
        y_pred_post,
        labels=['High Risk', 'Medium Risk', 'Low Risk'],
        target_names=['High Risk', 'Medium Risk', 'Low Risk'],
        zero_division=0,
    )
)

# ^ QUESTION 6

## * 6.1 

cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Full Champion and 3 Controlled Ablation Pipelines
ablation_pipelines = {
    'Full Champion Pipeline (Locked)': ImbPipeline([
        ('fe', WSAFeatureEngineer()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation='tanh',
            alpha=0.0001,
            learning_rate_init=0.01,
            max_iter=1000,
            random_state=42
        )),
    ]),

    'Ablation 1: Omit Feature Engineering': ImbPipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation='tanh',
            alpha=0.0001,
            learning_rate_init=0.01,
            max_iter=1000,
            random_state=42
        )),
    ]),

    'Ablation 2: Omit SMOTE Oversampling': ImbPipeline([
        ('fe', WSAFeatureEngineer()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation='tanh',
            alpha=0.0001,
            learning_rate_init=0.01,
            max_iter=1000,
            random_state=42
        )),
    ]),

    'Ablation 3: Omit Tanh & Weight Decay': ImbPipeline([
        ('fe', WSAFeatureEngineer()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation='relu',
            alpha=0.0,
            learning_rate_init=0.01,
            max_iter=1000,
            random_state=42
        )),
    ]),
}

# Evaluation of Cross-Validation and Test Metrics
ablation_results = []
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

for name, pipe in ablation_pipelines.items():
    # Cross-Validation on Training Set
    cv_res = cross_validate(
        pipe, X_train, y_train,
        cv=cv_stratified,
        scoring=['accuracy', 'recall_macro']
    )
    cv_rec = cv_res['test_recall_macro'].mean()
    cv_acc = cv_res['test_accuracy'].mean()

    # Single Evaluation on Held-Out Test Set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred, average='macro')
    high_risk_rec = recall_score(y_test, y_pred, labels=['High Risk'], average=None)[0]
    test_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    test_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')

    ablation_results.append({
        'Experiment Name': name,
        'CV Val Macro Recall': f'{cv_rec:.4f}',
        'CV Val Accuracy': f'{cv_acc:.4f}',
        'Test Accuracy': f'{test_acc:.4f}',
        'Test Macro Recall': f'{test_rec:.4f}',
        'Test High Risk Recall': f'{high_risk_rec:.4f}',
        'Test Macro Precision': f'{test_prec:.4f}',
        'Test Macro F1': f'{test_f1:.4f}',
        'Test ROC-AUC': f'{test_auc:.4f}',
    })

# Controlled Ablation Result
df_ablation = pd.DataFrame(ablation_results)
print(' QUESTION 6.1: CONTROLLED ABLATION STUDY RESULTS')
print(df_ablation.to_string(index=False))

# ^-^
# 1. Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# 2. Save best_model cleanly
joblib.dump(best_model, "models/best_model.pkl")

print("[SUCCESS] best_model.pkl saved successfully in 'models/' directory!")
