import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

df=pd.read_csv('diabetes.csv')  

print(df.head())

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# 4. TARGET DISTRIBUTION
# ==========================================

print("\nOutcome Distribution:")
print(df["Outcome"].value_counts())

sns.countplot(x="Outcome", data=df)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Number of Patients")

plt.show()

# ==========================================
# 5. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()

# ==========================================
# 6. FEATURE DISTRIBUTIONS
# ==========================================

df.hist(
    figsize=(12, 10),
    bins=20
)

plt.tight_layout()
plt.show()
# ==========================================
# 7. HANDLE MISSING VALUES
# ==========================================

missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace invalid zero values with NaN
df[missing_columns] = df[missing_columns].replace(0, np.nan)

print("\nMissing Values After Replacing Zeros:")

print(df.isnull().sum())
# ==========================================
# 8. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)
# ==========================================
# 9. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
# ==========================================
# 10. LOGISTIC REGRESSION
# ==========================================

logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_test)

y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression trained successfully!")

# ==========================================
# 11. RANDOM FOREST
# ==========================================

rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest trained successfully!")# ==========================================
# 12. XGBOOST
# ==========================================

xgb_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost trained successfully!")
# ==========================================
# 13. NEURAL NETWORK
# ==========================================

nn_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=1000,
        random_state=42
    ))
])

nn_model.fit(X_train, y_train)

y_pred_nn = nn_model.predict(X_test)

y_prob_nn = nn_model.predict_proba(X_test)[:, 1]

print("Neural Network trained successfully!")

# ==========================================
# 14. MODEL EVALUATION
# ==========================================

def evaluate_model(name, y_true, y_pred, y_prob):

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }


results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        y_pred_lr,
        y_prob_lr
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        y_pred_rf,
        y_prob_rf
    )
)

results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        y_pred_xgb,
        y_prob_xgb
    )
)

results.append(
    evaluate_model(
        "Neural Network",
        y_test,
        y_pred_nn,
        y_prob_nn
    )
)


results_df = pd.DataFrame(results)

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(results_df.round(3))
# ==========================================
# 15. MODEL COMPARISON GRAPH
# ==========================================

results_df.set_index("Model").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Model Performance Comparison")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend(loc="lower right")

plt.tight_layout()

plt.show()
# ==========================================
# 16. XGBOOST CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("XGBoost Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
# ==========================================
# 17. CLASSIFICATION REPORT
# ==========================================

print("\nXGBoost Classification Report:")

print(
    classification_report(
        y_test,
        y_pred_xgb
    )
)
