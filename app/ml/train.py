import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from app.ml.features import NUMERIC, CATEGORICAL, FEATURES

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

df = pd.read_csv("data/student-mat.csv", sep=";")

# Positive class = at risk of failing
df["at_risk"] = (df["G3"] < 10).astype(int)
print("At-risk share:", round(df["at_risk"].mean(), 3))

X = df[FEATURES]
y = df["at_risk"]

categorical = CATEGORICAL
numeric = NUMERIC

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric),
])

models = {
    "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scoring = ["roc_auc", "f1", "recall", "precision"]

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    results[name] = (pipe, s["test_roc_auc"].mean())
    print(f"{name:26s} AUC={s['test_roc_auc'].mean():.3f}  "
          f"F1={s['test_f1'].mean():.3f}  "
          f"recall={s['test_recall'].mean():.3f}  "
          f"precision={s['test_precision'].mean():.3f}")

# Pick best by ROC-AUC
best_name = max(results, key=lambda k: results[k][1])
best_pipe = results[best_name][0]
print(f"\nBest model: {best_name}")

best_pipe.fit(X_train, y_train)
preds = best_pipe.predict(X_test)
print(classification_report(y_test, preds, target_names=["pass", "at risk"]))
print("Test ROC-AUC:", round(roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:, 1]), 3))

joblib.dump(best_pipe, "app/ml/model.joblib")
print("Model saved.")