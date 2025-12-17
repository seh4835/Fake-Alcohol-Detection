from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import (
    train_test_split, KFold, GridSearchCV, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, make_scorer
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# -------------------- Flask App --------------------
app = Flask(__name__)

# -------------------- Load Dataset --------------------
DATA_PATH = r"C:\Users\Seher\Downloads\fake_alcohol_classsification\fake_alcohol_classsification\fake_alcohol_dataset.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["is_fake"])
y = df["is_fake"]

# -------------------- Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------- Cross Validation Setup --------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0)
}

# -------------------- Pipelines & Models --------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(random_state=42))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(random_state=42))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),

    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True))
    ]),

    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingClassifier(random_state=42))
    ])
}

# -------------------- Hyperparameter Grids --------------------
param_grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10]
    },
    "Decision Tree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10]
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10]
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"]
    },
    "SVM (RBF)": {
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", "auto"]
    },
    "Gradient Boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__n_estimators": [100, 200]
    }
}

# -------------------- Hyperparameter Tuning --------------------
best_models = {}

for name, pipeline in models.items():
    grid = GridSearchCV(
        pipeline,
        param_grids[name],
        scoring="f1",
        cv=kfold,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_

# -------------------- Evaluation & Plots --------------------
os.makedirs("static/plots", exist_ok=True)
results = []

for name, model in best_models.items():

    cv_results = cross_validate(
        model, X, y, cv=kfold, scoring=scoring, n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 3),
        "CV_Accuracy": round(cv_results["test_accuracy"].mean(), 3),
        "CV_Precision": round(cv_results["test_precision"].mean(), 3),
        "CV_Recall": round(cv_results["test_recall"].mean(), 3),
        "CV_F1": round(cv_results["test_f1"].mean(), 3)
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", cbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"static/plots/{name}_confusion.png")
    plt.close()

    # Decision Tree Plot
    if name == "Decision Tree":
        plt.figure(figsize=(14, 6))
        plot_tree(
            model.named_steps["model"],
            feature_names=X.columns,
            class_names=["Genuine", "Fake"],
            filled=True,
            max_depth=3
        )
        plt.savefig("static/plots/DecisionTree.png")
        plt.close()

    # Random Forest Feature Importance
    if name == "Random Forest":
        rf = model.named_steps["model"]
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=importances, y=importances.index, palette="crest")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig("static/plots/FeatureImportance.png")
        plt.close()

# -------------------- Results Summary --------------------
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
best_model_name = results_df.iloc[0]["Model"]

joblib.dump(best_models[best_model_name], "best_model.pkl")

# -------------------- Performance Comparison Line Plot --------------------
plt.figure(figsize=(8, 5))
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    plt.plot(results_df["Model"], results_df[metric], marker="o", label=metric)
plt.title("Performance Metrics Comparison")
plt.ylim(0, 1)
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("static/plots/PerformanceComparison.png")
plt.close()

# -------------------- Feature Distributions --------------------
for col in X.columns[:5]:
    plt.figure(figsize=(6, 3))
    sns.kdeplot(data=df, x=col, hue="is_fake", fill=True, alpha=0.4)
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig(f"static/plots/{col}_distribution.png")
    plt.close()

# -------------------- Scatter Plot --------------------
if len(X.columns) >= 2:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df, x=X.columns[0], y=X.columns[1],
        hue="is_fake", alpha=0.7
    )
    plt.title("Feature Scatter Plot")
    plt.tight_layout()
    plt.savefig("static/plots/Scatter.png")
    plt.close()

# -------------------- Flask Routes --------------------
@app.route("/")
def dashboard():
    images = [f for f in os.listdir("static/plots") if f.endswith(".png")]
    return render_template(
        "index.html",
        results=results_df.to_dict(orient="records"),
        best_model=best_model_name,
        images=images
    )

@app.route("/results")
def results_page():
    return render_template(
        "results.html",
        results=results_df.to_dict(orient="records")
    )

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
