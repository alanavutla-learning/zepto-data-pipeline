import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Load the same Titanic dataset saved during EDA
df = pd.read_csv("cleaned_titanic.csv")

print("Dataset shape:", df.shape)

# Features and target
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
y = df["survived"]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining set:", X_train.shape)
print("Test set:", X_test.shape)

print("\nClass distribution:")
print(y.value_counts(normalize=True))


# Numeric and categorical columns
numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

# Numeric preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully.")


# Create complete Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train using training data only
logistic_pipeline.fit(X_train, y_train)

print("\nLogistic Regression trained successfully.")



# Predictions on test data
y_pred = logistic_pipeline.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Probability of survival
y_prob = logistic_pipeline.predict_proba(X_test)[:, 1]

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print("\nLogistic Regression AUC:", auc_score)

plt.figure()
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("charts/logistic_roc.png")
plt.close()

print("Logistic Regression ROC curve saved.")



# Create Decision Tree pipeline
decision_tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Train using training data only
decision_tree_pipeline.fit(X_train, y_train)

print("\nDecision Tree trained successfully.")

# Predictions
y_pred_tree = decision_tree_pipeline.predict(X_test)

print("\n--- Decision Tree Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall   :", recall_score(y_test, y_pred_tree))
print("F1 Score :", f1_score(y_test, y_pred_tree))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

# ROC and AUC
y_prob_tree = decision_tree_pipeline.predict_proba(X_test)[:, 1]

fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, y_prob_tree)
auc_tree = roc_auc_score(y_test, y_prob_tree)

print("\nDecision Tree AUC:", auc_tree)

plt.figure()
plt.plot(
    fpr_tree,
    tpr_tree,
    label=f"Decision Tree (AUC = {auc_tree:.2f})"
)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.tight_layout()
plt.savefig("charts/decision_tree_roc.png")
plt.close()

print("Decision Tree ROC curve saved.")



# Get feature names after preprocessing
feature_names = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

plt.figure(figsize=(20, 10))

plot_tree(
    decision_tree_pipeline.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Titanic Survival")
plt.tight_layout()
plt.savefig("charts/decision_tree.png")
plt.close()

print("\nDecision Tree visualization saved.")


# Create Random Forest pipeline
random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train using training data only
random_forest_pipeline.fit(X_train, y_train)

print("\nRandom Forest trained successfully.")

# Predictions
y_pred_rf = random_forest_pipeline.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# ROC and AUC
y_prob_rf = random_forest_pipeline.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\nRandom Forest AUC:", auc_rf)

# Combined ROC Curve for all three classifiers

# Decision Tree AUC
auc_tree = roc_auc_score(
    y_test,
    decision_tree_pipeline.predict_proba(X_test)[:, 1]
)

# Logistic Regression AUC
auc_logistic = roc_auc_score(
    y_test,
    logistic_pipeline.predict_proba(X_test)[:, 1]
)

# Random Forest AUC
auc_rf = roc_auc_score(
    y_test,
    random_forest_pipeline.predict_proba(X_test)[:, 1]
)

# ROC values
fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    logistic_pipeline.predict_proba(X_test)[:, 1]
)

fpr_tree, tpr_tree, _ = roc_curve(
    y_test,
    decision_tree_pipeline.predict_proba(X_test)[:, 1]
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    random_forest_pipeline.predict_proba(X_test)[:, 1]
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression (AUC = {auc_logistic:.2f})"
)

plt.plot(
    fpr_tree,
    tpr_tree,
    label=f"Decision Tree (AUC = {auc_tree:.2f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.2f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig("charts/roc_comparison.png")
plt.close()

print("\nCombined ROC curve saved.")

# Side-by-side comparison of all three classifiers

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, logistic_pipeline.predict(X_test)),
        accuracy_score(y_test, decision_tree_pipeline.predict(X_test)),
        accuracy_score(y_test, random_forest_pipeline.predict(X_test))
    ],
    "Precision": [
        precision_score(y_test, logistic_pipeline.predict(X_test)),
        precision_score(y_test, decision_tree_pipeline.predict(X_test)),
        precision_score(y_test, random_forest_pipeline.predict(X_test))
    ],
    "Recall": [
        recall_score(y_test, logistic_pipeline.predict(X_test)),
        recall_score(y_test, decision_tree_pipeline.predict(X_test)),
        recall_score(y_test, random_forest_pipeline.predict(X_test))
    ],
    "F1": [
        f1_score(y_test, logistic_pipeline.predict(X_test)),
        f1_score(y_test, decision_tree_pipeline.predict(X_test)),
        f1_score(y_test, random_forest_pipeline.predict(X_test))
    ],
    "AUC": [
        auc_logistic,
        auc_tree,
        auc_rf
    ]
})

print("\n--- Model Comparison Table ---")
print(comparison.round(4).to_string(index=False))

# Logistic Regression with class_weight='balanced'

balanced_logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

balanced_logistic_pipeline.fit(X_train, y_train)

y_pred_balanced = balanced_logistic_pipeline.predict(X_test)

print("\n--- Balanced Logistic Regression ---")
print("Accuracy :", accuracy_score(y_test, y_pred_balanced))
print("Precision:", precision_score(y_test, y_pred_balanced))
print("Recall   :", recall_score(y_test, y_pred_balanced))
print("F1 Score :", f1_score(y_test, y_pred_balanced))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))

# Logistic Regression with SMOTE
smote_logistic_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# SMOTE is applied only to the training data
smote_logistic_pipeline.fit(X_train, y_train)

y_pred_smote = smote_logistic_pipeline.predict(X_test)

print("\n--- SMOTE Logistic Regression ---")
print("Accuracy :", accuracy_score(y_test, y_pred_smote))
print("Precision:", precision_score(y_test, y_pred_smote))
print("Recall   :", recall_score(y_test, y_pred_smote))
print("F1 Score :", f1_score(y_test, y_pred_smote))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_smote))

# Random Forest GridSearchCV

rf_grid_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        oob_score=True,
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=rf_grid_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n--- Random Forest GridSearchCV ---")
print("Best parameters:", grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)

best_rf_pipeline = grid_search.best_estimator_
best_rf_model = best_rf_pipeline.named_steps["classifier"]

print("OOB score:", best_rf_model.oob_score_)

# Evaluate the tuned Random Forest

y_pred_tuned = best_rf_pipeline.predict(X_test)
y_prob_tuned = best_rf_pipeline.predict_proba(X_test)[:, 1]

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_precision = precision_score(y_test, y_pred_tuned)
tuned_recall = recall_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned)
tuned_auc = roc_auc_score(y_test, y_prob_tuned)

print("\n--- Tuned Random Forest Test Results ---")
print("Accuracy:", tuned_accuracy)
print("Precision:", tuned_precision)
print("Recall:", tuned_recall)
print("F1:", tuned_f1)
print("AUC:", tuned_auc)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))


# Regression: Predict Fare

regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = df[regression_features]
y_reg = df["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

reg_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

reg_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

reg_preprocessor = ColumnTransformer([
    ("numeric", reg_numeric_pipeline, reg_numeric_features),
    ("categorical", reg_categorical_pipeline, reg_categorical_features)
])

regression_pipeline = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression())
])

regression_pipeline.fit(X_reg_train, y_reg_train)

y_reg_pred = regression_pipeline.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

n = len(y_reg_test)
p = len(
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\n--- Fare Regression Results ---")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)

# Residual Plot


residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_reg_pred, residuals, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Fare Regression - Residual Plot")
plt.tight_layout()
plt.savefig("charts/fare_residuals.png")
plt.close()

print("\nResidual plot saved to charts/fare_residuals.png")

# Save and reload the complete best pipeline

joblib.dump(random_forest_pipeline, "best_titanic_pipeline.joblib")

print("\nBest pipeline saved as best_titanic_pipeline.joblib")

loaded_pipeline = joblib.load(
    "best_titanic_pipeline.joblib"
)

sample = X_test.iloc[[0]]

prediction = loaded_pipeline.predict(sample)

print("Reloaded pipeline prediction:", prediction)
# Final Classifier Comparison

final_comparison = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(
            y_test, logistic_pipeline.predict(X_test)
        ),
        "Precision": precision_score(
            y_test, logistic_pipeline.predict(X_test)
        ),
        "Recall": recall_score(
            y_test, logistic_pipeline.predict(X_test)
        ),
        "F1": f1_score(
            y_test, logistic_pipeline.predict(X_test)
        ),
        "AUC": auc_logistic
    },
    {
        "Model": "Decision Tree",
        "Accuracy": accuracy_score(
            y_test, decision_tree_pipeline.predict(X_test)
        ),
        "Precision": precision_score(
            y_test, decision_tree_pipeline.predict(X_test)
        ),
        "Recall": recall_score(
            y_test, decision_tree_pipeline.predict(X_test)
        ),
        "F1": f1_score(
            y_test, decision_tree_pipeline.predict(X_test)
        ),
        "AUC": auc_tree
    },
    {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(
            y_test, random_forest_pipeline.predict(X_test)
        ),
        "Precision": precision_score(
            y_test, random_forest_pipeline.predict(X_test)
        ),
        "Recall": recall_score(
            y_test, random_forest_pipeline.predict(X_test)
        ),
        "F1": f1_score(
            y_test, random_forest_pipeline.predict(X_test)
        ),
        "AUC": auc_rf
    },
    {
        "Model": "Tuned Random Forest",
        "Accuracy": tuned_accuracy,
        "Precision": tuned_precision,
        "Recall": tuned_recall,
        "F1": tuned_f1,
        "AUC": tuned_auc
    }
])

print("\n--- Final Classifier Comparison ---")
print(final_comparison.round(4).to_string(index=False))


print("\n--- Multivariate Chart Interpretations ---")

print("""
1. Survival by Sex and Passenger Class:
Female passengers survived more than male passengers.
Passengers in higher classes also had better survival chances.
This shows that both gender and passenger class affected survival.
""")

print("""
2. Age Distribution by Survival and Sex:
The age of survivors and non-survivors is very different in some groups.
However, there is also a lot of overlap between them.
So, age only  cannot explain who survived.
""")

print("""
3. Fare by Passenger Class and Survival:
First-class passengers usually paid higher fares than second- and third-class passengers.
Survivors also had higher fares than non-survivors.
This shows a relationship between fare, passenger class, and survival.
""")

print("""
4. Age and Fare by Survival and Sex:
Most passengers paid lower fares, while only a few paid very high fares.
More survivors can be seen in passengers with higher fares.
Survival also varies with age and sex, so fare only cannot explain survival.
""")
