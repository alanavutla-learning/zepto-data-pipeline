import seaborn as sns
import matplotlib.pyplot as plt

df=sns.load_dataset("titanic")
df.to_csv("titanic.csv",index=False)

print("Titanic dataset loaded and saved")
print("shape:",df.shape)
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Missing Percentage ---")
print(df.isnull().mean() * 100)

# Handle missing values for EDA

df_clean = df.copy()

# Age: 19.87% missing → median imputation
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

# Embarked and embark_town: less than 5% missing → drop rows
df_clean = df_clean.dropna(subset=["embarked", "embark_town"])

# Deck: 77.22% missing → drop column
df_clean = df_clean.drop(columns=["deck"])

print("\n--- Cleaned Dataset ---")
print("Shape:", df_clean.shape)
print("\nRemaining missing values:")
print(df_clean.isnull().sum())

df_clean.to_csv("cleaned_titanic.csv", index=False)

print("Cleaned dataset saved as cleaned_titanic.csv")

# Univariate numerical analysis

print("\n--- Age Statistics ---")
print("Mean:", df_clean["age"].mean())
print("Median:", df_clean["age"].median())

print("\n--- Fare Statistics ---")
print("Mean:", df_clean["fare"].mean())
print("Median:", df_clean["fare"].median())
print("Mode:", df_clean["fare"].mode()[0])

# IQR outlier counts
def iqr_outlier_count(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return ((series < lower) | (series > upper)).sum()

print("\n--- IQR Outliers ---")
print("Age outliers:", iqr_outlier_count(df_clean["age"]))
print("Fare outliers:", iqr_outlier_count(df_clean["fare"]))



# Age histogram
plt.figure()
sns.histplot(df_clean["age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("charts/age_histogram.png")
plt.close()

# Age boxplot
plt.figure()
sns.boxplot(x=df_clean["age"])
plt.title("Age Boxplot")
plt.xlabel("Age")
plt.savefig("charts/age_boxplot.png")
plt.close()

# Fare histogram
plt.figure()
sns.histplot(df_clean["fare"], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.savefig("charts/fare_histogram.png")
plt.close()

# Fare boxplot
plt.figure()
sns.boxplot(x=df_clean["fare"])
plt.title("Fare Boxplot")
plt.xlabel("Fare")
plt.savefig("charts/fare_boxplot.png")
plt.close()

print("\n--- Charts saved successfully ---")
# Bivariate Analysis - Survival Rates

print("\n--- Survival Rate by Sex ---")

male_survival = df_clean[df_clean["sex"] == "male"]["survived"].mean()
female_survival = df_clean[df_clean["sex"] == "female"]["survived"].mean()

print("Male survival rate:", male_survival)
print("Female survival rate:", female_survival)


print("\n--- Survival Rate by Pclass ---")

class_1_survival = df_clean[df_clean["pclass"] == 1]["survived"].mean()
class_2_survival = df_clean[df_clean["pclass"] == 2]["survived"].mean()
class_3_survival = df_clean[df_clean["pclass"] == 3]["survived"].mean()

print("Pclass 1:", class_1_survival)
print("Pclass 2:", class_2_survival)
print("Pclass 3:", class_3_survival)


print("\n--- Survival Rate by Sex and Pclass ---")

for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        rate = df_clean[
            (df_clean["sex"] == sex) &
            (df_clean["pclass"] == pclass)
        ]["survived"].mean()

        print(
            f"{sex.title()} + Pclass {pclass}: {rate:.4f}"
        )
# Correlation Matrix

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df_clean[correlation_columns].corr()

print("\n--- Correlation Matrix ---")
print(corr_matrix)

# Find two strongest unique off-diagonal correlations

corr_pairs = corr_matrix.where(
    __import__("numpy").triu(
        __import__("numpy").ones(corr_matrix.shape),
        k=1
    ).astype(bool)
).stack()

strongest_pairs = corr_pairs.abs().sort_values(ascending=False).head(2)

print("\n--- Two Strongest Unique Correlations ---")

for pair in strongest_pairs.index:
    print(pair, ":", corr_pairs[pair])

# Correlation Heatmap

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Titanic Variables")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png")
plt.close()

print("\nCorrelation heatmap saved.")


# Multivariate Chart 1: Survival by Sex and Pclass

plt.figure(figsize=(8, 6))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("charts/survival_sex_pclass.png")
plt.close()


# Multivariate Chart 2: Age by Survival and Sex

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df_clean,
    x="survived",
    y="age",
    hue="sex"
)

plt.title("Age Distribution by Survival and Sex")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("charts/age_survival_sex.png")
plt.close()


# Multivariate Chart 3: Fare by Pclass and Survival

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df_clean,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title("Fare Distribution by Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("charts/fare_pclass_survival.png")
plt.close()


# Multivariate Chart 4: Age vs Fare by Survival

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived",
    style="sex"
)

plt.title("Age vs Fare by Survival and Sex")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("charts/age_fare_survival.png")
plt.close()

print("\n--- Multivariate charts saved successfully ---")

# Exploratory Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_standardized = df_clean.copy()

df_standardized[["age", "fare"]] = scaler.fit_transform(
    df_clean[["age", "fare"]]
)

print("\n--- Before Standardization ---")
print("Age mean:", df_clean["age"].mean())
print("Age std:", df_clean["age"].std())
print("Fare mean:", df_clean["fare"].mean())
print("Fare std:", df_clean["fare"].std())

print("\n--- After Standardization ---")
print("Age mean:", df_standardized["age"].mean())
print("Age std:", df_standardized["age"].std())
print("Fare mean:", df_standardized["fare"].mean())
print("Fare std:", df_standardized["fare"].std())