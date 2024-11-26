import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
import os


# Part 1: API Pulling, Data Wrangling, and Visualizations


# Part 1.1: API Setup
def fetch_fred_yield(series_id, start_date="2023-01-01", end_date="2023-12-31", api_key="your_api_key"):
    # Define the API URL and parameters here
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }

    # Make an API request and handle errors
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    # Convert response JSON into a DataFrame, setting 'date' as index
    data = response.json().get("observations", [])
    df = pd.DataFrame(data)

    # Convert the 'value' column to numeric and rename it based on series_id
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.rename(columns={"value": series_id}, inplace=True)

    return df[[series_id]]  # Final DataFrame with date index and yield column

# Part 1.2: Data Pulling


# List of Treasury yield series IDs on FRED
tenor_series_ids = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1",  # Short-term yields
    "DGS2", "DGS3", "DGS5",  # Medium-term yields
    "DGS7", "DGS10", "DGS20", "DGS30"  # Long-term yields
]

# Initialize API key
api_key = "015eaf80d4964ba2d2c184346cd4a567"

# Fetch data for each tenor, store in a dictionary of DataFrames
yield_data = {}
for series_id in tenor_series_ids:
    yield_data[series_id] = fetch_fred_yield(series_id, api_key=api_key)

# Part 1.3: Data Storage
# Combine all DataFrames into a single DataFrame, joining on the date index
combined_yield_data = pd.concat(yield_data.values(), axis=1)

# Print the number of rows in the final DataFrame
print(f"Number of rows in the final DataFrame: {len(combined_yield_data)}")
print(combined_yield_data.head())

latest_treasury_yields = combined_yield_data.iloc[-1].dropna()
latest_treasury_yields.index = [float(maturity.replace("DGS", "").replace("MO", "").replace("Y", ""))
                                for maturity in latest_treasury_yields.index]

treasury_curve = pd.DataFrame({
    "Maturity": latest_treasury_yields.index,
    "Yield": latest_treasury_yields.values
})

# Part 1.4: Spread Calculation
bonds_yields_path = '/Users/allenwang/Downloads/Part 1. bonds_yields.xlsx'
loan_data_path = '/Users/allenwang/Downloads/Part 2. loan_data_final.csv'

# Load Excel file (bonds yields)
bonds_yields = pd.read_excel(bonds_yields_path)

bonds_yields.rename(columns={"WAL (years)": "WAL", "Yield (%)": "Yield"}, inplace=True)


def interpolate_treasury_yield(wal, treasury_df):
    maturities = treasury_df["Maturity"].values
    yields = treasury_df["Yield"].values
    return np.interp(wal, maturities, yields)


def calculate_spread(row, treasury_df):
    interpolated_yield = interpolate_treasury_yield(row["WAL"], treasury_df)
    return row["Yield"] - interpolated_yield


# Apply spread calculation
bonds_yields["Spread"] = bonds_yields.apply(
    lambda row: calculate_spread(row, treasury_df=treasury_curve), axis=1
)

# Display the bonds data with spreads
print(bonds_yields)


# Part 1.5: Visualizations

# Spreads vs. Sector Boxplot

# Spread vs Sector
plt.figure(figsize=(12, 6))
sns.boxplot(data=bonds_yields, x="Sector", y="Spread", palette="Set2")
plt.title("Spread vs Sector", fontsize=16)
plt.xlabel("Sector", fontsize=14)
plt.ylabel("Spread (bps)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Spread vs WAL
plt.figure(figsize=(12, 6))
sns.scatterplot(data=bonds_yields, x="WAL", y="Spread", hue="Sector", palette="Set1")
plt.title("Spread vs Weighted Average Life (WAL)", fontsize=16)
plt.xlabel("Weighted Average Life (Years)", fontsize=14)
plt.ylabel("Spread (bps)", fontsize=14)
plt.legend(title="Sector", fontsize=10, loc="upper right")
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.show()


# Average Spread vs. Sector Barchart
plt.figure(figsize=(12, 6))
sns.barplot(data=bonds_yields, x="Sector", y="Spread", ci=None, palette="viridis")
plt.title("Average Spread by Sector", fontsize=16)
plt.xlabel("Sector", fontsize=14)
plt.ylabel("Average Spread (bps)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

#  Spreads Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=bonds_yields, x="Sector", y="Spread", palette="coolwarm", inner="quartile", scale="width")
plt.title("Spread Distribution by Sector (Violin Plot)", fontsize=16)
plt.xlabel("Sector", fontsize=14)
plt.ylabel("Spread (bps)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# Part 2: Loan Default Prediction Task

# Load CSV file (loan data)
loan_data = pd.read_csv(loan_data_path)
print(loan_data)

# Part 2.1 Data Exploration

# Search for missing values
print(loan_data.info())
print(loan_data.describe())
print(loan_data.isnull().sum())

# Drop duplicates
loan_data.drop_duplicates(inplace=True)

if "Unnamed: 0" in loan_data.columns:
    loan_data = loan_data.drop(columns=["Unnamed: 0"])

# Part 2.2 Feature engineering

# Explore categorical features
categorical_columns = loan_data.select_dtypes(include=["object", "category"]).columns
for col in categorical_columns:
    print(f"\nValue counts for {col}:")
    print(loan_data[col].value_counts())

# Explore the target variable
print("\nLoan Status Distribution:")
print(loan_data["loan_status"].value_counts())

print("Unique values in loan_status:", loan_data["loan_status"].unique())
print("Data type of loan_status:", loan_data["loan_status"].dtype)

# Handle binary categorical features
loan_data["previous_loan_defaults_on_file"] = loan_data["previous_loan_defaults_on_file"].map({"Yes": 1, "No": 0})

# One-hot encoding for categorical columns
categorical_columns = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "loan_type"]
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_features = encoder.fit_transform(loan_data[categorical_columns])
encoded_columns = encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=loan_data.index)
loan_data = loan_data.drop(columns=categorical_columns)
loan_data = pd.concat([loan_data, encoded_df], axis=1)

numerical_columns = ["person_age", "person_income", "person_emp_exp", "loan_int_rate",
                     "loan_percent_income", "cb_person_cred_hist_length",
                     "credit_score", "regional_unemployment_rate",
                     "borrower_risk_score", "loan_to_income_ratio"]

# Scale numerical features
scaler = StandardScaler()
loan_data[numerical_columns] = scaler.fit_transform(loan_data[numerical_columns])

# Part 2.3 Model Training
# Define target variable and features
X = loan_data.drop(columns=["loan_status"])
y = loan_data["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

# Predictions
y_pred_log = log_model.predict(X_test)

# Train random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)


# Evaluate Logistic Regression
print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_log, pos_label=1):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_log, pos_label=1):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_log, pos_label=1):.2f}")

# Evaluate Random Forest
print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, pos_label=1):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_rf, pos_label=1):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf, pos_label=1):.2f}")

# Classification report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Random forest feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)


plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_log)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Predict probabilities for the positive class
y_pred_prob = log_model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line for random chance
plt.title("ROC Curve for Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label="Logistic Regression")
plt.title("Precision-Recall Curve for Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.show()

# Compute calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)

# Plot calibration curve
plt.figure(figsize=(10, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Perfectly calibrated line
plt.title("Calibration Curve for Logistic Regression")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Probability")
plt.legend()
plt.grid()
plt.show()


# Extract feature importance (coefficients)
coefficients = log_model.coef_[0]
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x="Coefficient", y="Feature", data=feature_importance, palette="coolwarm")
plt.title("Feature Importance from Logistic Regression")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.show()

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda y, pred: precision_score(y, pred, pos_label=1),
    "Recall": lambda y, pred: recall_score(y, pred, pos_label=1),
    "F1 Score": lambda y, pred: f1_score(y, pred, pos_label=1)
}

# Part 2.4 Documentation

'''
Approach: Random forest
Looking at the feature importance table, the most important features seem to be history of loan 
defaults and loan interest rate, followed by personal income, loan percentage of income, and loan 
to income ratio. These five features explain the majority of the variation in the random forest
model. This aligns with what is expected - that someone's likelihood of default is dependent on their 
financial situation and ancillary variables such as the purpose of the loan and the education level of the
loaner have lower propensity to describe the likelihood that someone is to default

Approach: logistic regression
Analyzing the results of the logistic regression, the logistic regression performed nearly as well as the
random forest, with low levels of false positives and false negatives (type I and II errors). The model
classified correctly the majority of people. Overall the model seems to not be overfitted or underfitted.
The large area under the curve of the ROC indicates a well-performing model. Since the model is 
imbalanced in the sense that most people fulfilled their loan obligations, looking at the P-R curve
we can see that the slow drop-off indicates a good balance between precision and recall. The shape of the
calibration curve corroborate this perspective, with the closeness of the logistic regression outputs to the
line indicating that the predictions align with the real-world results.

Analyzing the feature importance graph provides an interesting lens into our analysis, as we can see the 
outsize bar on the graph for previous loan defaults indicates a large effect on possibility of future default.
One question that I have is why the bar is negative because intuition points towards a positive relationship
between the two variables.
'''

