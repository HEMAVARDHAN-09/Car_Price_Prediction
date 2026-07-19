import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Load Dataset
df = pd.read_csv("car.csv")

# Display First Five Rows
print("First Five Records")
print(df.head())

# Dataset Shape
print("\nDataset Shape:", df.shape)

# Dataset Information
print("\nDataset Information")
print(df.info())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# ============================================
# Phase 2 : Data Preprocessing
# ============================================

print("\n==============================")
print("DATA PREPROCESSING")
print("==============================")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("Dataset Shape After Removing Duplicates:", df.shape)

# Create a new feature: Car Age
current_year = 2025
df["Car_Age"] = current_year - df["Year"]

# Drop Car_Name because it is not useful for prediction
df.drop("Car_Name", axis=1, inplace=True)

# Drop Year column because Car_Age is more useful
df.drop("Year", axis=1, inplace=True)

print("\nUpdated Dataset")
print(df.head())

# ============================================
# Phase 3 : Encoding Categorical Columns
# ============================================

print("\n==============================")
print("ENCODING CATEGORICAL COLUMNS")
print("==============================")

# Convert categorical columns into numerical values
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

print("\nEncoded Dataset")
print(df.head())

print("\nUpdated Dataset Shape:", df.shape)

# ============================================
# Phase 4 : Exploratory Data Analysis (EDA)
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns

print("\n==============================")
print("EXPLORATORY DATA ANALYSIS")
print("==============================")

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df["Selling_Price"], bins=20, kde=True)
plt.title("Distribution of Selling Price")
plt.savefig("images/selling_price_distribution.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Selling_Price"])
plt.title("Selling Price Box Plot")
plt.savefig("images/selling_price_boxplot.png")
plt.show()

# Pair Plot
sns.pairplot(df[["Selling_Price","Present_Price","Kms_Driven","Car_Age"]])
plt.savefig("images/pairplot.png")
plt.show()

# ============================================
# Phase 5 : Train-Test Split
# ============================================

from sklearn.model_selection import train_test_split

print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================")

# Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("Features Shape :", X.shape)
print("Target Shape :", y.shape)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data :", X_train.shape)
print("Testing Data :", X_test.shape)
# ============================================
# Phase 6 : Linear Regression
# ============================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

# Train Model
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Linear Regression Model Trained Successfully!")

# Prediction
y_pred = lr.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# ============================================
# Phase 7 : Decision Tree Regressor
# ============================================

from sklearn.tree import DecisionTreeRegressor

print("\n==============================")
print("DECISION TREE REGRESSOR")
print("==============================")

# Train Model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

print("Decision Tree Model Trained Successfully!")

# Prediction
y_pred_dt = dt.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred_dt)
mse = mean_squared_error(y_test, y_pred_dt)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Results")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)
# ============================================
# Phase 8 : Random Forest Regressor
# ============================================

from sklearn.ensemble import RandomForestRegressor

print("\n==============================")
print("RANDOM FOREST REGRESSOR")
print("==============================")

# Create Model
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
rf.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")

# Prediction
y_pred_rf = rf.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# ============================================
# Phase 9 : Gradient Boosting Regressor
# ============================================

from sklearn.ensemble import GradientBoostingRegressor

print("\n==============================")
print("GRADIENT BOOSTING REGRESSOR")
print("==============================")

# Create Model
gb = GradientBoostingRegressor(
    random_state=42
)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
print("Gradient Boosting Model Trained Successfully!")

# Prediction
y_pred_gb = gb.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred_gb)
mse = mean_squared_error(y_test, y_pred_gb)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Results")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)
# ============================================
# Phase 10 : Save Best Model
# ============================================

import pickle

print("\n==============================")
print("SAVING BEST MODEL")
print("==============================")

# Save Linear Regression model
with open("models/car_price_model.pkl", "wb") as file:
    pickle.dump(lr, file)

print("Model saved successfully!")
print("Location : models/car_price_model.pkl")

import pickle

# Save the trained model
pickle.dump(gb, open("models/car_price_model.pkl", "wb"))

# Save the feature(column) names
pickle.dump(X.columns, open("models/columns.pkl", "wb"))

print("Model saved successfully!")
print("Columns saved successfully!")
print("Location : models/")