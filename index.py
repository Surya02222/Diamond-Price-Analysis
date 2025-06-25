# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (update path if needed)
url = 'C:\my projects\Diamond-Price-Analysis\diamonds.csv'
df = pd.read_csv(url)

# View data
print(df.head())
print(df.info())

# Feature Engineering: compute volume (size)
df['volume'] = df['x'] * df['y'] * df['z']

# Drop rows with 0 or missing dimensions
df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)]

# Encode categorical features
label_encoders = {}
for col in ['cut', 'color', 'clarity']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'volume']
target = 'price'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Model: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

# Plotting feature importances from Random Forest
importances = rf.feature_importances_
sns.barplot(x=importances, y=features)
plt.title('Feature Importance (Random Forest)')
plt.show()

# Optional: Save the trained model
import joblib
joblib.dump(rf, 'diamond_price_model.pkl')
