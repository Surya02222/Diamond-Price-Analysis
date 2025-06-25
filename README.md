# Diamond-Price-Analysis
🧠 Goal
Predict diamond prices using features like carat, cut, color, clarity, and physical dimensions.

🔧 Step-by-Step Code Explanation
🔹 1. Import Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pandas: for data manipulation.

numpy: for numerical operations.

matplotlib and seaborn: for plotting graphs.

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
train_test_split: splits data into training and test sets.

LabelEncoder: converts categorical variables to numbers.

LinearRegression, RandomForestRegressor: ML models.

mean_squared_error, r2_score: metrics to evaluate models.

🔹 2. Load Dataset
python
Copy
Edit
url = 'your_local_or_online_csv_path.csv'
df = pd.read_csv(url)
Replace 'your_local_or_online_csv_path.csv' with your file path or URL.

Reads the dataset into a pandas DataFrame.

🔹 3. Explore Data
python
Copy
Edit
print(df.head())
print(df.info())
head(): shows first 5 rows.

info(): shows data types and missing values.

🔹 4. Feature Engineering
python
Copy
Edit
df['volume'] = df['x'] * df['y'] * df['z']
Calculates the physical volume of the diamond using its dimensions.

python
Copy
Edit
df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)]
Removes rows where any of the dimensions (x, y, z) are 0 (invalid).

🔹 5. Encode Categorical Columns
python
Copy
Edit
label_encoders = {}
for col in ['cut', 'color', 'clarity']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
Converts non-numeric features (cut, color, clarity) into numbers using LabelEncoder.

🔹 6. Prepare Features and Target
python
Copy
Edit
features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'volume']
target = 'price'

X = df[features]
y = df[target]
X: input features (independent variables).

y: target variable (diamond price).

🔹 7. Train-Test Split
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Splits data into training (80%) and testing (20%) sets.

random_state=42 ensures consistent results.

🔹 8. Train ML Models
🔸 Linear Regression
python
Copy
Edit
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
Trains a basic regression model.

Predicts prices on the test set.

🔸 Random Forest Regressor
python
Copy
Edit
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
Uses a powerful ensemble model (Random Forest).

n_estimators=100: uses 100 decision trees.

🔹 9. Evaluate the Models
python
Copy
Edit
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))
R² Score: tells how well the model explains the variance in data (closer to 1 is better).

🔹 10. Feature Importance Visualization
python
Copy
Edit
importances = rf.feature_importances_
sns.barplot(x=importances, y=features)
plt.title('Feature Importance (Random Forest)')
plt.show()
Shows which features (e.g., carat, cut) are most important for predicting price.

🔹 11. Save the Model (Optional)
python
Copy
Edit
import joblib
joblib.dump(rf, 'diamond_price_model.pkl')
Saves the trained model to a file for later use in a web app or production system.

✅ Summary
Section	Description
Load Data	Load CSV and inspect data
Clean Data	Remove invalid entries
Feature Engineering	Create new features like volume
Encoding	Convert categories to numbers
Modeling	Train Linear and Random Forest models
Evaluation	Compare models using R² score
Visualization	Plot feature importances
Save Model	Store model for deployment

