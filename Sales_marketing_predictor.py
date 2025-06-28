"""
Marketing -> Sales Prediction 

Predict product sales ($K) from advertising spend on
TV, radio, and social media channels.
"""
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#  Load (or generate) the dataset

csv_text = """
TV,Radio,Social,Sales
230.1,37.8,69.2,2.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3, 9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9
8.7,48.9,75.0, 7.2
57.5,32.8,23.5,11.8
120.2,19.6,11.6,13.2
8.6, 2.1, 1.0, 4.8
199.8, 2.6,21.2,10.6
66.1, 5.8,24.2, 8.6
214.7,24.0, 4.0,17.4
23.8,35.1,65.9, 9.2
97.5, 7.6, 7.2, 9.7
204.1,32.9,46.0,19.0
195.4,47.7,52.9,22.4
67.8,36.6,114.0,12.5
281.4,39.6,55.8,24.4
69.2,20.5,18.3,11.3
147.3,23.9,19.1,14.6
"""
df = pd.read_csv(StringIO(csv_text.strip()))

print("First 5 rows:")
print(df.head(), "\n")


#  Prepare features & target

X = df[["TV", "Radio", "Social"]]   # marketing spend ($K)
y = df["Sales"]                     # product sales ($K)

# Train / test split (80 % train, 20 % test)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Linear Regression model

model = LinearRegression()
model.fit(X_train, y_train)

print("Model coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"  {col:6s}: {coef:6.3f}")
print(f"Intercept: {model.intercept_:6.3f}\n")

# Evaluate on the hold-out test set

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)

print("Test-set performance:")
print(f"  RMSE: {rmse:.2f}  (sales units in $K)")
print(f"  R²  : {r2:.3f}\n")

# Forecast: what if we spend X on each channel?

new_budget = pd.DataFrame({
    "TV":     [150.0],   # $150K on TV ads
    "Radio":  [30.0],    # $30K on radio
    "Social": [40.0]     # $40K boosting on social media
})

forecast = model.predict(new_budget)[0]
print("----- Forecast scenario -----")
print(f"With a budget of $150K (TV) / $30K (Radio) / $40K (Social),")
print(f"expected sales ≈ **{forecast:.1f} K units**\n")
