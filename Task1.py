import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', 
                            '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01']),
    'Sales': [1200, 1350, 1580, 1600, 1750, 1900, 2100, 2050, 2200, 2400]
}

df = pd.DataFrame(data)

df['Month_Num'] = np.arange(len(df))

X = df[['Month_Num']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

future_months = np.array([[10], [11], [12]])
predictions = model.predict(future_months)

future_dates = pd.to_datetime(['2024-11-01', '2024-12-01', '2025-01-01'])
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Sales': predictions})

train_preds = model.predict(X)
mae = mean_absolute_error(y, train_preds)
r2 = r2_score(y, train_preds)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")
print("\n--- Future Forecast ---")
print(forecast_df)

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sales'], marker='o', label='Historical Sales', color='blue')
plt.plot(future_dates, predictions, marker='s', linestyle='--', label='Forecast', color='orange')
plt.title('Sales & Demand Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales Units')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()