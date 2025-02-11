import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
file_path = "Online Retail.xlsx"
df = pd.read_excel(file_path)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Remove canceled orders (negative quantities)
df = df[df['Quantity'] > 0]

# Aggregate sales at daily level
df['TotalSales'] = df['Quantity'] * df['UnitPrice']
df_daily = df.groupby(df['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
df_daily.columns = ['Date', 'Sales']
df_daily['Date'] = pd.to_datetime(df_daily['Date'])

# Sort data
df_daily = df_daily.sort_values(by='Date')

# Train-test split
train_size = int(len(df_daily) * 0.8)
train, test = df_daily[:train_size], df_daily[train_size:]
train_sales, test_sales = train['Sales'], test['Sales']

# Train ARIMA model
arima_model = ARIMA(train_sales, order=(5,1,0)).fit()
arima_preds = arima_model.forecast(steps=len(test_sales))

# Evaluate ARIMA
mse_arima = mean_squared_error(test_sales, arima_preds)
rmse_arima = np.sqrt(mse_arima)
mae_arima = mean_absolute_error(test_sales, arima_preds)
print(f"ARIMA Performance - MSE: {mse_arima:.2f}, RMSE: {rmse_arima:.2f}, MAE: {mae_arima:.2f}")

# Feature Engineering for XGBoost
df_daily['Sales_Lag1'] = df_daily['Sales'].shift(1)
df_daily['Sales_Lag7'] = df_daily['Sales'].shift(7)
df_daily['Sales_Lag30'] = df_daily['Sales'].shift(30)
df_daily = df_daily.dropna()

# Train-test split for XGBoost
train_xgb, test_xgb = df_daily[:train_size], df_daily[train_size:]
X_train, y_train = train_xgb[['Sales_Lag1', 'Sales_Lag7', 'Sales_Lag30']], train_xgb['Sales']
X_test, y_test = test_xgb[['Sales_Lag1', 'Sales_Lag7', 'Sales_Lag30']], test_xgb['Sales']

# Train XGBoost Model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluate XGBoost
mse_xgb = mean_squared_error(y_test, xgb_preds)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, xgb_preds)
print(f"XGBoost Performance - MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}")

# Future Forecasting (ARIMA)
future_dates = pd.date_range(start=df_daily['Date'].max(), periods=30, freq='D')
future_arima_preds = arima_model.forecast(steps=30)

plt.figure(figsize=(10,5))
plt.plot(df_daily['Date'], df_daily['Sales'], label="Historical Sales", color="blue")
plt.plot(future_dates, future_arima_preds, label="ARIMA Forecast (Next 30 Days)", color="red")
plt.legend()
plt.show()

# Future Forecasting (XGBoost)
future_df = pd.DataFrame({'Sales_Lag1': [df_daily['Sales'].iloc[-1]],
                          'Sales_Lag7': [df_daily['Sales'].iloc[-7]],
                          'Sales_Lag30': [df_daily['Sales'].iloc[-30]]})

future_xgb_preds = []
for _ in range(30):
    pred = xgb_model.predict(future_df)[0]
    future_xgb_preds.append(pred)
    future_df = pd.DataFrame({'Sales_Lag1': [pred],
                              'Sales_Lag7': [future_xgb_preds[-7] if len(future_xgb_preds) >= 7 else pred],
                              'Sales_Lag30': [future_xgb_preds[-30] if len(future_xgb_preds) >= 30 else pred]})

plt.figure(figsize=(10,5))
plt.plot(df_daily['Date'], df_daily['Sales'], label="Historical Sales", color="blue")
plt.plot(future_dates, future_xgb_preds, label="XGBoost Forecast (Next 30 Days)", color="green")
plt.legend()
plt.show()
