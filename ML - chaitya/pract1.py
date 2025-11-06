
print("Hello World")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/content/uber.csv")

df.head(
    5
)

df.describe()

df.shape

df2 = df.copy()

df2.isnull().sum()

df2.info()

df2=df2.dropna()

df2.isnull().sum()

"""Converting Date to required pandas format

"""

df2["pickup_datetime"]=pd.to_datetime(df2["pickup_datetime"])

df2.info()

df2['hour'] = df2['pickup_datetime'].dt.hour
df2['day'] = df2['pickup_datetime'].dt.day
df2['month'] = df2['pickup_datetime'].dt.month
df2['year'] = df2['pickup_datetime'].dt.year
df2['day_of_week'] = df2['pickup_datetime'].dt.dayofweek

df2.info()

df2.head()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df2["distance_km"]=haversine(df2["pickup_latitude"],df2["pickup_longitude"],df2["dropoff_latitude"],df2["dropoff_longitude"])

df2.head()

df2.describe()

# Remove impossible fares and distances
df2 = df2[(df2['fare_amount'] > 0) &
        (df2['distance_km'] > 0) &
        (df2['distance_km'] < 100)]

# Check new range
df2['distance_km'].describe()

df2.shape

cols = ['fare_amount', 'distance_km', 'passenger_count', 'hour', 'day_of_week']

import seaborn as sns
import matplotlib.pyplot as plt

pearson_corr = df2[cols].corr(method='pearson')

plt.figure(figsize=(6,4))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Pearson Correlation Heatmap")
plt.show()

spearman_corr = df2[cols].corr(method='spearman')

plt.figure(figsize=(6,4))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='vlag')
plt.title("Spearman Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split

# Select features (X) and target (y)
X = df2[['distance_km', 'passenger_count', 'hour', 'day_of_week']]
y = df2['fare_amount']

# Split into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Initialize and fit the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)

# Evaluation metrics
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print("Linear Regression Results:")
print(f"R²: {r2_lr:.3f}")
print(f"RMSE: {rmse_lr:.3f}")
print(f"MAE: {mae_lr:.3f}")

from sklearn.ensemble import RandomForestRegressor

# Initialize and train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Evaluation metrics
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("Random Forest Results:")
print(f"R²: {r2_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")
print(f"MAE: {mae_rf:.3f}")

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R²': [r2_lr, r2_rf],
    'RMSE': [rmse_lr, rmse_rf],
    'MAE': [mae_lr, mae_rf]
})
print(results)

# Split based on hour
df3 = df2[df2['hour'] > 12]       # Afternoon + evening
df4 = df2[df2['hour'] <= 12]      # Morning + early day

print("Afternoon/evening rides:", len(df3))
print("Morning/midday rides:", len(df4))

avg_fare_df3 = df3['fare_amount'].mean()
avg_fare_df4 = df4['fare_amount'].mean()

print("Average fare (hour > 12):", avg_fare_df3)
print("Average fare (hour ≤ 12):", avg_fare_df4)

categories = ['Hour > 12', 'Hour ≤ 12']
avg_fares = [avg_fare_df3, avg_fare_df4]

plt.bar(categories, avg_fares, color=['orange', 'skyblue'])
plt.title('Average Fare Comparison: Morning vs Afternoon')
plt.ylabel('Average Fare Amount')
plt.xlabel('Time Period')
plt.show()
