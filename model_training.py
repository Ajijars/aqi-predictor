import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Generate sample training data
# In production, replace this with actual Delhi AQI data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Hour': np.random.randint(0, 24, n_samples),
    'Day': np.random.randint(1, 32, n_samples),
    'Month': np.random.randint(1, 13, n_samples),
    'DayOfWeek': np.random.randint(0, 7, n_samples),
    'Lag_1': np.random.uniform(50, 400, n_samples),
    'Lag_3': np.random.uniform(50, 400, n_samples),
    'Lag_24': np.random.uniform(50, 400, n_samples),
    'Rolling_6': np.random.uniform(50, 400, n_samples),
    'Rolling_24': np.random.uniform(50, 400, n_samples),
})


# Target AQI values
y = 50 + 0.3 * data['Lag_1'] + 0.2 * data['Lag_3'] + 0.1 * data['Rolling_6'] + np.random.normal(0, 20, n_samples)
y = np.clip(y, 0, 500)

# Features used for training
features = ['Hour', 'Day', 'Month', 'DayOfWeek', 'Lag_1', 'Lag_3', 'Lag_24', 'Rolling_6', 'Rolling_24']
X = data[features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"✅ Model trained successfully")
print(f"Mean Absolute Error: {mae:.2f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
