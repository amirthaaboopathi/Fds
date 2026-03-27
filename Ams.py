import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'hour': np.random.randint(0, 24, n),
    'distance_km': np.random.uniform(1, 20, n),
    'surge_multiplier': np.random.uniform(1, 3, n),
    'driver_availability': np.random.randint(1, 50, n)
})

data['fare'] = (
    50 * data['distance_km'] * data['surge_multiplier'] +
    np.random.normal(0, 100, n)
)

data['ride_completed'] = (data['driver_availability'] > 10).astype(int)

print("\n--- Dataset Created ---")
print(data.head())

arr = data[['hour', 'distance_km', 'surge_multiplier', 'driver_availability']].values
reshaped = arr.reshape(-1, 4)

print("\n--- NumPy Reshape ---")
print("Shape:", reshaped.shape)
print(reshaped[:5])

data.loc[0:20, 'distance_km'] = np.nan

print("\nMissing Before:\n", data.isnull().sum())


data['distance_km'] = data['distance_km'].fillna(data['distance_km'].mean())

print("Missing After:\n", data.isnull().sum())

plt.figure()
plt.hist(data['distance_km'])
plt.title("Distance Distribution")
plt.show()

sampled = data.sample(frac=0.3, random_state=42)

print("\nOriginal:", len(data))
print("Sampled:", len(sampled))

plt.figure()
plt.hist(data['fare'], alpha=0.5)
plt.hist(sampled['fare'], alpha=0.5)
plt.title("Sampling Comparison")
plt.show()

corr = sampled.corr()

print("\n--- Correlation ---\n", corr)

plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

X_lr = sampled[['distance_km']]
y_lr = sampled['fare']

lr = LinearRegression()
lr.fit(X_lr, y_lr)

print("\n--- Linear Regression ---")
print("Coefficient:", lr.coef_)
print("Intercept:", lr.intercept_)

plt.figure()
plt.scatter(X_lr, y_lr)
plt.plot(X_lr, lr.predict(X_lr))
plt.title("Distance vs Fare")
plt.show()

X = sampled[['hour', 'distance_km', 'surge_multiplier', 'driver_availability']]
y = sampled['ride_completed']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, pred))

plt.figure()
plt.hist(pred)
plt.title("Ride Completion Predictions")
plt.show()


def simulate(surge_increase):
    temp = sampled.copy()
    temp['surge_multiplier'] += surge_increase
    temp['fare'] = 50 * temp['distance_km'] * temp['surge_multiplier']
    return temp['fare'].sum()

base = sampled['fare'].sum()
new = simulate(0.5)

print("\n--- Simulation ---")
print("Base Revenue:", base)
print("New Revenue:", new)

plt.figure()
plt.bar(['Base', 'New'], [base, new])
plt.title("Revenue Comparison")
plt.show()
print("\n--- Final Insights ---")
print("1. Distance strongly impacts fare")
print("2. Surge pricing increases revenue")
print("3. Driver availability affects ride completion")
