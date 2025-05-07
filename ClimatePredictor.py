# Upload files manually using Google Colab file upload
from google.colab import files
uploaded = files.upload()

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the data
train_df = pd.read_csv('DailyDelhiClimateTrain.csv')
test_df = pd.read_csv('DailyDelhiClimateTest.csv')

# Prepare training and test datasets
X_train = train_df.drop(columns=['meantemp', 'date'])
y_train = train_df['meantemp']
X_test = test_df.drop(columns=['meantemp', 'date'])
y_test = test_df['meantemp']

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict temperatures on test data
predictions = model.predict(X_test)

# 1. Line Plot - Predicted Temperatures
plt.figure(figsize=(10, 5))
plt.plot(predictions, label='Predicted Temperature', color='blue')
plt.title('Predicted Temperature on Test Data')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart - Actual vs Predicted (First 30 Days)
plt.figure(figsize=(12, 6))
plt.bar(range(30), y_test[:30], label='Actual', alpha=0.6, color='green')
plt.bar(range(30), predictions[:30], label='Predicted', alpha=0.6, color='orange')
plt.title('Actual vs Predicted Temperatures (First 30 Days)')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Heatmap - Correlation of Features
plt.figure(figsize=(8, 6))
corr = train_df.drop(columns=['date']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Training Features')
plt.tight_layout()
plt.show()

