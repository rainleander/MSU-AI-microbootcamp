import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import the data
data = pd.read_csv("https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv")

# Create labels set (y) and features DataFrame (X)
y = data['spam']
X = data.drop('spam', axis=1)

# Check the balance of the labels variable (`y`)
print("Label balance:\n", y.value_counts())

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of StandardScaler and fit it with the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model with random_state=1
logistic_model = LogisticRegression(random_state=1)
logistic_model.fit(X_train_scaled, y_train)

# Make predictions using the testing data
logistic_predictions = logistic_model.predict(X_test_scaled)

# Calculate and print the accuracy score for Logistic Regression
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
print(f"Logistic Regression Accuracy Score: {logistic_accuracy}")

# Train a Random Forest Classifier model with random_state=1
random_forest_model = RandomForestClassifier(random_state=1)
random_forest_model.fit(X_train_scaled, y_train)

# Make predictions using the testing data
rf_predictions = random_forest_model.predict(X_test_scaled)

# Calculate and print the accuracy score for Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy Score: {rf_accuracy}")

# Evaluate which model performed better
print("Which model performed better? " + ("Random Forest" if rf_accuracy > logistic_accuracy else "Logistic Regression"))
