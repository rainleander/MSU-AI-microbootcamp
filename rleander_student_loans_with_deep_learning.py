# -*- coding: utf-8 -*-
"""rleander_student_loans_with_deep_learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xDHQu4-e61GV3hIH4gtoWSvUk3Vxay9L

# Student Loan Risk with Deep Learning
"""

# Imports
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""---

## Prepare the data to be used on a neural network model

### Step 1: Read the `student_loans.csv` file into a Pandas DataFrame. Review the DataFrame, looking for columns that could eventually define your features and target variables.
"""

# Read the csv into a Pandas DataFrame
file_path = "https://static.bc-edx.com/mbc/ai/m6/datasets/student_loans.csv"
df = pd.read_csv(file_path)

# Review the DataFrame
# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Display information about the DataFrame
print("\nDataFrame Information:")
print(df.info())

# Display statistical summaries of the DataFrame
print("\nDataFrame Statistical Summary:")
print(df.describe())

# Review the data types associated with the columns
print(df.dtypes)

"""### Step 2: Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “credit_ranking”. The remaining columns should define the features dataset."""

# Define the target set y using the credit_ranking column
y = df['credit_ranking'].values

# Display a sample of y
print(y[:5])  # Display the first 5 entries

# Define features set X by selecting all columns but credit_ranking
X = df.drop(columns=['credit_ranking'])

# Review the features DataFrame
print(X.head())  # Display the first few rows

"""### Step 3: Split the features and target sets into training and testing datasets.

"""

# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
from sklearn.model_selection import train_test_split

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Optionally, you can review the shapes of the datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

"""### Step 4: Use scikit-learn's `StandardScaler` to scale the features data."""

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the features training dataset
scaler.fit(X_train)

# Fit the scaler to the features training dataset
# Assumed this ^ was a typo and ignored it

# Scale the features training dataset
X_train_scaled = scaler.transform(X_train)

# Scale the features testing dataset
X_test_scaled = scaler.transform(X_test)

# Review the scaled data
print("Scaled X_train sample:\n", X_train_scaled[:5])
print("Scaled X_test sample:\n", X_test_scaled[:5])

"""---

## Compile and Evaluate a Model Using a Neural Network

### Step 1: Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

> **Hint** You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.
"""

# Define the number of inputs (features) to the model
number_input_features = X_train_scaled.shape[1]

# Review the number of features
print(number_input_features)

# Define the number of neurons in the output layer
number_output_neurons = 1

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 = (number_input_features + number_output_neurons) // 2

# Review the number hidden nodes in the first layer
print(hidden_nodes_layer1)

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 = hidden_nodes_layer1 // 2

# Review the number hidden nodes in the second layer
print(hidden_nodes_layer2)

# Create the Sequential model instance
model = Sequential()

# Add the first hidden layer
model.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation='relu'))

# Add the second hidden layer
model.add(Dense(units=hidden_nodes_layer2, activation='relu'))

# Add the output layer to the model specifying the number of output neurons and activation function
model.add(Dense(units=1, activation='linear'))

# Display the Sequential model summary
model.summary()

"""### Step 2: Compile and fit the model using the `mse` loss function, the `adam` optimizer, and the `mse` evaluation metric.

"""

# Compile the Sequential model
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

# Fit the model using 50 epochs and the training data
model.fit(X_train_scaled, y_train, epochs=50)

"""### Step 3: Evaluate the model using the test data to determine the model’s loss and accuracy.

"""

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

"""### Step 4: Save and export your model to an HDF5 file, and name the file `student_loans.h5`.

"""

# Set the model's file path
file_path = "student_loans.h5"

# Export your model to a HDF5 file
model.save(file_path)

"""---
## Predict Loan Repayment Success by Using your Neural Network Model

### Step 1: Reload your saved model.
"""

# Import the load_model function from the tensorflow.keras.models module
from tensorflow.keras.models import load_model

# Set the model's file path
file_path = "student_loans.h5"

# Load the model to a new object
nn_imported = load_model(file_path)

"""### Step 2: Make predictions on the testing data."""

# Make predictions on the testing data
predictions = nn_imported.predict(X_test_scaled)

"""### Step 3: Create a DataFrame to compare the predictions with the actual values."""

# Create a DataFrame to compare the predictions with the actual values
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions.ravel()
})

"""### Step 4: Display a sample of the DataFrame you created in step 3."""

# Display sample data
results_df.head()