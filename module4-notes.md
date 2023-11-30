## Module 4: Supervised Learning — Classification
### Classification Overview
4.1.1: Classification Overview  
**Introduction to Supervised Learning:**
- Learning the difference between supervised and unsupervised learning.
- Two types of models based on the nature of predicted outputs (continuous or discrete).

**Linear Regression for Continuous Values:**
- Linear regression is used to predict continuous values.
- Continuous values can be divided into smaller quantities.

**Classification Models for Discrete Values:**
- Classification models are used for predicting discrete values.
- Discrete values are finite and cannot be subdivided.

**Introduction to Logistic Regression:**
- Logistic regression is a classification model for predicting binary discrete values.
- Examples of binary outcomes include spam detection in emails.

**Considerations for Model Selection:**
- Besides known output type, other factors for model selection include data size and complexity.
- Addressing complexity includes methods like linear and non-linear data handling, support vector machines (SVMs), decision trees, and ensemble learning.

**Applications of Classification:**
- Organizations use classification models in various fields, including finance, insurance, health, and risk assessment.
- Different approaches to classification, including logistic regression, SVMs, and decision trees, will be discussed in the module.

This text provides an overview of supervised learning, focusing on linear regression for continuous values and introducing the concept of classification models for discrete values, particularly logistic regression. It also mentions considerations for model selection and real-world applications of classification models.

4.1.2: Getting Started  
**Files to Download:**
- Download module files from "Module 4 Demo and Activity Files."

**Required Installations:**
- Install the following packages:
  - Scikit-learn for building classification models.
  - PyDotPlus for visualizing decision trees.

**Install Scikit-learn:**
- Activate your dev virtual environment using "conda activate dev."
- Check for existing Scikit-learn installation with "conda show scikit-learn."
  - If installed, no further action is needed.
  - If not installed, follow the official Scikit-learn documentation for installation.

**Install PyDotPlus:**
- Ensure the dev virtual environment is active.
- Install PyDotPlus using "conda install -c conda-forge pydotplus."
- Verify the installation with "conda list pydotplus."

**Note:**
- PyDotPlus is required for the Decision Tree demonstration.
- If you encounter issues with PyDotPlus installation, consider using Google Colab or attending office hours for the demonstration.

### Classification and Logistic Regression
4.2.1: Overview  
- This section covers classification and logistic regression.
- The four steps for running a logistic regression model:
  1. Preprocessing
  2. Training
  3. Validation
  4. Prediction
- Key Concepts:
  - Classification categorizes based on shared qualities.
  - Focus on binary outcomes.
  - Logistic regression predicts discrete outcomes.
  - It uses multiple variables to make decisions.
  - Sigmoid curve represents the probability.
  - Classification threshold is typically 50%.
- Logistic regression converts data into a single probability.
- Video explains logistic regression using the example of animal classification.
- Applying logistic regression involves:
  1. Preprocessing
  2. Training
  3. Validation
  4. Prediction

4.2.2: Understand the Categorical Data Before Applying Logistic Regression  
- Using logistic regression in supervised learning follows the model-fit-predict process.
- The process involves deciding the model, fitting it to data, and making predictions.
- Visualizing data might be necessary based on data complexity.
- Example: Building a supervised learning model for a bank to approve loans.
- Data about two groups: healthy and unhealthy firms.
- Goal: Classify firms into these categories.
- Evaluation of model predictions is important.
- You can find the complete solution for this demonstration in the "demos/01-Logistic_Regression/Solved" folder.
- Importing Pandas and startup data.
- Visualizing data with a Pandas scatter plot.
- Data includes healthy and unhealthy firms.
- Preparing data by labeling categories as 1s and 0s.
- Splitting data into training and testing sets.

4.2.3: Preprocessing  
- Preprocessing data involves preparation and splitting into training and testing sets.
- Counting the number of firms in each category using "value_counts."
- 346 firms performed well (value = 1) and 978 went bankrupt (value = 0).
- Splitting data into training and testing sets for unbiased model evaluation.
- Using the train_test_split function from Scikit-learn.
- Creating features (X) and target (y) DataFrames.
- The X data includes variables/features, and the y data is the target variable.
- The train_test_split function defaults to using 75% of the data for training.
- You can adjust the proportion using the "train_size" parameter.
- Preparing to create and use a classifier for predicting startup health.

4.2.4: Training and Validation  
- Create a logistic regression model for prediction.
  - Import the LogisticRegression class from Scikit-learn.
  - Create an empty logistic regression model.
- Train the model using the fit function with training data (X_train, y_train).
  - The fit function determines which data belongs in which category.
  - The algorithm selects the best version of the model to distinguish categories.
- Validation is performed using the score function.
  - Score the model with training and testing data.
  - The score function compares predicted outcomes to actual outcomes.
- Training and testing scores are both 1.0, indicating perfect accuracy.
  - Achieving perfect accuracy in real-world data is uncommon.
- Scoring with training data compares accuracy against data used for training.
- Scoring with testing data gauges real-life prediction accuracy.
- A significant accuracy gap between training and testing scores may indicate overfitting.
- For classification models, the score function returns accuracy.
- The model is trained, validated, and ready for testing.

4.2.5: Prediction  
- Classify features with the model using the predict function.
  - Predict whether the model can assign features to the correct targets (startups' success).
  - Create a DataFrame to compare predictions and actual targets.
- The model predicts startup success based on financial performance and industry health scores.
- The model is expected to perform differently on new, previously unknown data in the real world.
- Testing the model on new data:
  - Use logistic_regression_model to predict with X_test DataFrame.
  - Save predictions and actual values to a DataFrame for evaluation.
- The model accurately predicts startup success for the testing data, indicating its potential usefulness for new startups.
- Evaluation of the model:
  - Introduce the accuracy_score function for model evaluation.
  - Calculate the accuracy of model predictions for testing data.
  - The model achieves an accuracy score of 1.0, indicating perfect accuracy in this example.
  - Perfect accuracy is rare and could be a sign of overfitting, which we'll discuss later.
- Summary of the steps to use a logistic regression model:
  1. Create a model with LogisticRegression().
  2. Train the model with model.fit().
  3. Make predictions with model.predict().
  4. Evaluate the model with accuracy_score().

4.2.6: Activity: Logistic Regression  
4.2.7: Recap and Knowledge Check  

### Model Selection
4.3.1: Introduction to Model Selection  
4.3.2: Linear vs. Non-linear Data  
4.3.3: Linear Models  
4.3.4: Demonstration: Predict Occupancy with SVM  
4.3.5: Non-linear Models  
4.3.6: Demonstration: Decision Trees  
4.3.7: Overfitting  
4.3.8: Recap and Knowledge Check  

### Ensemble Learning
4.4.1: Introduction to Ensemble Learning  
4.4.2: Random Forest  
4.4.3: Demonstration: Random Forest  
4.4.4: Boosting  
4.4.5: Recap and Knowledge Check  

### Summary: Supervised Learning — Classification
4.5.1: Summary: Supervised Learning — Classification  
4.5.2: Reflect on Your Learning  
4.5.3: References  
