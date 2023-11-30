## Module 3: Supervised Learning — Linear Regression
### Supervised Learning Overview
3.1.1: Introduction to Supervised Learning  
- Overview of Machine Learning Categories:
  - Unsupervised Learning:
    - Used for knowledge discovery and clustering.
    - Examples include news feed recommendations based on online habits.
  - Supervised Learning:
    - Used to predict outcomes based on trained data.
    - Examples include spam detection and weather forecasting.

- Introduction to Supervised Learning:
  - Supervised learning involves learning from data and known outcomes.
  - It entails feeding the model with examples and correct answers.
  - The model is trained to minimize prediction errors.
  - Useful for predicting numerical values or recognizing predetermined categories.

- Practical Example of Supervised Learning:
  - Categorizing fruit based on labeled data for training.
  - Application in predicting high-risk vs. low-risk loans using historical data.

- Key Learnings in the Module:
  - Understanding the differences between supervised and unsupervised learning.
  - Learning key concepts of supervised learning.
  - Determining the appropriateness of regression or classification models.
  - Applying the model-fit-predict process.
  - Making predictions with linear regression models in supervised learning.

3.1.2: Supervised vs. Unsupervised Learning  
- Distinction between Supervised and Unsupervised Learning:
  - Common Aspects:
    - Both use data and algorithms.
  - Differences:
    - Supervised Learning:
      - Requires clear answers for predictions.
      - Utilizes labeled information.
      - Involves high human intervention and expertise.
    - Unsupervised Learning:
      - Limited human intervention.
      - Works with unstructured, unlabeled data.
      - Focuses on finding patterns.

- Visualization of Data:
  - Supervised Learning:
    - Uses classification models to label data on graphs.
  - Unsupervised Learning:
    - Isolates patterns using clustering models on graphs.

- Key Differences Summarized:
  - Human Intervention:
    - Supervised: High
    - Unsupervised: Limited
  - Data Type:
    - Supervised: Labeled
    - Unsupervised: Unlabeled
  - Tools:
    - Supervised: Linear and logistic regression, decision trees, support vector machines
    - Unsupervised: Clustering and dimensionality reduction
  - Models:
    - Supervised: Regression and classification
    - Unsupervised: Clustering and association
  - Action:
    - Supervised: Training model with known results to update parameters.
    - Unsupervised: Segmenting data into groups or reducing data dimensionality.
  - Goal:
    - Supervised: Forecasting and predicting predefined outputs.
    - Unsupervised: Finding hidden patterns in data.
  - Results:
    - Supervised: Metrics reflecting the model’s ability to generalize data.
    - Unsupervised: Exploratory results identifying groups or anomalies.
  - Disadvantages:
    - Supervised: Requires labeled data; risk of overfitting or underfitting.
    - Unsupervised: May yield non-meaningful results; results are subjective.

- Module Focus:
  - Previous Module: Using clustering and association in unsupervised learning.
  - Current Module: Key concepts of supervised learning and their application to forecast and predict continuous values with regression models.
  - Next Module: Application of supervised learning concepts to discrete values using classification models.

3.1.3: Getting Started  
- Preparation for Module 3:

  - File Downloads:
    - Download "Module 3 Demo and Activity Files".

  - Installation Requirements:
    - Essential to install scikit-learn to complete activities and demonstrations.

  - Installation Instructions:
    - Check if scikit-learn is already installed:
      - Open a terminal window.
      - Activate the dev virtual environment with `conda activate dev`.
      - Run `conda list scikit-learn` to check for an existing installation.
    - If not installed:
      - Follow instructions in the official Scikit-learn documentation for installation.

  - Reminder:
    - Instructions included as a refresher, assuming scikit-learn may have been installed previously during the unsupervised learning module.

  - Outcome:
    - Post-installation, ready to create linear regression models in supervised learning.

### Supervised Learning Key Concepts
3.2.1: Features and Labels  
- Distinction Between Supervised and Unsupervised Learning:
  - Key Difference: Use of labeled data in supervised learning.

- Key Concepts in Supervised Learning:
  - Features:
    - Known as independent variables in statistics.
    - Used to predict changes in other variables.
    - In supervised learning, features predict labels.
    - Example: Shape and color of fruit in image identification.
  - Labels:
    - Known as dependent variables in statistics.
    - Outcome that is to be predicted, depending on features.
    - Sometimes referred to as target variables.
    - Example: Diagnosis of myopia based on various ocular features.

- Important Note:
  - Target Column: The column in a dataset that contains the labels.

- Practical Application:
  - Working with Pandas DataFrames where features and labels are already classified.
  - Features are represented in column headers, excluding the target column.
  - Labels are in the "outcome" column.

- Data Analysis in Supervised Learning:
  - Preliminary analysis to identify impactful features.
  - Reduction of datasets to significant features for label impact.

- Coding Conventions:
  - Shorthand Variable Names:
    - X represents features (uppercase due to multiple columns).
    - y represents class labels (lowercase for typically a single column).

- Supervised Learning Algorithms:
  - Utilizes features and labels for training.
  - Classification and regression algorithms are used for new data classification and outcome prediction.

3.2.2: Regression vs. Classification  
- Understanding Supervised Learning Goals:
  - Aim: To predict and forecast predefined outcomes.
  - Examples:
    - Regression for continuous variables (e.g., predicting rainfall amount for a farmer).
    - Classification for discrete outcomes (e.g., predicting rain occurrence for event planning).

- Types of Algorithms in Supervised Learning:
  - Regression:
    - Used for modeling and predicting continuous variables.
    - Example: Predicting a tree's growth based on environmental conditions.
  - Classification:
    - Used for predicting discrete outcomes.
    - Example: Classifying a property as an apartment or a house based on specific features.

- Characteristics of Variables:
  - Continuous Variables:
    - Represented by numbers divisible into smaller fractions.
    - Typically plotted as lines on a graph.
  - Discrete Variables:
    - Cannot be subdivided into smaller parts.
    - Represent finite values, often shown as scatter plots on graphs.

- Visualization of Regression vs. Classification:
  - Example with Housing Data:
    - Regression: Plotting housing prices against floor area to define a trend.
    - Classification: Distinguishing between apartments and houses as separate classes.
  - Both plots use the same data but with different objectives.

- Deep Dive into Classification:
  - Nonbinary Classification:
    - More than two discrete outcomes are possible (e.g., multiple types of fruit).
    - Adds complexity to the classification process.
  - Focus on Binary Outcomes:
    - Upcoming modules will concentrate on binary ("either-or", "yes-no") outcomes.

- Creating Supervised Learning Models:
  - Common Pattern: Model-fit-predict.
  - This approach is used regardless of whether the model is for regression or classification.

3.2.3: Model-Fit-Predict  
- Overview of Model-Fit-Predict in Supervised Learning:
  - A common three-stage pattern: model selection, fitting, and prediction.

- Model Stage:
  - Selection of an appropriate machine learning algorithm.
  - Represents a relationship in the real world (e.g., housing prices and floor area).
  - An untrained model is like a mathematical ball of clay, ready to be shaped.

- Fit Stage (Training Stage):
  - Adjusting the model to match patterns in the data.
  - Utilizes labeled data to fit the model closely to the data trend.
  - The model learns to make predictions that match the given data.

- Predict Stage:
  - Using a trained model to predict outcomes for new data.
  - The model's effectiveness depends on its similarity to the training data.

- Considerations and Questions in the Prediction Stage:
  - Evaluating the model's effectiveness and accuracy.
  - Addressing potential errors due to omitted features for simplicity.
  - Deciding whether to include more features in the model.
  - Assessing the model's performance with new data.
  - Considering the complexity of training the model.
  - All these factors prompt the need to evaluate the trained model.

3.2.4: Model Evaluation  
- Evaluating Supervised Learning Models:
  - Key Concern: Accuracy and effectiveness of the model with new data.
  - Various methods are employed for model evaluation.

- Evaluation Methods:
  - Resampling Methods:
    - Rearranging data samples to test model generalization.
  - Random Split Methods:
    - Dividing data into training and testing sets, sometimes including a validation set.
  - Time-Based Split Methods:
    - Reserving data for specific time intervals for testing.
    - Useful for time-series data where seasonal patterns are significant.
  - K-Fold Cross-Validation Methods:
    - Randomly shuffling the dataset and dividing it into k groups for testing and training.

- Evaluation Metrics vs. Methods:
  - Evaluation methods are different from evaluation metrics.
  - Choice of metric depends on the model type.
  - Common Metrics:
    - Mean Squared Error (MSE) and R-squared (R2) for linear regression.
    - Accuracy test for classification in the next module.

- Upcoming Learning:
  - Detailed exploration of model evaluation, including classification report and confusion matrix, in the machine learning optimization module.
  - Learning to create training and testing sets is a common practice in evaluating supervised learning models.

3.2.5: Training and Testing Data  
- Overview of Training and Testing in Supervised Learning:
  - Applicable to both regression and classification problems.
  - Involves dividing datasets into training and testing sets.

- Splitting the Data:
  - Training Dataset:
    - Used for fitting the machine learning model.
    - Model learns from this dataset.
  - Testing Dataset:
    - Used for evaluating model performance.
    - Tests the model's effectiveness on new data.

- Analogy:
  - Studying for an exam using a test bank of questions.
  - Studying with part of the questions and using the rest as a mock test to gauge readiness.

- Importance of Splitting Datasets:
  - Provides an understanding of how the model performs with unseen data.
  - Essential for training the model for future predictions without expected target values.

- Deep Dive into Train-Test Split Ratio:
  - Ratio determines the division of data for training and testing.
  - Common ratios include 80:20, 70:30, 60:40, and 50:50.

- Considerations for Effective Splitting:
  - Sufficient data for both training the model and evaluating its performance.
  - The original dataset must be large enough to represent the analysis scope adequately.
  - Adequate representation includes covering common and most uncommon scenarios.
  - Large datasets allow for efficient computational processing, especially in repeated evaluations.

3.2.6: Recap and Knowledge Check  
- Initial Steps in Supervised Learning:
  - Arranging data into features (independent variables) and labels (dependent variables).

- Model Selection Based on Label Type:
  - Continuous Outcome: Utilize a regression model.
  - Discrete Outcome: Employ a classification model.

- Criteria for Model Choice:
  - Selecting a model that fits well with the dataset.
  - Ensuring accurate and effective outcome prediction with unseen data.

- The Model-Fit-Predict Process:
  - A three-stage procedure involving model selection, fitting, and prediction.

- Model Evaluation:
  - Assessing the chosen model's performance is crucial.
  - Training the model and verifying its predictions with labeled data.

- Common Method in Supervised Learning:
  - Splitting the dataset into:
    - Training Data: For the model to learn from.
    - Testing Data: To validate the model's performance.

### Linear Regression
3.3.1: Introduction to Linear Regression  
3.3.2: Making Predictions with Linear Regression  
3.3.3: Model Evaluation: Quantifying Regression  
3.3.4: Activity: Predicting Sales with Linear Regression  
3.3.5: Recap and Knowledge Check  

### Summary: Supervised Learning — Linear Regression
3.4.1: Summary: Supervised Learning — Linear Regression  
3.4.2: Reflect on Your Learning  
3.4.3: References  
