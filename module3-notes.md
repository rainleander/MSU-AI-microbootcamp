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
  - Learning to create training and testing sets, a common practice in evaluating supervised learning models.

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
  - Study part of the questions and use the rest as a mock test to gauge readiness.

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
- Supervised Learning and Linear Regression:
  - Linear regression is used for predicting continuous values.
  - Suitable for both continuous and discrete variable predictions.

- Continuous Values:
  - Can always be divided into smaller ranges.
  - Examples include distance and price, with the possibility of finding smaller units.

- Linear Regression Explained:
  - Describes the relationship between a dependent variable and one/more independent variables.
  - Example: The contagion rate (dependent variable) depends on various factors (independent variables).

- Types of Linear Regression:
  - Simple Linear Regression:
    - Relationship between one dependent and one independent variable.
  - Multiple Linear Regression:
    - Relationship between one dependent variable and two or more independent variables.

- Formula and Components:
  - Formula represents the relationship between independent variable (x) and dependent variable (y).
  - Simple linear regression model is written as y = a + bx.
    - b: Slope of the relationship.
    - a: Y-intercept (value of y when x is 0).
  - Greek Notation in Equations:
    - Β0 (Beta zero) represents the y-intercept.
    - Β1 (Beta one) represents slope.
    - x is the independent variable.

- Trends in Linear Regression Data:
  - Positive Trends: Dependent value y increases as independent value x increases.
  - Negative Trends: Dependent value y decreases as independent value x increases.
  - No Trend: Random increases and decreases in y with no apparent pattern as x increases.

- Application in Supervised Learning:
  - Linear regression is a supervised learning model.
  - Predicts the value of y based on historical data.

3.3.2: Making Predictions with Linear Regression  
- Practical Application of Supervised Learning Concepts:
  - Example: Predicting salary based on years of Python and Scikit-learn experience.

- Steps for Implementing a Supervised Learning Model:
  - Model-Fit-Predict process involving creation, training, and prediction using the model.

- Getting Started with the Example:
  - Dataset: CSV file with salary data.
  - Python Setup: 
    - Use of Scikit-learn for linear regression.
    - Import necessary libraries: numpy, pandas, sklearn.linear_model.
  - Loading Data:
    - Read salary data into a Pandas DataFrame.
    - Inspect the relationship between years of experience and salary using a scatter plot.

- Data Preparation:
  - Formatting Data for Scikit-learn:
    - Reshape years of experience data to meet Scikit-learn requirements.
    - Assign the salary column as the target variable (y).

- Creating and Training the Model:
  - Instantiate a linear regression model.
  - Fit the model with input (X) and output (y) data.

- Model Parameters and Predictions:
  - Examine model parameters like slope and y-intercept.
  - Use the model's formula for salary prediction.
  - Make multiple predictions using the predict() method.

- Analyzing Predictions:
  - Compare original and predicted salary data in a DataFrame.
  - Visualize predictions with a line plot of the best fit.

- Extending Predictions Beyond Current Data:
  - Extrapolate predictions for years of experience not in the dataset.
  - Plot original data points and predicted values together.

- Summary of Supervised Learning Pattern:
  - Split data into input (X) and output (y).
  - Create a model instance and train it with the dataset.
  - Generate predictions using the trained model.

- Skill Drill:
  - Encouragement to rerun the notebook to understand linear regression predictions better.

3.3.3: Model Evaluation: Quantifying Regression  
- Evaluating Regression Model Performance:
  - Importance: Visual confirmation is not enough; quantification of the model is essential.

- Common Quantification Scores:
  - R-squared (R2):
    - Measures how well the model accounts for data variability.
    - Ranges between 0 and 1; higher values indicate better predictiveness.
    - Example: An R2 of 0.85 means the model accounts for 85% of data variability.
  - Mean Squared Error (MSE):
    - Indicates accuracy in predicting each data point.
    - Larger errors have a greater impact due to squaring, sensitive to outliers.
    - Always above 0 with no upper limit; varies widely between projects.
    - Used for comparing models trained on the same data.
  - Root Mean Squared Error (RMSE):
    - Aggregate of prediction errors' magnitude across data points.
    - Amplifies the importance of outliers.
    - Uses the same units as the training data.
    - "Good" scores for R2 and MSE depend on the problem domain.

- Understanding Errors:
  - Error refers to the distance between each data point and the model's line of best fit.

- Calculating Metrics in Python:
  - Using sklearn metrics functions: `mean_squared_error` and `r2_score`.
  - Code Example:
    - Import required sklearn functions.
    - Calculate R2, MSE, and RMSE for the salary prediction model.
    - Example Output:
      - Score: 0.95696
      - R2: 0.95696
      - Mean Squared Error: 31270951.7223
      - Root Mean Squared Error: 5592.0436

- Skill Drill:
  - Rerun the notebook for deeper understanding.
  - The code is available in the provided Jupyter notebook solution file.

3.3.4: Activity: Predicting Sales with Linear Regression  
- Activity Overview: Creating a Linear Regression Model for Sales Prediction
  - Objective: Write code to produce a linear regression model predicting sales based on ad display numbers.
  - Evaluation: Calculate MSE, RMSE, and R2 scores to assess the model's performance.

- Background Scenario:
  - Developed a unique child seat suitable for traveling families.
  - Ran a digital marketing campaign with varying ad displays to promote the product.
  - Aim to predict expected sales based on the number of ads displayed.

- Steps for Completing the Activity:
  - Access the activity files in `activities/01-Linear_Regression/Unsolved`.
  - Use the Jupyter notebook in the Unsolved folder.
  - Process:
    1. Load and visualize sales data.
    2. Prepare data for fitting into a linear regression model.
    3. Build the model using Scikit-learn's LinearRegression module.
    4. Plot the line of best fit for the sales prediction model.
    5. Manually predict sales with a hypothetical 100 ads.
    6. Use the predict function for automated predictions.
    7. Compute relevant metrics (MSE, RMSE, R2) for model evaluation.

- Post-Activity Review:
  - Access the solution in the Solved folder.
  - Compare your approach and code to the provided solution.
  - Identify any differences or areas of confusion.
  - Seek assistance during instructor-led office hours for further clarification.

3.3.5: Recap and Knowledge Check  
- Overview of Linear Regression in Supervised Learning:
  - Purpose: Describes the relationship between a numerical response and explanatory variables.

- Types of Linear Regression:
  - Simple Linear Regression:
    - Relationship between one dependent and one independent variable.
  - Multiple Linear Regression:
    - Relationship between one dependent variable and two or more independent variables.

- Linear Regression Formula:
  - Models the relationship between independent variable (x) and dependent variable (y).

- Practical Application Using Python:
  - Example: Predicting a person’s salary based on work experience.
  - Tool: Scikit-learn, a Python machine learning library.

- Basic Pattern for Linear Regression:
  1. Data Splitting:
     - Split data into inputs (X) and outputs (y).
  2. Model Creation:
     - Instantiate the model: `model = LinearRegression()`.
  3. Model Training:
     - Train the model using the dataset: `model.fit(X, y)`.
  4. Making Predictions:
     - Generate predictions: `y_predictions = model.predict(X)`.

### Summary: Supervised Learning — Linear Regression
3.4.1: Summary: Supervised Learning — Linear Regression  
3.4.2: Reflect on Your Learning  
3.4.3: References  
