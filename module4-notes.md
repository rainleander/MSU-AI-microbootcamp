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
**Background:**
- Nefarious actors attempt to steal private information through malware apps on electronic devices.
- The task is to use logistic regression to identify malware in a real-world problem.

**Instructions:**
1. **Prepare the Data:**
   - Load the "app-data.csv" file into a Pandas DataFrame.
   - Use value_counts to determine how many malware apps are in the dataset.

2. **Split the Data into Training and Testing Sets:**
   - Define the target variable (y) as "Result" and features (X) as all columns except "Result."
   - Split the data into training and testing datasets (X_train, X_test, y_train, y_test) using train_test_split.

3. **Model and Fit the Data to a Logistic Regression:**
   - Declare a LogisticRegression model.
   - Fit the model to the training data and save it.
   - Validate the model.

4. **Predict the Testing Labels:**
   - Use the testing dataset to make predictions about malware and save them.

5. **Calculate the Performance Metrics:**
   - Calculate the accuracy score by comparing y_test to testing_predictions.
   - Determine how well the model predicted actual malware for this dataset.

**Resources:**
- Logistic regression
- train_test_split
- Accuracy score

**Solution:**
- Compare your work to the solution in the Solved folder in the activity folder.
- Evaluate your performance and check for differences between your approach and the solution.

4.2.7: Recap and Knowledge Check  
- Logistic regression predicts discrete outcomes based on probability.
- Probability is represented using a sigmoid curve.
- The model assigns a sample to a class if its probability exceeds a cutoff point.
- The same model-fit-predict process as linear regression is followed in supervised learning.
- Steps: 
  1. Create a model with LogisticRegression().
  2. Train the model with model.fit().
  3. Make predictions with model.predict().
  4. Evaluate the model with accuracy_score().

### Model Selection
4.3.1: Introduction to Model Selection  
- Linear regression models are suitable for predicting continuous variables.
- Logistic regression models are used for predicting binary classifications.
- In cases where logistic regression doesn't perform well, other models like support vector machines (SVM) may be more effective.
- Complex data that linear models cannot handle may require non-linear models, such as decision trees.
- Factors to consider when choosing a classification model include the data type, data size, and problem complexity.
- Linear models like logistic regression and linear SVM are suitable for well-separated, linearly separable data.
- Non-linear models like SVM and decision trees are necessary for non-linear or highly overlapping data.
- Scikit-learn provides a consistent model-fit-predict interface, making it easy to try different models with minimal code changes.
- Tools like Amazon's SageMaker can automatically evaluate and rank different models.
- Future modules will cover more techniques for model evaluation and tuning.

4.3.2: Linear vs. Non-linear Data  
- Linear classification models have a linear relationship between input variables and the outcome.
- Non-linear classification models have a non-linear relationship between input variables and the outcome.
- Plotting data is a good practice to determine if the relationship is linear or non-linear.
- A straight line can often separate linear data, while non-linear data may require different shapes to separate classes.
- Non-linear models are more efficient for classifying data into more than two categories.
- Non-linear models suit complex relationships between input features and output predictions.
- The SVM model is explored to improve prediction accuracy over logistic regression, considering both linear and non-linear models.

4.3.3: Linear Models  
- Logistic regression defines a line to separate two sets of data.
- SVMs are useful for classification analysis in multidimensional space.
- SVMs can employ both linear and non-linear approaches.
- SVMs create a hyperplane to separate data points.
- The optimal hyperplane maximizes boundaries between groups.
- Support vectors are data points closest to the hyperplane margin.
- Support vectors can be errors when they fall within the margin.
- SVMs can operate in multidimensional space, creating 3D hyperplanes.
- Zero tolerance with perfect partition is used for non-linear hyperplanes.
- SVM is beneficial for classifying outliers and overlapping data points.
- SVM can provide higher accuracy with less computation power.

4.3.4: Demonstration: Predict Occupancy with SVM  
- An SVM classifier predicts office space occupancy based on conditions.
- Steps similar to building a logistic regression model.
- Import required dependencies (pandas, train_test_split, accuracy_score, SVC).
- Import data from an external source.
- Preprocess data by splitting it into features and target variables.
- Split data into training and testing sets.
- Create an SVM model with a linear kernel.
- Fit the model to the training data.
- Validate the model's accuracy on both training and testing data.
- Make predictions with the testing data.
- Evaluate the model using the accuracy_score function, confirming its accuracy.
- Introduction to using SVM for linear models and considerations for non-linear data.

4.3.5: Non-linear Models  
- Examination of two non-linear models: Non-linear SVM and Decision trees.
- Decision trees can be used for multi-class classification and to improve prediction accuracy.
- Non-linear SVM uses a circular hyperplane in three-dimensional space to separate overlapping data points.
- Decision trees are a method for encoding true-false questions to map non-linear relationships in data.
- Decision tree concepts: root node, parent node, child node, decision node, leaf or terminal node, branch or subtree, splitting, pruning, tree depth.
- Complex and deep decision trees tend to overfit the data.
- Example of using a decision tree model to predict the success of a crowdfunding campaign.

4.3.6: Demonstration: Decision Trees  
- Introduction to using a decision tree model to answer questions about crowdfunding campaigns.
- Importing necessary libraries for decision tree modeling and visualization.
- Loading data from the "crowdfunding-data.csv" file into a DataFrame.
- Preprocessing the data: defining features and target variables, splitting data into training and testing sets, and scaling features using StandardScaler.
- Creating and training a decision tree classifier model.
- Making predictions and evaluating model accuracy.
- Visualizing the decision tree model.
- Note the possibility of overfitting when decision trees are too precise with the training data.

4.3.7: Overfitting  
- Complexity of a model increases with the number of independent variables or features.
- This complexity is particularly evident in non-linear data and multi-class classification problems.
- Models with many variables are harder to train.
- Balancing complexity with the goals of accurate pattern capture and consistent predictions is crucial.
- Decision trees are complex models that can accurately capture patterns in training data but may overfit new data.
- Overfitting can lead to poor generalization of new data.
- Overfitting occurs when the model's learned patterns are too specific to the training dataset.
- High variance is a result of overfitting.
- Reducing model complexity can help reduce overfitting.
- Some ensemble learning methods use decision trees to address overfitting.

4.3.8: Recap and Knowledge Check  
- Selecting the suitable machine learning model is crucial in the model-fit-predict process.
- Model choice is influenced by data type, data size, and problem complexity.
- Trying multiple models is feasible as many Scikit-learn models follow the model-fit-predict process.
- Linear classification models represent relationships using straight lines, and logistic regression is commonly used for such problems.
- Support Vector Machines (SVM) can help find hyperplanes that maximize data separation in cases with unclear class boundaries.
- The SVC module in Scikit-learn is used for SVM models, and a linear kernel is suitable for linear data.
- Non-linear data may require non-linear SVM models, which can separate data in three-dimensional space.
- Decision trees are ideal for multi-class and binary classification, representing possible solutions based on conditions.
- Decision trees can become complex, leading to overfitting, where the model performs well on training data but poorly on new data.
- Overfitting results in high variance between training and testing set predictions.

### Ensemble Learning
4.4.1: Introduction to Ensemble Learning  
- **Ensemble Learning Overview:**
  - Basic concept: Combining multiple models to enhance accuracy and robustness.
  - Reduces variance and increases overall performance.
  - Involves feeding data to multiple algorithms and aggregating their predictions.

- **Techniques in Ensemble Learning:**
  - **Random Forest:**
    - Combines multiple decision trees to form a more powerful model.
  - **Boosting:**
    - Another ensemble technique discussed later in the lesson.

- **Advantages of Ensemble Learning:**
  - Combines predictions from two or more models.
  - Improves the final model's accuracy and robustness.

- **Weak Learners:**
  - Definition: Algorithms that perform poorly independently due to limitations like insufficient data.
  - Role in Ensemble Learning:
    - Even though individually weak, they contribute to a stronger combined model.
    - Example: Different algorithms for speech emotion recognition in different languages can be combined for better performance in multilingual environments.
  - Utilization in Techniques:
    - Incorporated methods like random forest and boosting.

4.4.2: Random Forest  
- **Random Forest Characteristics:**
  - Composition: Collection of smaller, simpler decision trees.
  - Operation: Trees work together as an ensemble.
  - Decision Making: Majority vote determines the final prediction.

- **Functioning of a Random Forest:**
  - Each tree predicts for a specific class.
  - Built from a random subset of features.
  - Example: In a forest of nine trees, if six trees predict 1 and three predict 0, the final prediction is 1.

- **Role of Weak Learners:**
  - Simple trees are weak learners.
  - Created by randomly sampling data and focusing on a small portion.
  - Slightly better than a random guess individually.
  - Combined to form a strong learner with superior decision-making ability.
  - Protect against individual errors as long as trees don’t consistently make the same mistakes.

- **Benefits of Random Forest Algorithms:**
  - Resistance to overfitting due to training on different data segments.
  - Natural ranking of input variables' importance.
  - Capability to handle thousands of input variables without needing to delete any.
  - Robustness against outliers and non-linear data.
  - Efficient performance on large datasets.

4.4.3: Demonstration: Random Forest  
- **Random Forest Model with Malware Classification Dataset:**
  - Previous success: Achieved 96% accuracy using logistic regression.
  - Dataset details: 86 features, 29,332 applications.
  - Objective: Explore if a random forest model can enhance prediction accuracy.

- **Setting Up the Random Forest Model:**
  - Source: Available in the `demos/04-Random_Forest/Solved` folder.
  - Tools: Utilizes Jupyter Notebook or Google Colab for coding.
  - Libraries: Uses Scikit-learn's ensemble module.
  - Dependency changes: Replaces logistic regression import with random forest classifier import.

- **Data Preparation and Preprocessing:**
  - Data Source: `app-data.csv` from previous logistic regression activity.
  - Process:
    - Split data into training and testing sets.
    - Define feature and target sets (using `ravel()` method for the target set).

- **Fitting the Random Forest Model:**
  - Creation: Instantiate RandomForestClassifier with 128 trees.
  - Fitting: Model trained with training data (X_train, y_train).

- **Making Predictions and Evaluating the Model:**
  - Predictions: Made using testing data (X_test).
  - Evaluation: Accuracy score calculated and compared with logistic regression model.
  - Result: Random forest model shows a slightly higher accuracy (96.7%) than logistic regression (96%).

- **Feature Importance Analysis:**
  - Purpose: Determine the impact of each feature on the model.
  - Method: Listing the top ten features based on their importance.
  - Considerations: Identifying and removing overpowering or low-impact features to improve model accuracy.

- **Next Steps in Learning:**
  - Exploration of other ensemble learning methods under the boosting category.

4.4.4: Boosting  
- **Overview of Boosting in Ensemble Learning:**
  - Boosting enhances weak learners by focusing on correct solutions.
  - Mainly used for classification, though applicable to regression.

- **Boosting vs. Random Forest:**
  - Both use decision trees as a basis.
  - Difference lies in how decision trees are built and combined.
  - Random Forest: Combines separate trees, each learning from a data subset.
  - Boosting: Sequentially adds trees to learn from previous errors.

- **Boosting Methodology:**
  - **Assign Weights:**
    - Each sample is weighted based on prediction difficulty.
    - Weights updated throughout the process.
  - **Algorithm Focus:**
    - Learns to make correct predictions for higher-weighted, difficult samples.
  - **Sequential Learner Addition:**
    - Learners added one after another, not simultaneously.
    - Focuses on minimizing errors and improving predictions with each iteration.
  - **Result:**
    - Final model is a weighted sum or vote of all learners.

- **Characteristics and Considerations:**
  - Boosting is time-consuming due to multiple iterations.
  - Often results in increased accuracy.
  - Appropriate for datasets that benefit from detailed error analysis and correction.

- **Advanced Boosting Methods:**
  - **AdaBoost (Adaptive Boosting):**
    - Iteratively gives more weight to incorrectly predicted observations.
    - Focuses on challenging outcomes to improve accuracy.
  - **Gradient Boosting:**
    - Trains shallow trees using pseudo residuals to reduce prediction error.
    - Aims for negligible error or near-zero log-likelihood in classification.
  - **XGBoost (eXtreme Gradient Boost):**
    - Optimizes speed and performance.
    - Builds trees in parallel and evaluates split quality at each data level.

- **Further Learning Resources:**
  - AdaBoostClassifier, GradientBoostingClassifier, and XGBoost documentation.

4.4.5: Recap and Knowledge Check  
- **Decision Trees and Their Limitations:**
  - Useful for non-linear data and multiple features.
  - Can become complex and deep, leading to overfitting.
  - Overfitting: Accurate on training data but poor on testing/new data.

- **Ensemble Learning as a Solution:**
  - Combines predictions from multiple models to create a stronger model.
  - Addresses overfitting and enhances prediction accuracy.
  - Models in the ensemble are "weak learners," slightly better than random guesses.
  - Weak learners, when combined, compensate for each other's errors.

- **Common Ensemble Learning Techniques:**
  - **Random Forest:**
    - Consists of multiple simple decision trees.
    - Each tree trained on a subset of data and predicts a specific class.
    - Predictions are combined, often through majority voting in classification.
    - Implemented via `RandomForestClassifier` in Scikit-learn.
    - `n_estimators` parameter controls the number of trees, recommended between 64–128.
  - **Boosting:**
    - Aggregation method enhancing weak learners based on correct solutions.
    - Errors are weighted, with each new learner focusing on these errors.
    - Sequential building of the ensemble, learning from the full training set.
    - Reduces bias by avoiding repetitive errors.
    - Advanced forms include Gradient Boosting and XGBoost for better performance and efficiency.

### Summary: Supervised Learning — Classification
4.5.1: Summary: Supervised Learning — Classification  
- **Completion of Supervised Learning — Classification:**
  - Achievement: Understanding how to apply various classification models.

- **Key Learning Outcomes:**
  - **Model Selection:**
    - Ability to determine appropriate situations for using classification models.
  - **Understanding Different Models:**
    - Logistic Regression: Suitable for binary output prediction.
    - Support Vector Machines (SVMs): Effective in separating data points into two classes using a hyperplane; capable of handling overlapping and non-linear data.
    - Decision Trees: Useful for multiple classes and features, but prone to overfitting and complexity.
  - **Data Types Recognition:**
    - Identifying linear versus non-linear data.

- **Application of Models:**
  - Mastering the model-fit-predict process for logistic regression, SVMs, decision trees, and random forests.

- **Concept of Ensemble Learning:**
  - Understanding how it combines multiple models for improved accuracy and performance.
  - Addressing issues like overfitting and complexity in decision trees.

4.5.2: Reflect on Your Learning  
- **Reflection on Learned Concepts:**
  - Identify challenging concepts or skills for further practice.
  - Consider practical applications of learned classification methods in professional or hobby-related contexts.

- **Application and Integration:**
  - Explore ways to utilize classification methods for making predictions in various scenarios.
  - Consider specific examples relevant to job or personal interests.

- **Growth in Understanding:**
  - Reflect on how understanding of classification models and ensemble learning has evolved.
  - Assess changes in comprehension and application of these models.

- **Curiosity and Further Learning:**
  - Identify areas of continued interest or curiosity.
  - Explore resources or experts for additional learning or clarification.

- **Future Learning Path:**
  - Anticipate building upon current skills and knowledge in subsequent course modules.
  - Prepare for learning optimization techniques for supervised learning models.

4.5.3: References  
Classifying Malware (app-data.csv): Mathur, A. 2022. NATICUSdroid (Android permissions) dataset [Dataset]. UCI Machine Learning Repository. Available: https://archive-beta.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset [2023, April 28].

Detecting Occupancy (occupancy.csv): Dua, D. & Graff, C. 2019. Occupancy detection data set [Dataset]. UCI Machine Learning Repository. Available: https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection [2023, April 28].

Additional datasets used in this module were generated by edX Boot Camps LLC, and are intended for educational purposes only.
