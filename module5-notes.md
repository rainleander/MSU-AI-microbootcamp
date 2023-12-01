## Module 5: Machine Learning Optimization
### Introduction to Machine Learning Optimization
5.1.1: Introduction to Machine Learning Optimization  
- **Machine Learning Overview**:
  - Subset of AI.
  - Utilizes algorithms and statistical models.
  - Aims for automatic performance improvement in tasks through experience.
  - Involves pattern recognition in data without explicit programming.
  - Relies on extensive data for training models.

- **Machine Learning Optimization**:
  - Process to enhance machine learning model performance.
  - Adjusts parameters and hyperparameters.
  - Utilizes training datasets for running models.
  - Involves performance evaluation on validation datasets.
  - Focuses on improving accuracy or other performance metrics.

- **Reflection on Previous Learning Modules**:
  - Explored unsupervised and supervised learning methods.
  - Questions about the completeness of covered content regarding model training.
  - Suggests potential gaps needing attention.

- **Reassessing Performance Metrics**:
  - Initially used accuracy to gauge model performance.
  - Questions if accuracy is the best success indicator.

- **Module Learning Goals**:
  - Teaches optimization of model performance.
  - Focuses on various metrics to assess model success.
  - Addresses challenges of imbalanced data.
  - Aims to equip learners with:
    - Skills to choose appropriate metrics for specific projects and datasets.
    - Techniques for preprocessing imbalanced data.
    - Methods for tuning models using hyperparameters.

5.1.2: Getting Started  
- **File Downloads**:
  - Download "Module 5 Demo and Activity Files" before starting the module.

- **Required Installations**:
  - Utilize Scikit-learn and imbalanced-learn for data processing, model training, evaluation, and optimization.

- **Installation Steps for Imbalanced-Learn**:
  - Open a terminal window.
  - Activate the 'dev' virtual environment using the command `conda activate dev`.
  - Check for existing imbalanced-learn installation:
    - Run `conda list imbalanced-learn`.
    - If a version appears, it's already installed.
  - If not installed, execute `pip install -U imbalanced-learn`.
  - Begin the module once the installation is complete.

### Evaluating Model Performance
5.2.1: What is a good model?  
- **Machine Learning Process Complexity**:
  - Numerous choices in data preprocessing and model training.
  - Challenges in determining model adequacy and comparing results.

- **Defining a "Good" Model**:
  - Traditionally focused on accuracy as a critical metric.
  - Importance of context in evaluating model quality.
  - For example: 60% accuracy is significant in stock predictions but needs to be improved for identifying edible mushrooms.

- **Relevance and Limitations of Accuracy**:
  - Accuracy as a metric is context-dependent.
  - Example: A model predicting all instances as Class A in a Class A-heavy dataset may show high accuracy but needs true class differentiation.
  - Importance varies based on class balance and the weight of correct/incorrect predictions.

- **Alternative Performance Metrics**:
  - Sensitivity, specificity, and precision are other important metrics.
  - Example: In edible mushroom identification, false negatives are less critical than false positives.

- **Selecting the Right Target Column**:
  - Choosing the target column that aligns with the problem being solved is critical.
  - Different target columns, like “profit” vs “revenue,” yield varied real-world outcomes.
  - The upcoming activity will provide practice in selecting appropriate target columns.

5.2.2: Overfitting and Underfitting  
- **Assessing Model Fit**:
  - Evaluate for overfitting or underfitting.
  - Overfit/underfit models introduce bias and variance, skewing results.

- **Understanding Underfitting**:
  - Occurs when a model can't capture the relationship between inputs and outputs.
  - Results from overly simplistic models.
  - Fails to represent the complexity of the problem.

- **Concept of Overfitting**:
  - Involves finding non-representative patterns in training data.
  - Model becomes too accurate for training data but fails on new data.
  - Focuses on specific quirks of training data, not generalizable principles.

- **Visual Representation**:
  - The image illustrates the difference between overfitting, underfitting, and good fit.

- **Impact of Overfitting and Underfitting**:
  - Introduces bias and variance, leading to prediction errors.
  - High-bias models oversimplify and miss nuances.
  - High variance models are overly complex and fail to generalize.

- **Examples of Model Biases**:
  - Underfit model: "Every fruit is an apple."
  - Overfit model: Extremely specific characteristics of an apple.

- **Balancing Bias and Variance**:
  - Aim for a balance to ensure accurate predictions.
  - Avoid overly simplistic or complex models.

- **Testing Model Performance**:
  - Test model on fresh, untrained data.
  - Compare new data scores with training data scores.
  - Identify occurrences of overfitting or underfitting.

- **Upcoming Content**:
  - Further exploration of measurements to avoid over- and underfitting.

5.2.3: Confusion Matrix  
- **Understanding Over- and Underfitting**:
  - Importance of evaluating models beyond accuracy.
  - Performance is measured by comparing predicted and actual values.

- **Types of Model Predictions**:
  - True Positives: Correctly identifying true cases (e.g., edible mushrooms as edible).
  - True Negatives: Correctly identifying false cases (e.g., poisonous mushrooms as not edible).
  - False Positives: Incorrectly identifying false cases as true.
  - False Negatives: Incorrectly identifying true cases as false.

- **Confusion Matrix**:
  - Tool to organize model results into true/false positives/negatives.
  - The example is shown in a 2x2 array format.
  - Scikit-learn includes a `confusion_matrix` function for easy creation.

- **Using a Confusion Matrix**:
  - Example output of a confusion matrix:
    - `array([[11, 1], [0, 13]])`
  - Indicates the number of each type of result.

- **Practical Application Activity**:
  - Activity to practice generating a confusion matrix.
  - Located in the `activities/01_Confusion_Matrix/Unsolved` folder.
  - Steps include reading starter code, adding code for a confusion matrix, and interpreting results.

- **Evaluation and Comparison**:
  - Compare personal results with the solution in the "Solved" folder.
  - Reflect on differences in approach and understanding.

5.2.4: Accuracy  
- **Accuracy in Generative AI Models**:
  - Acknowledges limitations of accuracy as a success metric.
  - Accuracy: How many correct results does a model return?

- **Calculating Accuracy**:
  - Derived from confusion matrix: \( \text{Accuracy} = \frac{\text{True Positives (TP) + True Negatives (TN)}}{\text{Total Results}} \).

- **Applicability of Accuracy**:
  - Suitable when all outcomes have similar importance.
  - Example: Identifying salads or sandwiches in images.

- **Limitations of Accuracy**:
  - Inadequate in scenarios where outcomes have varying importance.
  - Example: Detecting fraudulent credit card transactions.
  - Potential misleading high accuracy in models that fail in their primary function.

- **Need for Alternative Measurements**:
  - Necessary in cases like fraud detection where traditional accuracy is insufficient.
  - Example: The model shows 99.99% accuracy but fails to identify fraudulent transactions.

5.2.5: Other Metrics  
- **Sensitivity (Recall)**:
  - Measures how many actual true data points are correctly identified.
  - Ideal for situations where avoiding false negatives is crucial (e.g., identifying fraudulent transactions, poisonous mushrooms, or cancer patients).

- **Specificity**:
  - Assesses how many actually false data points are correctly identified as negative.
  - Important for models where false positives are undesirable (e.g., identifying patients or edible mushrooms).

- **Precision**:
  - Focuses on avoiding false positives.
  - Useful for scenarios like identifying credit risks or high-risk areas for violent crime.
  - Determines how many predicted true results are actually true.

- **Balancing Risk and Measurement Selection**:
  - Importance of choosing the right metric for a given problem.
  - Sensitivity vs. Precision:
    - High sensitivity: Good for diagnosing diseases like cancer, where missing a positive result can have severe consequences.
    - High precision: Important in criminal justice to avoid wrongful convictions.

- **F1 Score**:
  - Balances sensitivity and precision.
  - The harmonic mean of sensitivity and precision.
  - Higher scores indicate a better balance between sensitivity and precision.

- **Deep Dive and Further Research**:
  - Weighted F1 scores can be more informative in specific contexts.
  - Encouraged to research weighted F1 scores for future projects.

5.2.6: Classification Report  
- **Using a Classification Report in Scikit-learn**:
  - Collates precision, sensitivity (recall), F1 score, and accuracy.
  - Conveniently summarizes various success metrics.

- **Example of a Classification Report**:
  - Test data with 192 data points: 125 false and 67 true.
  - Precision scores: 0.78 for negative predictions, 0.75 for positive.
  - Sensitivity (recall) scores: 0.9 for negative predictions, 0.54 for positive.
  - F1 scores and overall accuracy detailed.
  - Includes macro and weighted averages.

- **Observations from the Report**:
  - Similar precision for positive and negative predictions.
  - Lower sensitivity for positive predictions, affecting F1 score.

- **Generating a Classification Report**:
  - Code snippet to produce and display the report: `print(report)`.

- **Importance of Model Evaluation**:
  - Crucial to use these metrics for model improvement.

- **Pro Tip on Imbalanced Data**:
  - Imbalanced data can skew model predictions.
  - Example shows low sensitivity for positive predictions due to data imbalance.
  - An upcoming lesson will explore imbalanced data in more detail.

5.2.7: The Importance of Metric and Target Selection  
- **Selecting Target and Metric in Machine Learning**:
  - Emphasizes careful selection of target and metric before starting a project.
  - Different problems necessitate different outcome priorities.

- **Ethical Responsibility in Predictive Modeling**:
  - Importance of ethical considerations in model creation.
  - Targets and metrics selection can impact people's lives.

- **Examples of Ethical Considerations**:
  - Using "loan approved" as a target could perpetuate existing biases.
  - A model identifying all edible mushrooms but including inedible ones is impractical and dangerous.

- **Prioritizing Correct Identification**:
  - More useful to correctly identify inedible mushrooms, even if some edible ones are mistakenly excluded.

- **Importance of Metric Selection**:
  - Correct metric selection doesn't guarantee performance but helps identify and understand problems.

- **Upcoming Focus on Imbalanced Data**:
  - Next lesson to address how imbalanced data affects model performance.

5.2.8: Recap and Knowledge Check  
- **Summary of Lesson Content**:
  - Emphasis on choosing the right target and model.
  - Understanding overfitting and underfitting in models.
  - Utilization of confusion matrices and understanding true/false positives and negatives.
  
- **Different Success Metrics for Model Evaluation**:
  - Accuracy: How many correct results does a model return?
  - Sensitivity (Recall): Measures correct identification of true cases.
  - Specificity: Focuses on correctly identifying false cases as negative.
  - Precision: Assesses avoidance of false positives.
  - F1 Score: Balances sensitivity and precision.

- **Using Classification Reports**:
  - Classification reports summarize precision, sensitivity, F1 score, and accuracy.
  
- **Preparation for Model Optimization**:
  - Recognizes the need for additional data preparation in optimizing models.
  - Upcoming lesson to address challenges related to imbalanced data.

### Case Study: Imbalanced Data
5.3.1: Introduction to Imbalanced Data  
- **Preparing Data Before Model Training**:
  - Ensure sufficient, complete, and appropriate data for the problem domain.
  - Classification reports help identify model weaknesses in predicting different classes.

- **Imbalanced Data in Machine Learning**:
  - Occurs when some classes are more or less frequent than others.
  - Affects both binary and multi-class classification tasks.
  - Can mislead the effectiveness of a model when judged by accuracy alone.

- **Risk of Imbalanced Data**:
  - Models may favor predicting the majority class, minimizing total incorrect classifications.
  - Example: An unsophisticated model might show high accuracy in an imbalanced dataset by always predicting the majority class.

- **Scenarios Illustrating Imbalance**:
  - Fraudulent bank transactions are rare but crucial to identify.
  - Cancer diagnosis: fewer patients have cancer, but failing to identify them is problematic.

- **Addressing Imbalance in Data**:
  - Techniques like over- and undersampling can help models predict the minority class more effectively.

5.3.2: Oversampling and Undersampling  
- **Techniques for Handling Imbalanced Training Data**:
  - Generally categorized under oversampling and undersampling methods.

- **Oversampling**:
  - Involves creating more instances of the minority class to balance the dataset.
  - Helps to match the minority class quantity with the majority class.

- **Undersampling**:
  - Reduces instances of the majority class to equalize class representation.
  - Requires sufficient data in the majority class to remain effective after reduction.

- **Approaches to Sampling**:
  - Random Sampling: Chooses random instances from the existing dataset, used in oversampling and undersampling.
  - Synthetic Sampling: Generates new instances based on observations from existing data, e.g., using K-nearest neighbors for simulation.

- **Sampling Techniques Overview**:
  - Random Oversampling: Directly replicates minority class instances.
  - Synthetic Minority Oversampling Technique (SMOTE): Creates synthetic samples for the minority class.
  - Random Undersampling: Removes instances from the majority class.
  - Cluster Centroid: Reduces majority class by clustering and representing clusters with centroids.
  - SMOTE and Edited Nearest Neighbors (SMOTEENN): Combines SMOTE with cleaning of majority and minority class using ENN.

- **Imbalanced Classification Tree**:
  - Outlines different sampling techniques: Random Oversampling, SMOTEENN, and Random Undersampling.
  - Categorizes techniques by type for easier understanding and application.

5.3.3: Applying Random Sampling Techniques  
- **Random Under- and Oversampling Techniques**:
  - Import modules like pandas, sklearn, and StandardScaler for data processing.
  - Train random forest models on raw, undersampled, and oversampled data.
  - Compare models trained on different datasets.

- **Data Preparation and Analysis**:
  - Load and preprocess data from a CSV file.
  - Split data into features (X) and target (y), encode categorical variables.
  - Notice the imbalanced dataset in the training set.
  - Scale the data using StandardScaler.

- **Training a Base Model for Comparison**:
  - Train a RandomForestClassifier on the scaled training data.
  - Make predictions on the testing set.

- **Random Undersampling Method**:
  - Reduce instances of the majority class using RandomUnderSampler.
  - Fit a new RandomForestClassifier to the undersampled data.
  - Compare classification reports for original and undersampled data.
  - Notice increased recall for the minority class in the undersampled model.

- **Random Oversampling Method**:
  - Increase instances of the minority class using RandomOverSampler.
  - Fit another RandomForestClassifier to the oversampled data.
  - Compare classification reports for original, undersampled, and oversampled data.
  - Observe changes in identifying minority class instances.

- **Evaluating Sampling Techniques**:
  - Undersampling increased minority class detection but reduced majority class performance.
  - Oversampling maintained majority class results while improving minority class detection.
  - Choice of technique depends on specific project needs and the balance of class detection.

5.3.4: Synthetic Resampling  
- **Exploration of Synthetic Resampling Methods**:
  - Focuses on cluster centroids, SMOTE, and SMOTEENN.
  
- **Cluster Centroids Method**:
  - Uses clustering algorithms to group similar data points.
  - Synthetic sampling to create new points using KNN within clusters.
  - Aimed to balance majority and minority classes.
  - Involves generating points in majority class clusters and then undersampling.
  - `ClusterCentroids` model in `imblearn.under_sampling` is used for this process.
  - Implementation shows balanced class counts after resampling.

- **SMOTE (Synthetic Minority Oversampling Technique)**:
  - Generates synthetic data points for the minority class using KNN.
  - Oversamples minority class to match the majority class size.
  - Utilizes `SMOTE` from `imblearn.over_sampling`.
  - The resulting dataset shows an equal count for both classes.
  - Demonstrates an improvement in minority class results, though not always the best performance.

- **SMOTEENN (Combination of SMOTE and ENN)**:
  - Integrates SMOTE oversampling with undersampling by removing misclassified points using Edited Nearest Neighbors (ENN).
  - `SMOTEENN` from `imblearn.combine` was used for this approach.
  - Results in non-equal value counts due to ENN application.
  - Shows significant improvement in recall of “yes” class without major sacrifices in other metrics.
  - Highlights the variability of results depending on the dataset and resampling strategy.

- **Comparing Resampling Techniques**:
  - Cluster centroids dramatically improve recall but at the cost of precision and overall accuracy.
  - SMOTEENN provides a balanced improvement in recall without heavily impacting other metrics.
  - Emphasizes the need to try multiple methods for different datasets to find the most effective approach.

5.3.5: Balanced Models  
- **Exploring BalancedRandomForestClassifier**:
  - Alternative to RandomForestClassifier, found in `imblearn`.
  - Automatically performs random undersampling.
  - Useful for handling imbalanced datasets.

- **Data Preparation**:
  - Import necessary modules, including pandas, StandardScaler, train_test_split, and classification_report.
  - Load and preprocess data from a CSV file.
  - Encode categorical variables and split data into training and testing sets.
  - Scale data using StandardScaler.

- **Implementing BalancedRandomForestClassifier**:
  - Import `BalancedRandomForestClassifier` from `imblearn.ensemble`.
  - Instantiate and fit the classifier to the scaled training data.
  - Predict labels on the scaled test dataset.
  - Classifier integrates random undersampling into its algorithm.

- **Model Performance Analysis**:
  - Print classification report after model prediction.
  - The classifier shows comparable performance metrics to separate random sampling and model training steps.
  - Example performance metrics include precision, recall, and f1-score.

- **Overall Implications for Model Training**:
  - Emphasizes the importance of proper preprocessing and metric use in training models on imbalanced data.
  - Highlights the need for a thorough evaluation to ensure models perform well in practice beyond theoretical accuracy.

5.3.6: Activity: Improving Bank Marketing Campaigns with Synthetic Sampling  
- **Activity Overview**:
  - Fitting various models to small-business loan data from the US Small Business Administration (SBA).
  - Dataset includes loan-related information, like amount, term, and whether the loan defaulted.
  - The dataset is imbalanced, with fewer instances of loan defaults.

- **Dataset Details**:
  - Columns: Year, Month, Amount, Term, Zip, CreateJob, NoEmp, RealEstate, RevLineCr, UrbanRural, Default.
  - Default column indicates loan defaults (1 for default, 0 for no default).

- **Objective**:
  - Predict which SBA loans are most likely to default using different models.

- **Instructions for the Activity**:
  1. Read the CSV file into a Pandas DataFrame.
  2. Create a Series (y) for the Default column and a DataFrame (X) for the remaining columns.
  3. Split data into training and testing sets; apply StandardScaler to X data.
  4. Check imbalance in labels using value_counts.
  5. Fit two random forest models: a regular RandomForestClassifier and a BalancedRandomForestClassifier.
  6. Implement an additional resampling method (like RandomOverSampler, undersampling, or a synthetic technique).
  7. Print each model's confusion matrices, accuracy scores, and classification reports.
  8. Evaluate the models' effectiveness in predicting loan defaults.

- **Evaluation and Comparison**:
  - Compare your approach with the provided solution in the Solved folder.
  - Reflect on any differences in methods or results.

5.3.7: Recap and Knowledge Check  
- **Resampling as a Preprocessing Step**:
  - Resampling improves model results.
  - Other preprocessing techniques include scaling, PCA, etc., enhancing model inference capabilities.

- **Beyond Preprocessing: Additional Tuning Options**:
  - Post-preprocessing, further tuning through hyperparameters is possible.
  - The following lesson focuses on hyperparameter tuning.

- **Key Topics Covered in This Lesson**:
  - Oversampling and undersampling to address imbalanced data.
  - Understanding and handling imbalanced data.
  - Utilizing random sampling methods.
  - Synthetic resampling techniques.
  - Cluster centroids for data balancing.
  - SMOTE (Synthetic Minority Oversampling Technique).
  - SMOTEENN, combining SMOTE with Edited Nearest Neighbors.
  - Using balanced models to improve predictions on imbalanced datasets.

### Tuning
5.4.1: Eyes on the Prize  
- **Challenge of Choosing a Model in Predictive Analytics**:
  - Selecting the best model can be overwhelming.
  - There are more efficient models than academic knowledge alone.
  - Machine learning's uniqueness lies beyond just theoretical model understanding.

- **Dynamic Nature of Machine Learning**:
  - The field is constantly evolving with new techniques and models.
  - Abundance of models available for various projects.
  - Awareness of numerous models grows with experience in the field.

- **Learning About Models**:
  - Understanding each model's unique characteristics is beneficial.
  - Deep knowledge of every model is optional for beginners.
  - Practical experience over time leads to better understanding.

- **Approach for Initial Projects**:
  - For early projects, extensive knowledge of each model is optional.
  - Focus on defining and using appropriate metrics as a guide in the project.

5.4.2: Hyperparameter Tuning  
- **Choosing a Machine Learning Model**:
  - Selecting a machine learning algorithm is analogous to choosing a specific type of car for a task.
  - Machine learning models, like cars, have various types and custom options (hyperparameters).

- **Hyperparameters in Machine Learning**:
  - Hyperparameters allow customization of algorithms to specific datasets.
  - They differ from internal parameters as the user specifies them externally.
  - Changes to hyperparameters can range from minor tweaks to major adjustments.

- **Finding Optimal Hyperparameters**:
  - Utilize grid search or random search strategies to determine optimal hyperparameter values.
  - Grid search tests every combination of a list of hyperparameter values.
  - Random search tests a random sample of combinations from a specified range.

- **Practical Implementation of Grid Search**:
  - Example with `GridSearchCV` in Scikit-learn, using a base model and parameter dictionary.
  - Fits a model for each combination of parameters and averages the scores.
  - Best parameters and scores can be listed using the `best_params_` and `best_score_` attributes.

- **Random Search as an Efficient Alternative**:
  - Suitable for large parameter ranges where grid search becomes impractical.
  - Uses `RandomizedSearchCV` in Scikit-learn to sample a subset of parameter combinations.
  - Can still predict and score like a regular model despite testing fewer combinations.

- **Using RandomizedSearchCV**:
  - Create a larger parameter grid for C and gamma.
  - Randomly selects combinations to test, reducing the number of model runs.
  - Identifies best parameters and scores, demonstrating efficiency in handling larger parameter spaces.

5.4.3: Activity: Hyperparameter Tuning  
- **Activity Overview**:
  - Utilizing `GridSearchCV()` and `RandomizedSearchCV()` to determine parameters for a KNN model.
  - Focused on numeric columns of the bank dataset.

- **Instructions**:
  1. Train an untuned KNN model using the provided starter code; print the classification report.
  2. Create a parameters grid for n_neighbors, weights, and leaf_size. Suggested values are provided in the notebook comments.
  3. Train a `GridSearchCV()` model with the KNN model.
  4. Print the best parameters and classification report for the tuned model.
  5. Create a random parameters grid for n_neighbors, weights, and leaf_size.
  6. Train a `RandomizedSearchCV()` model with the KNN model.
  7. Print the best parameters and classification report for the tuned model.
  8. Document interpretations of the results, focusing on the best settings for each hyperparameter and the improvements in model accuracy.

- **Post-Activity Evaluation**:
  - Compare your completed work with the solution in the Solved folder.
  - Assess any differences in approach and results.

5.4.4: How Much is Enough?  
- **Decision-Making in Machine Learning**:
  - Recognizing the challenge of deciding when a machine learning model is sufficiently optimized.
  - Continual tweaking is possible, but knowing when to stop is crucial.

- **Project Constraints and Requirements**:
  - Understanding project constraints (time, budget) before model development is essential.
  - Determine necessary performance metrics and required scores for the model's usefulness.
  - Assess the value impact of minor improvements (e.g., 1% or 2% better performance).

- **Law of Diminishing Returns**:
  - Acknowledge the law of diminishing returns in model optimization.
  - Deciding the balance point for each project is key.

- **Approach Variances in Machine Learning**:
  - Traditional machine-learning models benefit more from extensive preprocessing and less from extensive tuning.
  - For neural networks, the focus may shift towards spending more time on tuning rather than preprocessing.

5.4.5: Realities and Limitations  
- **Limitations of Machine Learning Models**:
  - Each model has a limit to the complexity it can understand.
  - Complex relationships in data can surpass a model's capability.
  - Optimizing a model becomes difficult or ineffective beyond its complexity limit.

- **Practical Demonstration**:
  1. Load required libraries (Pandas, SVC from sklearn, etc.).
  2. Import and plot data from a spiral dataset.
  3. Split data into training and testing sets.
  4. Train an SVC model and predict on the test set.
  5. Generate and review the classification report.
  6. Plot predictions made by the model.

- **Observations from the Demonstration**:
  - The SVC model fails to learn the spiral shape of the dataset accurately.
  - Even though the SVC model struggles, other models like RandomForest can classify such data more effectively.
  - This example highlights the challenge of increasingly complex data in machine learning.

- **Future Learning: Neural Networks**:
  - The upcoming module will introduce neural networks.
  - Neural networks are some of the most potent algorithms currently used in machine learning, especially for complex data.

### Summary: Machine Learning Optimization
5.5.1: Summary: Machine Learning Optimization  
5.5.2: Reflect on Your Learning  
5.5.3: References  
