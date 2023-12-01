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
  - Weighted F1 scores can be more informative in certain contexts.
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
  - Upcoming lesson to explore imbalanced data in more detail.

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
  - Accuracy: How many correct results a model returns.
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
5.3.2: Oversampling and Undersampling  
5.3.3: Applying Random Sampling Techniques  
5.3.4: Synthetic Resampling  
5.3.5: Balanced Models  
5.3.6: Activity: Improving Bank Marketing Campaigns with Synthetic Sampling  
5.3.7: Recap and Knowledge Check  

### Tuning
5.4.1: Eyes on the Prize  
5.4.2: Hyperparameter Tuning  
5.4.3: Activity: Hyperparameter Tuning  
5.4.4: How Much is Enough?  
5.4.5: Realities and Limitations  

### Summary: Machine Learning Optimization
5.5.1: Summary: Machine Learning Optimization  
5.5.2: Reflect on Your Learning  
5.5.3: References  
