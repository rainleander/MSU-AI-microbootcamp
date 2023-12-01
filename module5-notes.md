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
5.2.6: Classification Report  
5.2.7: The Importance of Metric and Target Selection  
5.2.8: Recap and Knowledge Check  

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
