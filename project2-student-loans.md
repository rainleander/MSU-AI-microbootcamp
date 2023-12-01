## Background
You work at a company that specializes in student loan refinancing. If the company can predict whether a borrower will repay their loan, it can provide a more accurate interest rate for the borrower. Your team has asked you to create a model to predict student loan repayment.

The business team has given you a CSV file that contains information about previous student loan recipients. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a model that will predict the likelihood that an applicant will repay their student loans. The CSV file contains various information about these students, including their credit ranking.

### Files
[Project 2 Starter Code]()

## Instructions
The steps for this challenge are broken out into the following sections:  

- Prepare the data for use on a neural network model.  
- Compile and evaluate a model using a neural network.  
- Predict loan repayment success by using your neural network model.

### Prepare the Data for Use on a Neural Network Model
Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, preprocess the dataset so that you can use it to compile and evaluate the neural network model later.

Open the starter code file and complete the following data preparation steps:

1. Read the data from [https://static.bc-edx.com/mbc/ai/m6/datasets/student_loans.csv](https://static.bc-edx.com/mbc/ai/m6/datasets/student_loans.csv) into a Pandas DataFrame. Review the DataFrame, looking for columns that could eventually define your features and target variables.  
2. Create the features (`X`) and target (`y`) datasets. The “credit_ranking” column should define the target dataset. The remaining columns should define the features dataset.  
3. Split the features and target sets into training and testing datasets.  
4. Use scikit-learn's StandardScaler to scale the features data.

### Compile and Evaluate a Model Using a Neural Network
Use your knowledge of TensorFlow to design a deep neural network model. This model should use the dataset’s features to predict a student's credit quality based on the dataset's features. Consider the number of inputs before determining the number of layers your model will contain or the number of neurons on each layer. Then, compile and fit your model. Finally, evaluate the model to calculate its loss and accuracy.

To do so, complete the following steps:

1. Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.  

**HINT**: You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.

2. Compile and fit the model using the `mse` loss function, the `adam` optimizer, and the `mse` evaluation metric.  

**HINT**: When fitting the model, start with a few epochs, such as 50 or 100.  

3. Evaluate the model using the test data to determine the model’s loss and accuracy.
4. Save and export your model to an HDF5 file, and name the file `student_loans.h5`.

### Predict Loan Repayment Success by Using Your Neural Network Model
You can use the model you saved in the previous section to make predictions on your reserved testing data.

To do so, complete the following steps:  

1. Reload your saved model.  
2. Make predictions on the testing data.  
3. Create a DataFrame that includes predictions and the actual values.  
4. Display a sample of the DataFrame you created in Step 3. Compare this sample's prediction and actual values and describe what you notice.  

## Requirements
To receive all points, your Jupyter notebook file must have all of the following:  

### Prepare the Data for Use on a Neural Network Model (20 points)
- The `student_loans.csv` file was read into a Pandas DataFrame, and a dataset sample was shown. (5 points)
- Two datasets were created: a target (`y`) dataset, which includes the "credit_ranking" column, and a features (`X`) dataset, which includes the other columns. (5 points)
- The features and target sets have been split into training and testing datasets. (5 points)
- Scikit-learn's `StandardScaler` was used to scale the features data. (5 points)

### Compile and Evaluate a Model Using a Neural Network (40 points)
- A deep neural network was created with appropriate parameters. (10 points)
- The model was compiled and fit using the `mse` loss function, the `adam` optimizer, the `mse` evaluation metric, and a small number of epochs, such as 50 or 100. (10 points)
- The model was evaluated using the test data to determine its loss and accuracy. (10 points)
- The model was saved and exported to an HDF5 file named `student_loans.h5`. (10 points)

### Predict Loan Repayment Success by Using your Neural Network Model (20 points)
- The saved model was reloaded. (5 points)
- The reloaded model was used to make predictions on the testing data. (5 points)
- There is a DataFrame that includes both the predictions and the actual values. (5 points)
- A sample of the DataFrame created in Step 3 is displayed, and there is a comparison of the values shown in the sample. (5 points)

## Grading
This assignment will be evaluated against the requirements and assigned a grade according to the following table:

### Grade	Points
A (+/-)	90+
B (+/-)	80–89
C (+/-)	70–79
D (+/-)	60–69
F (+/-)	< 60

## Submission
Could you make sure to submit your work by the assignment due date? To do so, click Submit, then upload your project files. If you have any problems uploading your files, you may also provide a link to a folder within Google Drive, Dropbox, or a similar service. Could you set the sharing permissions so anyone with the link can view your files?
