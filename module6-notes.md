## Module 6: Neural Networks and Deep Learning
### Introduction to Neural Networks and Deep Learning
6.1.1: Introduction to Neural Networks and Deep Learning  
- **Introduction to Neural Networks**:
  - Artificial Neural Networks (ANNs) are a key innovation in AI, commonly used in applications like YouTube recommendations.
  - For this module, ANNs will be referred to as neural networks.

- **Applications of Neural Networks**:
  - Used by companies like Google and Tesla to develop self-driving cars.
  - Increasingly applied in various fields due to their ability to handle complex data.

- **Advantages Over Traditional Machine Learning Models**:
  - Traditional models need to be more comprehensive in understanding complex relationships in data, suitable for linearly separable data.
  - Neural networks excel in processing datasets that are too complex for other machine learning algorithms.
  - Designed based on human brain neurons, capable of weighting input information to inform other system parts.

- **Neural Networks in Industry**:
  - In finance: Used for fraud detection, risk management, and algorithmic trading.
  - In aerospace engineering: Crucial for flight simulations and auto-pilot systems.
  - In healthcare: Applied for cancer cell and electrocardiography analysis.

- **Learning Objectives for the Module**:
  - Understand what a neural network is.
  - Compare neural networks with other types of machine learning models.
  - Distinguish between neural network models and deep neural network models.
  - Learn to preprocess data for neural networks.
  - Implement neural network models using TensorFlow and Keras.
  - Save trained TensorFlow neural network models.
  - Implement deep neural network models using TensorFlow and Keras.

- **Preparation for Upcoming Modules**:
  - Completing this module is a foundation for success in future modules on natural language processing and emerging AI topics.

6.1.2: Your Neural Network and Deep Learning Background  
- **Reflecting on Previous Exposure to Neural Networks and Deep Learning**:
  - Consider personal experience with technology featuring neural network or deep learning capabilities in everyday life.
  - Reflect on whether neural networks or deep learning algorithms influence workplace decisions.
  - Identify what is perceived as the most disruptive application of deep learning or neural networks.
  - Assess current opinions and interest levels in neural networks and deep learning.

6.1.3: Getting Started  
- **File Downloads for the Module**:
  - Download "Module 6 Demo and Activity Files" before starting.

- **Installation Requirements**:
  - Install TensorFlow 2.0 and Keras to implement neural networks.
  - Follow tutorials in subsequent sections for installation guidance.

- **Important Note for Apple M1 Chip Users**:
  - Do not install TensorFlow and Keras directly if using an Apple computer with an M1 Chip.
  - Refer to the "Apple M1 Chip Users" section for alternative instructions.

- **Installing TensorFlow**:
  - TensorFlow 2.0 dependencies should be present in the default Conda environment.
  - Activate the Conda environment and install TensorFlow using `pip install --upgrade tensorflow`.
  - Verify TensorFlow installation by checking the version (`python -c "import tensorflow as tf;print(tf.__version__)"`).

- **Installing and Verifying Keras**:
  - Keras is included with TensorFlow 2.0.
  - Verify Keras installation (`python -c "import tensorflow as tf;print(tf.keras.__version__)"`).

- **Troubleshooting Installation Issues**:
  - If you have any problems, please refer to the [TensorFlow Install Guide](https://www.tensorflow.org/install/pip).
  - Google Colab is an alternative platform that supports TensorFlow and can run Jupyter Notebook files.

- **Using Google Colab for Apple M1 Chip Users**:
  - Use Google Colab to run module activities.
  - Upload activity notebook files to Google Colab and run the code as usual.

### Artificial Neural Networks
6.2.1: What is a Neural Network?  
- **Understanding Artificial Neural Networks (ANNs)**:
  - Inspired by neurons in the human brain.
  - Consists of layers of artificial neurons performing computations and communicating results.

- **Advantages of Neural Networks**:
  - Excel in processing large, complex datasets.
  - Capable of detecting complex relationships in data.
  - It is better to handle large, noisy datasets by learning to ignore noise.

- **Disadvantages of Neural Networks**:
  - Complexity results in a 'black box' problem, opaque the process from input to output.
  - Prone to overfitting, which limits their ability to generalize data trends beyond training data.

- **Examples of Neural Network Applications**:
  - Voice-activated assistants (Siri, Cortana, Google Assistant) for speech recognition.
  - OpenAI's ChatGPT for conversational response generation in NLP.
  - Self-driving cars and facial recognition using computer vision systems.
  - Recommendation engines (Netflix, YouTube) using deep neural networks for personalized content recommendations.

- **Google's Contributions to Neural Networks**:
  - Google has developed tools to make neural networks more accessible.
  - TensorFlow, created by Google Brain, is an open-source machine learning platform.
  - Encouragement to explore neural networks through interactive applications and games on Experiments with Google.

- **Next Steps in Learning**:
  - Move towards a deeper understanding of how neural networks are built and function.

6.2.2: Making an Artificial Brain  
- **Perceptron Inspired by Neurons in the Brain**:
  - Neural networks consist of layers of neurons.
  - Each neuron in a neural network is akin to a perceptron, inspired by brain neurons.

- **Perceptron as a Computational Equivalent of Neurons**:
  - Perceptrons make classification decisions based on input, similar to brain neurons.
  - Acts as a binary classifier, categorizing input data into two parts (1 or 0).

- **Working of a Perceptron**:
  - Receives inputs (xn) as numeric values representing characteristics (e.g., 1 for presence, 0 for absence of a trait).
  - Each input is multiplied by a weight (wn), a process known as weighting.
  - Weighting is akin to assigning importance or value to each characteristic.
  - Weighted values are summed, including a bias value (w0).
  - An activation function determines the final output, setting a threshold for decision-making.
  - If the weighted sum exceeds the threshold, the output is 1 ("Yes"); otherwise, it's 0 ("No").

- **Role of Perceptron in Neural Networks**:
  - Perceptrons are foundational units in neural networks.
  - They integrate input signals, weights, bias, and an activation function to produce binary output.

6.2.3: The Structure of a Neural Network  
- **Composition of Neural Networks**:
  - Neural networks consist of three interconnected layers: input, hidden, and output.
  - Input layer: Receives and transforms input values through weighting.
  - Hidden layer(s): Can contain one or more perceptrons.
  - Output layer: Reports the outcome of the neural network.

- **Role of Activation Functions**:
  - Activation functions are critical in producing a clear output from complex inputs.
  - Applied at the end of each perceptron in the hidden and output layers (not in the input layer).
  - Transforms perceptron output into a quantitative value, used as input for the next perceptron.
  - In neural network design, various activation function combinations are tested for optimal performance.

- **Further Exploration of Activation Functions**:
  - Detailed study of activation functions will be covered later in the module.
  - Encouragement to explore formulas and mathematical fundamentals of activation functions from external resources like Wikipedia and ML Glossary.

6.2.4: Recap and Knowledge Check  
- **Introduction to Neural Networks**:
  - Artificial neural networks simulate the functioning of the human brain.
  - Capable of processing large, complex, and noisy datasets.

- **Perceptron: Basic Unit of Neural Networks**:
  - Perceptron acts as a binary classifier in a neural network.
  - Receives inputs, assigns weights, and calculates a weighted sum.
  - Applies an activation function to produce output.

- **Structure of Neural Networks**:
  - Composed of multiple perceptrons forming various layers.
  - A neural network includes three distinct layers: input, hidden, and output.

### Make Predictions with a Neural Network Model
6.3.1: Create a Neural Network  
- **Introduction to Creating Neural Networks with Keras**:
  - Learn to use the Keras library for building neural networks.

- **Approaches to Coding Neural Networks**:
  1. Code from scratch using Python, Pandas, and NumPy.
  2. Use an API or framework for efficiency, focusing on model improvement.

- **Using TensorFlow and Keras**:
  - TensorFlow: Open-source platform for efficient machine learning.
  - Keras: Abstraction layer over TensorFlow for easier model building.
  - Follow the standard model -> fit -> predict interface.

- **Prerequisites**:
  - TensorFlow installation is required for this module.

- **Demonstration of Neural Network Creation**:
  - Import necessary modules, including Pandas, TensorFlow, and Keras.
  - Use the `Sequential` model and the `Dense` class in Keras.

- **Steps to Create a Neural Network**:
  1. Create a dummy dataset using `make_blobs` from sklearn.
  2. Preprocess data: Transform target variable `y` into a vertical vector.
  3. Create a DataFrame for visualization and plot the dummy data.
  4. Split the data into training and testing sets using `train_test_split`.
  5. Normalize the data using `StandardScaler` from scikit-learn for better neural network performance.
  6. Scale the feature data but not the target data.

6.3.2: Creating a Neural Network Model Using Keras  
- **Creating a Neural Network with Keras**:
  1. **Create the Model Structure**:
     - Initialize a Sequential model and store it in a variable named `neuron`.

  2. **Input and Hidden Layers**:
     - Define the number of inputs and hidden nodes.
     - Add input and hidden layers using the `add` function and `Dense` module.
     - Parameters for the `Dense` module:
       - `input_dim`: Number of inputs the neuron receives, set to `number_inputs` (2).
       - `units`: Number of neurons in the hidden layer, set to `number_hidden_nodes` (1).
       - `activation`: Type of activation function, `relu` (rectified linear unit), used for non-linearity in the first layer.

  3. **Output Layer**:
     - Add output layer using the `Dense` module with two parameters: `units` and `activation`.
     - `units`: Number of output neurons (1 for binary classification).
     - `activation`: `sigmoid` function to transform the output to a probability range between 0 and 1.
     - The model classifies data points as Class 1 or 0 based on a default threshold of 0.5.

- **Understanding the Model**:
  - Use the `summary` function to check the model structure.
  - Summary includes:
    - The first row shows the input and hidden layer.
    - Second row detailing the output layer.
    - Total of five parameters in the model.

6.3.3: Compile a Neural Network  
- **Understanding Neural Network Compilation in Keras**:
  - Creating a neural network is akin to designing a house's blueprints, specifying layers and activation functions.
  - Compiling the neural network is like building the house, specifying loss, optimization, and activation functions.

- **Loss Functions, Activation Functions, and Evaluation Metrics**:
  - Loss function: Indicates performance change over iterations during model training.
  - Optimization function: Shapes the neural network during training, reducing losses for accurate output.
  - Evaluation metric (accuracy): Tracks model predictive accuracy, especially for binary classification models. A higher value (closer to 1) indicates better accuracy.

- **Compiling the Neural Network Model**:
  - Utilize the `compile` function in Keras.
  - For binary classification, use `binary_crossentropy` as the loss function.
  - Set the optimizer to `adam` to balance speed and quality.
  - Set the evaluation metric to `accuracy`, aiming for it to be as close to 1 as possible.

6.3.4: Train a Neural Network  
- **Training the Neural Network Model**:
  - The next step is to train the compiled neural network using the `fit` function.
  - Training requires x and y values and the number of epochs.
  - An epoch is a complete pass of the training dataset through the model.
  - The optimizer and loss functions adjust the weights during each epoch.
  - Example: `model = neuron.fit(X_train_scaled, y_train, epochs=100)`.

- **Observing Training Results**:
  - The output shows loss and accuracy results for each epoch.
  - The goal is to minimize loss and maximize accuracy.

- **Visualizing Model Performance**:
  - Create a DataFrame from the model's history dictionary, storing loss and accuracy.
  - Plot loss and accuracy using Pandas to visualize improvements over epochs.
  - Ideal trend: Accuracy increases and loss decreases with more epochs.

- **Determining Optimal Number of Epochs**:
  - Start with 20 epochs and adjust based on performance.
  - Aim for loss nearing zero and accuracy approaching 1.
  - Increase epochs in increments until the desired performance is achieved.

- **Evaluating Model Performance with Test Data**:
  - Assess the model’s performance on test data to ensure reliability.
  - Use the `evaluate` function in TensorFlow for testing.
  - Example: Evaluate using `model_loss, model_accuracy = neuron.evaluate(X_test_scaled, y_test, verbose=2)`.
  - Evaluate results by loss and accuracy on the test data.

- **Next Steps**: 
  - Learn to use the trained neural network to predict new, unseen data.

6.3.5: Make Predictions with a Neural Network  
- **Reaching the Prediction Stage**:
  - Utilize the trained neural network model for predicting binary classifications on new datasets.
  - Employ the `predict` function in the neural network for generating predictions.

- **Predict Function Requirements**:
  - Supply the function with new data and a threshold value.
  - Classifications are based on the threshold: below the threshold is classified as 0, and above as 1.

- **Demonstration with Dummy Data**:
  - Create a dummy dataset for prediction demonstration.
  - Use a threshold of 0.5 in the `predict` function.
  - Example: `predictions = (neuron.predict(new_X) > 0.5).astype("int32")`.

- **Comparing Predictions to Actual Classifications**:
  - Create a DataFrame to compare model predictions with actual data point classifications.
  - Visualize the comparison in a table format.

- **Outcome of the Prediction Process**:
  - The model successfully classified all 10 data points in the dummy dataset.
  - Demonstrates the ability to create, train, and utilize a neural network for making accurate predictions.

- **Next Steps**:
  - Apply the learned skills in a practical activity involving neural network creation, training, and prediction.

6.3.6: Activity: Predict Credit Card Defaults  
- **Activity Overview: Building a Neural Network for Credit Default Prediction**:
  - Use Keras to build a neural network model predicting credit card debt default.
  - Apply knowledge from previous demonstrations to a dataset with 22 features and one target.

- **Background of the Task**:
  - Work on a project for a major credit card company.
  - Objective: Predict which customers will default on credit card debt.
  - Dataset: 30,000 records with 22 feature columns and one binary target column ("DEFAULT").

- **Features and Target in the Dataset**:
  - Features: Include demographic info, credit limit, past payment details, etc.
  - Target ("DEFAULT"): 1 for defaulted card, 0 for non-defaulted.

- **Instructions for the Activity**:
  1. Read the dataset into a Pandas DataFrame.
  2. Define features set `X` (excluding the "DEFAULT" column).
  3. Create target `y` from the "DEFAULT" column.
  4. Split data into training and testing sets using `train_test_split`.
  5. Scale features using `StandardScaler`.
  6. Create a neural network model with 22 inputs, one hidden layer (12 neurons), and an output layer (ReLU for hidden, sigmoid for output).
  7. Use the `summary` function to display the model structure.
  8. Compile the model with `binary_crossentropy`, `adam` optimizer, and `accuracy` metric.
  9. Fit the model with training data for 100 epochs.
  10. Plot loss function and accuracy over epochs.
  11. Evaluate the model with test data.
  12. Predict `y` values using the model and scaled X test data.
  13. Create and display a DataFrame comparing predicted and actual `y` values.
  14. Analyze model performance on test data and consider improvement strategies.

- **Post-Activity Evaluation**:
  - Review and compare the completed work with the solution in the Solved folder.
  - Reflect on the differences in approach and any challenges faced.

6.3.7: Recap and Knowledge Check  
- **Overview of Neural Network Process**:
  - Creating, compiling, and training a neural network is essential before using it for predictions.
  - Building a neural network is akin to designing a house's blueprint.

- **Key Components in Neural Network Training**:
  - Creation: Establishing the structure and layers of the neural network.
  - Compilation: Preparing the model for training.
  - Training: Fitting the model to the data.

- **Importance of Loss Function and Accuracy Metrics**:
  - Essential for evaluating the performance of classification models.
  - Indicators of how well the model is learning and predicting.

- **Role of Optimization Function**:
  - Shapes the neural network by adjusting input weights during each training epoch.

- **Understanding Epochs in Training**:
  - An epoch represents a complete pass of the entire training dataset through the model.
  - The number of epochs directly impacts model performance.
  - Training typically begins with 20 epochs, with adjustments based on loss and accuracy.

### Deep Learning
6.4.1: What is Deep Learning?  
- **Deep Neural Networks and Deep Learning**:
  - Deep neural networks are a subset of artificial neural networks characterized by multiple hidden layers.
  - More hidden layers allow for modeling complex relationships and concepts.

- **Understanding the Function of Layers in Deep Neural Networks**:
  - Each layer in a neural network calculates weights of input data and passes it to the next layer.
  - In deep learning, added hidden layers enhance the model's ability to interpret data.
  - Examples: Image recognition, speech recognition, and natural language processing.

- **Processing Complex Data in Deep Learning**:
  - Example: Identifying cats in images.
    - Input layer receives pixel data.
    - Each hidden layer progressively interprets more complex relationships, like color differences and shapes.
  - Deep learning models don't think like humans, but each layer understands more complex relationships.

- **Exploring Deep Learning Models with a Practical Example**:
  - Use deep neural networks to predict wine quality in a demonstration.

- **Demystifying the Black Box in Deep Learning**:
  - Deep learning models are complex and less interpretable (Black Box issue).
  - TensorFlow playground provides visualization to understand the effect of hidden layers on model performance and loss.

- **Skill Drill with TensorFlow Playground**:
  - Experiment with different settings in the TensorFlow playground.
  - Observe changes in loss metric with varying numbers of hidden layers and neurons.
  - Explore how different configurations impact model performance.

6.4.2: Predict Wine Quality with Deep Learning  
- **Using Deep Learning for Wine Quality Prediction**:
  - Tasked with predicting wine quality scores for a Spanish winery to assist in forecasting revenue and assessing expansion risks.

- **Building a Deep Learning Model for Wine Quality Prediction**:
  1. **Create a DataFrame**: 
     - Read wine quality data into a Pandas DataFrame.
     - The data includes 11 variables characterizing different aspects of wine.

  2. **Preprocessing the Data**:
     - Create features set (X) and target set (y).
     - Code example: `X = df.drop(columns=["quality"]).values` and `y = df["quality"].values`.
     - Wine quality is assessed on a scale from 1 to 10.

  3. **Creating Training and Testing Datasets**:
     - Use `train_test_split` to divide the data into training and testing sets.
     - Normalize the data using `StandardScaler`.

  4. **Note on Data Preparation for Neural Networks**:
     - Data for neural networks must be numerical and normalized to the same scale, regardless of the number of hidden layers.

  5. **Scaling the Features Data**:
     - Fit and transform the training data with `StandardScaler`.
     - Apply the same transformation to the testing data.

- **Next Steps**: 
  - Ready to create and train the deep learning model for wine quality prediction.

6.4.3: Create a Deep Learning Model  
- **Creating a Deep Learning Model with Keras**:
  - Similar process to simple neural networks, with additional hidden layers.
  - Optimize model performance after compiling.
  - Neuron count in each layer generally decreases or remains equal to the previous layer.

- **Designing the Layers of the Deep Learning Model**:
  1. **Input Layer**: 
     - Consists of 11 nodes for the 11 characteristics in the input data.

  2. **Hidden Layers**:
     - Two hidden layers with decreasing neuron counts.
     - The first hidden layer has 8 nodes, and the second has 4.
     - Activation function for both layers: ReLU (Rectified Linear Unit).

  3. **Output Layer**:
     - One neuron in the output layer for continuous output.
     - Suitable for regression problems using a linear activation function.

- **Coding the Deep Learning Model in Keras**:
  - Set up the number of inputs and hidden nodes.
  - Create a Sequential model (`nn`).
  - Add first and second hidden layers using `Dense`, specifying `units` and `activation`.
  - Add output layer with linear activation.
  - Important: Only define `units` and `activation` for the second hidden layer, not `input_dim`.

- **Next Steps**: 
  - Compile and fit the created deep learning model.

6.4.4: Train a Deep Learning Model  
- **Compiling and Fitting the Deep Neural Network Model**:
  - Use the `compile` and `fit` functions in Keras for the deep neural network.
  - Compile the model with `mean_squared_error` for the loss function and `adam` as the optimizer.
  - Set `mse` (mean squared error) as the evaluation metric.
  - Fit the model on scaled training data (`X_train_scaled`, `y_train`) for 100 epochs.

- **Understanding Key Components in the Training Process**:
  - **Epochs**: Represent one complete pass of the training dataset through the model.
  - **Loss Function**: Evaluates model performance after each epoch; `mean_squared_error` is used for regression models.
  - **Optimizer**: Adjusts the model's parameters to minimize the loss function.
  - **Evaluation Metric**: `mse` is used as it's suitable for regression models, unlike accuracy, which is used for classification models.

- **Aim of the Model**:
  - For regression models like the wine-quality prediction model, the goal is to achieve a mean squared error as close to zero as possible.

6.4.5: Test and Evaluate a Deep Learning Model  
- **Deciding on the Number of Hidden Layers in Deep Learning**:
  - More layers can increase performance but may only sometimes be necessary.
  - Sometimes, the initial layers sufficiently capture dataset complexity.

- **Demonstrating the Impact of Additional Hidden Layers**:
  - Comparison of a new model with four hidden layers against a previous model.

- **Creating a New Deep Learning Model with Four Hidden Layers**:
  - Step 1: Define a new model (`nn_2`) with increased hidden layers (22, 11, 8, and 6 nodes, respectively).
  - Step 2: Compile and fit the new model using `mean_squared_error` loss, `adam` optimizer, and `mse` metric for 100 epochs.

- **Evaluating and Comparing Model Performance**:
  - Use `evaluate` function with testing data to compare MSE values of both models.
  - Results indicate minimal performance gain with additional layers in Model 2.

- **Insight**:
  - Additional layers in Model 2 showed only slight improvement, suggesting Model 1's simplicity might be preferable.
  - Adding more layers isn’t always beneficial and can lead to overfitting.

- **Determining Optimal Model Depth**:
  - No set rule for the best number of layers; a trial and error approach is needed.
  - Train and evaluate models with increasing depth until no significant improvements are observed.

- **Making Predictions with the Chosen Model**:
  - Step 1: Predict wine quality using `nn.predict` on scaled test data.
  - Step 2: Compare predictions with actual values in a DataFrame.

- **Observations and Optimization Considerations**:
  - Several predicted scores are incorrect, indicating room for optimization.
  - Consider applying optimization techniques from previous modules to improve model accuracy.

6.4.6: Save and Load a Deep Learning Model  
- **Model Storage and Access for Complex Problems**:
  - For formal applications, training a model each time for data analysis could be more practical due to time and resource constraints.
  - Complex neural network training can be time-consuming and resource-intensive.
  - Data scientists save and access trained models outside the training environment for efficiency.

- **Sharing and Deploying Trained Models**:
  - Trained models are shared through scientific papers, software deployment, GitHub, and among colleagues.

- **Using Keras to Save and Load Models**:
  - Keras' Sequential model features a `save` function to export models in HDF5 format.
  - The `save` function exports model configurations, layer weights, activation functions, optimizers, losses, and metrics.
  - Keras `load_model` function allows importing trained models for analysis and predictions.

- **Demonstration with Wine Quality Model**:
  - Step 1: Save the trained model as an HDF5 file using `nn.save(file_path)`.
  - Step 2: Load the model using TensorFlow's `tf.keras.models.load_model` function.
  - Step 3: Test the performance of the imported model on test data to ensure it matches the original model's performance.

- **Validation of Imported Model**:
  - The imported model shows consistent performance metrics (loss and accuracy) with the original model.
  - This method allows the evaluation of imported Keras neural network models on any compatible dataset.

6.4.7: Activity: Detecting Myopia  
- **Objective**: Create a deep learning model to predict myopia diagnosis.

- **Dataset and Files**:
  - Access the dataset and files in the 'activities/02-Detecting_Myopia/Unsolved' subfolder.

- **Steps in the Activity**:
  1. **Data Preparation**:
     - Create features and target sets from the dataset, using the 'MYOPIC' column as the target.
     - Preprocess the input data using Scikit-learn's `StandardScaler`.
  2. **Deep Learning Model Design**:
     - First Dense layer: 14 inputs, 16 hidden nodes, ReLU activation function.
     - Second Dense layer: At least 16 neurons, ReLU activation function.
     - Output layer: One neuron, sigmoid activation function.
  3. **Compile and Train**:
     - Compile the model and train it for a maximum of 50 epochs.
  4. **Model Evaluation**:
     - Evaluate the model's performance by calculating test loss and accuracy.
  5. **Make Predictions**:
     - Use the model to predict myopia on test data.
  6. **Results Comparison**:
     - Create a dataframe to compare predictions with actual values.
     - Review the first 10 rows of this comparison dataframe.

- **Solution Review**:
  - Compare your completed work with the provided solution in the 'Solved' folder.
  - Assess if all steps were completed correctly and note any differences in approach.
  - Reflect on any challenges or confusion encountered during the activity.

6.4.8: Recap and Knowledge Check  
- **Deep Learning Networks vs. Neural Networks**:
  - **Multiple Hidden Layers**: Deep learning networks differ from neural networks because they have more than one hidden layer.
  - **Complex Data Handling**: They can manage complex, unstructured datasets like images, text, and human speech, thanks to additional hidden layers.
  - **Balance in Layers**: Adding more layers can increase performance, but there's a risk of overfitting. More layers sometimes yield significant performance improvements.
  - **Optimal Design Decisions**: Determining the optimal number of layers and neurons involves performance evaluation over a set number of epochs and adjusting layers/neurons in each trial.
  - **Regression Models**: The key performance metric for regression models with continuous output is a mean squared error (MSE).
  - **Performance Evaluation**: The goal is to minimize MSE and loss; the closer these values are to zero, the better the model performs.
  - **Model Saving and Sharing**: Keras facilitates the saving and uploading of trained models, allowing for efficient sharing and reuse among data scientists.

### Summary: Neural Networks and Deep Learning
6.5.1: Summary: Neural Networks and Deep Learning  
6.5.2: Reflect on Your Learning  
6.5.3: References  
