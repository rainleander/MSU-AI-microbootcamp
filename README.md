# MSU AI MicroBootCamp Notes and Projects
## Module 0: Getting Started
### Getting Started
0.1.1: Welcome  
0.1.2: Course Tools  
0.1.3: Google Colab  
0.1.4: Local Installations

### Navigating the Course
0.2.1: Course Overview  
0.2.2: Course Structure  
0.2.3: Submitting Assignments  
0.2.4: Grade Notifications  
0.2.5: Your Support Team  

## Module 1: Introduction to AI
### Introduction to Artificial Intelligence
1.1.1: Welcome to the AI Micro Boot Camp!  
1.1.2: What is AI?  
1.1.3: Narrow AI vs Artificial General Intelligence  
1.1.4: Ethics and AI  
1.1.5: Recap and Knowledge Check  

### The Impact of Machine Learning
1.2.1: Finance  
1.2.2: Business  
1.2.3: Medicine  
1.2.4: Daily Life  
1.2.5: Recap and Knowledge Check  

### Machine Learning Models and Methods
1.3.1: Overview of Machine Learning Models  
1.3.2: Unsupervised Learning  
1.3.3: Supervised Learning  
- Supervised learning involves "supervising" the algorithm's learning by providing data with known outcomes to make accurate predictions.
- The training cycle involves:
  - Giving the algorithm categories.
  - Feeding more data for better results.
  - Assessing and optimizing the model's performance.
- Supervised learning uses inputs of labeled data with features to predict outcomes on new unlabeled data.
- An example includes using a dataset of high-risk vs. low-risk loans to improve a model's prediction capability.
- A well-trained supervised learning model learns from its own errors, refining its predictions on new data.
- Supervised learning is categorized into regression and classification algorithms.
- Regression algorithms predict continuous variables, like predicting a person's weight based on height, age, and exercise or predicting prices in finance.
- Classification algorithms predict discrete outcomes, like predicting voting behavior based on traits or predicting buy vs. sell in finance.
- Despite its capabilities, supervised learning has limitations, especially when dealing with complex problems.
- Current AI research aims to develop even more sophisticated algorithms, building on existing supervised and unsupervised learning techniques.  

1.3.4: Machine Learning Optimization  
- Machine learning optimization enhances the performance of a model by adjusting its parameters and hyperparameters.
- The model is run on a training dataset, evaluated on a validation dataset, and adjustments are made to enhance its performance metrics.
- Continuous evaluation of machine learning models is crucial to minimize errors.
- Optimization refines and boosts the accuracy of machine learning models over time.  
  
**Metrics:**
- Models need to be assessed for performance, not just trained.
- **Accuracy** gives the ratio of correct predictions to total outcomes, indicating how often the model was correct.
- **Precision or PPV** reflects the model's confidence in its positive predictions.
- **Recall or Sensitivity** checks if the model identifies all positive instances (e.g., all fraudulent accounts).
- **F1 score** is a combined statistic of precision and recall.  
  
**Imbalanced Classes:**
- A common issue in classification is when one class size significantly surpasses the other.
- An example is detecting fraudulent transactions in credit card operations, where non-fraudulent transactions typically outnumber fraud.
- **Resampling** balances the class input during training to prevent bias towards the larger class.
  - **Oversampling:** Increasing instances of the smaller class.
  - **Undersampling:** Reducing instances of the larger class.  
  
**Model Tuning:**
- Crucial for machine learning optimization.
- Involves adjusting hyperparameters to find optimal values for the best model performance.
- Key components include hyperparameter tuning, kernel selection, and grid search.

1.3.5: Neural Networks and Deep Learning  
**Neural Networks:**

- Neural or artificial neural networks (ANN) are algorithms inspired by the human brain's structure and function.
- ANNs consist of artificial neurons (or nodes) that mimic biological neurons and are interconnected, mirroring brain synapses.
- Basic structure: layers of neurons that perform individual computations, with the results weighed and passed through layers until a final result is reached.
- Neural networks depend on training data to develop their algorithms, refining their accuracy as more data is inputted.
- Once trained, they quickly perform tasks on vast data sets, like classification and clustering.
- Neural networks can discern intricate data patterns, like predicting shopping behaviors or loan default probabilities.
- Benefits: Efficient at detecting complex data relationships and can handle messy data by learning to overlook noise.
- Challenges:
  - **Black box problem:** The complexity of neural network algorithms often makes them hard for humans to comprehend.
  - **Overfitting:** The model may perform too well on training data, impairing its generalization to new data.
- Specific model designs and optimization techniques can be applied to address these issues.  

**Deep Learning:**

- A specialized neural network with three or more layers, making it more efficient and accurate.
- Unlike most machine learning models, deep learning models can detect nonlinear relationships, excelling at analyzing intricate or unstructured data (e.g., images, text, voice).
- Neural networks weigh and transform input data into a quantified output. This data transformation process continues across layers until the final prediction.
- The distinction between regular neural networks and deep learning is typically based on the number of hidden layers. In this context, "deep" refers to networks with more than one hidden layer.
- Each additional neuron layer allows the modeling of intricate relationships and ideas, such as categorizing images.
- A practical example: A neural network classifying a picture containing a cat may first identify any animal, then specific features like paws or ears, breaking down the challenge until the image's individual pixels are analyzed.
- One prominent application for neural networks is natural language processing.  

1.3.6: Natural Language Processing (NLP) and Transformers  
- **Binary Data Representation:** Computers store and understand data in zeros and ones, termed binary code. This method represents various types of content, including text, sound, images, and video. To humans, binary code is typically indecipherable.
  
- **Human-Machine Communication:** Humans and machines have distinct ways of understanding data, necessitating the creation of methods for both to communicate effectively using a mutual "language."

- **Role of AI and ML:** AI technologies, combined with machine learning algorithms, allow computers to interpret and respond to written and spoken language in a human-like manner.

- **Natural Language Processing (NLP):**
  - Combines human linguistics rules with machine learning, particularly deep learning models.
  - Aims to not only translate but comprehend the essence behind words, recognizing intention, sentiment, ambiguities, emotions, and parts of speech.
  - Can convert spoken language into textual data.

- **Large Language Models:**
  - NVIDIA defines them as deep learning algorithms capable of recognizing, summarizing, translating, predicting, and generating text. They leverage insights from extensive datasets.
  
- **Transformer Models:**
  - Have been touched upon but will be elaborated on later in the course.
  - Defined as a neural network that discerns context and meaning by tracking relations in sequential data (e.g., words in a sentence).
  - Involves inputting text/spoken words into the algorithm, which then undertakes tokenization (breaking down into individual words/phrases). The algorithm subsequently classifies, labels, and uses statistical training to interpret the probable meaning of the data.
  
- **Growing Popularity of NLP:**
  - The rise in the usage of pre-trained models contributes to their growing popularity, as they minimize computational expenses and facilitate the implementation of advanced models.
  - Common applications include differentiating between spam and genuine emails, language translation, social media sentiment analysis, and powering chatbots/virtual agents.  

1.3.7: Emerging Technologies  
- **AI's Impact:** Artificial intelligence has significantly altered our lives and is evolving at a pace that's challenging to predict for the upcoming decades.

- **Generative AI:**
  - A rapidly progressing field within AI.
  - Beyond text generation, transformer technology in Generative AI can produce images (e.g., Stable Diffusion) and music.
  - Models are trained on data like image and audio files and then can create new content based on this data. With more data and time, these models increase in accuracy and efficiency.

- **Natural Human-Computer Interaction:**
  - AI is advancing towards enabling computers to engage more organically with humans in the real world.
  - Emergent technologies enable computers to visually perceive the world using advanced cameras and detect tactile information through sensors.
  - This facilitates more innovative interactions between humans and computers.
  - Early applications include autonomous vehicles, robots, and similar devices, with rapid ongoing development in this area.

- **Ethical and Regulatory Implications:**
  - The swift progress of AI technologies presents unforeseen ethical issues and challenges for regulatory frameworks.
  - These challenges were hard to anticipate even a short while ago, emphasizing the importance of ethical considerations in AI development.

- **Course Perspective:** Encouragement for learners to consistently ponder the potential and challenges AI brings to individual lives and broader society throughout the course.  

1.3.8: Recap and Knowledge Check  
1.3.9: References  

## Module 2: Unsupervised Learning
### Introduction to Unsupervised Learning
2.1.1: What is Unsupervised Learning?  
2.1.2: Recap  
2.1.3: Getting Started  
2.1.4: Clustering  
2.1.5: Recap and Knowledge Check  
2.1.6: Segmenting Data: The K-Means Algorithm  
2.1.7: Using the K-Means Algorithm for Customer Segmentation  
2.1.8: Activity: Spending Beyond Your K-Means  
2.1.9: Recap and Knowledge Check  

### Optimizing Unsupervised Learning
2.2.1: Finding Optimal Clusters: The Elbow Method  
2.2.2: Apply the Elbow Method  
2.2.3: Activity: Finding the Best k  
2.2.4: Recap and Knowledge Check  
2.2.5: Scaling and Transforming for Optimization  
2.2.6: Apply Standard Scaling  
2.2.7: Activity: Standardizing Stock Data  
2.2.8: Recap and Knowledge Check  

### Principal Component Analysis
2.3.1: Introduction to Principal Component Analysis  
2.3.2: Recap and Knowledge Check  
2.3.3: Activity: Energize Your Stock Clustering  
2.3.4: Recap and Knowledge Check  

### Summary: Unsupervised Learning
2.4.1: Summary: Unsupervised Learning  
2.4.2: Reflect on Your Learning  
2.4.3: References  

## Module 3: Supervised Learning — Linear Regression
### Supervised Learning Overview
3.1.1: Introduction to Supervised Learning  
3.1.2: Supervised vs. Unsupervised Learning  
3.1.3: Getting Started  

### Supervised Learning Key Concepts
3.2.1: Features and Labels  
3.2.2: Regression vs. Classification  
3.2.3: Model-Fit-Predict  
3.2.4: Model Evaluation  
3.2.5: Training and Testing Data  
3.2.6: Recap and Knowledge Check  

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

## Module 4: Supervised Learning — Classification
### Classification Overview
4.1.1: Classification Overview  
4.1.2: Getting Started  

### Classification and Logistic Regression
4.2.1: Overview  
4.2.2: Understand the Categorical Data Before Applying Logistic Regression  
4.2.3: Preprocessing  
4.2.4: Training and Validation  
4.2.5: Prediction  
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

## Project 1: Developing a Spam Detection Model

## Module 5: Machine Learning Optimization
### Introduction to Machine Learning Optimization
5.1.1: Introduction to Machine Learning Optimization  
5.1.2: Getting Started  

### Evaluating Model Performance
5.2.1: What is a good model?  
5.2.2: Overfitting and Underfitting  
5.2.3: Confusion Matrix  
5.2.4: Accuracy  
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

## Module 6: Neural Networks and Deep Learning
### Introduction to Neural Networks and Deep Learning
6.1.1: Introduction to Neural Networks and Deep Learning  
6.1.2: Your Neural Network and Deep Learning Background  
6.1.3: Getting Started  

### Artificial Neural Networks
6.2.1: What is a Neural Network?  
6.2.2: Making an Artificial Brain  
6.2.3: The Structure of a Neural Network  
6.2.4: Recap and Knowledge Check  

### Make Predictions with a Neural Network Model
6.3.1: Create a Neural Network  
6.3.2: Creating a Neural Network Model Using Keras  
6.3.3: Compile a Neural Network  
6.3.4: Train a Neural Network  
6.3.5: Make Predictions with a Neural Network  
6.3.6: Activity: Predict Credit Card Defaults  
6.3.7: Recap and Knowledge Check  

### Deep Learning
6.4.1: What is Deep Learning?  
6.4.2: Predict Wine Quality with Deep Learning  
6.4.3: Create a Deep Learning Model  
6.4.4: Train a Deep Learning Model  
6.4.5: Test and Evaluate a Deep Learning Model  
6.4.6: Save and Load a Deep Learning Model  
6.4.7: Activity: Detecting Myopia  
6.4.8: Recap and Knowledge Check  

### Summary: Neural Networks and Deep Learning
6.5.1: Summary: Neural Networks and Deep Learning  
6.5.2: Reflect on Your Learning  
6.5.3: References  

## Project 2: Predict Student Loan Repayment with Deep Learning

## Module 7: Natural Language Processing
### Introduction Natural Language Processing
7.1.1: Introduction to NLP  
7.1.2: Getting Started  
7.1.3: What is Text?  
7.1.4: Bag-of-Words Model  
7.1.5: What is a Language Model?  

### Tokenizers
7.2.1: Introduction to Tokenizers  
7.2.2: Tokenizer Example: Index Encoder  
7.2.3: Introduction to Hugging Face Tokenizers  
7.2.4: Similarity Measures  
7.2.5: Tokenizer Case Study: AI Search Engine  
7.2.6: Recap and Knowledge Check  

### Transformers
7.3.1: Introduction to Transformers  
7.3.2: Pre-trained models  
7.3.3: Language Translation  
7.3.4: Hugging Face Pipelines  
7.3.5: Text Generation  
7.3.6: Question and Answering  
7.3.7: Text Summarization  
7.3.8: Recap and Knowledge Check  

### AI Applications
7.4.1: AI Applications with Gradio  
7.4.2: Gradio Interfaces  
7.4.3: Gradio App: Text Summarization  
7.4.4: Other Gradio Components  
7.4.5: Activity: Question and Answering Textbox  
7.4.6: Introduction to Hugging Face Spaces  
7.4.7: Recap and Knowledge Check  
7.4.8: References  

## Module 8: Emerging Topics in AI
### Introduction to Emerging Topics in AI
8.1.1: Introduction  
8.1.2: Getting Started  

### AI the Creator
8.2.1: Introduction  
8.2.2: Interactive Text Generation  
8.2.3: Image Generation  
8.2.4: Music Generation  
8.2.5: Problems and Possibilities  
8.2.6: References  

### AI Outside the Computer
8.3.1: Introduction  
8.3.2: Autonomous Vehicles  
8.3.3: Robots  
8.3.4: The Internet of Things  
8.3.5: Mobile Deployment  
8.3.6: Problems and Possibilities  
8.3.7: References  

### Additional Areas of Active Research
8.4.1: Introduction  
8.4.2: One-Shot Learning  
8.4.3: Creating 3D Environments  
8.4.4: Algorithm Speed and Computational Resource Management  
8.4.5: Ethics and Regulations  
8.4.6: Problems and Possibilities  
8.4.7: References  

### Course Wrap-Up
8.5.1: Congratulations!  
