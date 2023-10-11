## Module 2: Unsupervised Learning
[Module 2 Notes](module2-notes.md)
### Introduction to Unsupervised Learning
2.1.1: What is Unsupervised Learning?  
- **Comparison with Supervised Learning**:
  - Two primary methods of machine learning are unsupervised and supervised learning.
  - Unlike supervised learning that utilizes labeled data, unsupervised learning works with unlabeled data.

- **Process & Use**:
  - Unsupervised learning identifies trends, relationships, and patterns (clusters) within data.
  - Its applications:
    1. Cluster/group data to discern patterns rather than predicting a class.
    2. Transform data for intuitive analysis or supervised/deep learning use.

- **Practical Applications**:
  - Recommending related products based on customer views.
  - Targeting specific customer groups in marketing.
  - Use cases include customer segmentation, fraud detection, outlier identification, spam detection, healthcare diagnoses, and sentiment analysis.

- **Challenges**:
  - No definitive means to verify the correctness of the output.
  - Understanding the clusters is challenging, as the algorithm self-generates categories.

- **Interplay with Other Learning Methods**:
  - Unsupervised learning can be used alongside supervised learning, deep learning, and NLP.
  - Techniques like PCA (Principal Component Analysis) can optimize predictions or classifications by reducing input variables.
  - It can be a pre-training step before applying deep learning to labeled data.
  - In NLP, unsupervised learning can group texts/documents based on similarity.

- **Learning Outcomes**:
  - By the module's end, learners will be able to:
    1. Understand unsupervised learning's role in AI.
    2. Define clustering in the context of machine learning.
    3. Utilize the K-means algorithm for clustering.
    4. Determine the optimal cluster number using the elbow method.
    5. Convert categorical variables to numerical representations.
    6. Understand and apply PCA for dimensionality reduction.
    7. Utilize K-means post-PCA for improved data analysis.  

2.1.2: Recap  
- **Purpose**:
  - Unsupervised learning is employed to cluster unlabeled datasets.
  - It can also provide an alternative data representation for subsequent analysis and machine learning processes.
  - Algorithms in unsupervised learning use test data to form models identifying relationships among data points.

- **Challenges**:
  1. Due to the absence of labels in the input dataset, verifying the accuracy of the output data is uncertain.
  2. The algorithm self-generates data categories, requiring an expert to assess the relevance and significance of these categories.

- **Value**:
  - Despite its challenges, unsupervised learning proves beneficial in a diverse range of applications.  

2.1.3: Getting Started  
- **Files**:
  - Before starting the module, download the required files: [Module 2 Demo and Activity Files](https://github.com/rainleander/MSU-AI-microbootcamp/tree/main/module-2-files).

- **Installations**:
  - The primary tools for this module are JupyterLab and the scikit-learn Python library.
  - Scikit-learn is typically pre-installed with Anaconda.

- **Confirmation Steps for scikit-learn Installation**:
  1. Activate the Conda development environment using the command: `conda activate dev`.
  2. To verify the installation of scikit-learn in the active environment, use: `conda list scikit-learn`.
  3. If installed correctly, the terminal will show scikit-learn as listed.

- **Installation Instructions** (if scikit-learn is missing):
  1. Install scikit-learn using the command: `pip install -U scikit-learn`.
  2. After installation, revert to the confirmation steps to ensure the library has been installed successfully.

2.1.4: [Clustering](https://github.com/rainleander/MSU-AI-microbootcamp/tree/main/module-2-files/demos/01-Clustering)   
- **Definition**: 
  - Clustering is the process of grouping similar data based on certain similarities.
  - Unsupervised learning models often use clustering algorithms to group objects.

- **Application Example**:
  - Cable services can use a clustering algorithm to group customers based on their viewing habits.

- **Practical Implementation**:
  - An example in the Interactive Python Notebook file is [demos/01-Clustering/Unsolved/clusters.ipynb](https://github.com/rainleander/MSU-AI-microbootcamp/blob/main/module-2-files/demos/01-Clustering/Unsolved/clusters.ipynb) in the [module-2-files folder](https://github.com/rainleander/MSU-AI-microbootcamp/tree/main/module-2-files).
  - A synthetic dataset is generated using `make_blobs` from scikit-learn.
    - It creates two features (X) and labels (y).
    - `random_state=1` ensures consistent random data generation.

- **Exploring the Data**:
  - The synthetic dataset created provides an array of X values, with a (100, 2) shape indicating 100 rows and two columns.
  - The y values are transformed to fit into a single column.
  - Data can be visualized using Pandas plot methods.
    - The X values (features) are named "Feature 1" and "Feature 2".
    - y values (labels) are added as "Target".

- **Visualization**:
  - Data can be visualized in a scatter plot, showcasing the different clusters.
  - Clusters indicate similar data points.
  - This is called "centering," which determines classes/groups in advanced analytics.
  - Centering also improves logistic regression models by ensuring data points have the same starting mean value.  
  
2.1.5: Recap and Knowledge Check  
- **Definition**:
  - Unsupervised learning models often use clustering algorithms.
  - The goal is to group similar objects/data points into clusters.

- **Cluster Assignment**:
  - In the given examples, clusters are pre-defined using the "clusters" parameter.
  - However, not all datasets come with predefined clusters.

- **Role of Unsupervised Learning**:
  - The primary role is to identify the distinct number of clusters within a dataset.
  - This task is achieved using the K-means algorithm, which will be discussed next.  

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
