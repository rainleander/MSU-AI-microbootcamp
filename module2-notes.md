## Module 2: Unsupervised Learning
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
  - The primary role is identifying a dataset's distinct number of clusters.
  - This task is achieved using the K-means algorithm, which will be discussed next.  

2.1.6: Segmenting Data: The K-Means Algorithm  
- **Challenges with Clustering**:
  - Identifying the correct number of clusters in data is often challenging.
  - Features distinguishing different groups might not always be evident.

- **K-Means Algorithm**:
  - It's an unsupervised learning algorithm used for clustering.
  - Simplifies the process of objectively and automatically grouping data.
  - K (uppercase) indicates the actual number of clusters.
  - k (lowercase) generally refers to a cluster.

- **How K-Means Works**:
  1. The algorithm first assigns points to the nearest cluster center.
  2. It adjusts the cluster center to the mean of data points in that cluster.
  3. This process is iterative, refining cluster assignments with each iteration.
  4. K-Means undergoes the following steps:
     - Randomly selecting K clusters.
     - Assigning each object to a similar centroid randomly.
     - Updating cluster centers based on new means.
     - Reassigning data points based on updated cluster centers.

- **Applications & Benefits**:
  - One significant application is **customer segmentation** in markets.
  - Customer segmentation is crucial in financial services for better market understanding.
  - Notable Example: **Netflix**:
    - Uses segmentation to recommend movies based on customer preferences.
    - Has improved product utility and reduced user cancellation rates through this.

- **Using K-Means with scikit-learn**:
  - K-Means can be implemented in Python using the **scikit-learn** library.
  - Scikit-learn is an open-source Python library offering various supervised and unsupervised learning algorithms.
  - Esteemed organizations like J.P. Morgan, Booking.com, Spotify, and Change.org use scikit-learn for their machine learning efforts.
  - It is the primary tool recommended for machine learning in the provided content.  

2.1.7: [Using the K-Means Algorithm for Customer Segmentation](https://github.com/rainleander/MSU-AI-microbootcamp/tree/main/module-2-files/demos/02-Kmeans)  
- The example discusses clustering average ratings for customer service, focusing on ratings from customers who evaluated the mobile application and in-person banking services.
- Users can follow along using a specific Jupyter notebook file ([`demos/02-Kmeans/Unsolved/services_clustering.ipynb`](https://github.com/rainleander/MSU-AI-microbootcamp/tree/main/module-2-files/demos/02-Kmeans/Unsolved)).
- The data is loaded using Python's Pandas library from a CSV file named `service-ratings.csv`.
- Initial data review shows two columns: “mobile_app_rating” and “personal_banker_rating”.
- A scatter plot is created to visualize the spread of these ratings.
- Data visualization reveals that clear clusters aren't immediately obvious; however, there's a noticeable congregation of points around specific values.
- The K-means clustering algorithm from the scikit-learn library will be used to identify clusters in this data.
- The K-means model is initialized targeting two clusters, with specific parameters for consistent outcomes (`random_state=1`) and automatic determination of starting centroids (`n_init='auto'`).
- The model is trained (fit) using the service ratings data.
- Predictions for cluster assignments are made for each data point.
- The predictions are added as a new column (“customer_rating”) to a copy of the original DataFrame.
- The predicted customer ratings create A new scatter plot, color-coded.
- The updated scatter plot reveals two distinct clusters, indicating preferences: one group leaning towards mobile banking and the other towards in-person banking.
  
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
