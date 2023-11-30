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
    - This has improved product utility and reduced user cancellation rates.

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
- Data visualization reveals that clear clusters aren't immediately apparent; however, there's a noticeable congregation of points around specific values.
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
- Cluster analysis requires trial and error to find optimal clusters.
- Different features are tested to achieve the desired number of clusters.
- K-means algorithm has been used for defining customer segments with limited features like "mobile_app_rating" and "personal_banker_rating".
- Identifying the optimal number of clusters (k) in a new, unlabeled dataset with many features is challenging.
- The elbow method is a well-known technique for determining the optimal value of k in the K-means algorithm.
- The elbow method is a heuristic for efficiently solving the problem of determining the number of clusters.
- In K-means, uppercase K represents the number of clusters, while lowercase k refers to a cluster in general.
- The elbow method involves running the K-means algorithm for various values of k and plotting an elbow curve.
- The elbow curve plots the number of clusters (k) on the x-axis and the measure of inertia on the y-axis.
- Inertia measures the distribution of data points within a cluster.
- Low inertia indicates data points are close together within a cluster, implying a slight standard deviation relative to the cluster mean.
- High inertia suggests data points are more spread out within a cluster, indicating a high standard deviation relative to the cluster mean.
- The optimal value for k is found at the elbow of the curve, where the inertia shows minimal change with each additional cluster added.

2.2.2: Apply the Elbow Method  
- The Jupyter notebook file 'elbow_method.ipynb' in the 'demos/03-Elbow_Method/Unsolved' directory applies the elbow method.
- Python and the K-means algorithm are used for customer segmentation analysis with customer service ratings data.
- Dependencies are imported, and the dataset is loaded into a Pandas DataFrame.
- An empty list is created to store inertia values, and a range of k-values (1 to 10) is used to test.
- A loop computes inertia for each k-value, which is then appended to the inertia list.
- A DataFrame is created to hold k-values and corresponding inertia values.
- The DataFrame is plotted as an elbow curve to observe the relationship between the number of clusters and inertia.
- The rate of decrease in inertia is analyzed to determine the elbow point, which is identified at k=4.
- The percentage decrease in inertia between each k-value is calculated to confirm the elbow point.
- K-means algorithm and plotting are rerun using four clusters.
- The elbow curve is emphasized as a guide, not a definitive answer, for determining the right number of clusters.

2.2.3: Activity: Finding the Best k  
- Background: This activity involves using the elbow method to determine the optimal number of clusters for segmenting stock pricing information.

Subtasks:
  - Establish an elbow curve.
  - Evaluate the two most likely values for k using K-means.
  - Generate scatter plots for each value of k.

- Files: Start with the files in `activities/02-Finding_the_Best_k/Unsolved/`.

- Instructions:
  1. Read the `option-trades.csv` file into a DataFrame using the "date" column as the DateTime Index. Parameters for `parse_dates` and `infer_datetime_format` should be included.
  2. Create two lists: one for lowercase k-values (1 to 11) and another for inertia scores.
  3. For each k-value, define and fit a K-means model, then append the model's inertia to the inertia list.
  4. Store the k-values and inertia in a DataFrame called `df_elbow_data`.
  5. Plot the `df_elbow_data` DataFrame using Pandas to visualize the elbow curve, ensuring the plot is styled and formatted.

- Solution:
  - Compare your work with the solution in `activities/02-Finding_the_Best_k/Solved/finding_the_best_k_solution.ipynb` in the `module-2-files` folder.
  - Reflect on differences between your approach and the solution, and identify confusing aspects.
  - If questions arise, attend instructor-led office hours for a detailed walkthrough.

2.2.4: Recap and Knowledge Check  
- Focus: Using the elbow method to find the optimal number of clusters (k) for data.

Subtasks:
  1. Running the K-means algorithm for a range of k-values.
  2. Plotting the results and the corresponding level of inertia for each k-value.

- Inertia: Measures the distribution of data points within a cluster.
  - Optimal k-value: Identified at the elbow of the curve where the rate of decrease in inertia slows down.

- Next Steps: Introduction to data optimization through data scaling.
  - Importance: Data scaling is crucial for applying PCA (Principal Component Analysis).
  - PCA combines learnings from unsupervised learning to enhance the algorithm and simplify data interpretation.
  - Further exploration of PCA will follow the scaling and transforming data activity.
    
2.2.5: Scaling and Transforming for Optimization  
- Overview: Applying PCA to enhance machine learning algorithms after optimizing data clustering with K-means.

Subtasks:
  1. Apply standard scaling to data features before using PCA.
  2. Combine PCA with K-means for better processing of large datasets.

- Preparing Data:
  - Manual data preparation, especially normalization or transformation, is time-consuming for multiple columns.
  - K-means requires numeric values in a DataFrame to be on the same scale to avoid bias towards any single variable.

- Standard Scaling:
  - Common method for data scaling, centering values around the mean.
  - Involves transforming data to eliminate measurement units and bring numeric values to a similar scale.
  - Use of functions from Pandas and scikit-learn simplifies data preparation.
  - Standard scaling centers bell curves around the same mean value, making them comparable.
  - Example: Scaling "Annual Income" using StandardScaler from scikit-learn.
    - StandardScaler calculates the column mean and scales data to a standard deviation of 1.
    - Formula: (value - mean)/standard deviation.
  - Scaling to a mean of 0 and standard deviation of 1 is essential for equalizing variable ranges in machine learning models.

- Importance:
  - Prevents machine learning models from giving undue importance to columns with larger or more widely ranging values.
  - Known as data standardization, a common practice before training machine learning models.

- Application:
  - Demonstration of applying standard scaling to a shopping dataset.

2.2.6: Apply Standard Scaling  
- Overview: Application of standard scaling using a shopping dataset from the "Spending beyond your K-Means" activity.

Steps for Standard Scaling:
  1. Import the StandardScaler module and Pandas.
  2. Load data into a DataFrame and drop unnecessary columns like "CustomerID".
  3. Scale numeric columns "Age", "Annual Income", and "Spending Score" using StandardScaler's fit_transform function.
  4. Create a new DataFrame with scaled data.

- Important Notes:
  - StandardScaler's fit_transform function returns an array, not a DataFrame.
  - The array needs conversion to a DataFrame, with columns named in the order they were scaled.
  - StandardScaler is suitable for continuously ranging data, not categorical data.

Handling Categorical Data:
  - Use Pandas' get_dummies function to transform categorical data into numerical format.
  - Example: Transform "Card Type" column into numerical values representing "Credit" and "Debit".
  - get_dummies converts categorical columns into separate numerical columns.

Concatenating Transformed Data:
  - Concatenated scaled numerical data and transformed categorical data into one DataFrame using Pandas' concat function.

Pro Tip:
  - Scaling numeric data and encoding categorical data are essential practices in data preparation for K-means clustering.
  - Ensures all data is on the same scale, preventing disproportionate weight to any variable.

- Application:
  - Practice standard scaling and encoding variables in the next activity.

2.2.7: Activity: Standardizing Stock Data  
- Background: Standardize stock data and use the K-means algorithm for clustering.

Files:
  - Begin with files in `activities/03-Standardizing_Stock_Data/Unsolved/`.

Instructions:
  1. Read the `tsx-energy-2018.csv` file into a DataFrame, setting the “Ticker” column as the index.
  2. Note: Stock data includes yearly mean prices, volume, annual return, and variance from TSX-listed energy companies.
  3. Use the StandardScaler module and fit_transform function to scale numerical columns.
  4. Review a sample of the scaled data.
  5. Create a new DataFrame, `df_stocks_scaled`, with the scaled data:
     - Retain original column labels.
     - Add and set the ticker column from the original DataFrame as the index.
  6. Review the new DataFrame.
  7. Encode the “EnergyType” column using pd.get_dummies, storing the result in `df_oil_dummies`.
  8. Concatenate `df_stocks_scaled` and `df_oil_dummies` using pd.concat function along axis=1.
  9. Review the concatenated DataFrame.
  10. Cluster the data using the K-means algorithm with a k value 3.
  11. Create a copy of the DataFrame and add a column with company segment values.

Solution:
  - Compare your work with the solution in `activities/03-Standardizing_Stock_Data/Solved/standardizing_stock_data_solution.ipynb`.
  - Reflect on the differences between your approach and the solution.
  - Please feel free to seek help during instructor-led office hours for any confusion or questions.

2.2.8: Recap and Knowledge Check  
- Overview: Scaling or normalizing datasets prepares data for future analysis.
  - Purpose: Adjust the scale of data for easier comparison.
  - Methods: This can be done manually or using functions from Pandas and scikit-learn.

- Recent Lesson: Explored data scaling by centering values around the mean, known as standard scaling.

- Upcoming Lesson: 
  - Focus: Understanding the importance of scaling data for PCA (Principal Component Analysis).
  - Content: Explore the steps involved in the PCA technique.

### Principal Component Analysis
2.3.1: Introduction to Principal Component Analysis  
2.3.2: Recap and Knowledge Check  
2.3.3: Activity: Energize Your Stock Clustering  
2.3.4: Recap and Knowledge Check  

### Summary: Unsupervised Learning
2.4.1: Summary: Unsupervised Learning  
2.4.2: Reflect on Your Learning  
2.4.3: References  
