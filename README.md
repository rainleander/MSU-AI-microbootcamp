# MSU AI MicroBootCamp Notes and Projects
## Module 0: Getting Started
### Getting Started
0.1.1: Welcome  
- **Real-world AI Applications:**
  - Netflix's recommendation system.
  - Google Maps' optimal route prediction.
  - Alexa's rapid playlist retrieval.

- **History and Progression of AI:**
  - The Logic Theorist, the first AI programming language, developed in the 1950s.
  - Evolution over decades with advancements like Large Language Models (LLMs) and Language Generation software such as ChatGPT.
  - AI's extensive impact on diverse sectors like entertainment, finance, medicine, and daily life.

- **Business Impact of AI:**
  - AI boosts productivity and efficiency by 40%.
  - A projected growth in Global GDP by $15.7 trillion due to AI by 2030 (Techjury, 2023).

- **Course Content and Objectives:**
  - A deep dive into machine learning as a core AI application.
  - Merges conceptual knowledge with technical proficiency.
  - Practical hands-on guidance, ensuring students develop AI models independently.
  - A glimpse into recent AI breakthroughs and potential future applications.  

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
- **Introduction to AI:**
  - AI is a branch of computer science that imitates human intelligence.
  - AI systems ingest vast amounts of data, learn from it, and leverage the acquired knowledge to forecast future data and tackle intricate issues.

- **Understanding Machine Learning:**
  - A subfield of AI that allows computer systems to learn from data and make informed decisions or predictions without human intervention.
  - Traditional programming needs explicit rules, whereas machine learning predicts using data-driven models.
  - Examples include weather forecasting, where past data aids in crafting models for future predictions.
  - Over time, machine learning models can self-improve by identifying and discarding outlier data.

- **Categories of Machine Learning:**
  - **Supervised Learning:** Predominantly utilizes labeled data.
  - **Unsupervised Learning:** The system independently categorizes data.
  - **Reinforcement Learning:** Learning occurs via a trial-and-error approach.
  - This course will concentrate on supervised and unsupervised learning.

- **Industry Insights:**
  - The machine learning sector will be valued at $209.9 billion by 2029 (McCain, 2023).

- **AI vs. Machine Learning:**
  - AI focuses on replicating human intelligence for problem-solving, while machine learning is centered on data-driven predictions and decision-making.
  - AI equips algorithms to mimic human-like behavior, whereas machine learning empowers algorithms to generate their intelligence.

- **Neural Networks and Deep Learning:**
  - Neural networks, inspired by the human brain's structure, help computers emulate human cognition.
  - Like brain neurons, these networks are crucial for data transmission and signal relay in machine learning.
  - Using neural networks, machine learning software processes data and crafts algorithms that enhance performance over time.
  - Deep learning, a machine learning subset, heavily relies on these neural networks.  

1.1.3: Narrow AI vs Artificial General Intelligence  
- **Narrow AI (ANI or Weak AI):**
  - Focuses on executing specific tasks and making decisions based on its training data.
  - Describing it as "weak" might be misleading, as ANI can solve intricate problems efficiently.
  - All present-day AI applications, such as chatbots, recommendation systems, facial recognition, self-driving cars, and voice assistants like Siri and Alexa, are considered narrow AI.
  - **Insight:** Voice assistants are gaining traction; around 35% of Americans utilize them daily for news and weather updates (Branka, 2023).

- **Artificial General Intelligence (AGI or Strong AI):**
  - Represents AI that has self-awareness and can match or surpass human intelligence.
  - AGI is more of a fictional concept seen in movies like The Terminator, Her, WALL-E, and 2001: A Space Odyssey.
  - Potential AGI would combine machine learning, artificial neural networks, NLP, deep learning, and technologies not yet developed.
  - It could possibly possess human-like attributes like imagination, deception, and inquisitiveness.
  - Some existing narrow AI tools, such as ChatGPT, simulate AGI characteristics by creating human-like interactions.
  - However, these powerful tools are not sentient and differ from human cognition.

- **Comparison of ANI and AGI:**
  - The provided table (not visible here) elaborates on the distinctions between narrow and general AI.

- **Artificial Super Intelligence (ASI):**
  - Represents the zenith of AI, outperforming human capabilities.
  - Renowned experts, including Elon Musk and Stephen Hawking, perceive AGI and ASI as potential threats to human existence.

- **Ethical Considerations:**
  - Adopting AI, especially advanced forms, poses ethical dilemmas and potential societal harm.
  - A human-centric approach is essential during AI model development to ensure conscious and responsible usage.
  - The subsequent section will address the ethics associated with AI and strategies to avert potential challenges.  

1.1.4: Ethics and AI  
**Data Ethics and Big Data**

Data ethics examines the ethical implications of data usage, especially big data. The core idea revolves around ethical principles guiding how data is used. 

- **Why is Data Ethics Important?** As data grows in volume and complexity, its misuse could harm individuals. Understanding how to manage this data ethically is crucial. Awareness of potential ethical pitfalls can lead to positive change through technology.

**Main Concerns with AI**

1. **Consent**: Consent is a cornerstone of ethical data usage. An infamous case involving Clearview AI demonstrated the dangers of neglecting consent. Clearview AI used personal images without the individual's permission, leading to a hefty fine. The takeaway? Consent is paramount.

2. **Algorithmic Bias**: At its core, an algorithm is a step-by-step process for accomplishing a task. Algorithms can be simple or complex and are often used to process data. Bias refers to unequal treatment. Combine the two, and you have "algorithmic bias" - when systems treat groups or individuals unequally.

   - *Types of Algorithms* (as listed by Nicholas Diakopoulos):
     - Prioritization
     - Classification
     - Association
     - Filtering

Bias in algorithms can accumulate quickly, especially when the system operates without much human intervention. 

**Algorithmic Bias in Action: Gender Shades**

- **The Study**: Researchers at MIT tested facial recognition software by IBM, Microsoft, and Face++.
- **Findings**: The software performed differently based on skin shade and gender, often misidentifying darker-skinned females.
- **Implications**: Even if a tool is largely accurate, it can still have biases that disproportionately affect certain groups.

**Understanding Causes of Algorithmic Bias**

1. **Background Disparities**: If developers have a homogenous background, their unconscious biases might manifest in the technologies they develop.
2. **Biased Training Data**: An algorithm learns from the data. If that data has built-in biases, so will the algorithm.

**Addressing Algorithmic Bias**

1. **Audits**: Both internal and external audits can identify biases. The Gender Shades study is an example of an external audit.
2. **Transparency**: Being open about how algorithms work and their data sources can aid in accountability. However, intellectual property and privacy concerns may limit this.
3. **Contestability**: Offering users the chance to contest or disagree with algorithmic results.

**Checklists for Addressing Algorithmic Bias**

- *Existing Systems*:
  1. Understand its workings and historical data biases.
  2. Compare with similar systems.
  3. Audit for varying results based on input.
  4. Ensure contestability.

- *Systems in Development*:
  1. Ensure developers understand the diverse groups the system impacts.
  2. Examine training and testing data for representation biases.
  3. Maintain clear documentation.
  4. Plan for internal audits and third-party testing.

Ethical considerations are paramount in the age of big data and AI. By understanding the potential pitfalls and actively working to mitigate them, we can ensure that these technologies benefit all members of society.  

1.1.5: Recap and Knowledge Check  
- Artificial intelligence (AI) is a branch of computer science that aims to replicate human intelligence in machines.
- Machine learning, a subset of AI, allows algorithms to learn from data and make decisions or predictions without specific programmer instructions.
- These technologies (AI and machine learning) significantly impact our daily lives and the world at large.
- There are distinct differences between AI and machine learning, yet they are interrelated.
- AI can be categorized as narrow AI (specialized in one task) or artificial general intelligence (capable of any intellectual task a human can do).
- There's an ongoing debate about the feasibility and desirability of achieving artificial superintelligence (an intelligence surpassing human capabilities).
- Ethical concerns in AI include issues like algorithmic bias.
- It's crucial to employ strategies to identify and reduce bias when developing AI systems.  

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
  - Aims to translate and comprehend the essence behind words, recognizing intention, sentiment, ambiguities, emotions, and parts of speech.
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
- **Course Overview:** This lesson provides a foundational understanding of various machine learning models featured in the course.

- **Types of Machine Learning:**
  - **Unsupervised Learning:** Uses unlabeled data for analysis and clustering.
  - **Supervised Learning:** Leverages labeled data for training and predictions.

- **Machine Learning Optimization:** 
  - An essential process to enhance machine learning model performance.
  - Involves tweaking both parameters and hyperparameters.

- **Neural Networks:** 
  - Algorithms inspired by the structure and function of the human brain.
  - **Deep Learning:** A subtype of neural networks consisting of three or more layers, enhancing efficiency and capability.

- **Computers vs. Humans:**
  - Distinct differences exist in data comprehension between computers (binary code) and humans.
  - **Natural Language Processing (NLP):** A tool to reconcile these differences, enabling more intuitive human-computer interaction.

- **Future Learning:** This lesson is a broad introduction, with more in-depth exploration and hands-on experiences planned for subsequent course sections.  

1.3.9: References  
AI.NL. 2022. Narrow AI vs Artificial General Intelligence – the key difference and future of AI. Available: https://www.ai.nl/knowledge-base/narrow-weak-ai-vs-artificial-general-intelligence/Links to an external site. [2023, March 29]

Azure Microsoft. n.d. Artificial intelligence (AI) vs. machine learning (ML). Available: https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/artificial-intelligence-vs-machine-learning/#introductionLinks to an external site. [2023, March 29].

Baduge, S.T., Thilakarathna, S., Perera, J.S., Arashpour, M., Sharafi, P., Teodosio, B., & Shringi, A. 2022. Artificial intelligence and smart vision for building and construction 4.0: Machine and deep learning methods and applications. Automation in Construction. Available: https://doi.org/10.1016/j.autcon.2022.104440Links to an external site..

Bailey. J. 2022. The Ethical and Legal Challenges of GitHub Copilot. Available: https://www.plagiarismtoday.com/2022/10/19/the-ethical-and-legal-challenges-of-github-copilot/Links to an external site. [2023, April 5].

Branka. 2023. Voice search statistics - 2023. Available: https://truelist.co/blog/voice-search-statistics/Links to an external site. [2023, April 21].

Brown. S. 2021. Machine learning, explained. Available: https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explainedLinks to an external site. [2023, April 6].

Bohr, A. & Memarzadeh, K. 2020. The rise of artificial intelligence in healthcare applications. Artificial Intelligence in Healthcare. 25-60.

Chakravarthy. S. 2020. Tokenization for natural language processing Available: https://towardsdatascience.com/tokenization-for-natural-language-processing-a179a891bad4Links to an external site. [2023, April 6].

Chang. C. 2020. 3 ways AI can help solve inventory management challenges Available: https://www.ibm.com/blog/3-ways-ai-solves-inventory-management-challenges/Links to an external site. [2023, April 5].

Finin. T. & Joshi. K.P. 2017. Teaching machines to understand – and summarize – text. Available: https://theconversation.com/teaching-machines-to-understand-and-summarize-text-78236Links to an external site. [2023, April 6].

Forbes. 2023. Artificial intelligence's new role in medicine, finance and other iIndustries - how computer learning is changing every corner of the market Available: https://www.forbes.com/sites/qai/2023/02/02/artificial-intelligences-new-role-in-medicine-finance-and-other-industrieshow-computer-learning-is-changing-every-corner-of-the-market/?sh=604580392e68Links to an external site. [2023, April 5].

Fougner. C. 2022. ‘We can invent new biology’: Molly Gibson on the power of AI. Available: https://www.mckinsey.com/industries/life-sciences/our-insights/we-can-invent-new-biology-molly-gibson-on-the-power-of-aiLinks to an external site. [2023, April 5].

Huang. S. 2021. Doctor.ai, an AI-powered virtual voice assistant for health care Available: https://towardsdatascience.com/doctor-ai-an-ai-powered-virtual-voice-assistant-for-health-care-8c09af65aabbLinks to an external site. [2023, April 5].

IBM (n.d.a). What is machine learning?. Available: https://www.ibm.com/topics/machine-learningLinks to an external site. [2023, April 6].

IBM. (n.d) What is artificial intelligence in medicine?. Available: https://www.ibm.com/topics/artificial-intelligence-medicineLinks to an external site. [2023, April 21].

IBM. (.n.d). What is a chatbot? Available: https://www.ibm.com/topics/chatbotsLinks to an external site. [2023, April 5].

IBM. (n.d). What is natural language processing (NLP)?. Available: https://www.ibm.com/topics/natural-language-processingLinks to an external site. [2023, April 6].

Joby. A. 2020. Narrow AI: not as weak as it sounds. Available: https://www.g2.com/articles/narrow-aiLinks to an external site. [2023, March 29]

Le, K. 2022. ‘Hi, can I help you?’ — How chatbots are changing customer service. Available: https://www.salesforce.com/blog/what-is-a-chatbot/Links to an external site. [2023, April 5].

Lee. A. 2023. What are large language models used for? Available: https://blogs.nvidia.com/blog/2023/01/26/what-are-large-language-models-used-for/Links to an external site. [2023, April 6].

Mack. J.L. 2022. Wealth managers embrace AI, machine learning faster than other financial services firms. Available: https://www.financial-planning.com/list/wealth-managers-embrace-ai-machine-learning-faster-than-other-financial-services-firmsLinks to an external site. [2023, April 5].

McCain, A. 2023. *25+ incredible machine learning statistics [2023]: key facts about the future of technology. Available: https://www.zippia.com/advice/machine-learning-statistics/Links to an external site. [2023, April 21].

Merritt. R. 2022. What is a transformer model? Available: https://blogs.nvidia.com/blog/2022/03/25/what-is-a-transformer-model/Links to an external site. [2023, April 5].

Microsoft. (n.d). How medtech helps accelerate drug discovery. Available: https://www.microsoft.com/en-us/industry/healthcare/resources/pharma-medtech-drug-discoveryLinks to an external site. [2023, April 5].

Mollman. S. 2023. OpenAI CEO Sam Altman warns that other A.I. developers working on ChatGPT-like tools won’t put on safety limits—and the clock is ticking. Available: https://fortune.com/2023/03/18/openai-ceo-sam-altman-warns-that-other-ai-developers-working-on-chatgpt-like-tools-wont-put-on-safety-limits-and-clock-is-ticking/Links to an external site. [2023, March 29].

NIHCM. 2021. Racial bias in health care artificial intelligence. Available: https://nihcm.org/publications/artificial-intelligences-racial-bias-in-health-careLinks to an external site. [2023, April 5].

NVIDA. n.d. Improving product quality with AI-based video analytics solution overview. Available: https://www.hpe.com/psnow/doc/a00119250enwLinks to an external site. [2023, April 5].

Rizzoli, A. 2023. 7 Out-of-the-box applications of AI in manufacturing. Available: https://www.v7labs.com/blog/ai-in-manufacturingLinks to an external site. [2023, April 5].

Real Finance. 2019. *Algo trading dominates 80% of stock market. Available: https://seekingalpha.com/article/4230982-algo-trading-dominates-80-percent-of-stock-marketLinks to an external site. [2023, April 21].

Rouse, M. 2022. Narrow artificial intelligence (Narrow AI). Available: https://www.techopedia.com/definition/32874/narrow-artificial-intelligence-narrow-aiLinks to an external site. [2023, March 29].

Sanghavi, A. 2022. 30 self-driving statistics to drive you crazy. Available: https://www.g2.com/articles/self-driving-vehicle-statisticsLinks to an external site. [2023, April 21].

Sodha, S. 2019. Look deeper into the Syntax API feature within Watson natural language understanding. Available: https://developer.ibm.com/articles/a-deeper-look-at-the-syntax-api-feature-within-watson-nlu/Links to an external site. [2023, April 21].

Son, H. 2017. JP Morgan software does in seconds what what took lawyers 360,000 hours. Available: https://www.independent.co.uk/news/business/news/jp-morgan-software-lawyers-coin-contract-intelligence-parsing-financial-deals-seconds-legal-working-hours-360000-a7603256.htmlLinks to an external site. [2023, April 21].

Techjury. 2023. 101 artificial intelligence statistics [updated for 2023]. Available: https://techjury.net/blog/ai-statistics/#grefLinks to an external site. [2023, April 21].

Topol, E. 2023. Doctors, get ready for your AI assistants. Available: https://www.wired.com/story/doctors-artificial-intelligence-medicine/Links to an external site. [2023, April 5].

Vacek, G. 2023. How AI is transforming genomics. Available: https://blogs.nvidia.com/blog/2023/02/24/how-ai-is-transforming-genomics/Links to an external site. [2023, April 5].

Verma, P. 2022. These robots were trained on AI. They became racist and sexist. Available: https://www.washingtonpost.com/technology/2022/07/16/racist-robots-ai/Links to an external site. [2023, March 29].

Vincent, J. 2022. The lawsuit that could rewrite the rules of AI copyright. Available: https://www.theverge.com/2022/11/8/23446821/microsoft-openai-github-copilot-class-action-lawsuit-ai-copyright-violation-training-dataLinks to an external site. [2023, March 29].

Vincent, J. 2023. Getty Images sues AI art generator Stable Diffusion in the US for copyright infringement. Available: https://www.theverge.com/2023/2/6/23587393/ai-art-copyright-lawsuit-getty-images-stable-diffusionLinks to an external site. [2023, March 29].

Visual Studio. (n.d). GitHub copilot. Available: https://marketplace.visualstudio.com/items?itemName=GitHub.copilotLinks to an external site. [2023, April 5].

WEF. 2023. 4 ways artificial intelligence could transform manufacturing. Available: https://www.weforum.org/agenda/2023/01/4-ways-artificial-intelligence-manufacturing-davos2023/Links to an external site. [2023, April 5].

Zakaryan, V. 2022. AI-enhanced predictive maintenance for manufacturing — What’s the point? Available: https://postindustria.com/ai-enhanced-predictive-maintenance-for-ml-manufacturing-whats-the-point/Links to an external site. [2023, April 5].  

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
