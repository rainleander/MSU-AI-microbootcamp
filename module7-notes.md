## Module 7: Natural Language Processing
### Introduction Natural Language Processing
7.1.1: Introduction to NLP  
- Introduction to Natural Language Processing (NLP) in the context of deep learning models, focusing on text and speech processing.
- Natural languages refer to human communication methods, unlike the structured programming languages of computers. NLP aims to enable computers to interpret language as humans do.
- Challenges in NLP include understanding homonyms, idioms, and varied contexts in human communication.
- NLP applications in enterprise settings are significant due to the prevalence of unstructured data (over 80% according to Gartner) in various formats like emails, social media posts, and documents.
- Historical NLP models relied on rules-based approaches, using syntax and grammar to understand text. However, these models need to be expanded and help with the variations and complexities of natural language.
- Modern NLP adopts a statistical approach, where machine learning and deep learning models assign probabilities to different text interpretations, improving language understanding accuracy.
- A critical development in NLP is the introduction of attention mechanisms, allowing models to understand word relationships more effectively.
- NLP is divided into two main subsections: 
  - Natural Language Understanding (NLU) focuses on comprehension and meaning extraction from text data.
  - Natural Language Generation (NLG) focuses on generating human-like language and responses.
- NLP applications include email spam filters, sentiment analysis, question answering, speech recognition, and language translation.
- Advanced NLP models, like GPT3, GPT3.5, GPT4, and Google's Bard, leverage NLU and NLG to interpret prompts and generate relevant, natural-sounding responses.
- The module aims to teach the fundamentals of NLP in AI, including understanding text in computing, language models, tokenization processes, similarity measures, and using transformer neural networks for various NLP tasks.

7.1.2: Getting Started  
- Instruction to download a specific file before beginning the module.
- File name: Module 7 Demo and Activity Files.

7.1.3: What is Text?  
- Humans use language comprising letters, symbols, words, and gestures for communication and conveying meaning.
- Computers interpret and "understand" language through numerical representations.
- Computer text is stored as binary digits (1's and 0's) representing characters based on ASCII or Unicode standards.
- Text data presents unique challenges for processing compared to structured data.
- Text data is inherently unstructured, unlike structured data like tabular datasets or image pixels.
- Specialized techniques, such as tokenization, are required to process text data.
- These techniques extract meaningful features from text and convert them into a numerical format.
- The bag-of-words (BoW) model is a popular and straightforward method for converting text to numerical format.

7.1.4: Bag-of-Words Model  
- The bag-of-words (BoW) model converts text into a numerical format, such as a vector or array, for computers to understand.
- BoW tallies the occurrence of words in text, ignoring grammar, punctuation, and word order.
- It develops a vocabulary from all words in the text and records the frequency of each word in sentences.
- Example sentences are used to demonstrate the BoW model:
  - "I want to invest for retirement."
  - "Should I invest in mutual funds, or should I invest in stocks?"
  - "I should schedule an appointment with a financial planner."
- The vocabulary generated includes words like 'appointment', 'financial', 'funds', 'invest', etc.
- BoW process involves stop-word removal and filtering out words like pronouns, prepositions, and articles to create a more meaningful vocabulary.
- Frequency scores are assigned to each sentence based on the occurrence of each vocabulary word.
- The frequency representation is stored as vectors compiled into an array.
- When applied to large datasets, the BoW model results in an extensive vocabulary and vectors with many zero values, known as sparse matrices.
- More advanced NLP techniques, like language models, have been developed based on the BoW concept to capture semantic meaning and contextual understanding in sentences.

7.1.5: What is a Language Model?  
- Language models are statistical models trained on text data to understand patterns and relationships between sentences, words, and characters.
- The process of using language models involves:
  - Preprocessing text into numerical form.
  - Providing this numerical data as input to the model.
  - Training the model, during which it updates its internal weights.
  - Monitoring the model’s performance through key evaluation metrics and loss function values.
  - Optimizing the model by adjusting parameters and observing the evaluation metrics and loss function.
- Language models typically make probabilistic assessments to predict the next word in a sequence, similar to predictive text features on smartphones or email clients.
- Large language models (LLMs) are distinguished by being trained on vast amounts of text data and having numerous parameters, allowing them to handle more complex problems.
- LLMs utilize transformer architecture and are powerful due to their ability to update and tune numerous parameters.
- These models use self-supervised learning, starting unsupervised and evolving into supervised models.
- Pre-trained LLMs save computational resources as they have been trained on extensive datasets using powerful hardware.
- Responses from LLMs can exhibit cognitive biases, reflecting biases in the training data.
- Upcoming lessons will cover using pre-trained LLMs for various NLP tasks and breaking down text into tokens for processing.

### Tokenizers
7.2.1: Introduction to Tokenizers  
- Preprocessing text-based, unstructured data for natural language processing (NLP) models is crucial, with tokenization being a key step.
- Tokenization involves breaking text down into smaller units, or "tokens," including words, subwords, punctuation, symbols, and emojis.
- Each token is converted into a numerical representation for neural networks to predict the next word in a text.
- Common types of tokenization:
  - Word tokenization: Treats each word as a token. Used for tasks like named entity recognition, part-of-speech tagging, and text classification. Not typically used with transformers and LLMs.
  - Subword tokenization: Breaks words down into smaller subwords. It is vital for LLMs as it helps build a language model's vocabulary and handle out-of-vocabulary words. Useful for tasks like text translation, summarization, and question answering.
    - Example: "I am learning about subword tokenization" becomes ['i', 'am', 'learning', 'about', 'sub', '##word', 'token', '##ization', '.'], where "##" indicates a continuation of the word.
  - Character tokenization: Considers each character as a token. Used for detailed tasks like text classification or sentiment analysis, capturing fine-grained patterns in text, including slang and emojis. Not used by LLMs due to performance issues.
- A Jupyter notebook file demonstration can showcase the tokens' conversion into numerical values.

7.2.2: Tokenizer Example: Index Encoder  
- Tokenizers divide text into units like words or subwords and encode these units into numerical values using index encoding.
- Follow along with the demonstration using the Jupyter notebook file "tokenization_encoding.ipynb" from the module-7-files folder.
- Example text: "I love my dog. I love my family."
- Process of index encoding:
  - Step 1: Create a unique word index for the vocabulary, assigning each word a number while removing duplicates and unnecessary characters.
    - Example output: {'I': 1, 'love': 2, 'my': 3, 'dog': 4, 'family': 5}.
  - Step 2: Split the text into sentences.
    - Result: ['I love my dog', 'I love my family'].
  - Step 3: Split the sentences into lists of words.
    - Result: [['I', 'love', 'my', 'dog'], ['I', 'love', 'my', 'family']].
  - Step 4: Encode the text using the integer index from the vocabulary.
    - Final output: [[1, 2, 3, 4], [1, 2, 3, 5]].
- This process creates lists of integers (vectors or embeddings) that can be used with machine learning models.
- Considerations for tokenization:
  - Choose the unit of tokenization (words, characters, subwords).
  - Decide on the size of the dictionary for encoding.
- The demonstration used word tokenization, but character, subword, or other text sequence encoders can also be used.
- Modern NLP libraries often use complex subword tokenizers like BERT.
- In practice, using a tokenizer library with a larger and more complex dictionary is preferable, compared to manually compiling a small one.

7.2.3: Introduction to Hugging Face Tokenizers  
- Hugging Face tokenizers are open-sourced, pre-trained tools that convert text into numerical tokens for machine learning models, especially suited for large datasets and large-scale models like Large Language Models (LLMs).
- These tokenizers are fast, accurate, and easy to integrate into Python applications.
- Embeddings are vector representations of text sequences that capture meaning for various NLP tasks like text classification and sentiment analysis.
- To demonstrate converting sentences into tokens and embeddings, the "sentence_tokenizer.ipynb" Jupyter notebook file is used.
- Step-by-step demonstration:
  - Step 1: Install the Sentence Transformer Package and use the pre-trained model 'all-MiniLM-L6-v2'.
  - Step 2: Generate tokens from a sample sentence, resulting in a list of strings representing tokens.
  - Step 3: Convert tokens to IDs using the `convert_tokens_to_ids` method; each token gets a unique numerical ID.
  - Step 4: Decode the IDs back into words, reconstructing the original sentence.
- The process also includes generating embeddings, arrays of numerical values representing sentences.
- Token IDs correlate with corresponding embeddings, capturing the meaning and context of words or sentences.
- These embeddings are used as inputs to the model's neural network for generating predictions or outputs.
- Transformers require numerical values (embeddings) rather than text for processing, making tokenization a critical first step.

7.2.4: Similarity Measures  
- Embeddings are crucial in NLP for representing the relationships between words or sentences in a model's internal data.
- Similarity measures like Euclidean distance, cosine similarity, and Pearson correlation coefficient are mathematical methods to calculate distances between vectors, indicating similarity.
  - **Euclidean distance** measures the distance between two data points in multi-dimensional space.
  - **Cosine similarity** assesses the cosine of the angle between vectors, ranging from -1.0 to 1.0.
  - **Pearson correlation coefficient** measures the linear correlation between variables, ranging from -1.0 to 1.0.
- The Jupyter notebook file "cosine_similarity_measures.ipynb" is used to demonstrate similarity measures.
- Steps in the demonstration:
  - **Step 1**: Install Transformers and import required classes.
  - **Step 2**: Create the model using "bert-base-uncased" and "bert-base-cased".
  - **Step 3**: Generate tokens for sentences, resulting in a list of strings.
  - **Step 4**: Convert tokens to IDs using tokenizer methods.
  - **Step 5**: Convert IDs to embeddings using PyTorch, printing the first 10 numerical values for each sentence's embedding.
  - **Step 6**: Determine similarity between sentences using a 3x3 matrix to calculate pairwise cosine similarities.
- The similarity matrix reveals the degree of similarity between each sentence.
- A DataFrame visualizes the cosine similarities for each embedding.
- The geometric representation of embeddings shows the proximity of sentences in space, indicating their similarity.

7.2.5: Tokenizer Case Study: AI Search Engine  
- Tokenizers play a crucial role in NLP by converting text into smaller units called tokens.
- Index encoder tokenizers can be implemented using pre-trained models to facilitate this process.
- Models determine similarity between embeddings, a vital aspect of understanding relationships in text.
- The upcoming demonstration involves creating an AI-driven search engine, leveraging the concept of similarity to determine relevance.
- Procedure for the AI search engine demonstration:
  - **Step 1**: Install the Sentence-transformer package.
  - **Step 2**: Define the text for the search using a variable.
  - **Step 3**: Split the text into sentences using Python's splitlines method.
  - **Step 4**: Define the SentenceTransformer model.
  - **Step 5**: Use the model to create embedding vectors for each sentence.
  - **Step 6**: Generate a query, encode it, and compare it to the existing text.
  - **Step 7**: Determine similarity between the query and sentences using cosine similarity.
- In the demonstration, the model encodes sentences from the text and a query into embeddings, then compared using cosine similarity to find the most relevant sentences.
- The output demonstrates the similarity scores between the query and each sentence, indicating which sentence in the text is most similar to the query.
- The similarity function could be expanded for more comprehensive analysis and comparisons.

7.2.6: Recap and Knowledge Check  
- The preprocessing step in machine learning, including unsupervised, supervised, and deep learning, aims to reduce data noise for improved predictions.
- In NLP, preprocessing extends to tokenization, a crucial process for converting human language into numerical forms understandable by computers.
- Tokenizers bridge the gap between natural human language and the numerical language of computers, enabling the development of NLP applications.
- Mathematical similarity measures are applied to text embeddings to compare sequences, which are functional in tasks like similarity search where similarity determines relevance.
- The lesson highlights the importance of Hugging Face tokenizers, offering powerful and user-friendly tools for NLP tasks, significantly easing the coding process.

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
