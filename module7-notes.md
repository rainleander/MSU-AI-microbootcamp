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
- Introduction to Transformers in NLP: Transformers are neural networks crucial for understanding context and meaning in natural language processing (NLP).
- Use Cases for Transformers: You will learn to use a large language model (LLM) transformer for tasks like language translation, text generation, question answering, and text summarization.
- Origin of Transformers: Transformers were introduced in 2017, primarily for language translation models.
- Transformer Architecture: A transformer consists of an encoder and a decoder. The encoder processes input sequences (e.g., a sentence in English), and the decoder generates outputs (e.g., the translated sentence in another language).
- Encoder Structure: Each encoder in a transformer has two layers - a self-attention layer and a feed-forward layer. Input sequences are converted to vectors (embeddings) and then processed by these layers.
- Self-attention Mechanism: Self-attention allows the model to assess and encode relationships between words in a sequence, enhancing the understanding of context. For example, identifying that "it" refers to "sun" and not "flowers" in a given sentence.
- Parallelism in Transformers: A key feature of transformers is parallelism, where each word vector is processed simultaneously rather than sequentially. This improves the speed and efficiency of training and inference.
- Feed-forward Layers: After the self-attention layer, each word vector is independently processed in the feed-forward layer, eliminating dependencies between them.
- Advantages of Transformers: Due to parallelism and self-attention, transformers are fast in training, can handle large datasets, and can accurately predict the following words in a sequence.
- Transformers as a Subset of Deep Learning: NLP models, including transformers, fall under the broader category of deep learning models, specifically tailored for natural language applications.

7.3.2: Pre-trained models  
- Importance of Pre-trained Transformers in NLP:
  - Neural network and deep learning models, including NLP models using transformers, require training on datasets.
  - Larger and more diverse datasets enhance the robustness and performance of models.
  - Developers often need more access to extensive datasets.
  - Training models on large datasets is resource-intensive and time-consuming.
- Using Pre-trained Models:
  - Common practice in NLP is to utilize pre-trained transformer models.
  - This approach allows developers to concentrate on analysis, predictions, and outputs rather than model training.
- Applications of Transformers:
  - Pre-trained transformers can be used to solve a variety of NLP problems.
  - Upcoming demonstrations will focus on using Hugging Face transformers for translation, text generation, question answering, and summarization tasks.
  - The image illustrates various NLP problems that are solvable with Hugging Face transformers.

7.3.3: Language Translation  
- Language Translation with AI and Transformers:
  - Language translation is crucial for global interactions like diplomacy, business, and knowledge sharing.
  - AI-driven language translation applications are becoming more accurate and capable of handling longer text.
  - Transformers, particularly sequence-to-sequence (Seq2Seq) models, are integral to AI language translation.

- Using T5 Transformer for Language Translation:
  - T5 (Text-to-Text Transfer Transformer) is used for demonstrations in language translation.
  - T5 can perform multiple NLP tasks but is applied explicitly for translating English to French in this demonstration.

- Steps to Translate Text Using T5 Transformer:
  1. **Install and Import Transformers and Tokenizers**: Use the `transformers` package and `AutoTokenizer` class.
  2. **Specify Input**: Define the text to be translated and store it in a variable.
  3. **Get Input IDs**: Add a prefix to indicate the translation task and retrieve input IDs for the text.
  4. **Import Seq2Seq Model**: Utilize `TFAutoModelForSeq2SeqLM` from the `transformers` module.
  5. **Create Seq2Seq Translation Model**: Construct the translation model using the pre-trained 't5-base' model.
  6. **Generate Outputs**: Pass inputs through the translation model to generate numerical outputs.
  7. **Decode Outputs**: Decode the numerical outputs back to the text, translating them into the target language.
  8. **Result**: The output is the text translated into French.

- Exercise to Translate English to German:
  - Use the provided Jupyter notebook to translate "I am celebrating my birthday." into German.
  - Compare the exercise results with the solution provided in the solved notebook.

7.3.4: Hugging Face Pipelines  
- Simplifying NLP Tasks with Hugging Face's Pipeline Library:
  - The pipeline library in Hugging Face provides a simplified way to perform NLP tasks, handling tokenization internally.
  - Ideal for scenarios where the choice between PyTorch or TensorFlow is flexible.

- Demonstration of Text Translation Using Pipeline:
  1. **Install and Import Transformers and Pipeline**: Install `transformers` and import the `pipeline` class.
  2. **Create the Translation Model**: Initialize the pipeline object for translation using the 't5-base' model.
  3. **Specify Input and Perform Translation**: Define the English text and translate it into German using the pipeline.

- Example Output:
  - The example demonstrates translating "I am celebrating my birthday" into German.
  - The German translation output is ‘Ich feiere meinen Geburtstag.’

- Efficiency and Versatility of Pipeline Library:
  - The pipeline library simplifies complex NLP tasks, reducing the coding process to a few lines.
  - Can be used for various NLP tasks like summarization, feature extraction, and question answering.
  - Offers access to publicly available pre-trained models and tools for streamlined NLP application development.
  - Encourages exploration of more capabilities available in the pipeline library on Hugging Face.

7.3.5: Text Generation  
- Overview of Text Generation with Transformer Models:
  - Generative pre-trained transformers (GPTs) are widely used for text generation tasks.
  - Users provide prompts, and the model generates text by predicting the next word in a sequence based on learned language patterns.

- Process of Text Generation:
  - The model uses a decoder to predict the next word in the sequence.
  - Pre-training on large text datasets enables the model to learn word probabilities in various contexts.
  - The transformer generates output based on this learned knowledge.
  - Generative models can produce inaccurate or biased responses; please let me know when using them.

- Demonstration Using EleutherAI's Pre-trained Transformer Model:
  - The demo uses the EleutherAI/gpt-neo-1.3B model for text generation.
  - Follow the demo in the Jupyter notebook file demos/09-Text_Generation/Unsolved/text_generation.ipynb.

- Steps for Using EleutherAI for Text Generation:
  1. **Install and Import Libraries**: Install `transformers` and import the `pipeline` class.
  2. **Create Text Generation Model**: Initialize a pipeline for text generation with the EleutherAI/gpt-neo-1.3B model.
  3. **Generate Text**: Provide a prompt and generate text with a specified maximum length using the model.

- Example Output:
  - An example output text is generated for the prompt "I like gardening because."
  - The model successfully creates coherent, natural-sounding language.
  - Smaller models may produce nonsensical results, as seen in the example with the rabbit.

- Experimenting with Larger Models:
  - Try using larger models like EleutherAI/gpt-j-6b, which often yield better results from training on more parameters.
  - Larger models demonstrate the complexity and the immense data scale required for NLP tasks.

7.3.6: Question and Answering  
- Overview of Transformer Models in Question and Answering:
  - Transformer models automate question and answering tasks in various applications, such as chatbots and online quizzes.
  - BERT (Bidirectional Encoder Representations from Transformers) is commonly used to answer questions.
  - BERT's bidirectional nature allows it to consider past and future tokens in a sequence for better context understanding.

- Practical Use of BERT:
  - BERT can respond to questions by referencing previous ones, unlike similarity searches where each query is independent.
  - Applications include sentiment analysis, question answering, and text summarization.
  - BERT sources answers from pre-trained datasets like Google BooksCorpus and English Wikipedia.

- Demonstration with BERT:
  - The demo uses a light version of BERT, distilbert-base-cased-distilled-squad, fine-tuned on Stanford Question Answering Dataset (SQuAD).
  - Follow along in the Jupyter notebook file demos/10-Question_Answering/Unsolved/question_answering.ipynb.

- Steps for Creating a Question and Answer Model:
  1. **Install and Import Libraries**: Install `transformers` and import the `pipeline` class.
  2. **Create Question and Answer Model**: Initialize the pipeline with the task 'question-answering' and use the distilbert-base-cased-distilled-squad model.
  3. **Provide Context for Answering**: Supply context from Wikipedia about transformers.
  4. **Generate Questions**: List questions related to transformers.
  5. **Answer Questions**: Function to generate answers from the provided text, including a DataFrame showing the question, answer, score, and position of the answer in the text.

- DataFrame Output:
  - The DataFrame displays the question, answer, score (probability match), and the starting and ending index of the answer in the provided context.
  - The probability score indicates how well the answer matches the question.

- Probability Score Calculation:
  - The question is tokenized and processed by the model.
  - The model predicts the start and end token indices of the answer.
  - Probabilities are calculated for all possible start and end index combinations.
  - The highest probability combination is chosen as the answer.

- Further Learning:
  - A detailed understanding of probability score calculation can be explored through additional resources.

7.3.7: Text Summarization  
- Overview of Text Summarization with BART:
  - Utilize BART (Bidirectional Auto-Regressive Transformer) for summarizing text.
  - BART is effective for tasks like text summarization, translation, text classification, and question answering.
  - Follow the demonstration in the Jupyter notebook file demos/11-Text_Summarization/Unsolved/text_summarization.ipynb.

- Steps for Text Summarization:
  1. **Installation and Import**: Install `transformers`, and import `pipeline`.
  2. **Create Summarization Model**: Use pipeline for summarization task with `facebook/bart-large-cnn` model.
  3. **Input Text**: Provide a Wikipedia article on Deep Learning as the source text.
  4. **Perform Summarization**: Summarize the article using the BART model, specifying the range for the summary length.

- Text Summarization Outputs:
  - **Most Likely Summary**: Generated by setting `do_sample=False`, yielding a concise summary.
  - **Diverse Summary**: Created by setting `do_sample=True` or removing the `do_sample` parameter, offering a more varied summary.

- Example Summaries:
  - Most Likely Summary: Focuses on deep learning as part of machine learning, its applications in various fields, and types of learning.
  - Diverse Summary: Highlights deep learning's broad machine learning context, multi-layer network usage, and types of learning.

- Additional Learning:
  - Explore modifications in text generation and summarization strategies through available resources.

7.3.8: Recap and Knowledge Check  
- Summary of Key Learnings in NLP Models:
  - **Understanding Tokenization and Transformers**: Gained foundational knowledge of NLP models, focusing on tokenization and transformer architecture.
  - **Components of Transformers**: Explored transformers' encoder and decoder components.
  - **Self-Attention and Parallelism**: Learned about the crucial roles of self-attention and parallelism in transformers, enhancing contextual understanding and operational speed.
  - **Use of Pre-Trained Models**: Extended the use of pre-trained models to include transformer models like BERT and Eleuther AI’s gpt-j-6b.
  - **Applications of Hugging Face Pipelines**: Utilized Hugging Face pipelines in various NLP tasks, including text generation, question answering, and text summarization.
  - **Appreciation for NLP Complexity**: Developed an understanding of the complexity involved in NLP tasks and the extensive training parameters needed for practical outputs.

- Next Steps:
  - **Focus on Accessibility**: The upcoming learning phase will concentrate on making NLP models more accessible to a broader audience.
  - **Knowledge Check**: Engage in a knowledge check to assess understanding of the lesson’s content.

### AI Applications
7.4.1: AI Applications with Gradio  
- We've used transformers for NLP applications like language translation, text generation, question answering, and text summarization.
- These NLP applications are commonly used in everyday tools like Google Translate and search engines.
- In demonstrations, we've interacted with these models using Python code.
- In real-world applications, users typically interact with these models through a graphical user interface (GUI), not directly with code.
- For everyday use, such as finding a coffee shop, users would type queries into a search bar, utilizing the search engine's GUI.
- The lesson will focus on converting the text summarization code into an AI application with an easy-to-use interface, requiring no coding from the user.
- Gradio, a Python library, will be used to build the GUI for the machine learning model, making the application accessible and user-friendly.
- The lesson aims to demonstrate the Gradio Interface function and its application in creating user-friendly interfaces for AI and machine learning models.

7.4.2: Gradio Interfaces  
- Gradio interfaces act as wrappers for Python functions, allowing dynamic user input and displaying code outputs.
- An example demonstrates Gradio with a simple non-AI function that takes text input and returns modified text.
- To use Gradio, first install it using `!pip install gradio`.
- Create a function `run(msg)` that returns a string with the input message.
- In an example, calling `run("Hello")` in Python code returns "Running with message: Hello".
- However, modifying code for user interaction is impractical, so Gradio Interface is used to create a GUI.
- Import Gradio as `gr` and create an app with `gr.Interface`, specifying the function, inputs, and outputs.
- The `app.launch()` function launches the GUI, allowing users to interact without coding.
- The GUI lets users enter a message in a text field, click Submit, and see the function's output on the page.
- A public URL for the GUI can be created by setting the `share` parameter to `True` in the launch function.
- The public link is temporary, valid for 72 hours, with options to make it permanently available.
- A skill drill suggests creating a `greeter` function using Python and Gradio to greet a user by name.

7.4.3: Gradio App: Text Summarization  
- Utilizing Gradio, a user-friendly interface is created for a text summarization example based on an article about Deep Learning from Wikipedia.
- The original code for text summarization uses the `pipeline` function from the `transformers` library, specifically using the 'facebook/bart-large-cnn' model.
- The provided Wikipedia article on Deep Learning covers various aspects of deep learning, artificial neural networks, and their applications.
- The summarization model `summarizer` is created using the `pipeline` function with specified maximum and minimum lengths for the summary.
- A new function `summarize` is refactored to accept an article as input, run the transformer model, and return the text summary.
- The `summarize` function is tested with the same article to ensure it functions as expected.
- Gradio is then imported to create a dynamic user interface for the `summarize` function.
- The Gradio interface is created with `gr.Interface`, specifying `summarize` as the function and inputs and outputs as text.
- The application is launched using the `app.launch()` method, allowing users to dynamically input text and receive a summary.
- Users can now paste text into the Gradio app, submit it, and receive a summarized version, demonstrating the successful creation of an AI application using Gradio.

7.4.4: Other Gradio Components  
- The Gradio library allows for creating of user-friendly interfaces with additional versatility, using pre-built components for inputs and outputs.
- Number Components: 
  - Previously, the minimum and maximum word lengths for text summarization were hard coded. Gradio can be used to add fields for users to input these values dynamically.
  - In a demonstration, a function `summarize` is modified to accept a user-defined maximum output length.
  - A Gradio app is created, allowing inputs of text and a maximum word count for the summary.
  - Default values for number components can be set, with an example showing a default of 150 words for the maximum length of the summary.

- Default Values:
  - Gradio's number component can be initialized with a default value, improving usability.
  - The Gradio interface is updated to include this default value for the maximum word count.

- Checkbox Components:
  - Gradio's checkbox component is introduced for boolean inputs.
  - The `summarize` function is modified to accept a third parameter for the `do_sample` boolean value.
  - The Gradio app is updated to include a slider for summary length and a checkbox to toggle the `do_sample` parameter.
  - Users can now interact with the app, adjusting the slider and toggling the checkbox to influence the summary output.

- The lesson highlights Gradio's ability to transform machine learning models into easy-to-use applications without requiring coding knowledge from the user.
- Gradio apps (demos) help broaden the audience for models, facilitating feedback from a diverse pool of users, which can uncover points of failure and algorithmic biases in models.

7.4.5: Activity: Question and Answering Textbox  
- Background: The activity involves refactoring code from the Hugging Face Question and Answering solution to create a Gradio app with two textbox components. The app will allow users to input source text for search and ask questions, displaying the results upon submission.

- Files: The activity requires using the `gradio_textbox.ipynb` file from the module-7-files folder to be uploaded and run in Google Colab.

- Instructions:
  - Install transformers and Gradio.
  - Import the transformers pipeline and Gradio.
  - Initialize the pipeline with the `distilbert-base-cased-distilled-squad` model for question and answering.
  - Refactor the `question_answer` function to return the question, answer, score, and starting and ending index of the answer.
  - Create a Gradio app with two textbox components:
    - The first textbox for inputting the text to be searched.
    - The second textbox for asking a question.
  - Ensure the output displays the question, answer, probability score, and the starting and ending index of the answer.
  - Use text from the Wikipedia article on transformers as the source for testing.

- Solution Evaluation:
  - Compare the completed activity with the provided solution in `gradio_textbox_solution.ipynb`.
  - Could you please assess if all steps were completed correctly and note any differences in approach?
  - Reflect on any areas of confusion or difficulty.

7.4.6: Introduction to Hugging Face Spaces  
- In the previous module, Save and Load functions in Keras and GitHub were introduced for sharing trained neural networks among researchers and developers.
- Similarly, having acquired the knowledge of creating machine learning apps using Gradio, you can now quickly build, deploy, and share these apps with other developers.
- Sharing machine learning apps can be accomplished through Hugging Face Spaces with a few simple clicks.

7.4.7: Recap and Knowledge Check  
- This lesson focused on using Gradio to create graphical user interfaces (GUIs) for applications.
- Demonstrated how to refactor NLP models to create user-friendly apps with Gradio interfaces.
- Emphasized including interactive components for user input, such as text fields, number fields, checkboxes, and sliders.
- Explored the use of default values for these components and how to handle multiple inputs.
- In an activity, a question-answering model was refactored to include a Gradio interface, essentially creating a simple "search" engine.
- Highlighted the importance of sharing code and models using Hugging Face Spaces, facilitating collaboration and innovation in AI and NLP fields.
- Mentioned that Gradio can create interfaces for any Python-based application, making machine learning models more accessible to a broader audience.

7.4.8: References  
Abzug, Z. & Juracek, C. 2022. Bye-bye BERT-ie: why we built our own neural network language model instead of using BERT or GPT [Blog, 7 September]. Available: https://www.proofpoint.com/us/blog/engineering-insights/using-neural-network-language-model-instead-of-bert-gpt [2023, April 17].

Alammar, J. 2018. Visualizing a neural machine translation model (mechanics of Seq2seq models with attention). Available: https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/ [2023, April 13].

Oberoi, A. 2020. What are language models in NLP? [Blog, 15 July]. Available: https://insights.daffodilsw.com/blog/what-are-language-models-in-nlp [2023, April 14].

Devopedia. 2022. Natural language processing. Available: https://devopedia.org/natural-language-processing [2023, April 12].

LinkedIn. n.d. How do you evaluate the quality and accuracy of the texts generated by transformers and GPT-3 models? Available: https://www.linkedin.com/advice/0/how-do-you-evaluate-quality-accuracy-texts-generated [2023, April 11].

Gillham, J. 2023. What are transformer models: how do they relate to AI content creation? Available: https://originality.ai/what-are-transformer-models/ [2023, April 13].

Godalle, E. 2022. What is BART model in transformers? Available: https://www.projectpro.io/recipes/what-is-bart-model-transformers [2023, April 5].

Great Learning Team. 2022. An Introduction to Bag of Words (BoW): | What is Bag of Words? Available: https://www.mygreatlearning.com/blog/bag-of-words/ [2023, April 13].

Joshi, K. & Finin, T. 2017. Teaching machines to understand – and summarize – text. Available: https://theconversation.com/teaching-machines-to-understand-and-summarize-text-78236 [2023, April 13].

Menzli, A. 2023. Tokenization in NLP: types, challenges, examples, tools. Available: https://neptune.ai/blog/tokenization-in-nlp [2023, April 14].

Mohan, A. 2023. Question-answering using BERT. Available: https://www.kaggle.com/code/arunmohan003/question-answering-using-bert [2023, April 5].

Nishanth Analytics Vidhya. 2020. Question Answering System with BERT.. Available: https://medium.com/analytics-vidhya/question-answering-system-with-bert-ebe1130f8def#:~=BERT% [2023, April 4].

Rizkallah, J. 2017. The big (unstructured) data problem. Forbes, 5 June. Available: https://www.forbes.com/sites/forbestechcouncil/2017/06/05/the-big-unstructured-data-problem/?sh=42df4b25493a [2023, April 24].

Rouse, M. 2023. Large language model (LLM). Available: https://www.techopedia.com/definition/34948/large-language-model-llm [2023, April 15].

Singh, A. 2022. Hugging Face: understanding tokenizers. Available: https://medium.com/@awaldeep/hugging-face-understanding-tokenizers-1b7e4afdb154 [2023, April 17].
