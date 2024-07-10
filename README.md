<!-- markdown-disable MD041 -->
<div style="text-align: center;">
 <a href="">
  <img src="https://github.com/bfium/text11-data-science-RAG/blob/main/input/img/images11-data%20science-RAG.png" />
</a>

</div>

<hr />
<p align="center">
    <a href="#definition">What is RAG?</a> •
    <a href="#advantages of RAG">Advantages of RAG</a> •
    <a href="#Application Example">Application Example</a> •
    <a href="#Associated Models">Associated Models</a> •
    <a href="#RAG in Data Science">RAG in Data Science</a> •
    <a href="#Glossar">Glossar</a> •
    <a href="#about the author">About the Author</a> •
</p>
<hr />

# What is RAG?

The term "Retrieval-Augmented Generation" (RAG) is an approach in natural language processing (NLP) that combines retrieval mechanisms with text generation networks to improve the quality and relevance of automatically generated responses. It was popularized by models developed by researchers at [Facebook AI Research](https://ai.facebook.com/) (FAIR).

Here’s how it generally works:

1. Retrieval:
   - The first step involves using a retrieval component to fetch relevant information from a large knowledge base. This component can be a traditional search model or a neural retrieval model that extracts relevant passages from a vast collection of documents.

2. Augmentation:
   - The retrieved information is then used to augment the context available to a text generation model. In other words, the retrieved passages serve as additional context that the text generator can use to produce a more informed and accurate response.

3. Generation:
   - In the final step, a text generation model, often based on architectures like GPT (Generative Pre-trained Transformer), uses both the original input (e.g., a question or query) and the augmented information to generate a coherent and relevant response.
<div style="text-align: center;">
 <a href="">
  <img src="https://github.com/bfium/text11-data-science-RAG/blob/main/input/img/images11-data%20science-RAG-illustration.png" />
</a>

# Advantages of RAG

- Improved relevance: By using retrieved information passages to feed the generator, the model can produce responses that are not only grammatically correct but also informed and factually relevant.
  
- Handling vast knowledge: RAG allows combining the power of generative models with large information databases, which is particularly useful for applications like chatbots, question-answering systems, and informative text generation.

# Application Example

Imagine you have a question-answering system for an educational website. When a user asks a question on a specific topic like "What is Data Science?", the retrieval component of the RAG system could fetch relevant information from various sources (like articles, books, or technical documents). This information would then be passed to a generative model that uses this context to produce a precise and well-formulated response.

# Associated Models

- OpenAI GPT-3 with an integrated retrieval component.
- Models developed by FAIR such as DPR ([Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)) coupled with [BART](https://arxiv.org/abs/1910.13461) or another generative architecture.

# RAG in Data Science

Using "Retrieval-Augmented Generation" (RAG) in the context of data science can pave the way for enhanced text generation applications offering more accurate and relevant responses. As a data scientist, you can leverage this combination of retrieval and generation in various use cases such as intelligent virtual assistants, question-answering systems (Q&A), document summarization, and more. Here is a systematic approach to applying RAG in a data science project:

For example analyzing nutritional data from a `.csv` file involves combining retrieval and text generation processes to provide detailed analyses and answers to specific questions. Here’s a step-by-step guide to achieve this:

### Step 1: Data Preparation

1. Read the `.csv` file and load the data into a DataFrame.
2. Prepare the data so that it’s easily usable for retrieval.

```python
import pandas as pd

# Read the CSV file
file_path = "path/to/nutritional_data.csv"
data = pd.read_csv(file_path)

# Examine the first few lines of the DataFrame
print(data.head())
```

### Step 2: Creating Passages for Retrieval

1. Transform each row (or group of rows) into text passages describing the foods and their nutritional values.
2. Store these passages in a structure that allows quick retrieval, such as a dictionary or an index.

```python
passages = []
for index, row in data.iterrows():
    passage = f"The food item {row['Food']} contains {row['Calories']} calories, {row['Proteins']}g of proteins, {row['Fats']}g of fats, and {row['Carbohydrates']}g of carbohydrates."
    passages.append(passage)

# Example of a passage
print(passages[0])
```

### Step 3: Indexing and Retrieval

1. Use a retrieval model like TF-IDF or a dense model to index the passages.
2. Install and use Faiss for fast vector-based search (optional but recommended for large datasets).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Use TF-IDF for indexing and retrieval
vectorizer = TfidfVectorizer().fit_transform(passages)
vectorizer_dense = vectorizer.toarray()
```

### Step 4: Text Generation

1. Use a generation model like GPT-3 (or any similar model) to generate responses based on the retrieved information.

```python
from transformers import pipeline

# Load the text generation pipeline (GPT-3 in this case)
generator = pipeline('text-generation', model="gpt-3")

def retrieve_and_generate(query, top_k=5):
    """
    Function to retrieve the most relevant passages and generate a response.
    """
    # Searches in passages using cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectorizer).flatten()
    relevant_indices = similarities.argsort()[-top_k:][::-1]
    
    relevant_passages = [passages[i] for i in relevant_indices]
    context = " ".join(relevant_passages)
    response = generator(f"{query} {context}", max_length=150)
    return response[0]['generated_text']

# Example usage
query = "What are high-protein foods?"
print(retrieve_and_generate(query))
```

### Step 5: Evaluation and Optimization

1. Use quantitative and qualitative metrics to evaluate the relevance and accuracy of the generated responses.
2. Optimize the model and indexing to improve performance.

---

### Key Points:

- Data Preparation: The quality of the data entered as text passages is crucial for effective retrieval.
- Retrieval and Indexing: Use suitable techniques like TF-IDF for smaller datasets or dense retrieval models for larger datasets.
- Text Generation: The generative model should effectively use the context provided by the retrieved passages to generate accurate responses.
- Evaluation: Assess the results and adjust the pipeline accordingly to refine performance and relevance of responses.

This approach will enable you to use RAG to analyze nutritional data efficiently, providing relevant responses based on a combination of retrieval and text generation.

---

### Glossary
- TF-IDF: Term Frequency-Inverse Document Frequency, a statistical measure used to evaluate the importance of a word in a document relative to a corpus.

---
### About the Author

Meet [Barth. Feudong](https://www.linkedin.com/in/barth-feudong/), a talented individual with a passion for combining technology, health and resources optimization. 
With a diploma in Computer Science, Barth. Feudong has honed their skills in data analysis, programming, and software development.

While pursuing their academic interests, Barth. Feudong discovered a parallel interest in nutrition and wellness. 
As they delved deeper into the world of health and fitness, they realized that technology could play a crucial role in empowering individuals to make informed decisions about their well-being.

With this intersection of computer science and health in mind, Dipl.-Ing Barth. Feudong began creating content that bridges the gap between tech-heaviness and 
nutritional know-how. Their expertise lies in crafting accessible and actionable advice for readers seeking to optimize their health and wellness 
through evidence-based recommendations.

When not geeking out over code or exploring the latest advancements in healthcare, Dipl.-Ing Barth. Feudong can be found experimenting with new recipes, 
practicing yoga, or simply enjoying nature.


--- 