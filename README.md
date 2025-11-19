Document Question-Answering System using LangChain & FAISS
🔹 Introduction

This project demonstrates how to build an AI-powered Document Question-Answering (Doc-QA) system that allows users to upload a file (PDF, DOCX, or TXT) and interact with it like a chatbot.
Using modern tools like:

LangChain

FAISS Vector Database

HuggingFace Embeddings

Transformers (Flan-T5 model)

the system can read documents, convert them into chunks, store them in a vector index, and answer user queries based on document content.

This is an example of Retrieval-Augmented Generation (RAG)—a powerful technique used in AI chatbots, enterprise assistants, and knowledge-based systems.

📌 Objective of the Project

The primary objectives are:

Allow users to upload a PDF/DOCX/TXT file

Extract and preprocess document text

Convert text into embeddings for search

Store embeddings inside a FAISS vector database

Use an LLM to answer user questions based on retrieved chunks

Build an interactive Q&A system

📌 Step-by-Step Project Description
1. Installing Required Libraries

The following libraries are installed:

✓ LangChain

Used for chaining LLMs with retrieval systems.

✓ FAISS

Efficient vector search engine for embedding storage.

✓ pypdf / python-docx

Extracts text from PDF and DOCX files.

✓ sentence-transformers

Used for generating embeddings using the all-MiniLM-L6-v2 model.

✓ transformers

Loads the text generation model (Flan-T5).

These installations prepare the environment for building the full RAG pipeline.

2. Uploading a Document
uploaded = files.upload()


The user uploads any of these:

.pdf

.docx

.txt

The script automatically detects the filetype and chooses the correct loader:

PyPDFLoader for PDF

Docx2txtLoader for DOCX

TextLoader for TXT files

3. Loading and Splitting the Document
🔹 Text Extracting

The loader reads the entire document.

🔹 Text Splitting

Using RecursiveCharacterTextSplitter, the document is broken into small text chunks:

chunk_size = 500

chunk_overlap = 100

Chunking is important because:

LLMs cannot process large documents at once

Small text chunks improve retrieval accuracy

4. Creating Vector Embeddings

The embedding model used:

sentence-transformers/all-MiniLM-L6-v2


✔ Fast
✔ Lightweight
✔ High accuracy for semantic search

Each chunk is converted into an embedding vector.

5. Storing Chunks in FAISS Vector Database
vectorstore = FAISS.from_documents(documents, embeddings)


FAISS enables:

Fast similarity search

High accuracy

Efficient storage

This vectorstore is the “memory” of the chatbot—it tells the model where to search inside the document.

6. Loading the Language Model (LLM)

The model used is:

google/flan-t5-base

Why this model?

Small and fast

Good for short QA tasks

Works well in Google Colab

Supports instruction following

The LLM is wrapped using:

HuggingFacePipeline


so it can be used inside LangChain.

7. Building the RetrievalQA Chain

The workflow:

User asks a question

FAISS retrieves top 3 relevant chunks

LLM generates the answer based on those chunks

This is the core of the Retrieval-Augmented Generation (RAG) pipeline.

8. Interactive Chat Loop

A while loop allows continuous questioning:

Ask a question (or 'exit'):


The model responds using document-based information.

This creates a full AI chatbot capable of:

Summaries

Explanations

Extracting answers

Searching document content

📌 Conclusion

This project successfully implements a complete Document Q&A System using RAG.
It demonstrates:

Document uploading and preprocessing

Text splitting

Embedding creation using SentenceTransformers

Vector search using FAISS

LLM-based answer generation

Real-time interaction with user queries

This system is a practical application of AI used in:

Corporate knowledge assistants

PDF summarizers

Research paper Q&A tools

Legal document assistants

Student study assistants
