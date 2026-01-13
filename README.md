# Semantic Book Recommendation System (Learning Project)

This repository is a **code-along learning project** focused on understanding
semantic search, vector databases, and transformer-based NLP models using Python.

The goal is educational rather than production-ready deployment.

---

## Overview

This project builds a semantic recommendation system for books using:

- OpenAI embeddings for semantic similarity
- Chroma as a vector database
- Zero-shot classification with a transformer model
- Exploratory data analysis on a real-world book dataset

Users can query the system with natural language (e.g. *"a thrilling mystery novel"*)
and receive semantically similar books.

---

## Core Functionality

- Download and clean a book metadata dataset
- Analyze missing data and correlations
- Generate semantic embeddings for book descriptions
- Store embeddings in a vector database
- Perform similarity search for recommendations
- Classify books as Fiction / Non-Fiction using zero-shot learning

---

## Technologies Used

- Python
- pandas, numpy, seaborn, matplotlib
- LangChain
- OpenAI Embeddings
- Chroma Vector Database
- HuggingFace Transformers
- KaggleHub

---

## Limitations

This project is intentionally simple and has several known limitations:
No evaluation metrics for recommendation quality
Embeddings are regenerated instead of cached across environments
Zero-shot classification is coarse (Fiction vs Non-Fiction only)
Dataset bias (overrepresentation of certain genres)
Not optimized for scale or production use

----

## Purpose:

This repository exists to:
Learn how semantic search systems work end-to-end
Explore vector databases and transformer models
Understand practical NLP pipelines
It is not intended to be a production system.

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
