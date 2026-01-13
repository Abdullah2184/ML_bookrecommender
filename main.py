import os
import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # loads .env into environment variables

# Download latest version of the data set
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
books = pd.read_csv(f"{path}/books.csv")

# Visualizing missing data
plt.figure(figsize=(10, 4))
sns.heatmap(books.isna().transpose(), cbar=False)
plt.xlabel("Rows")
plt.ylabel("Columns with missing values")
plt.tight_layout()
plt.show()

# Creating new features for correlation analysis
books["missing_description"] = books["description"].isna().astype(int)
books["age_of_book"] = 2024 - books["published_year"]


# For correlation heatmap
columns_of_interest = [
    "num_pages",
    "age_of_book",
    "missing_description",
    "average_rating",
]

# Calculate Spearman correlation matrix
correlation_matrix = books[columns_of_interest].corr(method="spearman")

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"label": "Spearman correlation"},
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Extracting the books that have all four columns filled i.e. removing
# books that have any even of the data columns missing from the dataset
valid_books = books.loc[
    books["description"].notna()
    & books["published_year"].notna()
    & books["average_rating"].notna()
    & books["num_pages"].notna()
].copy()

# Alternatively, we can do it this way as well
'''valid_books = books[ ~(books["description"].isna() ) &
       ~(books["published_year"].isna() ) &
      ~(books["average_rating"].isna() ) &
      ~(books["num_pages"].isna() )
      ]
'''


# Here the issue is that "Fiction" has a very large representation, but there are many
# overly-specific categories that have a very low book count.
valid_books["categories"].value_counts().reset_index().sort_values("count", ascending=False)

# Remove the books with excessively short descriptions (this is based on self-analysis)
# since those are useless and save to a new dataset
valid_books_desc = valid_books.loc[
    valid_books["description"].str.len() >= 25
].copy()

# Can view those useless books by running this. Changing the between() parameters allows us
# to check for other ranges as well
# valid_books.loc[valid_books["description"].between(1, 4), "description"]

# Creating a new column that combines title and subtitle for better context
valid_books_desc["title_subtitle"] = np.where(
    valid_books_desc["subtitle"].isna(),
    valid_books_desc["title"],
    valid_books_desc["title"].astype(str) + ": " + valid_books_desc["subtitle"].astype(str),
)

valid_books_desc["tagged_description"] = (
    valid_books_desc["isbn13"].astype(str) + ": " +
    valid_books_desc["description"].astype(str)
)

cleaned_books = valid_books_desc.copy()

# Save the tagged descriptions to a text file for processing (LangChain)
TEXT_FILE = "tagged_description.txt"
cleaned_books["tagged_description"].to_csv(
    TEXT_FILE,
    sep="\n",
    index=False,
    header=False,
)


# Loading the text file and splitting into documents
raw_documents = TextLoader(TEXT_FILE).load()

# Splitting the documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000, # Max tokens for OpenAI Embeddings
    chunk_overlap=0, # No overlap needed for this use case
    separator="\n"
)

documents = text_splitter.split_documents(raw_documents)
print("Sample document:\n", documents[0])


#Creates a vector database that stores like-description books together
db_books = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_books" #Directory to store the DB
)

#Persisting the database to disk
db_books.persist()


#Running this will return the 5 most similar books to the query
#It will return a list of Documnent objects which just have the "tagged description" field
# as the page content, so from that we have to find the corresponding book title
# results_docs = db_books.similarity_search("A thrilling mystery novel", k=5)

#Will filter the books and return the one that matches the isbn13 from the search result
# cleaned_books[cleaned_books["isbn13"] == int(results_docs[0].page_content.split()[0].strip()) ]

#Making it modular
# Function that takes in a query and returns top_n similar books
def semantic_recommender(query: str, top_n: int = 5):
    # Perform similarity search
    results = db_books.similarity_search(query, k=top_n)

    isbns = []
    # Extract ISBNs from the results
    for doc in results:
        try:
            isbn = int(doc.page_content.split(":", 1)[0])
            isbns.append(isbn)
        except ValueError:
            continue

    # Return the corresponding books from the cleaned_books DataFrame
    return cleaned_books.loc[
        cleaned_books["isbn13"].isin(isbns)
    ].head(top_n)


# Zero-shot classification to categorize books into Fiction or Non-Fiction
from transformers import pipeline

pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

book_categories = ["Fiction", "Non-Fiction"]

def classify_book(description: str, labels=book_categories):
    """
    Classifies a book description using zero-shot learning.
    Returns the top predicted label and confidence score.
    """
    result = pipe(
        description,
        candidate_labels=labels,
        multi_label=False
    )

    return {
        "label": result["labels"][0],
        "confidence": result["scores"][0],
    }

'''Example usage
sample_desc = cleaned_books.iloc[0]["description"]
classify_book(sample_desc)
'''
