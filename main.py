import arxiv
import faiss
import numpy as np
import os
import openai
import pickle
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_community.adapters.openai import ChatCompletion
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

INDEX_FILE_PATH = "faiss_index.bin"
DOCSTORE_FILE_PATH = "docstore.pkl"
INDEX_TO_DOCSTORE_ID_FILE_PATH = "index_to_docstore_id.pkl"

# Save FAISS index to file
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

# Load FAISS index from file
def load_faiss_index(file_path):
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    else:
        return faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("hello world")))

# Save docstore and index_to_docstore_id to file
def save_docstore_and_mapping(docstore, index_to_docstore_id, docstore_path, mapping_path):
    with open(docstore_path, 'wb') as f:
        pickle.dump(docstore, f)
    with open(mapping_path, 'wb') as f:
        pickle.dump(index_to_docstore_id, f)

# Load docstore and index_to_docstore_id from file
def load_docstore_and_mapping(docstore_path, mapping_path):
    if os.path.exists(docstore_path) and os.path.exists(mapping_path):
        with open(docstore_path, 'rb') as f:
            docstore = pickle.load(f)
        with open(mapping_path, 'rb') as f:
            index_to_docstore_id = pickle.load(f)
    else:
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}
    return docstore, index_to_docstore_id

# Initialize FAISS-based Vector Store
embedding_model = OpenAIEmbeddings()
index = load_faiss_index(INDEX_FILE_PATH)
docstore, index_to_docstore_id = load_docstore_and_mapping(DOCSTORE_FILE_PATH, INDEX_TO_DOCSTORE_ID_FILE_PATH)

vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Fetch research papers from ArXiv
def fetch_papers(query, max_results=5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return list(search.results())

# Summarize research papers
def summarize_paper(paper):
    content = f"Title: {paper.title}\nAbstract: {paper.summary}"
    response = ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Summarize the following research paper."},
                  {"role": "user", "content": content}]
    )
    summary = response["choices"][0]["message"]["content"]
    print(f"Summary for {paper.title}:\n{summary}\n")
    return summary

# Store summaries in FAISS
def store_summary(title, summary):
    doc = Document(page_content=summary, metadata={"title": title})
    vector_store.add_documents([doc])
    save_faiss_index(index, INDEX_FILE_PATH)
    save_docstore_and_mapping(docstore, index_to_docstore_id, DOCSTORE_FILE_PATH, INDEX_TO_DOCSTORE_ID_FILE_PATH)

# Query stored summaries
def search_summaries(query, top_k=3):
    results = vector_store.similarity_search(query, k=top_k)
    return results

# Main Function
def main():
    while True:
        choice = input("Choose an option: (1) Research a new topic (2) Search through saved summaries (3) Exit: ")
        
        if choice == '1':
            query = input("Enter research topic: ")
            papers = fetch_papers(query)
            
            for paper in papers:
                print(f"Summarizing: {paper.title}")
                summary = summarize_paper(paper)
                store_summary(paper.title, summary)
                print(f"Stored Summary for: {paper.title}\n")
            
            print("Stored summaries are ready for retrieval.")
        
        elif choice == '2':
            query = input("Enter search query: ")
            results = search_summaries(query)
            
            print("Search Results:")
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")
        
        elif choice == '3':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()