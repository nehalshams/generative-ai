import chromadb
chroma_client = chromadb.Client()

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tempfile
import shutil

from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Collections are where you’ll store your embeddings, documents, and any additional metadata.
collection = chroma_client.create_collection(name="my_collection")


# Add some text documents to the collection
documents = [
    { "id": "doc1", "text": "This is a document about hawaii", "metadata": {"source": "web"} },
    { "id": "doc2", "text": "This is a document about machine learning", "metadata": {"source": "web"} },
    { "id": "doc3", "text": "This is a document about cooking", "metadata": {"source": "web"} }
]


# for doc in documents:
#     collection.upsert(
#         ids=doc["id"],
#         documents=doc["text"],
#         metadatas=doc["metadata"]
#     )


# # Query the collection
# # You can query the collection with a list of query texts, and Chroma will return the n most similar results.

# results = collection.query(
#     query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
#     n_results=2 # how many results to return
# )
# print(results)


# ---------------------------------------------------------------




SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]



def chroma_basics():
    with tempfile.TemporaryDirectory() as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents = SAMPLE_DOCS, embedding = embeddings_model, persist_directory = tmpdir
        )

        print(
            f"Vector store created with {vectorstore._collection.count()} documents"
        )


        # perform similarity search
        query = "What is langchain?"
        results = vectorstore.similarity_search(query, k=2)

        print(f"Top 2 results for query '{query}':")
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result.page_content} (source: {result.metadata['source']})")




def similarity_search_with_scores():
    with tempfile.TemporaryDirectory(dir='/') as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        # perform similarity search with scores
        query = "Explain vector stores."
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        print(f"Top 3 results with scores for query '{query}':")
        for i, (doc, score) in enumerate(results_with_scores):
            final_score = 1 / (1 + score)  # Convert distance to similarity
            print(
                f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
            )

        

if __name__ == '__main__':
    chroma_basics()