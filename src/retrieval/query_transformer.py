# src/retrieval/query_transformer.py
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

def create_query_transformer(retriever, llm):
    """
    Creates a MultiQueryRetriever to generate multiple queries from different perspectives.
    """
    return MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
    )
