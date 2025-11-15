# src/retrieval/web_search.py
from langchain_community.tools.tavily_search import TavilySearchResults
from config.settings import TAVILY_MAX_RESULTS

def get_web_search_tool():
    """Initializes and returns the Tavily web search tool."""
    return TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
