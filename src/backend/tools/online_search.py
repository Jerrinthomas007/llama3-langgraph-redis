from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def duckduckgo(query: str) -> str:
    return search.run(query)



