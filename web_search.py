"""Web search functionality using Serper API."""

import os
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool


@tool
def web_search(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Search the web using Google (via Serper API) for genes, pathways, diseases, or any biomedical term.

    Useful for:
    - Finding background information on genes or proteins
    - Discovering disease associations and syndromes
    - Looking up pathway descriptions and mechanisms
    - Finding recent publications or news about biological topics

    Args:
        query: Search query string (e.g., "CTNNB1 syndrome", "p53 pathway cancer").
        num_results: Maximum number of results to return. Defaults to 10.

    Returns:
        Dict with query, optional knowledge_graph, and list of search results.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return {
            "error": "SERPER_API_KEY not configured",
            "query": query,
            "results": [],
        }

    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num_results}

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Parse knowledge graph if present
        knowledge_graph = None
        if kg := data.get("knowledgeGraph"):
            knowledge_graph = {
                "title": kg.get("title"),
                "type": kg.get("type"),
                "description": kg.get("description"),
                "source": kg.get("descriptionSource"),
                "source_url": kg.get("descriptionLink"),
            }

        # Parse organic results
        results: List[Dict[str, Any]] = []
        for item in data.get("organic", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "date": item.get("date"),
                    "position": item.get("position"),
                }
            )

        return {
            "query": query,
            "knowledge_graph": knowledge_graph,
            "results": results,
        }
    except requests.RequestException as e:
        return {"error": str(e), "query": query, "results": []}
