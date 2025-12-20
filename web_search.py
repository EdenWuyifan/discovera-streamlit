"""Web search functionality using SerpAPI with Serper fallback."""

import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool


_CUTOFF_DATE = date(2025, 10, 1)


def _parse_result_date(value: Any) -> Optional[date]:
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    # Handle ISO-like timestamps (e.g., 2025-10-01T12:34:56Z)
    if len(text) >= 10 and text[4] in {"-", "/"} and text[7] in {"-", "/"}:
        candidate = text[:10]
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                pass

    # Some providers append separators like "·" after the date.
    if "·" in text:
        text = text.split("·", 1)[0].strip()

    for fmt in (
        "%b %d, %Y",  # Oct 1, 2025
        "%B %d, %Y",  # October 1, 2025
        "%d %b %Y",  # 1 Oct 2025
        "%d %B %Y",  # 1 October 2025
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

    return None


def _is_after_cutoff(result_date: Any) -> bool:
    parsed = _parse_result_date(result_date)
    return parsed is not None and parsed > _CUTOFF_DATE


def _search_serpapi(query: str, num_results: int) -> Dict[str, Any]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY not configured")

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    knowledge_graph = None
    if kg := data.get("knowledge_graph"):
        source = kg.get("source")
        source_name = None
        source_url = None
        if isinstance(source, dict):
            source_name = source.get("name")
            source_url = source.get("link")
        elif isinstance(source, str):
            source_name = source

        knowledge_graph = {
            "title": kg.get("title"),
            "type": kg.get("type"),
            "description": kg.get("description"),
            "source": source_name,
            "source_url": source_url,
        }

    results: List[Dict[str, Any]] = []
    for item in data.get("organic_results", []) or []:
        result = {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
            "date": item.get("date"),
            "position": item.get("position"),
        }
        if _is_after_cutoff(result.get("date")):
            continue
        results.append(result)
        if len(results) >= num_results:
            break

    return {"query": query, "knowledge_graph": knowledge_graph, "results": results}


def _search_serper(query: str, num_results: int) -> Dict[str, Any]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not configured")

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    knowledge_graph = None
    if kg := data.get("knowledgeGraph"):
        knowledge_graph = {
            "title": kg.get("title"),
            "type": kg.get("type"),
            "description": kg.get("description"),
            "source": kg.get("descriptionSource"),
            "source_url": kg.get("descriptionLink"),
        }

    results: List[Dict[str, Any]] = []
    for item in data.get("organic", []) or []:
        result = {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
            "date": item.get("date"),
            "position": item.get("position"),
        }
        if _is_after_cutoff(result.get("date")):
            continue
        results.append(result)
        if len(results) >= num_results:
            break

    return {"query": query, "knowledge_graph": knowledge_graph, "results": results}


@tool
def web_search(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Search the web using Google (via SerpAPI, with Serper fallback) for genes, pathways, diseases, or any biomedical term.

    Filters out results with a parsed date after 2025-10-01.

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
    try:
        return _search_serpapi(query=query, num_results=num_results)
    except Exception as serpapi_error:
        try:
            return _search_serper(query=query, num_results=num_results)
        except Exception as serper_error:
            return {
                "error": f"SerpAPI failed: {serpapi_error}; Serper failed: {serper_error}",
                "query": query,
                "results": [],
            }
