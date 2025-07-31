

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import chromadb
import os
from typing import TypedDict, Optional
import json
import re

load_dotenv()

# Configuration
CONFIG = {
    "max_results": 10,
    "max_doc_chars": 2000,
    "max_total_chars": 20000,
    "temperature": 0.2
}

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=CONFIG["temperature"],
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize ChromaDB
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = {
        "flood": client.get_or_create_collection("flood_knowledge"),
        "earthquake": client.get_or_create_collection("earthquake_knowledge"),
        "landslide": client.get_or_create_collection("landslide_knowledge")
    }
    print("[INFO] ChromaDB collections initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize ChromaDB: {e}")
    collections = {}

# Define state structure
class DisasterState(TypedDict):
    query: str
    response: str
    locations: list

# Utility function for safe JSON parsing
def safe_json_parse(text: str) -> list:
    """Safely parse JSON from LLM response with fallback strategies."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Initial JSON parse failed. Trying to extract JSON array...")
        # Try to find JSON array pattern
        json_patterns = [
            r'\[[\s\S]*?\]',  # Match [ ... ]
            r'\[.*?\]'        # Simple bracket match
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    continue
        
        print("All JSON parsing attempts failed. Returning empty list.")
        return []

# Load disaster-specific documents
def load_documents():
    """Load disaster documents into ChromaDB collections."""
    if not collections:
        print("[ERROR] Collections not initialized. Skipping document loading.")
        return
        
    base_path = "documents/rag_documents"
    
    if not os.path.exists(base_path):
        print(f"[WARNING] Base path {base_path} does not exist.")
        return
    
    for dtype in ["flood", "earthquake", "landslide"]:
        path = os.path.join(base_path, f"{dtype}.txt")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    if content.strip():
                        collections[dtype].add(
                            documents=[content],
                            ids=[dtype]
                        )
                        print(f"[INFO] Loaded {dtype}.txt successfully.")
                    else:
                        print(f"[WARNING] {dtype}.txt is empty.")
            else:
                print(f"[WARNING] {dtype}.txt not found in {base_path}.")
        except Exception as e:
            print(f"[ERROR] Could not load {dtype}.txt: {e}")

# Load documents on startup
load_documents()

# LLM-powered router
def llm_router(llm, query: str) -> str:
    """Route query to appropriate agent based on content."""
    routing_prompt = f"""
You are an expert disaster response router. Analyze the query and determine which specialized agent should handle it.

AGENTS:
- flood_agent: Handles flooding, inundation, water damage, flash floods, river overflow, drainage issues, flood warnings, evacuation due to water
- earthquake_agent: Handles earthquakes, seismic activity, tremors, aftershocks, structural damage from quakes, earthquake preparedness, tsunami risks
- landslide_agent: Handles landslides, mudslides, rockslides, slope failures, hillside collapses, erosion-related disasters, debris flows
- fallback: Use ONLY when query is clearly unrelated to any disaster type above

KEYWORDS TO CONSIDER:
Flood: water, rain, storm, drainage, river, dam, overflow, inundation, waterlogged, monsoon
Earthquake: shake, tremor, seismic, fault, magnitude, aftershock, building collapse, ground motion
Landslide: slope, hill, mud, rock, debris, erosion, unstable ground, hillside, mountain

RULES:
1. Choose the most specific agent that matches the disaster type
2. If multiple disasters are mentioned, pick the PRIMARY one
3. Only use fallback for non-disaster queries (weather forecasts, general info, etc.)
4. Even vague disaster-related queries should go to the appropriate agent, not fallback

Query: "{query}"

Return ONLY: flood_agent | earthquake_agent | landslide_agent | fallback
"""
   
    try:
        result = llm.invoke(routing_prompt)
        return result.content.strip().lower()
    except Exception as e:
        print(f"[ERROR] Router LLM call failed: {e}")
        return "fallback"

# Routing key function for conditional edges
def routing_key(state: DisasterState) -> str:
    """Determine routing key for conditional edges."""
        
    query = state["query"]
    agent_name = llm_router(llm, query)
    valid_agents = {
        "flood_agent",
        "earthquake_agent",
        "landslide_agent",
        "fallback"
    }
    return agent_name if agent_name in valid_agents else "fallback"

# Router node function
def router_node(state: DisasterState):
    """Router node - passes state through unchanged."""
    return state

# Disaster agent implementations
def flood_agent(state: DisasterState):
    """Handle flood-related queries."""
    return _disaster_agent(
        state,
        dtype="flood",
        prompt_header="You are a helpful assistant answering questions about floods."
    )

def earthquake_agent(state: DisasterState):
    """Handle earthquake-related queries."""
    return _disaster_agent(
        state,
        dtype="earthquake",
        prompt_header="You are a helpful assistant answering questions about earthquakes."
    )

def landslide_agent(state: DisasterState):
    """Handle landslide-related queries."""
    return _disaster_agent(
        state,
        dtype="landslide",
        prompt_header="You are a helpful assistant answering questions about landslides."
    )

def _disaster_agent(state: DisasterState, dtype: str, prompt_header: str):
    """Core disaster agent logic with RAG and location extraction."""
    if not state.get("query"):
        return {**state, "response": "No query provided.", "locations": []}
    
    query = state["query"]
    
    # Check if collection exists
    if dtype not in collections:
        return {**state, "response": f"No knowledge base available for {dtype}.", "locations": []}
    
    collection = collections[dtype]
    
    try:
        # Query the knowledge base
        results = collection.query(query_texts=[query], n_results=CONFIG["max_results"])
        docs = results["documents"][0] if results["documents"] else []
        
        # Build context from documents
        context = ""
        current_len = 0
        for doc in docs:
            if current_len + len(doc) <= CONFIG["max_total_chars"]:
                context += doc[:CONFIG["max_doc_chars"]] + "\n\n"
                current_len += len(doc)
            else:
                remaining = CONFIG["max_total_chars"] - current_len
                if remaining > 0:
                    context += doc[:min(remaining, CONFIG["max_doc_chars"])]
                break
        
        # Generate response
        response_prompt = f"""You are a {dtype} expert. Answer the question with specific factual details about the {dtype} events mentioned.

{context}

Question: {query}
Note : 
   1] Dont explain the context, just answer the question.Just provide the relevant information that should be precise.
   2] Dont mention any sources, just answer the question.
   3] Should be concise and precise and informative.

Provide only the essential facts: dates, locations, casualties, affected populations, and basic technical details. Focus solely on what happened, when, and where. Keep the response brief and factual."""
        response = llm.invoke(response_prompt).content.strip()
        
        # Extract locations
        location_prompt = f"""
Based on the context below, extract all relevant affected locations mentioned.

Context:
{context}

User Query: "{query}"

Return only a JSON array like this and no extra text:
[
  {{
    "name": "Place Name",
    "latitude": 12.34,
    "longitude": 56.78,
    "magnitude": 4.5,
    "total_affected": 30002
  }}
]
"""
        
        locs_raw = llm.invoke(location_prompt).content.strip()
        print(f"[DEBUG] Location extraction for {dtype}: {locs_raw}")
        
        locations = safe_json_parse(locs_raw)
        
        # Enrich locations with color coding
        enriched = []
        for loc in locations:
            if not isinstance(loc, dict):
                continue
                
            total_affected = loc.get("total_affected", 0)
            if isinstance(total_affected, (int, float)):
                if total_affected < 1000:
                    color = "yellow"
                elif total_affected < 10000:
                    color = "orange"
                else:
                    color = "red"
            else:
                color = "gray"
            
            enriched.append({
                **loc,
                "color": color
            })
        
        print(f"[DEBUG] Enriched locations for {dtype}: {enriched}")
        
        return {
            **state,
            "response": response,
            "locations": enriched
        }
        
    except Exception as e:
        print(f"[ERROR] Error in {dtype} agent: {e}")
        return {
            **state,
            "response": f"Sorry, I encountered an error processing your {dtype} query: {str(e)}",
            "locations": []
        }


def fallback(state: DisasterState):
    """Handle queries that don't match any disaster type using web search."""
    if not state.get("query"):
        return {**state, "response": "No query provided."}
    
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return {
                **state,
                "response": "I can help with flood, earthquake, and landslide questions. For other topics, please configure Tavily API key."
            }
        
        # Use TavilySearch directly
        tool = TavilySearch(max_results=2, tavily_api_key=tavily_api_key)
        search_results = tool.invoke({"query": state["query"]})
        
        # Format the results into a response
        if search_results:
            response = llm.invoke(f"""
You are a knowledgeable assistant with expertise across various domains. Answer the following question naturally and confidently as if drawing from your own knowledge and understanding.

Question: {state['query']}

Context information:
{search_results}

Instructions:
- Provide a comprehensive, helpful answer to the question
- Write in first person or as a direct response without mentioning sources
- Be confident and authoritative in your tone
- Structure your response clearly and logically
- If the information is insufficient, acknowledge limitations naturally
- Do not mention searching, looking up information, or external sources
- Respond as if this knowledge is part of your expertise

Answer:
""").content.strip()
        else:
            response = "I couldn't find relevant information for your query."
        
        return {
            **state,
            "response": response,
            "locations": []
        }
        
    except Exception as e:
        print(f"[ERROR] Fallback agent error: {e}")
        return {
            **state,
            "response": "I can help with questions about floods, earthquakes, and landslides. Please ask about one of these disaster types.",
            "locations": []
        }


# Build the LangGraph disaster workflow
def build_disaster_graph():
    """Build and compile the disaster response graph."""
    graph = StateGraph(DisasterState)
    
    # Add node functions
    graph.add_node("router", router_node)
    graph.add_node("flood_agent", flood_agent)
    graph.add_node("earthquake_agent", earthquake_agent)
    graph.add_node("landslide_agent", landslide_agent)
    graph.add_node("fallback", fallback)
    
    # Set entry point and conditional edges
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", routing_key, {
            "flood_agent": "flood_agent",
            "earthquake_agent": "earthquake_agent",
            "landslide_agent": "landslide_agent",
            "fallback": "fallback"
        }
    )
    
    # All agent nodes end after response
    graph.add_edge("flood_agent", END)
    graph.add_edge("earthquake_agent", END)
    graph.add_edge("landslide_agent", END)
    graph.add_edge("fallback", END)
    
    return graph.compile()

