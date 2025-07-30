from langgraph.graph import StateGraph, END
from agents import agent_response
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import chromadb
import os
from typing import TypedDict, Optional
import json
import re
import ast

load_dotenv()

llm = ChatGroq(
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)

# Use the recommended PersistentClient factory method
client = chromadb.PersistentClient(path="./chroma_db")

collections = {
    "flood": client.get_or_create_collection("flood_knowledge"),
    "earthquake": client.get_or_create_collection("earthquake_knowledge"),
    "landslide": client.get_or_create_collection("landslide_knowledge")
}

# Define state structure
class DisasterState(TypedDict):
    query: str
    response: str
    locations: list

# Load disaster-specific documents
def load_documents():
    base_path = "documents/rag_documents"
    for dtype in ["flood", "earthquake", "landslide"]:
        path = os.path.join(base_path, f"{dtype}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                if content.strip():
                    try:
                        collections[dtype].add(
                            documents=[content],
                            ids=[dtype]
                        )
                    except Exception as e:
                        print(f"[WARNING] Could not load {dtype}.txt: {e}")
                else:
                    print(f"[WARNING] {dtype}.txt is empty.")
        else:
            print(f"[WARNING] {dtype}.txt not found in {base_path}.")

load_documents()

# LLM-powered router (returns agent name as string)
def llm_router(llm, query: str) -> str:
    routing_prompt = f"""
You are a router assistant in a disaster response system.

Decide which agent should handle the user's query. Your choices are:
- flood_agent
- earthquake_agent
- landslide_agent
- map_agent (for anything related to map or location)
- fallback (if none match)

Respond with only the agent name from the list above. Do not explain.

User Query: "{query}"
Your Response:"""
    result = llm.invoke(routing_prompt)
    return result.content.strip().lower()

# This function is **only** for determining the routing key for conditional edges
def routing_key(state: DisasterState) -> str:
    query = state["query"]
    agent_name = llm_router(llm, query)
    valid_agents = {
        "flood_agent",
        "earthquake_agent",
        "landslide_agent",
        "fallback"
    }
    return agent_name if agent_name in valid_agents else "fallback"

# Node function: always return state dict!!
def router_node(state: DisasterState):
    return state


def flood_agent(state: DisasterState):
    return _disaster_agent(
        state,
        dtype="flood",
        prompt_header="You are a helpful assistant answering questions about floods."
    )

def earthquake_agent(state: DisasterState):
    return _disaster_agent(
        state,
        dtype="earthquake",
        prompt_header="You are a helpful assistant answering questions about earthquakes."
    )

def landslide_agent(state: DisasterState):
    return _disaster_agent(
        state,
        dtype="landslide",
        prompt_header="You are a helpful assistant answering questions about landslides."
    )

def _disaster_agent(state: DisasterState, dtype: str, prompt_header: str):
    query = state["query"]
    collection = collections[dtype]

    results = collection.query(query_texts=[query], n_results=10)
    docs = results["documents"][0] if results["documents"] else []
    context = "\n".join(doc[:2000] for doc in docs)

    response_prompt = f"""{prompt_header}

Use the following context to answer the user's question:
{context}

User's question: {query}
Answer:"""
    response = llm.invoke(response_prompt).content.strip()

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
    "total_affected": 30002,
  }},
  ...
]
"""


    locs_raw = llm.invoke(location_prompt).content.strip()
    # print("LLM location extraction raw output:", locs_raw)
    try:
        print(locs_raw)
        locations = json.loads(locs_raw)
    except json.JSONDecodeError:
        print("Failed to parse locations as JSON. Trying to extract JSON array...")
        match = re.search(r"\[.*\]", locs_raw, re.DOTALL)
        if match:
            try:
                locations = json.loads(match.group(0))
                # locations = ast.literal_eval(match.group(0))
            except Exception as e:
                print("Still failed to parse locations:", e)
                locations = []
        else:
            locations = []
    # print(locations)

    enriched = []
    for loc in locations:
        total_affected = loc.get("total_affected", 0)
        if total_affected < 1000:
            color = "yellow"
        elif total_affected < 10000:
            color = "orange"
        else:
            color = "red"

        enriched.append({
            **loc,
            "color": color
        })
    print(enriched)

    return {
        **state,
        "response": response,
        "locations": enriched
    }


# Fallback agent
def fallback(state: DisasterState):
    return {
        **state,
        "response": "Sorry, I couldn't understand your query. Please mention 'flood', 'earthquake', or 'landslide'."
    }

# Build the LangGraph disaster workflow
def build_disaster_graph():
    graph = StateGraph(DisasterState)

    # Add node functions - all must return state dict!
    graph.add_node("router", router_node)
    graph.add_node("flood_agent", flood_agent)
    graph.add_node("earthquake_agent", earthquake_agent)
    graph.add_node("landslide_agent", landslide_agent)
    graph.add_node("fallback", fallback)

    # Entry point router, conditional edges
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

