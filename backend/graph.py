# from langgraph.graph import StateGraph, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from chromadb.api.types import EmbeddingFunction
# from langchain_tavily import TavilySearch
# from dotenv import load_dotenv
# import chromadb
# import os
# from typing import TypedDict
# import json
# import re

# load_dotenv()

# CONFIG = {
#     "max_results": 10,
#     "max_doc_chars": 2000,
#     "max_total_chars": 20000,
#     "temperature": 0.2
# }

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-pro",
#     temperature=CONFIG["temperature"],
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )


# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )


# class LangChainEmbeddingWrapper(EmbeddingFunction):
#     def __init__(self, embeddings):
#         self.embeddings = embeddings

#     def __call__(self, texts):
#         return self.embeddings.embed_documents(list(texts))

# embedding_fn = LangChainEmbeddingWrapper(embeddings)

# try:
#     client = chromadb.PersistentClient(path="./chroma_db")
#     collections = {
#         "flood": client.get_or_create_collection(
#             "flood_knowledge", embedding_function=embedding_fn
#         ),
#         "earthquake": client.get_or_create_collection(
#             "earthquake_knowledge", embedding_function=embedding_fn
#         ),
#         "landslide": client.get_or_create_collection(
#             "landslide_knowledge", embedding_function=embedding_fn
#         ),
#     }
#     print("[INFO] ChromaDB collections initialized successfully.")
# except Exception as e:
#     print(f"[ERROR] Failed to initialize ChromaDB: {e}")
#     collections = {}



# class DisasterState(TypedDict):
#     query: str
#     response: str
#     locations: list

# def safe_json_parse(text: str) -> list:
#     """Safely parse JSON from LLM response with fallback strategies."""
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         print("Initial JSON parse failed. Trying to extract JSON array...")
#         json_patterns = [
#             r'\[[\s\S]*?\]', 
#             r'\[.*?\]'   
#         ]
        
#         for pattern in json_patterns:
#             matches = re.findall(pattern, text, re.DOTALL)
#             for match in matches:
#                 try:
#                     parsed = json.loads(match)
#                     if isinstance(parsed, list):
#                         return parsed
#                 except:
#                     continue
        
#         print("All JSON parsing attempts failed. Returning empty list.")
#         return []

# def load_documents():
#     """Load disaster documents into ChromaDB collections."""
#     if not collections:
#         print("[ERROR] Collections not initialized. Skipping document loading.")
#         return
        
#     base_path = "documents/rag_documents"
    
#     if not os.path.exists(base_path):
#         print(f"[WARNING] Base path {base_path} does not exist.")
#         return
    
#     for dtype in ["flood", "earthquake", "landslide"]:
#         collection = collections[dtype]
#         if collection.count() > 0:
#             print(f"{dtype} collection already populated, skipping.")
#             continue
#         path = os.path.join(base_path, f"{dtype}.txt")
#         try:
#             if os.path.exists(path):
#                 with open(path, "r", encoding="utf-8") as file:
#                     content = file.read()
#                     if content.strip():
#                         try:
#                             docs = [content]
#                             collections[dtype].add(
#                                 documents=docs,
#                                 ids=[dtype]
#                             )
#                         except Exception as e:
#                             print(f"[WARNING] Could not load {dtype}.txt: {e}")
#                     else:
#                         print(f"[WARNING] {dtype}.txt is empty.")
#             else:
#                 print(f"[WARNING] {dtype}.txt not found in {base_path}.")
#         except Exception as e:
#             print(f"[ERROR] Could not load {dtype}.txt: {e}")

# load_documents()

# def llm_router(llm, query: str) -> str:
#     routing_prompt = f"""
# You are an expert disaster response router. Analyze the query and determine which specialized agent should handle it.

# AGENTS:
# - flood_agent: Handles flooding, inundation, water damage, flash floods, river overflow, drainage issues, flood warnings, evacuation due to water
# - earthquake_agent: Handles earthquakes, seismic activity, tremors, aftershocks, structural damage from quakes, earthquake preparedness, tsunami risks
# - landslide_agent: Handles landslides, mudslides, rockslides, slope failures, hillside collapses, erosion-related disasters, debris flows
# - fallback: Use ONLY when query is clearly unrelated to any disaster type above

# KEYWORDS TO CONSIDER:
# Flood: water, rain, storm, drainage, river, dam, overflow, inundation, waterlogged, monsoon
# Earthquake: shake, tremor, seismic, fault, magnitude, aftershock, building collapse, ground motion
# Landslide: slope, hill, mud, rock, debris, erosion, unstable ground, hillside, mountain

# RULES:
# 1. Choose the most specific agent that matches the disaster type
# 2. If multiple disasters are mentioned, pick the PRIMARY one
# 3. Only use fallback for non-disaster queries (weather forecasts, general info, etc.)
# 4. Even vague disaster-related queries should go to the appropriate agent, not fallback

# Query: "{query}"

# Return ONLY: flood_agent | earthquake_agent | landslide_agent | fallback
# """
   
#     try:
#         result = llm.invoke(routing_prompt)
#         return result.content.strip().lower()
#     except Exception as e:
#         print(f"[ERROR] Router LLM call failed: {e}")
#         return "fallback"

# def routing_key(state: DisasterState) -> str:
#     """Determine routing key for conditional edges."""
        
#     query = state["query"]
#     agent_name = llm_router(llm, query)
#     valid_agents = {
#         "flood_agent",
#         "earthquake_agent",
#         "landslide_agent",
#         "fallback"
#     }
#     return agent_name if agent_name in valid_agents else "fallback"

# def router_node(state: DisasterState):
#     return state

# def flood_agent(state: DisasterState):
#     return _disaster_agent(
#         state,
#         dtype="flood",
#         prompt_header="You are a helpful assistant answering questions about floods."
#     )

# def earthquake_agent(state: DisasterState):
#     return _disaster_agent(
#         state,
#         dtype="earthquake",
#         prompt_header="You are a helpful assistant answering questions about earthquakes."
#     )

# def landslide_agent(state: DisasterState):
#     return _disaster_agent(
#         state,
#         dtype="landslide",
#         prompt_header="You are a helpful assistant answering questions about landslides."
#     )
# from langchain.retrievers import MultiQueryRetriever
# from langchain.vectorstores.base import VectorStore
# from langchain.schema import Document

# class CollectionVectorStoreWrapper(VectorStore):
#     def __init__(self, collection):
#         self.collection = collection
    
#     def similarity_search(self, query: str, k: int = 4, **kwargs):
#         results = self.collection.query(query_texts=[query], n_results=k)
#         docs = []
#         if results["documents"]:
#             for i, doc_content in enumerate(results["documents"][0]):
#                 metadata = {}
#                 if results.get("metadatas") and len(results["metadatas"]) > 0:
#                     metadata = results["metadatas"][0][i] if i < len(results["metadatas"][0]) else {}
#                 docs.append(Document(page_content=doc_content, metadata=metadata))
#         return docs
    
#     def add_texts(self, texts, metadatas=None, **kwargs):
#         pass
    
#     @classmethod
#     def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
#         pass

# def _disaster_agent(state: DisasterState, dtype: str, prompt_header: str):
#     if not state.get("query"):
#         return {**state, "response": "No query provided.", "locations": []}
    
#     query = state["query"]
    
#     if dtype not in collections:
#         return {**state, "response": f"No knowledge base available for {dtype}.", "locations": []}
    
#     collection = collections[dtype]
    
#     try:
#         # Use wrapper with MultiQueryRetriever
#         vector_store = CollectionVectorStoreWrapper(collection)
#         multi_query_retriever = MultiQueryRetriever.from_llm(
#             retriever=vector_store.as_retriever(search_kwargs={"k": CONFIG["max_results"]}),
#             llm=llm
#         )
        
#         # Get documents using MultiQueryRetriever
#         retrieved_docs = multi_query_retriever.get_relevant_documents(query)
#         docs = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs]

#         context = ""
#         current_len = 0
#         for doc in docs:
#             if current_len + len(doc) <= CONFIG["max_total_chars"]:
#                 context += doc[:CONFIG["max_doc_chars"]] + "\n\n"
#                 current_len += len(doc)
#             else:
#                 remaining = CONFIG["max_total_chars"] - current_len
#                 if remaining > 0:
#                     context += doc[:min(remaining, CONFIG["max_doc_chars"])]
#                 break
        
#         response_prompt = f"""You are a {dtype} expert. Answer the question with specific factual details about the {dtype} events mentioned.

# {context}

# Question: {query}
# Note : 
#    1] Dont explain the context, just answer the question.Just provide the relevant information that should be precise.
#    2] Dont mention any sources, just answer the question.
#    3] Should be concise and precise and informative.

# Provide only the essential facts: dates, locations, casualties, affected populations, and basic technical details. Focus solely on what happened, when, and where. Keep the response brief and factual."""
#         response = llm.invoke(response_prompt).content.strip()
        
#         location_prompt = f"""
# Based on the context below, extract all relevant affected locations mentioned.

# Context:
# {context}

# User Query: "{query}"

# Return only a JSON array like this and no extra text:
# [
#   {{
#     "name": "Place Name",
#     "latitude": 12.34,
#     "longitude": 56.78,
#     "magnitude": 4.5,
#     "total_affected": 30002
#   }}
# ]
# """
        
#         locs_raw = llm.invoke(location_prompt).content.strip()
#         print(f"[DEBUG] Location extraction for {dtype}: {locs_raw}")
        
#         locations = safe_json_parse(locs_raw)
        
#         enriched = []
#         for loc in locations:
#             if not isinstance(loc, dict):
#                 continue
                
#             total_affected = loc.get("total_affected", 0)
#             if isinstance(total_affected, (int, float)):
#                 if total_affected < 1000:
#                     color = "yellow"
#                 elif total_affected < 10000:
#                     color = "orange"
#                 else:
#                     color = "red"
#             else:
#                 color = "gray"
            
#             enriched.append({
#                 **loc,
#                 "color": color
#             })
        
#         print(f"[DEBUG] Enriched locations for {dtype}: {enriched}")
        
#         return {
#             **state,
#             "response": response,
#             "locations": enriched
#         }
        
#     except Exception as e:
#         print(f"[ERROR] Error in {dtype} agent: {e}")
#         return {
#             **state,
#             "response": f"Sorry, I encountered an error processing your {dtype} query: {str(e)}",
#             "locations": []
#         }

# def fallback(state: DisasterState):
#     """Handle queries that don't match any disaster type using web search."""
#     if not state.get("query"):
#         return {**state, "response": "No query provided."}
    
#     try:
#         tavily_api_key = os.getenv("TAVILY_API_KEY")
#         if not tavily_api_key:
#             return {
#                 **state,
#                 "response": "I can help with flood, earthquake, and landslide questions. For other topics, please configure Tavily API key."
#             }
        
#         tool = TavilySearch(max_results=2, tavily_api_key=tavily_api_key)
#         search_results = tool.invoke({"query": state["query"]})
        
#         if search_results:
#             response = llm.invoke(f"""
# You are a knowledgeable assistant with expertise across various domains. Answer the following question naturally and confidently as if drawing from your own knowledge and understanding.

# Question: {state['query']}

# Context information:
# {search_results}

# Instructions:
# - Provide a comprehensive, helpful answer to the question
# - Write in first person or as a direct response without mentioning sources
# - Be confident and authoritative in your tone
# - Structure your response clearly and logically
# - If the information is insufficient, acknowledge limitations naturally
# - Do not mention searching, looking up information, or external sources
# - Respond as if this knowledge is part of your expertise

# Answer:
# """).content.strip()
#         else:
#             response = "I couldn't find relevant information for your query."
        
#         return {
#             **state,
#             "response": response,
#             "locations": []
#         }
        
#     except Exception as e:
#         print(f"[ERROR] Fallback agent error: {e}")
#         return {
#             **state,
#             "response": "I can help with questions about floods, earthquakes, and landslides. Please ask about one of these disaster types.",
#             "locations": []
#         }


# def build_disaster_graph():
#     """Build and compile the disaster response graph."""
#     graph = StateGraph(DisasterState)
    
#     graph.add_node("router", router_node)
#     graph.add_node("flood_agent", flood_agent)
#     graph.add_node("earthquake_agent", earthquake_agent)
#     graph.add_node("landslide_agent", landslide_agent)
#     graph.add_node("fallback", fallback)
    
#     graph.set_entry_point("router")
#     graph.add_conditional_edges(
#         "router", routing_key, {
#             "flood_agent": "flood_agent",
#             "earthquake_agent": "earthquake_agent",
#             "landslide_agent": "landslide_agent",
#             "fallback": "fallback"
#         }
#     )
    
#     graph.add_edge("flood_agent", END)
#     graph.add_edge("earthquake_agent", END)
#     graph.add_edge("landslide_agent", END)
#     graph.add_edge("fallback", END)
    
#     return graph.compile()

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.api.types import EmbeddingFunction
from langchain_tavily import TavilySearch
from langchain.retrievers import MultiQueryRetriever
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from dotenv import load_dotenv
import chromadb
import os
from typing import TypedDict
import json
import re

load_dotenv()

CONFIG = {
    "max_results": 10,
    "max_doc_chars": 2000,
    "max_total_chars": 20000,
    "temperature": 0.2
}

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=CONFIG["temperature"],
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class LangChainEmbeddingWrapper(EmbeddingFunction):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, texts):
        return self.embeddings.embed_documents(list(texts))

embedding_fn = LangChainEmbeddingWrapper(embeddings)

try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = {
        "flood": client.get_or_create_collection(
            "flood_knowledge", embedding_function=embedding_fn
        ),
        "earthquake": client.get_or_create_collection(
            "earthquake_knowledge", embedding_function=embedding_fn
        ),
        "landslide": client.get_or_create_collection(
            "landslide_knowledge", embedding_function=embedding_fn
        ),
    }
    print("[INFO] ChromaDB collections initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize ChromaDB: {e}")
    collections = {}

class DisasterState(TypedDict):
    query: str
    response: str
    locations: list

def safe_json_parse(text: str) -> list:
    """Safely parse JSON from LLM response with fallback strategies."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Initial JSON parse failed. Trying to extract JSON array...")
        json_patterns = [
            r'\[[\s\S]*?\]', 
            r'\[.*?\]'   
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
        collection = collections[dtype]
        if collection.count() > 0:
            print(f"{dtype} collection already populated, skipping.")
            continue
        path = os.path.join(base_path, f"{dtype}.txt")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    if content.strip():
                        try:
                            docs = [content]
                            collections[dtype].add(
                                documents=docs,
                                ids=[dtype]
                            )
                        except Exception as e:
                            print(f"[WARNING] Could not load {dtype}.txt: {e}")
                    else:
                        print(f"[WARNING] {dtype}.txt is empty.")
            else:
                print(f"[WARNING] {dtype}.txt not found in {base_path}.")
        except Exception as e:
            print(f"[ERROR] Could not load {dtype}.txt: {e}")

load_documents()

def llm_router(llm, query: str) -> str:
    routing_prompt = f"""
You are an expert disaster response router. Analyze the query and determine which specialized agent should handle it.

AGENTS:
- flood_agent: Handles flooding, inundation, water damage, flash floods, river overflow, drainage issues, flood warnings, evacuation due to water
- earthquake_agent: Handles earthquakes, seismic activity, tremors, aftershocks, structural damage from quakes, earthquake preparedness, tsunami risks
- landslide_agent: Handles landslides, mudslides, rockslides, slope failures, hillside collapses, erosion-related disasters, debris flows
- fallback: Use ONLY when query is clearly unrelated to any flood,landslide and earthquake type above


RULES:
1. Choose the most specific agent that matches the disaster type
2. If multiple disasters are mentioned, pick the fallback one
3. Only use fallback for non-disaster queries (weather forecasts, general info, etc.)

Query: "{query}"

Return ONLY: flood_agent | earthquake_agent | landslide_agent | fallback
"""
   
    try:
        result = llm.invoke(routing_prompt)
        return result.content.strip().lower()
    except Exception as e:
        print(f"[ERROR] Router LLM call failed: {e}")
        return "fallback"

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

def router_node(state: DisasterState):
    return state

class CollectionVectorStoreWrapper(VectorStore):
    def __init__(self, collection):
        self.collection = collection
    
    def similarity_search(self, query: str, k: int = 4, **kwargs):
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = []
        if results["documents"]:
            for i, doc_content in enumerate(results["documents"][0]):
                # Initialize metadata as empty dict to avoid None values
                metadata = {}
                if results.get("metadatas") and len(results["metadatas"]) > 0:
                    if i < len(results["metadatas"][0]) and results["metadatas"][0][i] is not None:
                        metadata = results["metadatas"][0][i]
                
                docs.append(Document(page_content=doc_content, metadata=metadata))
        return docs
    
    def add_texts(self, texts, metadatas=None, **kwargs):
        pass
    
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        pass

def _disaster_agent(state: DisasterState, dtype: str, prompt_header: str):
    if not state.get("query"):
        return {**state, "response": "No query provided.", "locations": []}
    
    query = state["query"]
    
    if dtype not in collections:
        return {**state, "response": f"No knowledge base available for {dtype}.", "locations": []}
    
    collection = collections[dtype]
    
    try:
        # Use wrapper with MultiQueryRetriever
        vector_store = CollectionVectorStoreWrapper(collection)
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            llm=llm
        )
        
        # Get documents using MultiQueryRetriever
        retrieved_docs = multi_query_retriever.get_relevant_documents(query)
        docs = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs]

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
        
        response_prompt = f"""You are a {dtype} expert. Answer the question with specific factual details about the {dtype} events mentioned.

{context}

Question: {query}
Note : 
   1] Dont explain the context very long, just answer the question and brief a little bit .Just provide the relevant information that should be precise.
   2] Do not mention searching, looking up information, or external sources
   3] Respond as if this knowledge is part of your expertise
   4] Should be concise and precise and informative.

Provide only the essential facts: dates, locations, casualties, affected populations, and basic technical details. Focus solely on what happened, when, and where. Keep the response brief and factual."""
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
    "total_affected": 30002
  }}
]
"""
        
        locs_raw = llm.invoke(location_prompt).content.strip()
        print(f"[DEBUG] Location extraction for {dtype}: {locs_raw}")
        
        locations = safe_json_parse(locs_raw)
        
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
        
        tool = TavilySearch(max_results=2, tavily_api_key=tavily_api_key)
        search_results = tool.invoke({"query": state["query"]})
        
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

def build_disaster_graph():
    """Build and compile the disaster response graph."""
    graph = StateGraph(DisasterState)
    
    graph.add_node("router", router_node)
    graph.add_node("flood_agent", flood_agent)
    graph.add_node("earthquake_agent", earthquake_agent)
    graph.add_node("landslide_agent", landslide_agent)
    graph.add_node("fallback", fallback)
    
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", routing_key, {
            "flood_agent": "flood_agent",
            "earthquake_agent": "earthquake_agent",
            "landslide_agent": "landslide_agent",
            "fallback": "fallback"
        }
    )
    
    graph.add_edge("flood_agent", END)
    graph.add_edge("earthquake_agent", END)
    graph.add_edge("landslide_agent", END)
    graph.add_edge("fallback", END)
    
    return graph.compile()
