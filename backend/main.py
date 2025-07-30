from fastapi import FastAPI, Request
from graph import build_disaster_graph
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
graph = build_disaster_graph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_query(req: Request):
    data = await req.json()
    query = data.get("query", "")
    result = graph.invoke({"query": query})
    # return {"response": result["response"]}
    return {
        "response": result.get("response"),
        "locations": result.get("locations", []),
        "query": result.get("query")
    }
