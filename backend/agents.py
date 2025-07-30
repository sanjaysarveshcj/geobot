def get_document_by_type(disaster_type, collection, query):
    results = collection.query(query_texts=[query], where={"type": disaster_type}, n_results=1)
    return results['documents'][0][0] if results['documents'] else "No relevant data found."

def build_prompt(disaster_type, doc, query):
    return f"You are an expert in {disaster_type} disasters.\n\nContext:\n{doc}\n\nQuestion:\n{query}"

# def agent_response(llm, disaster_type, query, collection):
#     doc = get_document_by_type(disaster_type, collection, query)
#     prompt = build_prompt(disaster_type, doc, query)
#     return llm.invoke(prompt)

# def agent_response(llm, dtype, query, collection, top_k=5, max_chars_per_doc=1000):
#     results = collection.query(query_texts=[query], n_results=top_k)

#     docs = results['documents'][0] if results['documents'] and results['documents'][0] else []
    
#     # Truncate each document to a safe length (optional)
#     docs = [doc[:max_chars_per_doc] for doc in docs]

#     context = "\n\n".join(docs)         

#     prompt = f"""You are a helpful assistant answering questions about {dtype}s.

# Use the following context to answer the user's question:
# {context}

# User's question: {query}
# Answer:"""

#     try:
#         return llm.invoke(prompt).content.strip()
#     except Exception as e:
#         return f"Error calling LLM: {e}"

def agent_response(llm, dtype, query, collection, top_k=10, max_total_chars=20000):
    results = collection.query(query_texts=[query], n_results=top_k)

    # Flatten and clean documents
    docs = results['documents'][0] if results['documents'] and results['documents'][0] else []
    
    context = ""
    current_len = 0

    for doc in docs:
        doc = doc.strip()
        if not doc:
            continue

        if current_len + len(doc) <= max_total_chars:
            context += doc + "\n\n"
            current_len += len(doc)
        else:
            # Truncate the last one to fit remaining space
            remaining = max_total_chars - current_len
            if remaining > 0:
                context += doc[:remaining]
            break

    prompt = f"""You are a helpful assistant answering questions about {dtype}s.

Use the following context to answer the user's question:
{context}

User's question: {query}
Answer:"""

    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"Error calling LLM: {e}"
