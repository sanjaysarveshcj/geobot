def get_document_by_type(disaster_type, collection, query):
    results = collection.query(query_texts=[query], where={"type": disaster_type}, n_results=10)
    print(results['documents'][0][0])
    return results['documents'][0][0] if results['documents'] else "No relevant data found."

def build_prompt(disaster_type, doc, query):
    return f"You are an expert in {disaster_type} disasters.\n\nContext:\n{doc}\n\nQuestion:\n{query}"

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
            # Truncate the last one to fit remaining spacein
            remaining = max_total_chars - current_len
            if remaining > 0:
                context += doc[:remaining]
            break

    prompt = f"""You are a highly knowledgeable expert on {dtype} phenomena. Your understanding comes from extensive research, field observations, and analysis of these natural events across different regions and time periods.

When answering questions, draw upon your comprehensive knowledge to:
- Explain the underlying mechanisms and processes
- Provide historical context and patterns you've observed
- Discuss regional variations and unique characteristics
- Connect individual events to broader climatic or geological trends
- Share insights about frequency, intensity, and seasonal patterns
- Explain the interconnected factors that contribute to these events

Current context for reference:
{context}

Question: {query}

Share your expertise in a conversational, informative manner. Explain not just what happened, but why it's significant, what patterns emerge, and what insights can be drawn from the available information. Write as if you're explaining to an interested colleague who wants to understand the deeper aspects of these phenomena."""

    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"Error calling LLM: {e}"


