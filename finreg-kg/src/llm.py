import anthropic


def query_llm(question: str, subgraph: dict, chat_history: list, api_key: str) -> str:
    # Format subgraph context
    context_lines = ["KNOWLEDGE GRAPH CONTEXT", "Nodes:"]
    for node in subgraph.get("nodes", []):
        node_type = node.get("node_type", "UNKNOWN")
        node_id = node.get("id", "")
        mentioned_in = node.get("mentioned_in", [])
        if mentioned_in:
            context_lines.append(f'- [{node_type}] "{node_id}" (mentioned in: {", ".join(mentioned_in)})')
        else:
            context_lines.append(f'- [{node_type}] "{node_id}"')

    context_lines.append("Edges:")
    for edge in subgraph.get("edges", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        rel = edge.get("rel", "")
        weight = edge.get("weight", 1.0)
        if rel in ("CO_OCCURS", "RELATED_TO"):
            context_lines.append(f'- "{source}" --{rel}--> "{target}" (weight: {weight:.2f})')
        else:
            context_lines.append(f'- "{source}" --{rel}--> "{target}"')

    context = "\n".join(context_lines)

    system_prompt = (
        "You are a financial regulation research assistant with expertise in compliance, "
        "regulatory frameworks, and AI governance. "
        "Use the knowledge graph context provided to answer questions accurately. "
        "When you identify connections across documents, explain the path through the graph. "
        "Be precise about which documents support which claims. "
        "If the graph context does not contain enough information, say so clearly.\n\n"
        + context
    )

    messages = []
    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text
