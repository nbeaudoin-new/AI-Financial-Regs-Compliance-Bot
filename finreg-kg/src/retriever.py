import numpy as np
import networkx as nx
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")


class Retriever:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.model = load_sentence_transformer()

    def _get_candidate_nodes(self):
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") in ("ENTITY", "CONCEPT")
        ]

    def add_semantic_edges(self, threshold: float = 0.75):
        nodes = self._get_candidate_nodes()
        if len(nodes) < 2:
            return

        embeddings = self.model.encode(nodes, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if sim_matrix[i][j] >= threshold:
                    n1, n2 = nodes[i], nodes[j]
                    if not self.graph.has_edge(n1, n2):
                        self.graph.add_edge(n1, n2, rel="RELATED_TO", weight=float(sim_matrix[i][j]))

    def retrieve_subgraph(self, query: str, top_k: int = 15) -> dict:
        nodes = self._get_candidate_nodes()
        if not nodes:
            return {"nodes": [], "edges": []}

        query_emb = self.model.encode([query], show_progress_bar=False)
        node_embs = self.model.encode(nodes, show_progress_bar=False)
        sims = cosine_similarity(query_emb, node_embs)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        top_nodes = [nodes[i] for i in top_indices]

        # Expand with 1-hop neighbors
        neighbor_nodes = set(top_nodes)
        for node in top_nodes:
            for neighbor in self.graph.neighbors(node):
                neighbor_nodes.add(neighbor)

        # Build subgraph dicts
        result_nodes = []
        for node in neighbor_nodes:
            attrs = dict(self.graph.nodes[node])
            attrs["id"] = node
            # Truncate full_text for context
            if "full_text" in attrs:
                attrs["full_text"] = attrs["full_text"][:500] + "..."
            result_nodes.append(attrs)

        result_edges = []
        subgraph = self.graph.subgraph(neighbor_nodes)
        for u, v, data in subgraph.edges(data=True):
            result_edges.append({
                "source": u,
                "target": v,
                "rel": data.get("rel", ""),
                "weight": float(data.get("weight", 1.0)),
            })

        return {"nodes": result_nodes, "edges": result_edges}
