import pickle
import networkx as nx


def save_graph(graph: nx.Graph) -> bytes:
    return pickle.dumps(graph)


def load_graph(file_bytes: bytes) -> nx.Graph:
    return pickle.loads(file_bytes)
