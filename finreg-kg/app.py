import tempfile
import os
import streamlit as st
import networkx as nx
from pyvis.network import Network

from src.extractor import extract_pdf
from src.graph_builder import GraphBuilder
from src.graph_store import save_graph, load_graph
from src.retriever import Retriever
from src.llm import query_llm


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FinReg Knowledge Graph", layout="wide")

# ── API key ────────────────────────────────────────────────────────────────────
api_key = None
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

if not api_key:
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Anthropic API Key", type="password", key="api_key_input")

# ── Session state init ─────────────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


@st.cache_resource
def get_graph_builder():
    return GraphBuilder()


def autoload_data_folder():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.isdir(data_dir):
        return
    pdf_paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_paths:
        return
    builder = get_graph_builder()
    for path in pdf_paths:
        with open(path, "rb") as f:
            file_bytes = f.read()
        doc_data = extract_pdf(file_bytes, os.path.basename(path))
        st.session_state.graph = builder.build_graph(doc_data, st.session_state.graph)
    retriever = Retriever(st.session_state.graph)
    retriever.add_semantic_edges(threshold=0.75)
    st.session_state.retriever = retriever
    st.session_state.data_loaded = True


if not st.session_state.data_loaded:
    with st.spinner("Loading documents from data/ folder..."):
        autoload_data_folder()


def render_pyvis(G: nx.Graph) -> str:
    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut()

    color_map = {
        "DOCUMENT": "#003B71",
        "ENTITY": "#FF6C0C",
        "CONCEPT": "#888888",
    }

    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "CONCEPT")
        color = color_map.get(node_type, "#888888")
        title = f"Type: {node_type}"
        if "mentioned_in" in data:
            title += f"<br>In: {', '.join(data['mentioned_in'])}"
        if "ent_type" in data:
            title += f"<br>Entity type: {data['ent_type']}"
        net.add_node(node, label=node[:40], color=color, title=title)

    for u, v, data in G.edges(data=True):
        rel = data.get("rel", "")
        weight = data.get("weight", 1.0)
        net.add_edge(u, v, title=f"{rel} (w={weight:.1f})", label=rel)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        tmp_path = f.name

    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp_path)
    return html


def graph_stats(G: nx.Graph):
    from collections import Counter
    type_counts = Counter(d.get("node_type", "UNKNOWN") for _, d in G.nodes(data=True))
    rel_counts = Counter(d.get("rel", "UNKNOWN") for _, _, d in G.edges(data=True))
    return type_counts, rel_counts


# ── Layout ─────────────────────────────────────────────────────────────────────
st.title("FinReg Knowledge Graph — Research Assistant")

left_col, right_col = st.columns([0.4, 0.6])

# ─────────────── LEFT: Graph Explorer ─────────────────────────────────────────
with left_col:
    st.header("Knowledge Graph")

    if st.session_state.graph and st.session_state.graph.number_of_nodes() > 0:
        G = st.session_state.graph
        html = render_pyvis(G)
        st.components.v1.html(html, height=520, scrolling=False)

        # Stats
        type_counts, rel_counts = graph_stats(G)
        st.subheader("Graph Stats")
        cols = st.columns(len(type_counts) if type_counts else 1)
        for col, (ntype, count) in zip(cols, type_counts.items()):
            col.metric(ntype, count)

        st.caption("Edge types: " + ", ".join(f"{r}={c}" for r, c in rel_counts.items()))

        # Download
        graph_bytes = save_graph(G)
        st.download_button(
            label="Download Graph (.pkl)",
            data=graph_bytes,
            file_name="finreg_graph.pkl",
            mime="application/octet-stream",
        )
    else:
        st.info("Upload PDFs on the right to build the knowledge graph.")

    # Upload graph
    st.subheader("Upload Saved Graph")
    uploaded_graph = st.file_uploader("Upload .pkl graph file", type=["pkl"], key="graph_upload")
    if uploaded_graph is not None:
        loaded_g = load_graph(uploaded_graph.read())
        st.session_state.graph = loaded_g
        st.session_state.retriever = Retriever(loaded_g)
        st.success(f"Graph loaded: {loaded_g.number_of_nodes()} nodes, {loaded_g.number_of_edges()} edges")
        st.rerun()

# ─────────────── RIGHT: Chat ──────────────────────────────────────────────────
with right_col:
    st.header("Research Assistant")

    # PDF uploader
    uploaded_pdfs = st.file_uploader(
        "Upload financial regulation or AI complaint PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload",
    )

    if uploaded_pdfs:
        builder = get_graph_builder()
        new_entity_count = 0
        for pdf_file in uploaded_pdfs:
            with st.spinner(f"Processing {pdf_file.name}..."):
                doc_data = extract_pdf(pdf_file.read(), pdf_file.name)
                st.session_state.graph = builder.build_graph(doc_data, st.session_state.graph)
                new_entity_count += sum(
                    1 for _, d in st.session_state.graph.nodes(data=True)
                    if d.get("node_type") == "ENTITY"
                )

        with st.spinner("Computing semantic similarity edges..."):
            retriever = Retriever(st.session_state.graph)
            retriever.add_semantic_edges(threshold=0.75)
            st.session_state.retriever = retriever

        st.success(
            f"Processed {len(uploaded_pdfs)} PDF(s). "
            f"Graph has {st.session_state.graph.number_of_nodes()} nodes "
            f"({new_entity_count} entities) and {st.session_state.graph.number_of_edges()} edges."
        )
        st.rerun()

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Chat input (must be top-level, not inside a column) ───────────────────────
user_question = st.chat_input("Ask about the regulations...")

if user_question:
    if not api_key:
        st.error("Please provide an Anthropic API key in the sidebar.")
    elif st.session_state.retriever is None:
        st.error("Please upload at least one PDF first to build the knowledge graph.")
    else:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge graph..."):
                subgraph = st.session_state.retriever.retrieve_subgraph(user_question)
            with st.spinner("Generating answer..."):
                answer = query_llm(
                    user_question,
                    subgraph,
                    st.session_state.chat_history,
                    api_key,
                )
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
