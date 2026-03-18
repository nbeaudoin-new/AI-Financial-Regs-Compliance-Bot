import re
import spacy
import networkx as nx
import streamlit as st


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_lg")


class GraphBuilder:
    ENTITY_LABELS = {"ORG", "LAW", "GPE", "PERSON", "DATE", "PRODUCT"}

    def __init__(self):
        self.nlp = load_spacy_model()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _get_or_create_node(self, G: nx.Graph, label: str, node_type: str, **attrs) -> str:
        norm = self._normalize(label)
        # Check if normalized version already exists
        for node, data in G.nodes(data=True):
            if self._normalize(node) == norm and data.get("node_type") == node_type:
                return node
        G.add_node(label, node_type=node_type, **attrs)
        return label

    def build_graph(self, doc_data: dict, existing_graph: nx.Graph | None) -> nx.Graph:
        G = existing_graph if existing_graph is not None else nx.Graph()

        filename = doc_data["filename"]
        full_text = doc_data["full_text"]

        # Add DOCUMENT node
        doc_node = self._get_or_create_node(G, filename, "DOCUMENT", filename=filename, full_text=full_text)

        # Process with spaCy (limit to first 100k chars for performance)
        text_sample = full_text[:100000]
        spacy_doc = self.nlp(text_sample)

        entity_nodes = []

        # Extract named entities
        for ent in spacy_doc.ents:
            if ent.label_ in self.ENTITY_LABELS:
                label = ent.text.strip()
                if not label:
                    continue
                ent_node = self._get_or_create_node(G, label, "ENTITY", ent_type=ent.label_)
                # Add mentioned_in attribute
                if "mentioned_in" not in G.nodes[ent_node]:
                    G.nodes[ent_node]["mentioned_in"] = []
                if filename not in G.nodes[ent_node]["mentioned_in"]:
                    G.nodes[ent_node]["mentioned_in"].append(filename)

                # DOCUMENT -> ENTITY edge
                if not G.has_edge(doc_node, ent_node):
                    G.add_edge(doc_node, ent_node, rel="MENTIONS", weight=1.0)

                entity_nodes.append(ent_node)

        # Extract noun chunks as CONCEPT nodes
        seen_concepts = set()
        for chunk in spacy_doc.noun_chunks:
            concept_text = chunk.text.strip().lower()
            if not concept_text or len(concept_text) < 3:
                continue
            # Skip if it's already an entity
            norm_concept = self._normalize(concept_text)
            is_entity = any(
                self._normalize(n) == norm_concept and G.nodes[n].get("node_type") == "ENTITY"
                for n in G.nodes
            )
            if is_entity or norm_concept in seen_concepts:
                continue
            seen_concepts.add(norm_concept)

            concept_node = self._get_or_create_node(G, concept_text, "CONCEPT")
            if "mentioned_in" not in G.nodes[concept_node]:
                G.nodes[concept_node]["mentioned_in"] = []
            if filename not in G.nodes[concept_node]["mentioned_in"]:
                G.nodes[concept_node]["mentioned_in"].append(filename)

            if not G.has_edge(doc_node, concept_node):
                G.add_edge(doc_node, concept_node, rel="MENTIONS", weight=1.0)

        # CO_OCCURS edges: entities appearing in the same sentence
        for sent in spacy_doc.sents:
            sent_ents = [
                ent.text.strip()
                for ent in sent.ents
                if ent.label_ in self.ENTITY_LABELS and ent.text.strip()
            ]
            # Resolve to actual node IDs
            sent_nodes = []
            for ent_text in sent_ents:
                norm = self._normalize(ent_text)
                for node, data in G.nodes(data=True):
                    if self._normalize(node) == norm and data.get("node_type") == "ENTITY":
                        sent_nodes.append(node)
                        break

            for i in range(len(sent_nodes)):
                for j in range(i + 1, len(sent_nodes)):
                    n1, n2 = sent_nodes[i], sent_nodes[j]
                    if G.has_edge(n1, n2):
                        G[n1][n2]["weight"] = G[n1][n2].get("weight", 1) + 1
                    else:
                        G.add_edge(n1, n2, rel="CO_OCCURS", weight=1)

        return G
