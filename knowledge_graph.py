import os
import spacy
import networkx as nx
import pandas as pd
import glob

# --- Configuration ---
MODELS_DIR = "models"
DATA_DIR = "data"
GRAPH_FILE = os.path.join(MODELS_DIR, "financial_knowledge_graph.gml")

def load_real_documents():
    """
    Loads documents from the news articles and reports in the data directory.
    """
    print("Loading real-world documents from the 'data' directory...")
    documents = []
    
    # 1. Load news articles from CSV files
    news_files = glob.glob(os.path.join(DATA_DIR, "news_*.csv"))
    if not news_files:
        print("Warning: No news files found in 'data' directory. The knowledge graph will be sparse.")
    
    for file in news_files:
        try:
            df = pd.read_csv(file)
            # Combine title and description for richer context, handling potential missing values
            df['full_text'] = df['title'].fillna('') + ". " + df['description'].fillna('')
            documents.extend(df['full_text'].tolist())
        except Exception as e:
            print(f"Error reading or processing {file}: {e}")

    # 2. Load macroeconomic data from text files (e.g., RBI reports)
    macro_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    for file in macro_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print(f"Loaded a total of {len(documents)} documents for graph construction.")
    return documents


def build_knowledge_graph(documents):
    """
    Builds a knowledge graph from a list of documents using spaCy for NER.
    This is a simplified simulation of GraphRAG's principles.
    """
    print("Building knowledge graph from real documents...")
    # Load a pre-trained spaCy model
    # To download: python -m spacy download en_core_web_sm
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
        return None

    G = nx.Graph()
    
    # Increase max_length if documents are very long
    nlp.max_length = 2000000

    for doc_text in documents:
        if not isinstance(doc_text, str) or not doc_text.strip():
            continue

        doc = nlp(doc_text)
        # Focus on more relevant entities for finance
        entities = [ent for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT", "MONEY", "EVENT"]]
        
        # Add entities as nodes
        for ent in entities:
            node_name = ent.text.strip()
            if node_name and not G.has_node(node_name):
                G.add_node(node_name, label=ent.label_)
                
        # Create edges between co-occurring entities in the same sentence
        if len(entities) > 1:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    ent1 = entities[i].text.strip()
                    ent2 = entities[j].text.strip()

                    if not ent1 or not ent2 or ent1 == ent2:
                        continue

                    if not G.has_edge(ent1, ent2):
                        G.add_edge(ent1, ent2, weight=1)
                    else:
                        # Increase weight for more frequent co-occurrence
                        G.edges[ent1, ent2]['weight'] += 1
                        
    print(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def run_graph_build():
    """Runs the graph construction and saves it to a file."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")
        
    # Load documents dynamically instead of using a hardcoded list
    documents = load_real_documents()
    
    if not documents:
        print("No documents were loaded. Aborting knowledge graph creation.")
        return

    graph = build_knowledge_graph(documents)
    if graph:
        nx.write_gml(graph, GRAPH_FILE)
        print(f"Knowledge graph saved to {GRAPH_FILE}")

if __name__ == "__main__":
    # To run this script: python knowledge_graph.py
    run_graph_build()

