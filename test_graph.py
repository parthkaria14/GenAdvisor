# test_graph.py
import matplotlib.pyplot as plt
import networkx as nx
from advanced_rag_system import AdvancedRAGSystem
import time

def visualize_full_graph():
    """
    Initializes the RAG system and plots its entire knowledge graph.
    """
    print("Initializing RAG system to build graph...")
    # Make sure to use enable_redis=False if you're not running Redis
    rag_system = AdvancedRAGSystem(enable_redis=False)
    
    G = rag_system.knowledge_graph
    
    if G.number_of_nodes() == 0:
        print("Graph is empty! Check your _build_knowledge_graph function.")
        return

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Preparing visualization... (this may take a moment for large graphs)")
    
    start_time = time.time()

    # --- Set up colors for different node types ---
    color_map = {
        'stock': 'skyblue',
        'sector': 'lightgreen',
        'news': 'salmon',
        'index': 'gold',
        'market_indicator': 'orange',
        'unknown': 'gray'
    }
    
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        node_colors.append(color_map.get(node_type, 'gray'))

    # --- Set up plot ---
    plt.figure(figsize=(40, 40))  # Increase size for a large graph
    
    # Use a layout that tries to spread nodes
    pos = nx.spring_layout(G, k=0.3, iterations=40)
    
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=800,
        font_size=10,
        font_weight='bold',
        arrows=True,
        edge_color='gray'
    )
    
    plt.title("Full Knowledge Graph Visualization", size=40)
    
    # --- Save the file ---
    output_filename = "full_knowledge_graph.png"
    plt.savefig(output_filename, format="PNG", dpi=100)
    
    end_time = time.time()
    print(f"\nDone! Graph saved to '{output_filename}'")
    print(f"Visualization took {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    visualize_full_graph()