# test_graph.py
import matplotlib.pyplot as plt
import networkx as nx
from advanced_rag_system import AdvancedRAGSystem

def visualize_subgraph(target_node="RELIANCE.NS"):
    """
    Plots a subgraph centered around a single target node.
    """
    print("Initializing RAG system to build graph...")
    rag_system = AdvancedRAGSystem(enable_redis=False)
    
    G = rag_system.knowledge_graph
    
    if not G.has_node(target_node):
        print(f"Error: Node '{target_node}' not found in the graph.")
        print("Available stock nodes:", [n for n, d in G.nodes(data=True) if d.get('type') == 'stock'])
        return

    print(f"Generating subgraph for '{target_node}'...")

    # Create a 2-hop neighborhood (node, its neighbors, and their neighbors)
    nodes_to_plot = set([target_node])
    for neighbor in G.neighbors(target_node):
        nodes_to_plot.add(neighbor)
        for second_hop_neighbor in G.neighbors(neighbor):
            nodes_to_plot.add(second_hop_neighbor)
            
    # Create the subgraph from the selected nodes
    sub_G = G.subgraph(nodes_to_plot)

    # --- Set up colors ---
    color_map = {
        'stock': 'skyblue',
        'sector': 'lightgreen',
        'news': 'salmon',
        'index': 'gold',
        'market_indicator': 'orange',
        'peer_stock': 'lightblue',
        'unknown': 'gray'
    }
    
    node_colors = []
    labels = {}
    for node in sub_G.nodes():
        # Differentiate the target node
        if node == target_node:
            node_colors.append('red')
        else:
            node_type = G.nodes[node].get('type', 'unknown')
            node_colors.append(color_map.get(node_type, 'gray'))
        
        # Shorten news labels to avoid clutter
        if G.nodes[node].get('type') == 'news':
            labels[node] = "News"
        else:
            labels[node] = node
            
    # --- Set up plot ---
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(sub_G, k=0.8)
    
    nx.draw(
        sub_G,
        pos,
        labels=labels, # Use our custom labels
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        font_size=12,
        font_weight='bold',
        edge_color='gray'
    )
    
    plt.title(f"Subgraph for {target_node}", size=20)
    
    # --- Save the file ---
    output_filename = f"subgraph_{target_node}.png"
    plt.savefig(output_filename, format="PNG")
    print(f"Graph saved to '{output_filename}'")
    
    # Or display it directly in a popup window
    # plt.show() 

if __name__ == "__main__":
    # You can change the target node here
    visualize_subgraph(target_node="RELIANCE.NS")
    # visualize_subgraph(target_node="TCS.NS")
    # visualize_subgraph(target_node="NIFTY50")