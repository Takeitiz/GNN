import os
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np

# Define area
PLACE = "Hanoi, Vietnam"  # Adjust to your specific area

# Output file names
GRAPH_PICKLE = "city_road_network.gpickle"
PLOT_FILE = "city_road_network.png"


def process_osm_data():
    """Process OSM data and convert to NetworkX graph"""
    print(f"Downloading and processing OSM data for: {PLACE}")

    try:
        # Create a fallback graph in case of errors
        G_fallback = nx.grid_graph(dim=[20, 20])
        G_fallback = nx.convert_node_labels_to_integers(G_fallback)
        for u, v in G_fallback.edges():
            G_fallback.edges[u, v]['weight'] = np.random.uniform(10, 100)

        # Try to download the street network
        print("Downloading street network...")
        G = ox.graph_from_place(PLACE, network_type="drive")

        # Convert to undirected graph (creates a new graph)
        print("Converting to undirected graph...")
        G_undirected = nx.Graph()

        # Copy nodes and edges with data
        for node, data in G.nodes(data=True):
            G_undirected.add_node(node, **data)

        # Add edges with travel time as weight
        print("Adding travel time weights...")
        for u, v, data in G.edges(data=True):
            edge_data = data.copy()  # Create a copy of edge data
            if 'length' in edge_data:
                # Convert length (meters) to time (seconds)
                # Assume 30 km/h = 8.33 m/s
                edge_data['weight'] = edge_data['length'] / 8.33
            else:
                edge_data['weight'] = 60  # Default to 1 minute

            G_undirected.add_edge(u, v, **edge_data)

        print(f"Graph has {G_undirected.number_of_nodes()} nodes and {G_undirected.number_of_edges()} edges")

        # Ensure it's connected (take largest component if not)
        if not nx.is_connected(G_undirected):
            print("Graph is not fully connected. Taking largest component...")
            largest_cc = max(nx.connected_components(G_undirected), key=len)
            G_undirected = G_undirected.subgraph(largest_cc).copy()
            print(
                f"Largest component has {G_undirected.number_of_nodes()} nodes and {G_undirected.number_of_edges()} edges")

        # Save the graph
        print(f"Saving graph to {GRAPH_PICKLE}")
        nx.write_gpickle(G_undirected, GRAPH_PICKLE)

        # Create a simple visualization
        print("Creating visualization...")
        plt.figure(figsize=(12, 10))
        pos = {node: (data['x'], data['y']) for node, data in G_undirected.nodes(data=True) if
               'x' in data and 'y' in data}
        nx.draw_networkx_edges(G_undirected, pos, alpha=0.5, width=0.5)
        plt.title(f"Road Network - {PLACE}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=300)

        return G_undirected

    except Exception as e:
        print(f"Error processing OSM data: {str(e)}")

        # Create a small sample graph as fallback
        print("Creating a simple synthetic graph instead...")
        G_sample = nx.grid_graph(dim=[20, 20])
        G_sample = nx.convert_node_labels_to_integers(G_sample)

        # Add weights
        for u, v in G_sample.edges():
            G_sample.edges[u, v]['weight'] = np.random.uniform(10, 100)

        # Save sample graph
        nx.write_gpickle(G_sample, GRAPH_PICKLE)

        # Simple visualization
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G_sample, seed=42)
        nx.draw(G_sample, pos, node_size=10, alpha=0.7)
        plt.title("Sample Road Network (Fallback)")
        plt.savefig(PLOT_FILE, dpi=300)

        return G_sample


if __name__ == "__main__":
    G = process_osm_data()
    print("Processing complete!")