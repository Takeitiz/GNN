import os
import pyrosm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Path to your .osm.pbf file
OSM_FILE = "HN.osm.pbf"  # Replace with your actual file name

# Output file names
GRAPH_PICKLE = "city_road_network.gpickle"
PLOT_FILE = "city_road_network.png"


def process_osm_file():
    print(f"Processing OSM file: {OSM_FILE}")

    # Load OSM data
    osm = pyrosm.OSM(OSM_FILE)

    # Extract the road network
    # Use 'driving' for car roads, or 'all' for all roads
    print("Extracting road network...")
    road_network = osm.get_network(network_type="driving")

    # Convert to NetworkX graph
    print("Converting to NetworkX graph...")
    G = road_network.to_networkx()

    # Make sure the graph is undirected
    G = G.to_undirected()

    # Remove disconnected components, keep only the largest one
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    G = G.subgraph(largest_component).copy()

    # Ensure the graph has weights (using length as proxy for travel time)
    for u, v, data in G.edges(data=True):
        if 'length' in data:
            # Convert length to approximate travel time (in seconds)
            # Assuming average speed of 30 km/h = 8.33 m/s
            travel_time = data['length'] / 8.33
            G[u][v]['weight'] = travel_time
        else:
            # Default value if length is not available
            G[u][v]['weight'] = 60  # Default to 1 minute

    # Print graph info
    print(f"Processed graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Save the graph
    print(f"Saving graph to {GRAPH_PICKLE}")
    nx.write_gpickle(G, GRAPH_PICKLE)

    # Create a simple visualization
    print("Creating visualization...")
    plt.figure(figsize=(12, 10))

    # Get node positions - this can be slow for large networks
    # We'll use a simplified approach with random positions if too large
    if G.number_of_nodes() > 5000:
        pos = {node: (G.nodes[node].get('x', np.random.rand()),
                      G.nodes[node].get('y', np.random.rand()))
               for node in G.nodes()}
    else:
        pos = {node: (G.nodes[node].get('x', 0), G.nodes[node].get('y', 0))
               for node in G.nodes()}

    # Draw the network
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)

    plt.title(f"Road Network - {os.path.basename(OSM_FILE)}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"Visualization saved to {PLOT_FILE}")

    return G


if __name__ == "__main__":
    G = process_osm_file()
    print("Processing complete!")