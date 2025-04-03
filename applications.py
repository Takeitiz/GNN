import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from edge_gnn import EdgeGNN, evaluate_model, create_modified_edge_adjacency, generate_node_embeddings


def network_efficiency(G, weight=None):
    """
    Calculate network efficiency as described in the paper.

    Args:
        G: NetworkX graph
        weight: Edge attribute to use as weight

    Returns:
        Network efficiency value
    """
    n = G.number_of_nodes()
    efficiency_sum = 0.0

    # Calculate shortest path lengths between all pairs of nodes
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))

    # Sum reciprocals of distances
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                # If path exists
                if j in path_lengths[i]:
                    distance = path_lengths[i][j]
                    if distance > 0:
                        efficiency_sum += 1.0 / distance

    # Normalize
    if n > 1:
        efficiency = efficiency_sum / (n * (n - 1))
    else:
        efficiency = 0.0

    return efficiency


def resilience_assessment(model, G, edge_list, edge_weights, embedding_dim=256):
    """
    Assess network resilience by identifying critical road segments.

    Args:
        model: Trained GNN model
        G: NetworkX graph
        edge_list: List of edges
        edge_weights: Dictionary of edge weights
        embedding_dim: Dimension of embeddings

    Returns:
        critical_edges: List of critical edges and their importance scores
    """
    # Generate edge embeddings
    import torch
    edge_features = generate_node_embeddings(G, edge_list, embedding_dim)
    device = next(model.parameters()).device
    edge_features = edge_features.to(device)

    # Create modified adjacency matrices
    adj_degree, adj_weight = create_modified_edge_adjacency(G, edge_list, edge_weights)
    adj_degree_tensor = torch.FloatTensor(adj_degree).to(device)
    adj_weight_tensor = torch.FloatTensor(adj_weight).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        edge_scores = model(edge_features, adj_degree_tensor, adj_weight_tensor)

    # Convert to numpy array
    edge_scores = edge_scores.cpu().numpy().flatten()

    # Create list of edges with their importance scores
    critical_edges = [(edge_list[i], edge_scores[i]) for i in range(len(edge_list))]

    # Sort by importance (highest first)
    critical_edges.sort(key=lambda x: x[1], reverse=True)

    return critical_edges


def visualize_critical_edges(G, critical_edges, top_n=10, title="Critical Road Segments"):
    """
    Visualize the top critical edges in the network.

    Args:
        G: NetworkX graph
        critical_edges: List of (edge, score) tuples
        top_n: Number of top critical edges to highlight
        title: Plot title
    """
    plt.figure(figsize=(12, 10))

    # Get top N critical edges
    top_critical = [edge for edge, _ in critical_edges[:top_n]]

    # Create color map for edges
    edge_colors = []
    edge_widths = []

    for edge in G.edges():
        if edge in top_critical or (edge[1], edge[0]) in top_critical:
            edge_colors.append('red')
            edge_widths.append(3.0)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.0)

    # Position nodes
    pos = nx.spring_layout(G, seed=42)

    # Draw the network
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='blue', alpha=0.7)

    # Add labels for top critical edges
    edge_labels = {}
    for i, (edge, score) in enumerate(critical_edges[:top_n]):
        edge_labels[edge] = f"{i + 1}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    plt.title(title)
    plt.axis('off')

    return plt.gcf()


def post_disaster_recovery_sequencing(model, G, disrupted_edges, edge_list, edge_weights, embedding_dim=256):
    """
    Determine the optimal sequence for recovering disrupted edges.
    Implementation of Algorithm 2 from the paper.

    Args:
        model: Trained GNN model
        G: Original NetworkX graph
        disrupted_edges: List of disrupted (removed) edges
        edge_list: List of all edges in original graph
        edge_weights: Dictionary of edge weights
        embedding_dim: Dimension of embeddings

    Returns:
        recovery_sequence: Ordered list of edges for recovery
    """
    # Create a copy of the graph with disrupted edges removed
    G_disrupted = G.copy()
    G_disrupted.remove_edges_from(disrupted_edges)

    recovery_sequence = []
    remaining_disrupted = disrupted_edges.copy()

    # Iterate until all disrupted edges are recovered
    while remaining_disrupted:
        # For each possible recovery option, create a graph with that edge recovered
        recovery_options = []

        for edge in remaining_disrupted:
            # Create a graph with this edge recovered
            G_recovered = G_disrupted.copy()
            G_recovered.add_edge(edge[0], edge[1], weight=edge_weights.get(edge, 1.0))

            # Get the current list of edges
            current_edges = list(G_recovered.edges())

            # Calculate network efficiency
            efficiency = network_efficiency(G_recovered, weight='weight')

            # Calculate edge importance using the model
            _, _, predicted_scores, _ = evaluate_model(model, G_recovered, current_edges,
                                                       {e: G_recovered[e[0]][e[1]]['weight'] for e in current_edges},
                                                       embedding_dim)

            # Get the rank of the recovered edge
            recovered_edge_idx = current_edges.index(edge) if edge in current_edges else -1
            if recovered_edge_idx >= 0:
                edge_rank = predicted_scores[recovered_edge_idx]
            else:
                edge_rank = 0

            recovery_options.append((edge, efficiency, edge_rank))

        # Sort recovery options by network efficiency (primary) and edge rank (secondary)
        recovery_options.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Select the best option
        best_edge = recovery_options[0][0]
        recovery_sequence.append(best_edge)

        # Update the graph and remaining disrupted edges
        G_disrupted.add_edge(best_edge[0], best_edge[1], weight=edge_weights.get(best_edge, 1.0))
        remaining_disrupted.remove(best_edge)

    return recovery_sequence


def visualize_recovery_sequence(G_original, G_disrupted, recovery_sequence,
                                edge_weights, step=None, title="Post-Disaster Recovery Sequence"):
    """
    Visualize the recovery sequence.

    Args:
        G_original: Original NetworkX graph
        G_disrupted: Disrupted NetworkX graph
        recovery_sequence: Ordered list of edges for recovery
        edge_weights: Dictionary of edge weights
        step: Step in the recovery sequence to visualize (None for all steps)
        title: Plot title
    """
    # Position nodes - use the same layout for all visualizations
    pos = nx.spring_layout(G_original, seed=42)

    if step is None:
        # Visualize all steps
        n_steps = len(recovery_sequence)
        n_cols = min(3, n_steps)
        n_rows = (n_steps + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Initial state
        ax = axes[0]
        ax.set_title("Initial Disrupted State")

        nx.draw_networkx_edges(G_disrupted, pos, ax=ax, edge_color='gray', alpha=0.7)
        disrupted_edges = set(G_original.edges()) - set(G_disrupted.edges())

        for edge in disrupted_edges:
            nx.draw_networkx_edges(G_original, pos, edgelist=[edge], ax=ax,
                                   edge_color='red', width=2, style='dashed', alpha=0.7)

        nx.draw_networkx_nodes(G_original, pos, ax=ax, node_size=30, node_color='blue', alpha=0.7)
        ax.axis('off')

        # Recovery steps
        G_current = G_disrupted.copy()

        for i, edge in enumerate(recovery_sequence):
            ax_idx = i + 1
            if ax_idx < len(axes):
                ax = axes[ax_idx]
                ax.set_title(f"Recovery Step {i + 1}: Edge {edge}")

                # Add the recovered edge
                G_current.add_edge(edge[0], edge[1], weight=edge_weights.get(edge, 1.0))

                # Draw current state
                nx.draw_networkx_edges(G_current, pos, ax=ax, edge_color='gray', alpha=0.7)

                # Highlight the newly recovered edge
                nx.draw_networkx_edges(G_current, pos, edgelist=[edge], ax=ax,
                                       edge_color='green', width=3, alpha=1.0)

                # Draw remaining disrupted edges
                remaining_disrupted = set(G_original.edges()) - set(G_current.edges())
                for e in remaining_disrupted:
                    nx.draw_networkx_edges(G_original, pos, edgelist=[e], ax=ax,
                                           edge_color='red', width=2, style='dashed', alpha=0.7)

                nx.draw_networkx_nodes(G_original, pos, ax=ax, node_size=30, node_color='blue', alpha=0.7)
                ax.axis('off')

        # Remove any unused subplots
        for i in range(len(recovery_sequence) + 1, len(axes)):
            fig.delaxes(axes[i])

    else:
        # Visualize a specific step
        fig, ax = plt.subplots(figsize=(12, 10))

        if step == 0:
            # Initial state
            ax.set_title("Initial Disrupted State")
            nx.draw_networkx_edges(G_disrupted, pos, ax=ax, edge_color='gray', alpha=0.7)
            disrupted_edges = set(G_original.edges()) - set(G_disrupted.edges())

            for edge in disrupted_edges:
                nx.draw_networkx_edges(G_original, pos, edgelist=[edge], ax=ax,
                                       edge_color='red', width=2, style='dashed', alpha=0.7)
        else:
            # Recovery state at step
            G_current = G_disrupted.copy()
            for i, edge in enumerate(recovery_sequence[:step]):
                G_current.add_edge(edge[0], edge[1], weight=edge_weights.get(edge, 1.0))

            ax.set_title(f"Recovery Step {step}: Edge {recovery_sequence[step - 1]}")

            # Draw current state
            nx.draw_networkx_edges(G_current, pos, ax=ax, edge_color='gray', alpha=0.7)

            # Highlight the newly recovered edge
            nx.draw_networkx_edges(G_current, pos, edgelist=[recovery_sequence[step - 1]], ax=ax,
                                   edge_color='green', width=3, alpha=1.0)

            # Draw remaining disrupted edges
            remaining_disrupted = set(G_original.edges()) - set(G_current.edges())
            for edge in remaining_disrupted:
                nx.draw_networkx_edges(G_original, pos, edgelist=[edge], ax=ax,
                                       edge_color='red', width=2, style='dashed', alpha=0.7)

        nx.draw_networkx_nodes(G_original, pos, ax=ax, node_size=30, node_color='blue', alpha=0.7)
        ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    return fig


def demo_resilience_assessment(model, G, edge_list, edge_weights):
    """
    Demonstrate the resilience assessment application.

    Args:
        model: Trained GNN model
        G: NetworkX graph
        edge_list: List of edges
        edge_weights: Dictionary of edge weights
    """
    print("Performing resilience assessment...")

    # Identify critical road segments
    critical_edges = resilience_assessment(model, G, edge_list, edge_weights)

    # Print top 10 critical edges
    print("\nTop 10 critical road segments:")
    for i, (edge, score) in enumerate(critical_edges[:10]):
        print(f"{i + 1}. Edge {edge}: Score = {score:.4f}")

    # Visualize critical edges
    fig = visualize_critical_edges(G, critical_edges, top_n=10,
                                   title="Critical Road Segments for Resilience Enhancement")

    # Calculate network efficiency with all edges
    original_efficiency = network_efficiency(G, weight='weight')
    print(f"\nOriginal network efficiency: {original_efficiency:.4f}")

    # Calculate efficiency after removing top critical edges
    G_disrupted = G.copy()
    G_disrupted.remove_edges_from([edge for edge, _ in critical_edges[:5]])
    disrupted_efficiency = network_efficiency(G_disrupted, weight='weight')

    print(f"Network efficiency after removing top 5 critical edges: {disrupted_efficiency:.4f}")
    print(f"Efficiency reduction: {(original_efficiency - disrupted_efficiency) / original_efficiency * 100:.2f}%")

    # Save the figure
    fig.savefig("resilience_assessment.png")

    return critical_edges


def demo_post_disaster_recovery(model, G, edge_list, edge_weights):
    """
    Demonstrate the post-disaster recovery sequencing application.

    Args:
        model: Trained GNN model
        G: NetworkX graph
        edge_list: List of edges
        edge_weights: Dictionary of edge weights
    """
    print("\nSimulating post-disaster recovery scenario...")

    # Identify critical edges as potential disruption targets
    critical_edges = resilience_assessment(model, G, edge_list, edge_weights)

    # Select a subset of edges to simulate disruption (mix of critical and non-critical)
    num_disrupted = min(5, len(edge_list) // 20)  # Disrupt about 5% of edges, max 5

    # Select some critical and some random edges
    disrupted_edges = [edge for edge, _ in critical_edges[:num_disrupted // 2]]

    # Add some random edges
    random_edges = []
    import random
    while len(random_edges) < (num_disrupted - len(disrupted_edges)):
        edge = random.choice(edge_list)
        if edge not in disrupted_edges and (edge[1], edge[0]) not in disrupted_edges:
            random_edges.append(edge)

    disrupted_edges.extend(random_edges)

    # Create disrupted graph
    G_disrupted = G.copy()
    G_disrupted.remove_edges_from(disrupted_edges)

    # Calculate original and disrupted efficiency
    original_efficiency = network_efficiency(G, weight='weight')
    disrupted_efficiency = network_efficiency(G_disrupted, weight='weight')

    print(f"Disrupted {len(disrupted_edges)} edges:")
    for i, edge in enumerate(disrupted_edges):
        print(f"  {i + 1}. Edge {edge}")

    print(f"\nOriginal network efficiency: {original_efficiency:.4f}")
    print(f"Disrupted network efficiency: {disrupted_efficiency:.4f}")
    print(f"Efficiency reduction: {(original_efficiency - disrupted_efficiency) / original_efficiency * 100:.2f}%")

    # Determine optimal recovery sequence
    print("\nDetermining optimal recovery sequence...")
    recovery_sequence = post_disaster_recovery_sequencing(
        model, G, disrupted_edges, edge_list, edge_weights
    )

    print("\nRecovery sequence:")
    for i, edge in enumerate(recovery_sequence):
        print(f"  {i + 1}. Recover Edge {edge}")

        # Calculate efficiency after this recovery
        G_partial = G_disrupted.copy()
        for j in range(i + 1):
            recovered_edge = recovery_sequence[j]
            G_partial.add_edge(recovered_edge[0], recovered_edge[1],
                               weight=edge_weights.get(recovered_edge, 1.0))

        partial_efficiency = network_efficiency(G_partial, weight='weight')
        efficiency_gain = (partial_efficiency - disrupted_efficiency) / (
                    original_efficiency - disrupted_efficiency) * 100

        print(f"     Efficiency after recovery: {partial_efficiency:.4f} ({efficiency_gain:.2f}% of total restoration)")

    # Visualize recovery sequence
    fig = visualize_recovery_sequence(G, G_disrupted, recovery_sequence, edge_weights,
                                      title="Post-Disaster Recovery Sequencing")

    # Save the figure
    fig.savefig("recovery_sequence.png")

    return recovery_sequence


if __name__ == "__main__":
    # This script is designed to be imported and used with a trained model
    print("This script provides applications for resilience assessment and post-disaster recovery.")
    print("Import and use with a trained model to demonstrate these applications.")