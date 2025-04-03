import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import spearmanr
import random
import os
from tqdm import tqdm

# Make results reproducible
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class EdgeGNN(nn.Module):
    """
    Graph Neural Network for edge importance ranking as described in the paper.
    """

    def __init__(self, embedding_dim=256, num_layers=5, activation=F.leaky_relu):
        super(EdgeGNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.activation = activation

        # GNN layer weights
        self.weights = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])

        # MLP for each layer as described in the paper
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, 1)
            ) for _ in range(num_layers)
        ])

    def forward(self, edge_features, adj_degree, adj_weight):
        """
        Forward pass of the GNN.

        Args:
            edge_features: Initial edge features [num_edges, embedding_dim]
            adj_degree: Modified edge adjacency matrix based on node degree
            adj_weight: Modified edge adjacency matrix based on edge weight

        Returns:
            Edge importance scores
        """
        # Initialize features
        h_degree = edge_features
        h_weight = edge_features

        s_degree_list = []
        s_weight_list = []

        # Process through GNN layers
        for k in range(self.num_layers):
            # Update features using both adjacency matrices
            h_degree = self.activation(
                torch.matmul(adj_degree, h_degree) @ self.weights[k].weight.t() + self.weights[k].bias)
            h_weight = self.activation(
                torch.matmul(adj_weight, h_weight) @ self.weights[k].weight.t() + self.weights[k].bias)

            # Get scores from each layer
            s_degree = self.mlp_layers[k](h_degree)
            s_weight = self.mlp_layers[k](h_weight)

            s_degree_list.append(torch.abs(s_degree))
            s_weight_list.append(torch.abs(s_weight))

        # Sum scores from all layers
        s_degree_sum = torch.sum(torch.cat(s_degree_list, dim=1), dim=1, keepdim=True)
        s_weight_sum = torch.sum(torch.cat(s_weight_list, dim=1), dim=1, keepdim=True)

        # Final edge importance score (multiplication of the two scores)
        edge_importance = s_degree_sum * s_weight_sum

        return edge_importance


def create_modified_edge_adjacency(G, edge_list, edge_weights=None):
    """
    Create the two modified edge adjacency matrices described in the paper:
    1. Modified based on node degree
    2. Modified based on edge weight

    Args:
        G: NetworkX graph
        edge_list: List of edges in the graph
        edge_weights: Dictionary of edge weights

    Returns:
        adj_degree: Modified edge adjacency matrix based on node degree
        adj_weight: Modified edge adjacency matrix based on edge weight
    """
    num_edges = len(edge_list)

    # Create edge index mapping for faster lookup
    edge_to_idx = {edge: i for i, edge in enumerate(edge_list)}

    # Initialize adjacency matrices
    adj_degree = np.zeros((num_edges, num_edges))
    adj_weight = np.zeros((num_edges, num_edges))

    # Compute the two adjacency matrices
    for i, edge1 in enumerate(edge_list):
        for j, edge2 in enumerate(edge_list):
            # Skip self-loops
            if i == j:
                continue

            # Check if edges are adjacent (share a node)
            if edge1[0] == edge2[0] or edge1[0] == edge2[1] or edge1[1] == edge2[0] or edge1[1] == edge2[1]:
                # Get the shared node
                shared_nodes = set(edge1).intersection(set(edge2))

                if shared_nodes:
                    # Original adjacency is 1 if edges are adjacent
                    original_adj = 1

                    # Modified adjacency based on node degree
                    for node in shared_nodes:
                        node_degree = G.degree(node)
                        adj_degree[i, j] = original_adj / node_degree

                    # Modified adjacency based on edge weight
                    if edge_weights:
                        weight1 = edge_weights.get(edge1, 1.0)
                        weight2 = edge_weights.get(edge2, 1.0)
                        adj_weight[i, j] = original_adj / ((weight1 + weight2) / 2)
                    else:
                        adj_weight[i, j] = original_adj

    return adj_degree, adj_weight


def calculate_edge_betweenness_centrality(G, weight=None):
    """
    Calculate edge betweenness centrality using NetworkX.
    This is used as the ground truth for training.

    Args:
        G: NetworkX graph
        weight: Edge attribute to use as weight

    Returns:
        Edge betweenness centrality dictionary
    """
    return nx.edge_betweenness_centrality(G, weight=weight, normalized=True)


def create_synthetic_graph(graph_type='erdos_renyi', n_nodes=100, p=0.1, m=None, k=4, weight_range=(1, 100)):
    """
    Create a synthetic graph for testing.

    Args:
        graph_type: Type of graph ('erdos_renyi', 'watts_strogatz')
        n_nodes: Number of nodes
        p: Probability of edge creation (for Erdos-Renyi) or rewiring (for Watts-Strogatz)
        m: Number of edges (for Erdos-Renyi variant II)
        k: Mean degree (for Watts-Strogatz)
        weight_range: Range of edge weights

    Returns:
        G: NetworkX graph
        edge_weights: Dictionary of edge weights
    """
    if graph_type == 'erdos_renyi':
        if m is not None:
            # Erdos-Renyi variant II (fixed number of edges)
            G = nx.gnm_random_graph(n_nodes, m)
        else:
            # Erdos-Renyi variant I (probability-based)
            G = nx.erdos_renyi_graph(n_nodes, p)
    elif graph_type == 'watts_strogatz':
        # Watts-Strogatz small-world graph
        G = nx.watts_strogatz_graph(n_nodes, k, p)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Add random weights to edges
    edge_weights = {}
    for edge in G.edges():
        weight = np.random.uniform(weight_range[0], weight_range[1])
        G[edge[0]][edge[1]]['weight'] = weight
        edge_weights[edge] = weight

    return G, edge_weights


def generate_node_embeddings(G, edge_list, embedding_dim=256):
    """
    Generate node embeddings using Node2Vec.
    In a real implementation, we would use PecanPy here, but for
    simplicity, we'll use random embeddings for demonstration.

    Args:
        G: NetworkX graph
        edge_list: List of edges
        embedding_dim: Dimension of embeddings

    Returns:
        Edge embeddings tensor
    """
    # In reality, we would use:
    # from pecanpy import Node2Vec
    # model = Node2Vec(G, dimensions=embedding_dim, walk_length=30, num_walks=200, p=1, q=2)
    # model.train()
    # node_embeddings = {node: model.wv[str(node)] for node in G.nodes()}

    # But for this demo, we'll just use random embeddings
    node_embeddings = {node: np.random.normal(0, 0.1, embedding_dim) for node in G.nodes()}

    # Create edge embeddings by averaging node embeddings
    edge_embeddings = np.zeros((len(edge_list), embedding_dim))
    for i, (u, v) in enumerate(edge_list):
        edge_embeddings[i] = (node_embeddings[u] + node_embeddings[v]) / 2

    return torch.FloatTensor(edge_embeddings)


def margin_ranking_loss(predicted_scores, true_scores, margin=1.0, num_samples=None):
    """
    Margin ranking loss as described in the paper.

    Args:
        predicted_scores: Predicted importance scores
        true_scores: Ground truth importance scores
        margin: Margin for the loss function
        num_samples: Number of edge pairs to sample (for efficiency)

    Returns:
        Loss value
    """
    n_edges = predicted_scores.size(0)

    if num_samples is None:
        num_samples = 20 * n_edges  # As suggested in the paper

    total_loss = 0

    # Sample random pairs of edges
    for _ in range(num_samples):
        i = random.randint(0, n_edges - 1)
        j = random.randint(0, n_edges - 1)

        if i != j:
            # Determine which edge should be ranked higher
            if true_scores[i] > true_scores[j]:
                y = 1
            else:
                y = -1

            # Calculate the loss for this pair
            loss = max(0, -y * (predicted_scores[i] - predicted_scores[j]) + margin)
            total_loss += loss

    return total_loss / num_samples


def train_gnn(G, edge_list, edge_weights, num_epochs=50, embedding_dim=256,
              num_layers=5, learning_rate=0.0005, dropout_ratio=0.3):
    """
    Train the GNN model.

    Args:
        G: NetworkX graph
        edge_list: List of edges
        edge_weights: Dictionary of edge weights
        num_epochs: Number of training epochs
        embedding_dim: Dimension of embeddings
        num_layers: Number of GNN layers
        learning_rate: Learning rate for optimizer
        dropout_ratio: Dropout ratio

    Returns:
        Trained GNN model
    """
    # Calculate ground truth centrality
    true_centrality = calculate_edge_betweenness_centrality(G, weight='weight')
    true_scores = np.array([true_centrality[edge] for edge in edge_list])
    true_scores_tensor = torch.FloatTensor(true_scores).unsqueeze(1).to(device)

    # Generate edge embeddings
    edge_features = generate_node_embeddings(G, edge_list, embedding_dim).to(device)

    # Create modified adjacency matrices
    adj_degree, adj_weight = create_modified_edge_adjacency(G, edge_list, edge_weights)
    adj_degree_tensor = torch.FloatTensor(adj_degree).to(device)
    adj_weight_tensor = torch.FloatTensor(adj_weight).to(device)

    # Initialize model
    model = EdgeGNN(embedding_dim, num_layers).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    losses = []

    for epoch in range(num_epochs):
        # Forward pass
        edge_scores = model(edge_features, adj_degree_tensor, adj_weight_tensor)

        # Calculate loss
        loss = margin_ranking_loss(edge_scores, true_scores_tensor)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model, losses


def evaluate_model(model, G, edge_list, edge_weights, embedding_dim=256):
    """
    Evaluate the trained model.

    Args:
        model: Trained GNN model
        G: NetworkX graph
        edge_list: List of edges
        edge_weights: Dictionary of edge weights
        embedding_dim: Dimension of embeddings

    Returns:
        Spearman correlation coefficient between predicted and true rankings
    """
    # Set model to evaluation mode
    model.eval()

    # Generate edge embeddings
    edge_features = generate_node_embeddings(G, edge_list, embedding_dim).to(device)

    # Create modified adjacency matrices
    adj_degree, adj_weight = create_modified_edge_adjacency(G, edge_list, edge_weights)
    adj_degree_tensor = torch.FloatTensor(adj_degree).to(device)
    adj_weight_tensor = torch.FloatTensor(adj_weight).to(device)

    # Forward pass
    with torch.no_grad():
        predicted_scores = model(edge_features, adj_degree_tensor, adj_weight_tensor)

    # Convert to numpy arrays
    predicted_scores = predicted_scores.cpu().numpy().flatten()

    # Calculate true scores
    true_centrality = calculate_edge_betweenness_centrality(G, weight='weight')
    true_scores = np.array([true_centrality[edge] for edge in edge_list])

    # Calculate Spearman correlation
    correlation, p_value = spearmanr(predicted_scores, true_scores)

    return correlation, p_value, predicted_scores, true_scores


def create_modified_graph_for_disruption(G, disruption_type='edge_weights',
                                         weight_change_range=(0.8, 1.2),
                                         edge_removal_fraction=0.01):
    """
    Create a modified graph to simulate disruptions as described in the paper.

    Args:
        G: Original NetworkX graph
        disruption_type: Type of disruption ('edge_weights' or 'edge_removal')
        weight_change_range: Range for weight modification
        edge_removal_fraction: Fraction of edges to remove

    Returns:
        Modified graph and edge weights
    """
    G_modified = G.copy()
    modified_weights = {(u, v): G[u][v]['weight'] for u, v in G.edges()}

    if disruption_type == 'edge_weights':
        # Simulate minor congestion by changing edge weights
        for u, v in G.edges():
            weight_change = np.random.uniform(weight_change_range[0], weight_change_range[1])
            G_modified[u][v]['weight'] = G[u][v]['weight'] * weight_change
            modified_weights[(u, v)] = G_modified[u][v]['weight']

    elif disruption_type == 'edge_removal':
        # Simulate major disruption by removing edges
        num_edges_to_remove = int(edge_removal_fraction * G.number_of_edges())
        edges_to_remove = random.sample(list(G.edges()), num_edges_to_remove)

        G_modified.remove_edges_from(edges_to_remove)
        for u, v in edges_to_remove:
            if (u, v) in modified_weights:
                del modified_weights[(u, v)]

    return G_modified, modified_weights


def visualize_results(G, edge_list, predicted_scores, true_scores, title="Edge Importance Ranking"):
    """
    Visualize the graph with edges colored by their importance.

    Args:
        G: NetworkX graph
        edge_list: List of edges
        predicted_scores: Predicted importance scores
        true_scores: True importance scores
        title: Plot title
    """
    # Normalize scores for visualization
    predicted_scores_norm = (predicted_scores - predicted_scores.min()) / (
                predicted_scores.max() - predicted_scores.min())
    true_scores_norm = (true_scores - true_scores.min()) / (true_scores.max() - true_scores.min())

    # Create score dictionaries
    predicted_dict = {edge_list[i]: predicted_scores_norm[i] for i in range(len(edge_list))}
    true_dict = {edge_list[i]: true_scores_norm[i] for i in range(len(edge_list))}

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Draw the network with predicted scores
    pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout

    # Draw predicted importance
    edges = nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=[predicted_dict.get(e, 0) for e in G.edges()],
                                   width=3, edge_cmap=plt.cm.viridis)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=50)
    ax1.set_title(f"Predicted Importance")

    # Draw true importance
    edges = nx.draw_networkx_edges(G, pos, ax=ax2, edge_color=[true_dict.get(e, 0) for e in G.edges()],
                                   width=3, edge_cmap=plt.cm.viridis)
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=50)
    ax2.set_title(f"True Importance (EBC)")

    fig.colorbar(edges, ax=[ax1, ax2], label="Normalized Importance Score")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    return fig


# Main demo function
def demo_edge_gnn():
    """
    Demonstrate the Edge-based GNN for critical road ranking.
    """
    print("Generating synthetic graph...")
    G, edge_weights = create_synthetic_graph(
        graph_type='erdos_renyi',
        n_nodes=100,
        p=0.05,
        weight_range=(1, 100)
    )

    edge_list = list(G.edges())
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    print("\nTraining GNN model...")
    model, losses = train_gnn(
        G,
        edge_list,
        edge_weights,
        num_epochs=50,
        embedding_dim=256,
        num_layers=5
    )

    print("\nEvaluating model on original graph...")
    correlation, p_value, predicted_scores, true_scores = evaluate_model(
        model,
        G,
        edge_list,
        edge_weights
    )
    print(f"Spearman correlation: {correlation:.4f} (p-value: {p_value:.4f})")

    # Visualize results
    fig = visualize_results(G, edge_list, predicted_scores, true_scores,
                            title="Edge Importance Ranking - Original Graph")

    # Simulate disruptions
    print("\nSimulating minor disruption (edge weight changes)...")
    G_minor, weights_minor = create_modified_graph_for_disruption(
        G,
        disruption_type='edge_weights',
        weight_change_range=(0.8, 1.2)
    )

    edge_list_minor = list(G_minor.edges())
    correlation_minor, p_value_minor, pred_minor, true_minor = evaluate_model(
        model,
        G_minor,
        edge_list_minor,
        weights_minor
    )
    print(f"Spearman correlation after minor disruption: {correlation_minor:.4f}")

    # Visualize minor disruption results
    fig_minor = visualize_results(G_minor, edge_list_minor, pred_minor, true_minor,
                                  title="Edge Importance Ranking - After Minor Disruption")

    print("\nSimulating major disruption (edge removal)...")
    G_major, weights_major = create_modified_graph_for_disruption(
        G,
        disruption_type='edge_removal',
        edge_removal_fraction=0.01
    )

    edge_list_major = list(G_major.edges())
    correlation_major, p_value_major, pred_major, true_major = evaluate_model(
        model,
        G_major,
        edge_list_major,
        weights_major
    )
    print(f"Spearman correlation after major disruption: {correlation_major:.4f}")

    # Visualize major disruption results
    fig_major = visualize_results(G_major, edge_list_major, pred_major, true_major,
                                  title="Edge Importance Ranking - After Major Disruption")

    print("\nDemo completed!")

    # Save figures
    fig.savefig("original_graph_ranking.png")
    fig_minor.savefig("minor_disruption_ranking.png")
    fig_major.savefig("major_disruption_ranking.png")

    return model


if __name__ == "__main__":
    demo_edge_gnn()