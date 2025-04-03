"""
Edge-based Graph Neural Network for Ranking Critical Road Segments

This script implements the approach described in the paper:
"Edge-based graph neural network for ranking critical road segments in a network"
by Jana et al. (2023)

Usage:
    python main.py --network [synthetic/us_freeway/minnesota] --demo [train/resilience/recovery]
"""

import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# Import modules
from edge_gnn import (
    EdgeGNN,
    train_gnn,
    evaluate_model,
    create_synthetic_graph,
    visualize_results,
    create_modified_graph_for_disruption
)
from network_data_loader import load_road_network
from applications import (
    demo_resilience_assessment,
    demo_post_disaster_recovery
)


# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Edge-based GNN for Critical Road Ranking")
    parser.add_argument("--network", type=str, default="synthetic",
                        choices=["synthetic", "us_freeway", "minnesota", "city"],
                        help="Network to use")
    parser.add_argument("--demo", type=str, default="train",
                        choices=["train", "resilience", "recovery"],
                        help="Demo to run")
    parser.add_argument("--nodes", type=int, default=100,
                        help="Number of nodes for synthetic network")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Dimension of embeddings")
    parser.add_argument("--num_layers", type=int, default=5,
                        help="Number of GNN layers")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate")
    parser.add_argument("--save_model", action="store_true",
                        help="Save the trained model")
    parser.add_argument("--load_model", action="store_true",
                        help="Load a trained model")

    args = parser.parse_args()

    # Set seeds for reproducibility
    set_seeds()

    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Load network
    print(f"\nLoading {args.network} network...")
    G, edge_weights = load_road_network(args.network, args.nodes)
    edge_list = list(G.edges())
    print(f"Network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Model path
    model_path = f"output/{args.network}_edge_gnn.pt"

    # Load or train model
    if args.load_model and os.path.exists(model_path):
        print(f"\nLoading trained model from {model_path}...")
        model = EdgeGNN(args.embedding_dim, args.num_layers).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    else:
        if args.demo == "train" or not os.path.exists(model_path):
            # Train model
            print("\nTraining GNN model...")
            model, losses = train_gnn(
                G,
                edge_list,
                edge_weights,
                num_epochs=args.epochs,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                learning_rate=args.learning_rate
            )

            # Plot training loss
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.title(f"Training Loss - {args.network} Network")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(f"output/{args.network}_training_loss.png")

            # Save model if requested
            if args.save_model:
                print(f"\nSaving model to {model_path}...")
                torch.save(model.state_dict(), model_path)
        else:
            print("Error: No trained model found. Please train a model first or use --load_model flag.")
            return

    # Evaluate model
    print("\nEvaluating model on original graph...")
    correlation, p_value, predicted_scores, true_scores = evaluate_model(
        model,
        G,
        edge_list,
        edge_weights,
        embedding_dim=args.embedding_dim
    )
    print(f"Spearman correlation: {correlation:.4f} (p-value: {p_value:.4f})")

    # Visualize ranking
    fig = visualize_results(G, edge_list, predicted_scores, true_scores,
                            title=f"Edge Importance Ranking - {args.network.title()} Network")
    fig.savefig(f"output/{args.network}_ranking.png")

    # Run demos
    if args.demo == "resilience":
        # Resilience assessment
        critical_edges = demo_resilience_assessment(model, G, edge_list, edge_weights)

    elif args.demo == "recovery":
        # Post-disaster recovery
        recovery_sequence = demo_post_disaster_recovery(model, G, edge_list, edge_weights)

    print("\nDemo completed! Check the output directory for results.")


if __name__ == "__main__":
    main()