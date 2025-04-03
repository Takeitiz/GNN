# Edge-based GNN for Ranking Critical Road Segments

This repository implements the approach described in the paper:
**"Edge-based graph neural network for ranking critical road segments in a network"** by Jana et al. (2023).

## Overview

This implementation provides a Graph Neural Network (GNN) framework to rank critical road segments in transportation networks. It leverages the network structure and travel times to identify the most important road segments for network resilience and post-disaster recovery planning.

## Features

- GNN model for edge importance ranking
- Support for synthetic and real-world transportation networks
- Applications for resilience assessment and post-disaster recovery sequencing
- Visualization tools for network analysis

## Requirements

- Python 3.8+
- PyTorch
- NetworkX
- NumPy
- Matplotlib
- PecanPy (for Node2Vec embeddings)
- tqdm (for progress bars)

## Installation

```bash
# Clone the repository
git clone https://github.com/username/edge-gnn-critical-roads.git
cd edge-gnn-critical-roads

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a model

```bash
python main.py --network synthetic --demo train --epochs 50 --save_model
```

### Resilience assessment

```bash
python main.py --network us_freeway --demo resilience --load_model
```

### Post-disaster recovery planning

```bash
python main.py --network minnesota --demo recovery --load_model
```

## Command Line Arguments

- `--network`: Network to use (`synthetic`, `us_freeway`, or `minnesota`)
- `--demo`: Demo to run (`train`, `resilience`, or `recovery`)
- `--nodes`: Number of nodes for synthetic network (default: 100)
- `--epochs`: Number of training epochs (default: 50)
- `--embedding_dim`: Dimension of embeddings (default: 256)
- `--num_layers`: Number of GNN layers (default: 5)
- `--learning_rate`: Learning rate (default: 0.0005)
- `--save_model`: Save the trained model
- `--load_model`: Load a trained model

## Implementation Details

### Module Structure

- `edge_gnn.py`: Core GNN model for edge importance ranking
- `network_data_loader.py`: Functions to load and preprocess transportation networks
- `applications.py`: Applications for resilience assessment and post-disaster recovery
- `main.py`: Main script to run the demos

### Network Data

The implementation supports:

1. **Synthetic networks**: Random graphs with configurable parameters
2. **US Freeway network**: Interstate highway system (simplified version)
3. **Minnesota road network**: State transportation network

For real-world applications, you can add your own transportation network data.

### GNN Architecture

The GNN model consists of:

- Edge feature embedding using Node2Vec
- Modified edge adjacency matrices (node degree-based and edge weight-based)
- Multiple GNN layers for message passing
- MLP units for scoring at each layer

## Results

The implementation includes visualization tools to analyze:

- Edge importance ranking
- Critical road segments for resilience
- Optimal recovery sequences after disasters

## Extending the Implementation

To use your own transportation network:

1. Prepare your network data in a format compatible with NetworkX
2. Add a loader function to `network_data_loader.py`
3. Update the main script to include your network option

## Citation

If you use this implementation in your research, please cite the original paper:

```
@article{jana2023edge,
  title={Edge-based graph neural network for ranking critical road segments in a network},
  author={Jana, Debasish and Malama, Sven and Narasimhan, Sriram and Taciroglu, Ertugrul},
  journal={PLOS ONE},
  volume={18},
  number={12},
  pages={e0296045},
  year={2023}
}
```

## License

This implementation is provided for research purposes only.