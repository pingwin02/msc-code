import time
import torch
from torch_geometric.loader import NeighborLoader
from data_processing import load_data
from models import GCN, GAT, MLP
import torch_sparse

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

def run_sampled_training(train_time):
    data = load_data()

    data.adj_t = torch_sparse.SparseTensor.from_edge_index(data.edge_index).t()

    # Common model settings 
    num_layers = 3
    activation = torch.relu
    model_classes = {
        "GCN": GCN,
        "GAT": GAT,
        "MLP": MLP,
    }

    for model_name, ModelClass in model_classes.items():
        print(f"\n--- Training {model_name} with sampling ---")

        # Create model
        model = ModelClass(
            in_channels=data.num_node_features, 
            out_channels=torch.unique(data.y).size(0),
            num_layers=num_layers,
            activation=activation,
            heads=4 if model_name == "GAT" else None,
        )

        # Define NeighborLoader for sampling
        loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            num_neighbors=[15, 10, 5],  # 3-layer neighborhood
            batch_size=32,
            shuffle=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        start = time.time()

        stop_training = False
        n_iters = 0
        while not stop_training:
            for batch in loader: 
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)

                # Only compute loss on original target nodes (first batch_size entries)
                loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
                loss.backward()

                elapsed = time.time() - start
                if elapsed > train_time[model_name]:
                    stop_training = True
                    break  # Exit early before optimizer step

                optimizer.step() 
                n_iters += 1

        acc = test(model, data)
        print(f"{model_name} Training time for {n_iters} iterations (sampling): {elapsed:.4f} seconds")
        print(f"{model_name} Accuracy after {n_iters} iterations (sampling): {acc:.4f}")

if __name__ == "__main__":
    train_time = {"GCN": 0.0222, "GAT": 0.0271, "MLP": 0.0100}
    run_sampled_training(train_time)

# --- Training GCN with sampling ---
# GCN Training time for 5 iterations (sampling): 0.0257 seconds
# GCN Accuracy after 5 iterations (sampling): 0.6380

# --- Training GAT with sampling ---
# GAT Training time for 5 iterations (sampling): 0.0297 seconds
# GAT Accuracy after 5 iterations (sampling): 0.7200

# --- Training MLP with sampling ---
# MLP Training time for 3 iterations (sampling): 0.0127 seconds
# MLP Accuracy after 3 iterations (sampling): 0.3470