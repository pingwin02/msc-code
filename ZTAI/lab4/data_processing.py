import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils

def load_data():
    dataset = Planetoid(root='data', name='Cora')
    data = dataset[0]
    return data

def calculate_graph_stats(data):
    print(data)
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    labels = data.y
    
    degree = pyg_utils.degree(edge_index[0], num_nodes)
    mean_degree = degree.float().mean().item()
    max_degree = degree.max().item()
    
    matrix_H_size = data.x.shape
    
    num_classes = torch.unique(labels).size(0)
    
    return mean_degree, max_degree, matrix_H_size, num_classes

if __name__ == "__main__":
    data = load_data()
    stats = calculate_graph_stats(data)
    print(f"Mean Degree: {stats[0]:.2f}, Max Degree: {stats[1]}, H size: {stats[2]}, num_classes: {stats[3]}")


# Mean Degree: Jest to średnia liczba krawędzi wychodzących z każdego wierzchołka w grafie.

# Max Degree: Jest to liczba krawędzi wychodzących z wierzchołka o największym stopniu.

# H size: Określa wymiary macierzy, gdzie wiersze odpowiadają wierzchołkom, a kolumny to cechy tych wierzchołków.

# num_classes: Określa liczbę różnych klas.
