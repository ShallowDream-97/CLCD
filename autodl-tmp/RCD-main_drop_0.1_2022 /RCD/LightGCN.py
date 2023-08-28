import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
from utils import CommonArgParser, construct_local_map

class LightGCNLayer(nn.Module):
    def __init__(self):
        super(LightGCNLayer, self).__init__()

    def forward(self, x, laplacian):
        return torch.sparse.mm(laplacian, x)
    


class LightGCN(nn.Module):
    def __init__(self, num_layers, laplacian):
        super(LightGCN, self).__init__()
        self.gcn_layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])
        self.laplacian = laplacian

    def forward(self, x):
        for gcn in self.gcn_layers:
            x = gcn(x, self.laplacian)
        return x



def load_edges_from_file(file_path):
    edges = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            src, dst = map(int, line.strip().split('\t'))
            edges.append((src, dst))
    return edges

def create_adjacency_matrix(edges, num_nodes):
    # Initialize the adjacency matrix as zeros
    print("Constructing Adjacency Matrix")
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))
    
    # Fill the matrix based on the edges
    for (src, dst) in edges:
        if dst<100:
            print(src)
            print(dst)
        adjacency_matrix[src, dst] = 1
        adjacency_matrix[dst, src] = 1  # This makes the graph undirected

    print("Adjacened Completed!")
    return adjacency_matrix

# Example usage:
u_from_e_edges = load_edges_from_file('../data/ASSIST/graph/u_from_e.txt')
e_from_u_edges = load_edges_from_file('../data/ASSIST/graph/e_from_u.txt')

all_edges = u_from_e_edges + e_from_u_edges

# Assuming you know the total number of nodes
num_nodes = 20737  # Replace this with the actual number of nodes
adj_matrix = create_adjacency_matrix(all_edges, num_nodes)

print(adj_matrix)

def compute_degree_matrix(adj_matrix):
    """Compute the degree matrix of the adjacency matrix."""
    degree = torch.sum(adj_matrix, dim=1)
    degree_matrix = torch.diag(degree)
    return degree_matrix

def normalized_laplacian(adj_matrix, degree_matrix):
    """Compute the normalized Laplacian matrix."""
    degree = torch.sum(adj_matrix, dim=1)
    # Adding a small value for nodes with degree 0
    degree[degree == 0] = 1e-10
    degree_matrix_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree))
    laplacian = torch.eye(adj_matrix.size(0)) - torch.mm(degree_matrix_inv_sqrt, torch.mm(adj_matrix, degree_matrix_inv_sqrt))
    return laplacian

# Example usage:

degree_matrix = compute_degree_matrix(adj_matrix)
normalized_laplacian_matrix = normalized_laplacian(adj_matrix, degree_matrix)

print(normalized_laplacian_matrix)

# Example:
student_features = torch.randn((17746, 123)) # Or some actual feature matrix
exercise_features = torch.randn((2991, 123))
node_features = torch.cat([student_features, exercise_features], dim=0)


# Example node features - You need to provide actual features for your nodes

# Initialize the LightGCN model
input_dim = node_features.shape[1]
model = LightGCN(input_dim, normalized_laplacian_matrix)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # For reconstruction loss, if that's what you're using

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    embeddings = model(node_features)
    
    # Assuming reconstruction loss, you can use other loss definitions based on your requirements
    loss = loss_fn(embeddings, node_features)
    
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
