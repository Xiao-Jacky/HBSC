import networkx as nx
import torch
import torch.nn as nn
import re
import torch.optim as optim
import os
import glob

torch.manual_seed(42)
def parse_dot_file(file_path):
    with open(file_path, 'r') as file:
        dot_str = file.read()

    graph = nx.DiGraph()
    for line in dot_str.split('\n'):
        if '->' in line:
            match = re.findall(r'(\d+) -> (-?\d+)', line)
            if match:
                src, dst = map(int, match[0])
                graph.add_edge(src, dst) 
                if dst == -1:
                    graph.nodes[src]['stop_block'] = True  
        elif '[label=' in line:
            match = re.findall(r'(\d+)\[label="\[([^]]*)\]"\]', line)
            if match:
                node, label = match[0]
                node = int(node)
                features = torch.tensor([float(x) for x in label.split(', ')])
                if node not in graph:
                    graph.add_node(node)
                graph.nodes[node]['feature'] = features
    return graph

# Structure2Vec 模型
class Structure2Vec(nn.Module):
    def __init__(self, feature_dim, hidden_dim, leaky_relu_slope=0.01):
        super(Structure2Vec, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, graph, num_iterations=3):  
        embeddings = {node: self.leaky_relu(self.fc1(data['feature'])) for node, data in graph.nodes(data=True) if 'feature' in data}
        for _ in range(num_iterations):  
            updated_embeddings = {}
            for node in graph.nodes():
                if 'feature' in graph.nodes[node]:
                    neighbor_feats = [embeddings[nbr] for nbr in graph.neighbors(node) if nbr in embeddings]
                    if neighbor_feats:
                        neighbor_feats = torch.mean(torch.stack(neighbor_feats), 0)
                    else:
                        neighbor_feats = torch.zeros(self.hidden_dim)
                    updated_embeddings[node] = self.leaky_relu(self.fc2(neighbor_feats) + embeddings[node])  # 使用LeakyReLU
            embeddings.update(updated_embeddings)
        return embeddings

def some_loss_function(embeddings, graph):
    loss = 0.0
    for node in graph.nodes():
        if 'feature' in graph.nodes[node]:
            node_embedding = embeddings[node]
            neighbor_embeddings = [embeddings[n] for n in graph.neighbors(node) if 'feature' in graph.nodes[n]]
            if neighbor_embeddings:
                neighbor_embeddings = torch.stack(neighbor_embeddings)
                average_neighbor_embedding = torch.mean(neighbor_embeddings, dim=0)
                loss += (node_embedding - average_neighbor_embedding).pow(2).mean()
    return loss / len(graph.nodes())

def process_dot_files(folder_path):
    file_paths = sorted(glob.glob(os.path.join(folder_path, '*.dot')))
    for file_path in file_paths:
        graph = parse_dot_file(file_path)
        model = Structure2Vec(feature_dim=10, hidden_dim=5)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        embeddings = model(graph)
        mean_embedding = torch.mean(torch.stack(list(embeddings.values())), 0)
        print(f"File: {os.path.basename(file_path)} - Mean Embedding: {mean_embedding}")


