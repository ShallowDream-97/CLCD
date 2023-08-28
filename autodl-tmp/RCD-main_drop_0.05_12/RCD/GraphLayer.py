import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class LightGCNLayer(nn.Module):
    def __init__(self):
        super(LightGCNLayer, self).__init__()

    def forward(self, graph, features):
        # 此处假设graph有一个方法norm_adj，它返回标准化的邻接矩阵。
        # 例如，对于简单的平均，标准化可以是将邻接矩阵的每一行除以其非零元素的数目。
        norm_adj = graph.norm_adj()
        return torch.mm(norm_adj, features)


class LightGCN(nn.Module):
    def __init__(self, num_layers, num_features, num_users, num_items):
        super(LightGCN, self).__init__()

        self.embedding_user = nn.Embedding(num_users, num_features)
        self.embedding_item = nn.Embedding(num_items, num_features)
        self.gcn_layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])

    def forward(self, graph):
        users_embeddings = self.embedding_user.weight
        items_embeddings = self.embedding_item.weight
        all_embeddings = [torch.cat([users_embeddings, items_embeddings], dim=0)]
        
        features = all_embeddings[0]
        for layer in self.gcn_layers:
            features = layer(graph, features)
            all_embeddings.append(features)
        
        # 求所有层的累计和
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1)
        
        users_embeddings, items_embeddings = torch.split(all_embeddings, [num_users, num_items], dim=0)
        return users_embeddings, items_embeddings
