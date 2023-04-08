import torch
from torch.nn import Linear, Sequential, ReLU

from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.glob import AttentionalAggregation

from torch.nn import functional as F

class MLAP_GIN(torch.nn.Module):
    def __init__(self, dim_h, depth, node_encoder, norm=False, residual=False):
        super(MLAP_GIN, self).__init__()

        self.dim_h = dim_h
        self.depth = depth

        self.node_encoder = node_encoder

        self.norm = norm
        self.residual = residual

        # GIN layers
        self.layers = torch.nn.ModuleList(
            [GINConv(Sequential(
                Linear(dim_h, dim_h),
                ReLU(),
                Linear(dim_h, dim_h))) for _ in range(depth)])
            
        # normalization layers
        self.norm = torch.nn.ModuleList([GraphNorm(dim_h) for _ in range(self.depth)])
        
        # layer-wise attention poolings
        self.att_poolings = torch.nn.ModuleList(
            [AttentionalAggregation(
                Sequential(Linear(self.dim_h, 2*self.dim_h), 
                           ReLU(), 
                           Linear(2*self.dim_h, 1))) for _ in range(depth)])
        
    def forward(self, batched_data):

        self.graph_embs = []

        x = batched_data.x
        edge_index = batched_data.edge_index
        node_depth = batched_data.node_depth
        batch = batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1,))

        for d in range(self.depth):
            x_in = x

            x = self.layers[d](x, edge_index)
            if (self.norm):
                x = self.norm[d](x, batch)
            if (d < self.depth - 1):
                x = F.relu(x)
            if (self.residual):
                x = x + x_in
            
            h_g = self.att_poolings[d](x, batch)
            self.graph_embs.append(h_g)
        
        agg = self.aggregate()
        self.graph_embs.append(agg)
        output = torch.stack(self.graph_embs, dim=0)
        return output
    
    def aggregate(self):
        pass

class MLAP_Sum(MLAP_GIN):
    def aggregate(self):
        return torch.stack(self.graph_embs, dim=0).sum(dim=0)

class MLAP_Weighted(MLAP_GIN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(torch.ones(self.depth, 1, 1))

    def aggregate(self):
        a = F.softmax(self.weight, dim=0)
        h = torch.stack(self.graph_embs, dim=0)
        return (a * h).sum(dim=0)