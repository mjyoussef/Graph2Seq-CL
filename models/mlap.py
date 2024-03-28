import torch
from torch.nn import Module, Linear, Sequential, ReLU
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.glob import AttentionalAggregation
from torch.nn import functional as F
from utils import get_contrastive_graph_pair

class MLAP_GIN(Module):
    def __init__(self, dim_h, batch_size, depth, node_encoder, norm=False, residual=False, dropout=False):
        super(MLAP_GIN, self).__init__()

        self.dim_h = dim_h
        self.batch_size = batch_size
        self.depth = depth
        self.node_encoder = node_encoder
        self.norm = norm
        self.residual = residual
        self.dropout = dropout

        # non-linear projection function for cl task
        self.projection = Sequential(
            Linear(dim_h, int(dim_h/8)),
            ReLU(),
            Linear(int(dim_h/8), dim_h)
        )

        # GIN layers
        self.layers = torch.nn.ModuleList(
            [GINConv(
                Sequential(
                    Linear(dim_h, dim_h),
                    ReLU(),
                    Linear(dim_h, dim_h)
                )
            ) for _ in range(depth)]
        )
            
        # normalization layers
        self.norm = torch.nn.ModuleList([GraphNorm(dim_h) for _ in range(self.depth)])
        
        # layer-wise attention poolings
        self.att_poolings = torch.nn.ModuleList(
            [AttentionalAggregation(
                Sequential(
                    Linear(dim_h, 2*dim_h),
                    ReLU(),
                    Linear(2*dim_h, 1)
                )
            ) for _ in range(depth)]
        )
        
    def contrastive_loss(self, g1_x, g2_x):

        # compute projections + L2 row-wise normalizations
        g1_projections = self.projection(g1_x)
        g1_projections = F.normalize(g1_projections, p=2, dim=1)
        g2_projections = self.projection(g2_x)
        g2_projections = F.normalize(g2_projections, p=2, dim=1)
        
        g1_proj_T = torch.transpose(g1_projections, 0, 1)
        g2_proj_T = torch.transpose(g2_projections, 0, 1)

        inter_g1 = torch.exp(torch.matmul(g1_projections, g1_proj_T))
        inter_g2 = torch.exp(torch.matmul(g2_projections, g2_proj_T))
        intra_view = torch.exp(torch.matmul(g1_projections, g2_proj_T))

        corresponding_terms = torch.diagonal(intra_view, 0) # main diagonal
        non_matching_intra = torch.diagonal(intra_view, -1).sum()
        non_matching_inter_g1 = torch.diagonal(inter_g1, -1).sum()
        non_matching_inter_g2 = torch.diagonal(inter_g2, -1).sum()

        # inter-view pairs using g1
        corresponding_terms_g1 = corresponding_terms / (corresponding_terms + non_matching_inter_g1 + non_matching_intra)
        corresponding_terms_g1 = torch.log(corresponding_terms_g1)

        # inter-view pairs using g2
        corresponding_terms_g2 = corresponding_terms / (corresponding_terms + non_matching_inter_g2 + non_matching_intra)
        corresponding_terms_g2 = torch.log(corresponding_terms_g2)

        loss = (corresponding_terms_g1.sum() + corresponding_terms_g2.sum()) / (g1_x.shape[0] + g2_x.shape[0])
        
        loss = loss / self.batch_size

        return loss
    
    def layer_loop(self, batched_data, cl=False, cl_all=False):

        x = batched_data.x
        edge_index = batched_data.edge_index
        node_depth = batched_data.node_depth
        batch = batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1,))

        cl_embs = []
        for d in range(self.depth):
            x_in = x

            x = self.layers[d](x, edge_index)
            if (self.norm):
                x = self.norm[d](x, batch)
            if (d < self.depth - 1):
                x = F.relu(x)
            if (self.dropout):
                x = F.dropout(x)
            if (self.residual):
                x = x + x_in

            if (not cl):
                h_g = self.att_poolings[d](x, batch)
                self.graph_embs.append(h_g)
                continue

            if ((cl and cl_all) or (cl and (d == self.depth-1))):
                cl_embs += [x]
            
        return cl_embs

    def forward(self, batched_data, cl=False, cl_all=False):

        self.graph_embs = []

        # contrastive learning task
        cl_loss = 0

        if (cl):
            for i in range(int(self.batch_size / 5)):
                g = batched_data.get_example(i)
                g1, g2 = get_contrastive_graph_pair(g)
                g1_data, g2_data = g.clone(), g.clone()

                g1_data.x = g1[0]
                g1_data.edge_index = g1[1]
                g1_embs = self.layer_loop(g1_data, cl=cl, cl_all=cl_all)

                g2_data.x = g2[0]
                g2_data.edge_index = g2[1]
                g2_embs = self.layer_loop(g2_data, cl=cl, cl_all=cl_all)

                batch_cl_loss = 0
                for j in range(len(g1_embs)):
                    batch_cl_loss += self.contrastive_loss(g1_embs[j], g2_embs[j])
                
                batch_cl_loss = batch_cl_loss / len(g1_embs)

                cl_loss = cl_loss + batch_cl_loss

        # non-augmented graph
        # note: this populates self.graph_embs
        self.layer_loop(batched_data)

        agg = self.aggregate()
        self.graph_embs.append(agg)
        output = torch.stack(self.graph_embs, dim=0)

        return output, cl_loss
    
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