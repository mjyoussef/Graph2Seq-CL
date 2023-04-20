import torch
from torch.nn import Linear, Sequential, ReLU, ELU, Sigmoid

from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.glob import AttentionalAggregation

from torch.nn import functional as F

from utils import get_contrastive_graph_pair


class DISC(torch.nn.Module):
    def __init__(self, dim_h):
        super(DISC, self).__init__()

        W = torch.empty(dim_h, dim_h)
        torch.nn.init.xavier_normal_(W)

        self.W = torch.nn.Parameter(W)
        self.W.requires_grad = True

        self.sig = Sigmoid()
    
    def forward(self, h, s):
        out = torch.matmul(self.W, s)
        out = torch.matmul(h, out.unsqueeze(-1))
        return self.sig(out)


class MLAP_GIN(torch.nn.Module):
    def __init__(self, dim_h, batch_size, depth, node_encoder, norm=False, residual=False, dropout=False):
        super(MLAP_GIN, self).__init__()

        self.dim_h = dim_h
        self.batch_size = batch_size
        self.depth = depth

        self.node_encoder = node_encoder

        self.norm = norm
        self.residual = residual
        self.dropout = dropout

        self.discriminator = DISC(dim_h)

        # non-linear projection function for cl task
        self.projection = Sequential(
            Linear(dim_h, int(dim_h/8)),
            ELU(),
            Linear(int(dim_h/8), dim_h)
        )

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
        
    def contrastive_loss(self, g1_x, g2_x):

        # compute projections + L2 row-wise normalizations
        g1_projections = self.projection(g1_x)
        g1_projections = torch.nn.functional.normalize(g1_projections, p=2, dim=1)
        g2_projections = self.projection(g2_x)
        g2_projections = torch.nn.functional.normalize(g2_projections, p=2, dim=1)
        
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
    
    def layer_loop(self, x, edge_index, batch, cl=False, cl_all=False, dgi_task=False):

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

            if ((cl and cl_all) or (cl and (d == self.depth-1)) or (dgi_task and (d == self.depth-1))):
                cl_embs += [x]
            
        return cl_embs

    def forward(self, batched_data, cl=False, cl_all=False, dgi_task=False):

        self.graph_embs = []

        # non-augmented graph
        # note: populates self.graph_embs

        node_depth = batched_data.node_depth
        x_emb = self.node_encoder(batched_data.x, node_depth.view(-1,))
        edge_index = batched_data.edge_index
        batch = batched_data.batch

        final_layer_embs = self.layer_loop(x_emb, edge_index, batch, dgi_task=dgi_task)

        agg = self.aggregate()
        self.graph_embs.append(agg)
        output = torch.stack(self.graph_embs, dim=0)

        # # dgi task
        # dgi_loss = 0
        # if (dgi_task):
        #     for i in range(int(self.batch_size / 5)):
        #         g = batched_data.get_example(i)
        #         g_diff = get_contrastive_graph_pair(g, dgi_task=True)
        #         g_diff_data = g.clone()
        #         g_diff_data.x = g_diff[0]
        #         g_diff_data.edge_index = g_diff[1]

        #         final_layer_embs = final_layer_embs[0]
        #         g_diff_embs = self.layer_loop(g_diff_data, dgi_task=True)[0]

        #         # dgi objective on final_layer_embs, g_diff_embs, and output
        #         agg = agg.clone()
        #         positive = torch.log(self.discriminator(final_layer_embs, agg[i]))
        #         negative = torch.log(1. - self.discriminator(g_diff_embs, agg[i]))

        #         dgi_loss += (positive.sum() + negative.sum()) / (positive.shape[0] + negative.shape[0])
            
        #     dgi_loss /= int(self.batch_size / 5)

        # contrastive learning task
        cl_loss = 0

        if (cl):
            for i in range(int(self.batch_size / 5)):
                g = batched_data.get_example(i)
                g_clone = g.clone()
                nd = g.node_depth
                g_clone.x = self.node_encoder(g_clone.x, nd.view(-1,).clone())
                g1, g2 = get_contrastive_graph_pair(g_clone)

                b1 = g.batch
                g1_x = g1[0].clone()
                g1_edge_index = g1[1]
                g1_embs = self.layer_loop(g1_x, g1_edge_index, b1, cl=cl, cl_all=cl_all)

                b2 = g.batch
                g2_x = g2[0].clone()
                g2_edge_index = g2[1]
                g2_embs = self.layer_loop(g2_x, g2_edge_index, b2, cl=cl, cl_all=cl_all)

                batch_cl_loss = 0
                for j in range(len(g1_embs)):
                    batch_cl_loss += self.contrastive_loss(g1_embs[j], g2_embs[j])
                    pass
                
                batch_cl_loss = batch_cl_loss / len(g1_embs)

                cl_loss = cl_loss + batch_cl_loss
            
            cl_loss /= int(self.batch_size / 5)

        return output, cl_loss, 0
    
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