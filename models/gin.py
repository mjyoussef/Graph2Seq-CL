from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops
from torch.nn import Linear, BatchNorm1d
class GINConv(MessagePassing):
    def __init__(self, dim_h, mlp, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)

        self.mlp = mlp

        self.bn = BatchNorm1d(dim_h)

        self.edge_encoder = Linear(2, dim_h)
    
    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)

        edge_index, _ = remove_self_loops(edge_index)

        output = self.mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return self.bn(output)
    
    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out, x):
        return aggr_out + x

    def __repr__(self):
        return self.__class__.__name__