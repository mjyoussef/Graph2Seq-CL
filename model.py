import torch
from mlap import MLAP_Weighted
from decoders import LSTMDecoder
from utils import get_contrastive_graph_pair
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, ELU

class Model(torch.nn.Module):
    def __init__(self, batch_size, depth, dim_h, max_seq_len, node_encoder, vocab2idx, device):
        super(Model, self).__init__()
        
        self.batch_size = batch_size

        self.depth = depth
        self.dim_h = dim_h
        self.max_seq_len = max_seq_len

        self.node_encoder = node_encoder

        self.vocab2idx = vocab2idx

        self.device = device
        
        self.gnn = MLAP_Weighted(dim_h, batch_size, depth, node_encoder, norm=True, residual=True, dropout=True)

        self.decoder = LSTMDecoder(dim_h, max_seq_len, vocab2idx, device)

    def forward(self, batched_data, labels, training=False, cl=False, cl_all=False):

        embeddings, cl_loss = self.gnn(batched_data, cl=cl, cl_all=cl_all)
        predictions = self.decoder(self.batch_size, embeddings, labels, training=training)

        # for each batch, the prediction for the ith word is a logit
        # decoding each prediction to a word is done in the evaluation task in main

        return predictions, cl_loss