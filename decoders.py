import torch
from torch import nn
from torch.nn import functional as F

from utils import encode_seq_to_arr


class LinearDecoder(torch.nn.Module):
    def __init__(self, dim_h, max_seq_len, vocab2idx, device):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab2idx = vocab2idx

        self.decoders = nn.ModuleList([nn.Linear(dim_h, len(vocab2idx)) for _ in range(max_seq_len)])

    def forward(self, batch_size, layer_reps, labels, training=False):
        return [d(layer_reps[-1]) for d in self.decoders]


class LSTMDecoder(torch.nn.Module):
    def __init__(self, dim_h, max_seq_len, vocab2idx, device):
        super(LSTMDecoder, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.vocab2idx = vocab2idx

        self.lstm = nn.LSTMCell(dim_h, dim_h)
        self.w_hc = nn.Linear(dim_h * 2, dim_h)
        self.layernorm = nn.LayerNorm(dim_h)
        self.vocab_encoder = nn.Embedding(len(vocab2idx), dim_h)
        self.vocab_bias = nn.Parameter(torch.zeros(len(vocab2idx)))

        self.device = device
    
    def forward(self, batch_size, layer_reps, labels, training=False):
        if (training):
            batched_label = torch.vstack([encode_seq_to_arr(label, self.vocab2idx, self.max_seq_len - 1) for label in labels])
            batched_label = torch.hstack((torch.zeros((batch_size, 1), dtype=torch.int64), batched_label))
            true_emb = self.vocab_encoder(batched_label.to(device=self.device))
        
        h_t, c_t = layer_reps[-1].clone(), layer_reps[-1].clone()

        layer_reps = layer_reps.transpose(0,1)
        output = []

        pred_emb = self.vocab_encoder(torch.zeros((batch_size), dtype=torch.int64, device=self.device))
        vocab_mat = self.vocab_encoder(torch.arange(len(self.vocab2idx), dtype=torch.int64, device=self.device))

        for i in range(self.max_seq_len):
            if (training): # teacher forcing
                input = true_emb[:, i]
            else:
                input = pred_emb
            
            h_t, c_t = self.lstm(input, (h_t, c_t))

            a = F.softmax(torch.bmm(layer_reps, h_t.unsqueeze(-1)).squeeze(-1), dim=1)  # (batch_size, L + 1)
            context = torch.bmm(a.unsqueeze(1), layer_reps).squeeze(1)
            pred_emb = torch.tanh(self.layernorm(self.w_hc(torch.hstack((h_t, context)))))  # (batch_size, dim_h)

            # (batch_size, len(vocab)) x max_seq_len
            output.append(torch.matmul(pred_emb, vocab_mat.T) + self.vocab_bias.unsqueeze(0))
        
        return output