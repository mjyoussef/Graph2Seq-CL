import numpy as np
import torch
from torch_geometric.utils import dropout_edge, degree, to_undirected, scatter, to_networkx
import networkx as nx

class ASTNodeEncoder(torch.nn.Module):
    '''
        Input:
            x: default node feature. the first and second column represents node type and node attributes.
            depth: The depth of the node in the AST.

        Output:
            emb_dim-dimensional vector

    '''
    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        mlp_input = torch.hstack((self.type_encoder(x[:,0]), self.attribute_encoder(x[:,1]), self.depth_encoder(depth)))
        return self.node_mlp(mlp_input)



def get_vocab_mapping(seq_list, num_vocab):
    '''
        Input:
            seq_list: a list of sequences
            num_vocab: vocabulary size
        Output:
            vocab2idx:
                A dictionary that maps vocabulary into integer index.
                Additioanlly, we also index '__UNK__' and '__EOS__'
                '__UNK__' : out-of-vocabulary term
                '__EOS__' : end-of-sentence

            idx2vocab:
                A list that maps idx to actual vocabulary.

    '''

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind = 'stable')[:num_vocab]

    print('Coverage of top {} vocabulary:'.format(num_vocab))
    print(float(np.sum(cnt_list[topvocab]))/np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert(idx == vocab2idx[vocab])

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert(vocab2idx['__EOS__'] == len(idx2vocab) - 1)

    return vocab2idx, idx2vocab

def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim = 0)
    edge_attr_ast_inverse = torch.cat([torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim = 1)


    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1,) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim = 0)
    edge_attr_nextoken = torch.cat([torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim = 1)


    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim = 0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat([edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim = 1)
    data.edge_attr = torch.cat([edge_attr_ast,   edge_attr_ast_inverse, edge_attr_nextoken,  edge_attr_nextoken_inverse], dim = 0)

    return data

def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data 
    '''

    # PyG >= 1.5.0
    seq = data.y

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data

def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]], dtype = torch.long)


def decode_arr_to_seq(arr, idx2vocab):
    '''
        Input: torch 1d array: y_arr
        Output: a sequence of words.
    '''

    eos_idx_list = (arr == len(idx2vocab) - 1).nonzero() # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)] # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))


# ---- CAP functions ----
# from: https://github.com/CRIPAC-DIG/GCA/blob/cd6a9f0cf06c0b8c48e108a6aab743585f6ba6f1/pGRACE/functional.py
# and: https://github.com/CRIPAC-DIG/GCA/blob/cd6a9f0cf06c0b8c48e108a6aab743585f6ba6f1/pGRACE/utils.py

def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    # page rank
    # interesting comment: https://github.com/CRIPAC-DIG/GCA/issues/4
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[drop_mask] = 0.

    return x

def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x

def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())

def graph_perturb(data, drop_scheme='pr'):
  if drop_scheme == 'degree':
      drop_weights = degree_drop_weights(data.edge_index)
      edge_index_ = to_undirected(data.edge_index)
      node_deg = degree(edge_index_[1])
      feature_weights = feature_drop_weights(data.x, node_c=node_deg)
  elif drop_scheme == 'pr':
      drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200)
      node_pr = compute_pr(data.edge_index)
      feature_weights = feature_drop_weights(data.x, node_c=node_pr)
  elif drop_scheme == 'evc':
      drop_weights = evc_drop_weights(data)
      node_evc = eigenvector_centrality(data)
      feature_weights = feature_drop_weights(data.x, node_c=node_evc)
  else:
      feature_weights = torch.ones((data.x.size(1),))
      drop_weights = None
  
  return feature_weights, drop_weights

def drop_edge(data, drop_edge_rate, drop_weights, drop_scheme='pr', drop_edge_weighted_threshold=0.7):
  if drop_scheme == 'uniform':
      return dropout_edge(data.edge_index, p=drop_edge_rate)[0]
  elif drop_scheme in ['degree', 'evc', 'pr']:
      return drop_edge_weighted(
          data.edge_index, 
          drop_weights, 
          p=drop_edge_rate, 
          threshold=drop_edge_weighted_threshold
      )
  else:
      raise Exception(f'undefined drop scheme: {drop_scheme}')

def get_contrastive_graph_pair(data, drop_scheme='pr', drop_feature_rates=(0.7, 0.7), drop_edge_rates=(0.5, 0.5)):
  # use augmentation scheme to determine the weights of each node
  # i.e. pagerank, eigenvector centrality, node degree
  feat_weights, drop_weights = graph_perturb(data, drop_scheme)

  # apply drop edge according to computed features
  dr_e_1, dr_e_2 = drop_edge_rates
  edge_index_1 = drop_edge(data, dr_e_1, drop_weights, drop_scheme)
  edge_index_2 = drop_edge(data, dr_e_2, drop_weights, drop_scheme)

  dr_f_1, dr_f_2 = drop_feature_rates

  if drop_scheme in ['pr', 'degree', 'evc']:
    # graph-aware drop feature
    x_1 = drop_feature_weighted_2(data.x, feat_weights, dr_f_1)
    e_1 = drop_feature_weighted_2(data.edge_attr, feat_weights, dr_f_1)
    
    x_2 = drop_feature_weighted_2(data.x, feat_weights, dr_f_2)
    e_2 = drop_feature_weighted_2(data.edge_attr, feat_weights, dr_f_2)
  else:
    # naive drop feature
    x_1 = drop_feature(data.x, dr_f_1)
    e_1 = drop_feature(data.edge_attr, dr_f_1)
    
    x_2 = drop_feature(data.x, dr_f_2)
    e_2 = drop_feature(data.edge_attr, dr_f_2)

  return (
      # graph 1
      (x_1, edge_index_1, e_1),
      # graph 2
      (x_2, edge_index_2, e_2)
  )
