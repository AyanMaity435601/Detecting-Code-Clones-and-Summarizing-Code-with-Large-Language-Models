import sys
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GatedGraphConv
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.utils import scatter

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec
special_args = [
    "edge_index",
    "edge_index_i",
    "edge_index_j",
    "size",
    "size_i",
    "size_j",
]
__size_error_msg__ = (
    "All tensors which should get mapped to the same source "
    "or target nodes must be of same size in dimension 0."
)


class GMNlayer(MessagePassing):
    def __init__(self, in_channels, out_channels, device):
        super(GMNlayer, self).__init__(aggr="add")  # "Add" aggregation.
        self.device = device
        self.out_channels = out_channels
        self.fmessage = nn.Linear(3 * in_channels, out_channels)
        self.fnode = torch.nn.GRUCell(2 * out_channels, out_channels, bias=True)
        self.__match_args__ = getargspec(self.match)[0][1:]
        self.__special_match_args__ = [
            (i, arg) for i, arg in enumerate(self.__match_args__) if arg in special_args
        ]
        self.__match_args__ = [
            arg for arg in self.__match_args__ if arg not in special_args
        ]

    def propagate_match(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        match_args = []
        for arg in self.__match_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    match_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    match_args.append(tmp)
            else:
                match_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs["edge_index"] = edge_index
        kwargs["size"] = size

        for idx, arg in self.__special_match_args__:
            if arg[-2:] in ij.keys():
                match_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                match_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]
        out_attn = self.match(*match_args)
        out_attn = scatter(self.aggr, out_attn, edge_index[i], dim_size=size[i])
        out_attn = self.update(out_attn, *update_args)
        return out_attn

    def forward(
        self, x1, x2, edge_index1, edge_index2, edge_weight1, edge_weight2, mode="train"
    ):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        # Step 3-5: Start propagating messages.
        m1 = self.propagate(
            edge_index1, size=(x1.size(0), x1.size(0)), x=x1, edge_weight=edge_weight1
        )
        m2 = self.propagate(
            edge_index2, size=(x2.size(0), x2.size(0)), x=x2, edge_weight=edge_weight2
        )
        scores = torch.mm(x1, x2.t())
        attn_1 = F.softmax(scores, dim=1)
        attn_2 = F.softmax(scores, dim=0).t()
        attnsum_1 = torch.mm(attn_1, x2)
        attnsum_2 = torch.mm(attn_2, x1)
        u1 = x1 - attnsum_1
        u2 = x2 - attnsum_2
        m1 = torch.cat([m1, u1], dim=1)
        h1 = self.fnode(m1, x1)
        m2 = torch.cat([m2, u2], dim=1)
        h2 = self.fnode(m2, x2)
        return h1, h2

    def message(self, x_i, x_j, edge_index, size, edge_weight=None):
        # x_j has shape [E, out_channels]
        # Step 3: Normalize node features.
        if type(edge_weight) == type(None):
            edge_weight = torch.ones(x_i.size(0), x_i.size(1)).to(self.device)
            m = F.relu(self.fmessage(torch.cat([x_i, x_j, edge_weight], dim=1)))
        else:
            m = F.relu(self.fmessage(torch.cat([x_i, x_j, edge_weight], dim=1)))
        return m

    def match(self, edge_index_i, x_i, x_j, size_i):
        return

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out


class GMNnet(torch.nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device):
        super(GMNnet, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocablen, embedding_dim)
        self.edge_embed = nn.Embedding(20, embedding_dim)
        self.gmnlayer = GMNlayer(embedding_dim, embedding_dim, self.device)
        self.mlp_gate = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)

    def forward(self, data, mode="train"):
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
        x1 = self.embed(x1)
        x1 = x1.squeeze(1)
        x2 = self.embed(x2)
        x2 = x2.squeeze(1)
        if type(edge_attr1) == type(None):
            edge_weight1 = None
            edge_weight2 = None
        else:
            edge_weight1 = self.edge_embed(edge_attr1)
            edge_weight1 = edge_weight1.squeeze(1)
            edge_weight2 = self.edge_embed(edge_attr2)
            edge_weight2 = edge_weight2.squeeze(1)
        for i in range(self.num_layers):
            x1, x2 = self.gmnlayer.forward(
                x1,
                x2,
                edge_index1,
                edge_index2,
                edge_weight1,
                edge_weight2,
                mode="train",
            )
        batch1 = torch.zeros(x1.size(0), dtype=torch.long).to(
            self.device
        )  # without batching
        batch2 = torch.zeros(x2.size(0), dtype=torch.long).to(self.device)
        hg1 = self.pool(x1, batch=batch1)
        hg2 = self.pool(x2, batch=batch2)
        return hg1, hg2


class GGNN(torch.nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device):
        super(GGNN, self).__init__()
        self.device = device
        self.embed = nn.Embedding(vocablen, embedding_dim)
        self.edge_embed = nn.Embedding(20, embedding_dim)
        self.ggnnlayer = GatedGraphConv(embedding_dim, num_layers)
        self.mlp_gate = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)

    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.embed(x)
        x = x.squeeze(1)
        if type(edge_attr) == type(None):
            edge_weight = None
        else:
            edge_weight = self.edge_embed(edge_attr)
            edge_weight = edge_weight.squeeze(1)
        x = self.ggnnlayer(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        hg = self.pool(x, batch=batch)
        return hg
