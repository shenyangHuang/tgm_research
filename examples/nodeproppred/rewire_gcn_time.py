r"""python -u gcn_rewired.py --epochs=100 --device=cuda:0
"""
import argparse
import time
from typing import Tuple
import os
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader
from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.util.seed import seed_everything
from cayley_construction import batched_augment_cayley, build_cayley_bank

parser = argparse.ArgumentParser(
    description='GCN NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
# parser.add_argument(
#     '--node-dim', type=int, default=256, help='node feat dimension if not provided'
# )
parser.add_argument(
    '--snapshot-time-gran',
    type=str,
    default='Y',
    help='time granularity to operate on for snapshots',
)
parser.add_argument("--wandb", action="store_true", default=False, help="now using wandb")



class RewiredGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        num_exp_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder = GCNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_channels=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.expander = GCNProp(
            in_channels=embed_dim,
            embed_dim=embed_dim,
            out_channels=out_channels,
            num_layers=num_exp_layers,
            dropout=dropout,
        )

    def forward(
        self, batch: DGBatch, node_feat: torch.Tensor, expander_edge_index: torch.Tensor, prev_embed: torch.Tensor,
    ):
        
        z = self.expander(prev_embed, expander_edge_index)
        z = z + node_feat
        z = self.encoder(batch, z)
        return z


class GCNProp(nn.Module):    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, embed_dim))
        self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(embed_dim, embed_dim))
            self.bns.append(torch.nn.BatchNorm1d(embed_dim))
        self.convs.append(GCNConv(embed_dim, out_channels))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x



class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, embed_dim))
        self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(embed_dim, embed_dim))
            self.bns.append(torch.nn.BatchNorm1d(embed_dim))
        self.convs.append(GCNConv(embed_dim, out_channels))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        x = node_feat
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, z_node: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z_node)
        h = h.relu()
        return self.fc2(h)

def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
    expander_edge_index: torch.Tensor = None,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0
    perf_list = []
    prev_embed = static_node_feats.detach().clone()
    for batch in tqdm(loader):
        opt.zero_grad()
        y_true = batch.dynamic_node_feats
        if y_true is None:
            continue

        z = encoder(batch, static_node_feats, expander_edge_index, prev_embed)
        z_node = z[batch.node_ids]
        y_pred = decoder(z_node)

        # compute train NDCG as well
        input_dict = {
            'y_true': y_true,
            'y_pred': y_pred,
            'eval_metric': [METRIC_TGB_NODEPROPPRED],
        }
        perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

        loss = F.cross_entropy(y_pred, y_true)
        loss.backward()
        opt.step()
        total_loss += float(loss)
        prev_embed = z.detach()

    return total_loss, float(np.mean(perf_list)), z

@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
    prev_embed: torch.Tensor,
    expander_edge_index: torch.Tensor = None,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []

    for batch in tqdm(loader):
        y_true = batch.dynamic_node_feats
        if y_true is None:
            continue

        z = encoder(batch, static_node_feats, expander_edge_index, prev_embed)
        z_node = z[batch.node_ids]
        y_pred = decoder(z_node)

        input_dict = {
            'y_true': y_true,
            'y_pred': y_pred,
            'eval_metric': [METRIC_TGB_NODEPROPPRED],
        }
        perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])
        prev_embed = z

    return float(np.mean(perf_list)), prev_embed

args = parser.parse_args()
seed_everything(args.seed)

if args.wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="rewiring",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "gcn_rewired",
        "dataset": args.dataset,
        "time granularity": args.snapshot_time_gran,
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "task": "node prop pred",
        }
    )

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_loader = DGDataLoader(train_dg, batch_unit=args.snapshot_time_gran)
val_loader = DGDataLoader(val_dg, batch_unit=args.snapshot_time_gran)
test_loader = DGDataLoader(test_dg, batch_unit=args.snapshot_time_gran)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.embed_dim), device=args.device
    ) #! has to align with hidden_dim for now

evaluator = Evaluator(name=args.dataset)
num_classes = train_dg.dynamic_node_feats_dim

#! load cached cayley graph if possible
cache_path = f'cayley_{args.dataset}.pt'
if (os.path.exists(cache_path)):
    cayley_g = torch.load(cache_path) 
    print('Cayley graph loaded from, ', cache_path)
else:
    #! Cayley construction time
    start_time = time.perf_counter()
    cayley_bank = build_cayley_bank()
    num_cayley = test_dg.num_nodes
    cayley_g, cayley_edge_attr = batched_augment_cayley(num_cayley, cayley_bank)
    cayley_g = torch.LongTensor(cayley_g).to(args.device)
    end_time = time.perf_counter()
    latency = end_time - start_time
    print(f'Cayley construction latency: {latency:.4f} seconds')
    torch.save(cayley_g, cache_path)  # Save to disk
    print('Cayley graph cached at, ', cache_path)



encoder = RewiredGCN(
    in_channels=static_node_feats.shape[1],
    embed_dim=args.embed_dim,
    out_channels=args.embed_dim,
    num_layers=args.n_layers,
    dropout=float(args.dropout),
).to(args.device)
decoder = NodePredictor(in_dim=args.embed_dim, out_dim=num_classes).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    start_time = time.perf_counter()
    loss, train_NDCG, prev_embed = train(train_loader, static_node_feats, encoder, decoder, opt, cayley_g)
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    start_time = time.perf_counter()
    val_ndcg, prev_embed = eval(val_loader, static_node_feats, encoder, decoder, evaluator, prev_embed, cayley_g)
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} Train {METRIC_TGB_NODEPROPPRED}={train_NDCG:.4f} Validation {METRIC_TGB_NODEPROPPRED}={val_ndcg:.4f}'
    )
    end_time = time.perf_counter()
    val_latency = end_time - start_time

    if (args.wandb):
        wandb.log({"train_loss":loss,
                   "train_" + METRIC_TGB_NODEPROPPRED: train_NDCG,
                    "val_" + METRIC_TGB_NODEPROPPRED: val_ndcg,
                    "train latency": latency,
                    "val latency": val_latency,
                    })

test_ndcg, prev_embed = eval(test_loader, static_node_feats, encoder, decoder, evaluator, prev_embed, cayley_g)
print(f'Test {METRIC_TGB_NODEPROPPRED}={test_ndcg:.4f}')
