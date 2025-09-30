r"""python rewire_gcn.py --epochs=100 --device=cuda:0
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
from tgb.linkproppred.evaluate import Evaluator
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph, RecipeRegistry
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.loader import DGDataLoader
from tgm.timedelta import TimeDeltaDG
from tgm.util.seed import seed_everything
from cayley_construction import batched_augment_cayley, build_cayley_bank


parser = argparse.ArgumentParser(
    description='GCN LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument(
    '--snapshot-time-gran',
    type=str,
    default='D',
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
        self, batch: DGBatch, node_feat: torch.Tensor, expander_edge_index: torch.Tensor
    ):
        z = self.encoder(batch, node_feat)
        z = self.expander(z, expander_edge_index)
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


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h).view(-1)
    

def train(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
    conversion_rate: int,
    expander_edge_index: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    encoder.train()
    decoder.train()
    total_loss = 0

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)
    z = encoder(snapshot_batch, static_node_feats, expander_edge_index)
    z = z.detach()

    for batch in tqdm(loader):
        opt.zero_grad()

        pos_out = decoder(z[batch.src], z[batch.dst])
        neg_out = decoder(z[batch.src], z[batch.neg])

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss) / batch.src.shape[0]

        # update the model if the prediction batch has moved to next snapshot.
        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
                z = encoder(snapshot_batch, static_node_feats, expander_edge_index)
                z = z.detach()
            except StopIteration:
                pass

    return total_loss, z



@torch.no_grad()
def eval(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    z: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
    conversion_rate: int,
    expander_edge_index: torch.Tensor,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)

    for batch in tqdm(loader):
        neg_batch_list = batch.neg_batch_list
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = batch.src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])

            y_pred = decoder(z[query_src], z[query_dst]).sigmoid()
            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        # update the model if the prediction batch has moved to next snapshot.
        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
                z = encoder(snapshot_batch, static_node_feats, expander_edge_index)
            except StopIteration:
                pass

    return float(np.mean(perf_list))

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
        "task": "link prop pred",
        }
    )

evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

snapshot_td = TimeDeltaDG(args.snapshot_time_gran)
conversion_rate = int(snapshot_td.convert(train_dg.time_delta))

train_data_discretized = train_data.discretize(args.snapshot_time_gran)
val_data_discretized = val_data.discretize(args.snapshot_time_gran)
test_data_discretized = test_data.discretize(args.snapshot_time_gran)


train_snapshots = DGraph(train_data_discretized, device=args.device)
val_snapshots = DGraph(val_data_discretized, device=args.device)
test_snapshots = DGraph(test_data_discretized, device=args.device)

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

train_snapshots_loader = DGDataLoader(
    train_snapshots, batch_unit=args.snapshot_time_gran
)
val_snapshots_loader = DGDataLoader(val_snapshots, batch_unit=args.snapshot_time_gran)
test_snapshots_loader = DGDataLoader(test_snapshots, batch_unit=args.snapshot_time_gran)


if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )


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
decoder = LinkPredictor(args.embed_dim).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        start_time = time.perf_counter()
        loss, z = train(
            train_loader,
            train_snapshots_loader,
            static_node_feats,
            encoder,
            decoder,
            opt,
            conversion_rate,
            cayley_g,
        )
        end_time = time.perf_counter()
        latency = end_time - start_time


    with hm.activate(val_key):
        start_time = time.perf_counter()
        val_mrr = eval(
            val_loader,
            val_snapshots_loader,
            static_node_feats,
            z,
            encoder,
            decoder,
            evaluator,
            conversion_rate,
            cayley_g,
        )
        end_time = time.perf_counter()
        val_latency = end_time - start_time
        print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} val_latency={val_latency:.4f} Validation {METRIC_TGB_LINKPROPPRED}={val_mrr:.4f}'
    )
        
    if (args.wandb):
        wandb.log({"train_loss":loss,
                    "val_" + METRIC_TGB_LINKPROPPRED: val_mrr,
                    "train latency": latency,
                    "val latency": val_latency,
                    })
    
with hm.activate(test_key):
    test_mrr = eval(
        test_loader,
        test_snapshots_loader,
        static_node_feats,
        z,
        encoder,
        decoder,
        evaluator,
        conversion_rate,
        cayley_g,
    )
    print(f'Test {METRIC_TGB_LINKPROPPRED}={test_mrr:.4f}')
