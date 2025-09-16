import argparse
import copy
import time
from typing import Callable, Tuple, Set
import os
from utils.json_handler import load_edges_from_jsonl
from utils.json_handler import add_to_jsonl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm.graph import DGBatch, DGData, DGraph
from tgm.hooks import (
    HookManager,
    NegativeEdgeSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.loader import DGDataLoader
from tgm.nn import DyGFormer, Time2Vec
from tgm.util.seed import seed_everything
from tgm.hooks import StatelessHook

class FullNegativeHook(StatelessHook):
    """Negative Sampler Hook for full evaluation. 

    Args:
        full_dst (torch.Tensor): all possible destination nodes for evaluation
    """
    requires: Set[str] = set()
    produces = {'neg', 'neg_time'}

    def __init__(self, all_dst: torch.Tensor) -> None:
        self.all_dst = all_dst

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.neg = self.all_dst.to(dg.device)  # type: ignore
        gen = torch.Generator(device=dg.device)
        gen.manual_seed(0)
        batch.neg_time = torch.randint(  # type: ignore
            int(batch.time.min().item()),
            int(batch.time.max().item()) + 1,
            (batch.neg.size(0),),  # type: ignore
            device=dg.device,
            generator=gen,
        )
        return batch

def test_edge_set2dict(test_edge_set):
    edge_dict = {}
    for i in range(test_edge_set.shape[0]):
        src = test_edge_set[i,0]
        dst = test_edge_set[i,1]
        ts = test_edge_set[i,2]
        edge_dict[(src,dst,ts)] = i
    return edge_dict


parser = argparse.ArgumentParser(
    description='DyGFormers TGB Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--max_sequence_length',
    type=int,
    default=32,
    help='maximal length of the input sequence of each node',
)
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--time_dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed_dim', type=int, default=172, help='attention dimension')
parser.add_argument('--node_dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--channel-embedding-dim',
    type=int,
    default=50,
    help='dimension of each channel embedding',
)
parser.add_argument('--patch-size', type=int, default=1, help='patch size')
parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
parser.add_argument(
    '--num_heads', type=int, default=2, help='number of heads used in attention layer'
)
parser.add_argument(
    '--num-channels',
    type=int,
    default=4,
    help='number of channels used in attention layer',
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h).sigmoid().view(-1)


class DyGFormer_LinkPrediction(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        channel_embedding_dim: int,
        output_dim: int = 172,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        num_channels: int = 4,
        time_encoder: Callable[..., nn.Module] = Time2Vec,
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.encoder = DyGFormer(
            node_feat_dim,
            edge_feat_dim,
            time_feat_dim,
            channel_embedding_dim,
            output_dim,
            patch_size,
            num_layers,
            num_heads,
            dropout,
            max_input_sequence_length,
            num_channels,
            time_encoder,
            device,
        )
        self.decoder = LinkPredictor(
            output_dim
        )  # @TODO: Make encoder/decoder to be explicit

    def forward(self, batch: DGBatch, static_node_feat: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src = batch.src
        dst = batch.dst
        neg = batch.neg
        time = batch.time
        nbr_nids = batch.nbr_nids[0]
        nbr_times = batch.nbr_times[0]
        nbr_feats = batch.nbr_feats[0]
        pos_batch_size = dst.shape[0]
        neg_batch_size = neg.shape[0]

        # positive edge
        edge_idx_pos = torch.stack((src, dst), dim=0)
        z_src_pos, z_dst_pos = self.encoder(
            static_node_feat,
            edge_idx_pos,
            time,
            nbr_nids[: pos_batch_size * 2],
            nbr_times[: pos_batch_size * 2],
            nbr_feats[: pos_batch_size * 2],
        )
        pos_out = self.decoder(z_src_pos, z_dst_pos)

        neg_nbr_nids = nbr_nids[
            -neg_batch_size:
        ]  # @TODO: Assume that batch.neg doesn't have duplicated records
        neg_nbr_times = nbr_times[-neg_batch_size:]
        neg_nbr_feats = nbr_feats[-neg_batch_size:]
        src_nbr_nids = nbr_nids[:pos_batch_size]
        src_nbr_times = nbr_times[:pos_batch_size]
        src_nbr_feats = nbr_feats[:pos_batch_size]

        if src.shape[0] != neg_batch_size:
            src = torch.repeat_interleave(src, repeats=neg_batch_size, dim=0)
            time = torch.repeat_interleave(time, repeats=neg_batch_size, dim=0)
            src_nbr_nids = torch.repeat_interleave(
                src_nbr_nids, repeats=neg_batch_size, dim=0
            )
            src_nbr_times = torch.repeat_interleave(
                src_nbr_times, repeats=neg_batch_size, dim=0
            )
            src_nbr_feats = torch.repeat_interleave(
                src_nbr_feats, repeats=neg_batch_size, dim=0
            )
            neg_nbr_nids = neg_nbr_nids.repeat(pos_batch_size, 1)
            neg_nbr_times = neg_nbr_times.repeat(pos_batch_size, 1)
            neg_nbr_feats = neg_nbr_feats.repeat(pos_batch_size, 1, 1)
            neg = neg.repeat(pos_batch_size)
        else:
            src_nbr_nids = nbr_nids[:pos_batch_size]
            src_nbr_times = nbr_times[:pos_batch_size]
            src_nbr_feats = nbr_feats[:pos_batch_size]

        edge_idx_neg = torch.stack((src, neg), dim=0)

        # negative edge
        z_src_neg, z_dst_neg = self.encoder(
            static_node_feat,
            edge_idx_neg,
            time,
            torch.cat([src_nbr_nids, neg_nbr_nids], dim=0),
            torch.cat([src_nbr_times, neg_nbr_times], dim=0),
            torch.cat([src_nbr_feats, neg_nbr_feats], dim=0),
        )
        neg_out = self.decoder(z_src_neg, z_dst_neg)

        return pos_out, neg_out


def train(
    loader: DGDataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    static_node_feat: torch.tensor,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch, static_node_feat)

        loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(
    evaluator: Evaluator,
    loader: DGDataLoader,
    model: nn.Module,
    eval_metric: str,
    static_node_feat: torch.tensor,
    test_edge_set: np.ndarray = None,
) -> float:
    if test_edge_set is not None:
        test_edge_dict = test_edge_set2dict(test_edge_set)

    model.eval()
    perf_list = []
    for batch in tqdm(loader):
        copy_batch = copy.deepcopy(batch)
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            if test_edge_set is not None:
                # only evaluate on the edges in test_edge_set
                if (int(batch.src[idx]), int(batch.dst[idx]), float(batch.time[idx])) not in test_edge_dict:
                    continue
            copy_batch.src = batch.src[idx].unsqueeze(0)
            copy_batch.dst = batch.dst[idx].unsqueeze(0)
            copy_batch.time = batch.time[idx].unsqueeze(0)
            copy_batch.neg = neg_batch
            neg_idx = (batch.neg == neg_batch[:, None]).nonzero(as_tuple=True)[1]

            # A tensor of index of src, dst and negative nodes to retrieve neighbor information
            all_idx = torch.cat(
                [
                    torch.Tensor([idx]).to(neg_batch.device),  # src idx
                    torch.Tensor([idx + batch.src.shape[0]]).to(
                        neg_batch.device
                    ),  # dst idx
                    neg_idx,
                ],
                dim=0,
            ).long()
            copy_batch.nbr_nids = [batch.nbr_nids[0][all_idx]]
            copy_batch.nbr_times = [batch.nbr_times[0][all_idx]]
            copy_batch.nbr_feats = [batch.nbr_feats[0][all_idx]]

            pos_out, neg_out = model(copy_batch, static_node_feat)

            input_dict = {
                'y_pred_pos': pos_out,
                'y_pred_neg': neg_out,
                'eval_metric': [eval_metric],
            }
            perf_list.append(evaluator.eval(input_dict)[eval_metric])

    return float(np.mean(perf_list))



@torch.no_grad()
def eval_full(
    evaluator: Evaluator,
    loader: DGDataLoader,
    model: nn.Module,
    eval_metric: str,
    static_node_feat: torch.tensor,
    test_edge_set: np.ndarray = None,
) -> float:
    test_edge_dict = test_edge_set2dict(test_edge_set)

    model.eval()
    perf_list = []
    per_link_rows = []
    for batch in tqdm(loader):
        copy_batch = copy.deepcopy(batch)
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            if (int(batch.src[idx]), int(batch.dst[idx]), float(batch.time[idx])) not in test_edge_dict:
                continue
            valid_dst = batch.neg
            copy_batch.src = batch.src[idx].unsqueeze(0)
            copy_batch.dst = batch.dst[idx].unsqueeze(0)
            copy_batch.time = batch.time[idx].unsqueeze(0)
            neg_mask = valid_dst != int(batch.src[idx].item())
            neg_batch = valid_dst[neg_mask]
            copy_batch.neg = neg_batch
            neg_idx = (batch.neg == neg_batch[:, None]).nonzero(as_tuple=True)[1]

            # A tensor of index of src, dst and negative nodes to retrieve neighbor information
            all_idx = torch.cat(
                [
                    torch.Tensor([idx]).to(neg_batch.device),  # src idx
                    torch.Tensor([idx + batch.src.shape[0]]).to(
                        neg_batch.device
                    ),  # dst idx
                    neg_idx,
                ],
                dim=0,
            ).long()
            copy_batch.nbr_nids = [batch.nbr_nids[0][all_idx]]
            copy_batch.nbr_times = [batch.nbr_times[0][all_idx]]
            copy_batch.nbr_feats = [batch.nbr_feats[0][all_idx]]

            pos_out, neg_out = model(copy_batch, static_node_feat)

            #! full MRR evaluation
            input_dict = {
                'y_pred_pos': pos_out,
                'y_pred_neg': neg_out,
                'eval_metric': [eval_metric],
            }

            full_mrr = float(evaluator.eval(input_dict)[eval_metric])
            perf_list.append(full_mrr)
            per_link_rows.append([
                    batch.src[idx].item(),
                    batch.dst[idx].item(),
                    batch.time[idx].item(),
                    full_mrr,
                ])

    return float(np.mean(perf_list)), perf_list, per_link_rows


args = parser.parse_args()
seed_everything(args.seed)

# loading negative sample from TGB
dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
eval_metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=args.dataset)
dataset.load_val_ns()
dataset.load_test_ns()

data = dataset.get_TemporalData()
valid_dst = torch.unique(data.dst).to(args.device)

dir_path ="../"
test_file = os.path.join(dir_path + "Real-TG/test", args.dataset, "test.jsonl")
test_edge_set = load_edges_from_jsonl(test_file)  # (1000, 3) these are the edges we want to record MRR for



full_data = DGData.from_tgb(args.dataset)
full_graph = DGraph(full_data)
num_nodes = full_graph.num_nodes
edge_feats_dim = full_graph.edge_feats_dim
train_data, val_data, test_data = full_data.split()

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

if train_dg.static_node_feats is not None:
    static_node_feat = train_dg.static_node_feats.to(args.device)
else:
    static_node_feat = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

_, dst, _ = train_dg.edges
nbr_hook = RecencyNeighborHook(
    num_nbrs=[args.max_sequence_length - 1],  # 1 remaining for seed node itself
    num_nodes=num_nodes,
    edge_feats_dim=edge_feats_dim,
)

hm = HookManager(keys=['train', 'val', 'test', 'test_full'])
hm.register_shared(nbr_hook)
hm.register('train', NegativeEdgeSamplerHook(low=int(dst.min()), high=int(dst.max())))
hm.register('val', TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val'))
hm.register('test', TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='test'))
hm.register('test_full', FullNegativeHook(valid_dst))


train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

model = DyGFormer_LinkPrediction(
    node_feat_dim=static_node_feat.shape[1],
    edge_feat_dim=edge_feats_dim,
    time_feat_dim=args.time_dim,
    channel_embedding_dim=args.channel_embedding_dim,
    output_dim=args.embed_dim,
    max_input_sequence_length=args.max_sequence_length,
    dropout=args.dropout,
    num_heads=args.num_heads,
    num_channels=args.num_channels,
    num_layers=args.num_layers,
    device=args.device,
    patch_size=args.patch_size,
).to(args.device)

opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))


best_val_mrr = 0.0
for epoch in range(1, args.epochs + 1):
    with hm.activate('train'):
        start_time = time.perf_counter()
        loss = train(train_loader, model, opt, static_node_feat)
        end_time = time.perf_counter()
        latency = end_time - start_time
    with hm.activate('val'):
        val_mrr = eval(evaluator, val_loader, model, eval_metric, static_node_feat)
        print(
            f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} Validation {eval_metric}={val_mrr:.4f}'
        )

    if (val_mrr > best_val_mrr):
        best_val_mrr = val_mrr
        with hm.activate('test_full'):
            test_mrr, full_mrr_list, per_link_rows = eval_full(evaluator, test_loader, model, eval_metric, static_node_feat, test_edge_set=test_edge_set)
            print(f'Test MRR Full {eval_metric}={test_mrr:.4f}')   

        if (args.seed == 1):
            try:
                # store both metrics per link (your writer should accept 5 columns under the chosen field)
                add_to_jsonl(per_link_rows, test_file, field_name="TPNet")
                print(f"\tWrote  TPNet per-link MRRs to {test_file} (field='tpnet_mrr').")
            except Exception as e:
                print(f"\tWARNING: failed to write TPNet per-link MRRs: {e}")

    # Clear memory state between epochs, except last epoch
    if epoch < args.epochs:
        hm.reset_state()

# with hm.activate('test'):
#     test_mrr = eval(evaluator, test_loader, model, eval_metric)
#     print(f'Test MRR:{eval_metric}={test_mrr:.4f}')
