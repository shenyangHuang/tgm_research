import argparse
import time
from typing import Set
import numpy as np
import torch
import os
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm
from utils.json_handler import load_edges_from_jsonl


from tgm import DGData, DGraph, DGBatch
from tgm.constants import METRIC_TGB_LINKPROPPRED
from tgm.hooks import HookManager, TGBNegativeEdgeSamplerHook, StatelessHook
from tgm.loader import DGDataLoader
from tgm.nn import EdgeBankPredictor
from tgm.util.seed import seed_everything


parser = argparse.ArgumentParser(
    description='EdgeBank LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--window-ratio', type=float, default=0.15, help='Window ratio')
parser.add_argument('--pos-prob', type=float, default=1.0, help='Positive edge prob')
parser.add_argument(
    '--memory-mode',
    type=str,
    default='unlimited',
    choices=['unlimited', 'fixed'],
    help='Memory mode',
)


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



def eval(
    loader: DGDataLoader,
    model: EdgeBankPredictor,
    evaluator: Evaluator,
) -> float:
    perf_list = []
    for batch in tqdm(loader):
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            query_src = batch.src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])

            y_pred = model(query_src, query_dst)
            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])
        model.update(batch.src, batch.dst, batch.time)

    return float(np.mean(perf_list))



def eval_full(
    loader: DGDataLoader,
    model: EdgeBankPredictor,
    evaluator: Evaluator,
    test_edge_set: np.ndarray = None,
) -> float:
    test_edge_dict = test_edge_set2dict(test_edge_set)
    perf_list = []
    for batch in tqdm(loader):
        for idx in range(len(batch.src)):
            if (int(batch.src[idx]), int(batch.dst[idx]), float(batch.time[idx])) not in test_edge_dict:
                continue
            valid_dst = batch.neg
            neg_mask = valid_dst != int(batch.src[idx].item())
            neg_batch = valid_dst[neg_mask]
            query_src = batch.src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])

            y_pred = model(query_src, query_dst)
            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }

            full_mrr = float(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])
            perf_list.append(full_mrr)
        model.update(batch.src, batch.dst, batch.time)

    return float(np.mean(perf_list))


args = parser.parse_args()
seed_everything(args.seed)

dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=args.dataset)
dataset.load_val_ns()
dataset.load_test_ns()

data = dataset.get_TemporalData()
valid_dst = torch.unique(data.dst)

dir_path ="../"
test_file = os.path.join(dir_path + "Real-TG/test", args.dataset, "test.jsonl")
test_edge_set = load_edges_from_jsonl(test_file)  # (1000, 3) these are the edges we want to record MRR for


train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)

train_data = train_dg.materialize(materialize_features=False)

hm = HookManager(keys=['val', 'test', 'test_full'])
hm.register('val', TGBNegativeEdgeSamplerHook(dataset_name=args.dataset, split_mode='val'))
hm.register('test', TGBNegativeEdgeSamplerHook(dataset_name=args.dataset, split_mode='test'))
hm.register('test_full', FullNegativeHook(valid_dst))

val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

model = EdgeBankPredictor(
    train_data.src,
    train_data.dst,
    train_data.time,
    memory_mode=args.memory_mode,
    window_ratio=args.window_ratio,
    pos_prob=args.pos_prob,
)

with hm.activate('val'):
    start_time = time.perf_counter()
    val_mrr = eval(val_loader, model, evaluator)
    end_time = time.perf_counter()
    latency = end_time - start_time
    print(f'Latency={latency:.4f} Validation {METRIC_TGB_LINKPROPPRED}={val_mrr:.4f}')


with hm.activate('test_full'):
    test_mrr = eval_full(test_loader, model, evaluator, test_edge_set=test_edge_set)
    print(f'Test {METRIC_TGB_LINKPROPPRED}={test_mrr:.4f}')

# with hm.activate('test'):
#     test_mrr = eval_full(test_loader, model, evaluator)
#     print(f'Test {METRIC_TGB_LINKPROPPRED}={test_mrr:.4f}')
