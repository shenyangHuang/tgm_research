import numpy as np
import torch 

def _as_structured(arr_2d):
    assert arr_2d.ndim == 2 and arr_2d.shape[1] == 3
    return arr_2d.view([('', arr_2d.dtype)] * 3).ravel()

def select_triplet_indices(data, edges_subset_3):
    """
    Return indices i where (src[i], dst[i], t[i]) is in edges_subset_3.
    - edges_subset_3: np.ndarray (M, 3) int64 [(src, dst, t)]
    """
    src_all = np.asarray(data['sources'])
    dst_all = np.asarray(data['destinations'])
    t_all   = np.asarray(data['timestamps'])

    idx_all = np.arange(len(src_all), dtype=np.int64)

    all_triplets = np.stack([src_all, dst_all, t_all], axis=1).astype(edges_subset_3.dtype, copy=False)
    target_triplets = edges_subset_3.astype(all_triplets.dtype, copy=False)

    matches = np.isin(_as_structured(all_triplets), _as_structured(target_triplets))
    return idx_all[matches]

def make_triplet_mask(
    td,                         # already sliced, e.g., test_data
    target_triplets: np.ndarray,# shape (M,3) int64: [src, dst, t]
) -> torch.Tensor:
    """
    Returns a boolean torch tensor mask of length len(td.src)
    """
    s = td.src.detach().cpu().numpy().astype(np.int64, copy=False)
    d = td.dst.detach().cpu().numpy().astype(np.int64, copy=False)
    t = td.t.detach().cpu().numpy().astype(np.int64, copy=False)

    triplets = np.stack([s, d, t], axis=1)

    targets = np.asarray(target_triplets, dtype=np.int64)

    # # de-duplicate targets 
    if len(targets) > 1:
        targets = np.unique(targets, axis=0)

    mask_np = np.isin(_as_structured(triplets), _as_structured(targets))
    return torch.from_numpy(mask_np).to(td.src.device)  