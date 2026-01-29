
import torch
from torch_geometric.data import HeteroData
from tgm.data import DGData

def from_relbench_heterodata(data: HeteroData) -> DGData:
    """
    Converts a RelBench HeteroData object to a TGM DGData object.
    
    RelBench HeteroData typically stores timestamps on event nodes.
    TGM DGData expects temporal edges.
    This adaptor propagates node timestamps to incident edges.
    """
    
    # 1. Map Heterogeneous Node IDs to Homogeneous Global IDs
    node_offsets = {}
    total_nodes = 0
    node_types = data.node_types
    
    # Sort node types to ensure deterministic ordering
    node_types.sort()
    
    for nt in node_types:
        num_nodes = data[nt].num_nodes
        node_offsets[nt] = total_nodes
        total_nodes += num_nodes
        
    # 2. Collect Edges and Timestamps
    edge_srcs = []
    edge_dsts = []
    edge_times = []
    edge_types_list = []
    
    # Iterate over all edge types
    edge_types = data.edge_types
    for i, et in enumerate(edge_types):
        src_type, rel_type, dst_type = et
        
        edge_index = data[et].edge_index
        src_local = edge_index[0]
        dst_local = edge_index[1]
        
        # Calculate global IDs
        src_global = src_local + node_offsets[src_type]
        dst_global = dst_local + node_offsets[dst_type]
        
        # Determine Timestamps
        # Strategy: Use time from src or dst logic.
        # Check if src or dst has 'time' attribute
        
        times = None
        if 'time' in data[src_type]:
            # Source is event/temporal
            # data[src_type].time is [num_nodes], we need to gather by src_local indices
            node_times = data[src_type].time
            times = node_times[src_local]
        elif 'time' in data[dst_type]:
            # Destination is event/temporal
            node_times = data[dst_type].time
            times = node_times[dst_local]
        else:
            # Static edge, timestamp 0?
            # Or assume global time 0.
            times = torch.zeros(src_global.size(0), dtype=torch.long) # RelBench usually uses long/int timestamps
            
        edge_srcs.append(src_global)
        edge_dsts.append(dst_global)
        edge_times.append(times)
        edge_types_list.append(torch.full((src_global.size(0),), i, dtype=torch.long))

    # 3. Concatenate and Sort
    if not edge_srcs:
        # Empty graph?
        return DGData(edge_index=torch.empty((2, 0)), edge_time=torch.empty(0))
        
    all_src = torch.cat(edge_srcs)
    all_dst = torch.cat(edge_dsts)
    all_time = torch.cat(edge_times)
    all_type = torch.cat(edge_types_list)
    
    # TGM expects edges sorted by time
    perm = all_time.argsort()
    all_src = all_src[perm]
    all_dst = all_dst[perm]
    all_time = all_time[perm]
    all_type = all_type[perm]
    
    edge_index = torch.stack([all_src, all_dst])
    
    # 4. Construct DGData
    # Note: DGData might expect floating point time in some versions, check if casting needed.
    # DGData usually takes float time, but RelBench has int timestamps (days/seconds).
    # DGData.from_raw allows specifying edge_time.
    
    return DGData.from_raw(
        edge_index=edge_index,
        edge_time=all_time.float(), # Convert to float for general compatibility
        edge_type=all_type
        # Add node features if needed later, complexity grows with heterogeneity
    )

if __name__ == "__main__":
    # Test Verification
    print("Verifying RelBench to TGM Adaptor...")
    try:
        from relbench.datasets import get_dataset
        import torch_geometric.transforms as T
        
        # Load a small dataset or mock one
        # For speed, let's mock one first
        print("Creating Mock HeteroData...")
        data = HeteroData()
        
        # Node Type A (Users) - Static
        data['user'].num_nodes = 10
        
        # Node Type B (Events) - Temporal
        data['event'].num_nodes = 20
        data['event'].time = torch.arange(20) # 0 to 19 timestamps
        
        # Edge A->B (User attends Event)
        # 50 random edges
        src = torch.randint(0, 10, (50,))
        dst = torch.randint(0, 20, (50,))
        data['user', 'attends', 'event'].edge_index = torch.stack([src, dst])
        
        print("Converting to DGData...")
        dg = from_relbench_heterodata(data)
        
        print(f"DGData Edge Count: {dg.num_edges}")
        print(f"DGData Node Count: {dg.num_nodes}") # Inferred from max index usually
        print("Edge Time Sample:", dg.edge_time[:10])
        
        assert dg.num_edges == 50
        assert dg.edge_time.is_sorted
        print("Verification Successful!")
        
    except ImportError:
        print("RelBench or PyG not installed, skipping full verification.")
    except Exception as e:
        print(f"Verification Failed: {e}")
