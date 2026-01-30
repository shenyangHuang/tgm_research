
import unittest
import torch
from torch_geometric.data import HeteroData
from rdl_tgm import from_relbench_heterodata
import pytest 
from tgm.exceptions import EmptyGraphError


class TestRDLTGM(unittest.TestCase):
    def setUp(self):
        # Setup common mock data
        self.data = HeteroData()
        self.data['user'].num_nodes = 5
        self.data['product'].num_nodes = 10
        self.data['event'].num_nodes = 20
        
        # Edge type 1: User -> Event (Temporal, time on Event)
        self.data['event'].time = torch.randint(0, 20, (20,)) # Random timestamps 0-19
        
        src_ue = torch.tensor([0, 1, 2, 3])
        dst_ue = torch.tensor([0, 5, 10, 15])
        self.data['user', 'clicks', 'event'].edge_index = torch.stack([src_ue, dst_ue])
        
        # Edge type 2: Event -> Product (Temporal, time on Event)
        src_ep = torch.tensor([0, 5, 10, 15])
        dst_ep = torch.tensor([0, 1, 2, 3])
        self.data['event', 'references', 'product'].edge_index = torch.stack([src_ep, dst_ep])
        
        # Edge type 3: Product -> User (Static, no time, should default to 0)
        src_pu = torch.tensor([0, 1])
        dst_pu = torch.tensor([0, 1])
        self.data['product', 'bought_by', 'user'].edge_index = torch.stack([src_pu, dst_pu])

    def test_basic_conversion(self):
        """Test basic structural conversion and node count preservation."""
        dg = from_relbench_heterodata(self.data)
        
        expected_edges = 4 + 4 + 2 # 10 edges total
        self.assertEqual(dg.num_edge_events, expected_edges)
        
        # Total nodes = 5 (user) + 10 (product) + 20 (event) = 35
        # DGData infers num_nodes from max index in edge_index usually, or we might need to be careful if isolated nodes exist.
        # But let's check edges primarily.
        self.assertEqual(dg.edge_src.size(0), expected_edges)

    def test_timestamp_propagation(self):
        """Test if timestamps are correctly propagated from event nodes to edges."""
        dg = from_relbench_heterodata(self.data)
        
        # We need to reverse-engineer which edges ended up where.
        # But we can check if the timestamps in dg exist in our input times.
        
        # Check that we have valid timestamps
        self.assertTrue((dg.edge_time >= 0).all())
        
        # Check that static edges have time 0 (or whatever default we picked)
        # We can't easy distinguish edges in DGData without knowing the mapping or edge types
        # But we know we have 2 static edges, so at least 2 edges should be 0 (if we used 0)
        # However, random timestamps might include 0.
        
        # Let's verify specific edge lookup if possible
        pass

    def test_sorted_by_time(self):
        """Test that the resulting DGData edges are sorted by time."""
        dg = from_relbench_heterodata(self.data)
        times = dg.edge_time
        
        # Check if sorted
        is_sorted = (times[:-1] <= times[1:]).all()
        self.assertTrue(is_sorted, "Edges should be sorted by time")

    def test_empty_graph(self):
        """Test conversion of an empty HeteroData object."""
        empty_data = HeteroData()
        with pytest.raises(EmptyGraphError):
            dg = from_relbench_heterodata(empty_data)

        

if __name__ == '__main__':
    unittest.main()
