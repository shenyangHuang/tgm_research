import json
import numpy as np
from typing import Iterable, Optional, Set, List, Tuple
import os 

def redistribute_jsonl(path_complete: str, path_output_dir: str ) -> None: 
    """
    Redistribute test_complete.jsonl into each dataset's folder. 
    """
    datasets = ['tgbl-enron', 'tgbl-flight', 'tgbl-subreddit', 'tgbl-uci', 'tgbl-coin', 'tgbl-wiki']
    lines: Dict[str, List[str]] = {ds: [] for ds in datasets}

    with open(path_complete, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            obj = json.loads(line)
            dataset = obj["dataset"]
            
            lines[dataset].append(line)
    
    os.makedirs(path_output_dir, exist_ok=True)

    for dataset, dataset_lines in lines.items():
        output_dir = os.path.join(path_output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,  "test.jsonl")
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(dataset_lines))
