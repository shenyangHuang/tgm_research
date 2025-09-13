import json
import numpy as np
from typing import Iterable, Optional, Set, List, Tuple
import os 
import json, tempfile, contextlib

def load_edges_from_jsonl(
    path: str
) -> np.ndarray:
    """
    Loads edges from a chosen .jsonl file (path)

    Returns an array of shape (N, 3): [src, dst, t].
    - Uses the 'link' field: [src, [dst...], t].
    """
    edges: List[Tuple[int, int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            obj = json.loads(line)
            src, dst_list, ts = obj["link"]
            dsts = [int(d) for d in dst_list]

            for d in dsts:
                edges.append((int(src), d, int(ts)))

    return np.asarray(edges, dtype=np.int64)


def add_to_jsonl(test_perf: np.ndarray, path: str, field_name: str) -> int:
    """
    Add per-link performance to existing jsonl. 
    """
    test_perf = np.asarray(test_perf)
    
    # mapping (src, dst, t) -> metric (float or None if NaN)
    perf_map = {}
    for row in test_perf:
        s, d, t, m = row
        # cast to native ints for reliable tuple keys
        key = (int(s), int(d), int(t))
        m_f = float(m)
        perf_map[key] = (None if np.isnan(m_f) else m_f)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_perf_", suffix=".jsonl", dir=dir_name)
    os.close(fd)

    lines_written = 0
    try:
        with open(path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip("\n")
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Preserve undecodable lines as-is
                    fout.write(line + "\n")
                    lines_written += 1
                    continue

                # Expect "link": [src, [dst...], t]
                link = obj.get("link", None)
                if (
                    isinstance(link, list) and
                    len(link) == 3 and
                    isinstance(link[1], list)
                ):
                    src = int(link[0])
                    dst_list = [int(x) for x in link[1]]
                    ts = int(link[2])

                    if len(dst_list) == 1:
                        key = (src, dst_list[0], ts)
                        obj[field_name] = perf_map.get(key, None)
                    else:
                        # Multiple candidate destinations: write a dict
                        per_dst = {}
                        for d in dst_list:
                            key = (src, d, ts)
                            per_dst[str(d)] = perf_map.get(key, None)
                        obj[field_name] = per_dst
                else:
                    pass

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                lines_written += 1

        # Atomic replace
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp on failure
        with contextlib.suppress(Exception):
            os.remove(tmp_path)
        raise

    return lines_written