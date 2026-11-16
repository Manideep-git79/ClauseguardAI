import json
from pathlib import Path

data_dir = Path(r"c:\Users\DELL\Downloads\clauseguard-ai\ml\artifacts\phase9_50k")

for split_name in ["train", "validation", "test"]:
    p = data_dir / f"{split_name}_preprocessed.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    for idx, item in enumerate(data):
        if "doc_id" not in item:
            print(f"Missing doc_id in {split_name} at index {idx}: keys = {list(item.keys())}")
            break

