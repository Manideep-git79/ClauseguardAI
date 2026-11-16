import json
from pathlib import Path

data_dir = Path(r"c:\Users\DELL\Downloads\clauseguard-ai\ml\artifacts\phase9_50k")

for split_name in ["train", "validation", "test"]:
    p = data_dir / f"{split_name}_preprocessed.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"{split_name} length: {len(data)}")
    print(f"{split_name}[0] keys:", list(data[0].keys()))
    print(f"{split_name}[0] sample:", {k: data[0][k] for k in list(data[0].keys())[:5]})

