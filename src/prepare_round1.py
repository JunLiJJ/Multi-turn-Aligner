from datasets import load_dataset
from pathlib import Path
import json

DATASET = "aligner/aligner-20K"
OUT1 = Path("data/round1/round1_single_turn.jsonl")
OUT2 = Path("data/round1/round1_history_seed.jsonl")
N = 200  # 先取前5条测试，想取全部可改为 None

def norm(ex):
    q = ex.get("question", "").strip()
    a = ex.get("answer", "").strip()
    c = ex.get("correction", "").strip()
    if not (q and a and c):
        return None   # 跳过空样本
    return {"question": q, "answer": a, "correction": c}

def main():
    ds_all = load_dataset(DATASET)
    split = "train" if "train" in ds_all else list(ds_all.keys())[0]
    ds = ds_all[split].map(norm, remove_columns=ds_all[split].column_names)
    # 删除空返回
    ds = ds.filter(lambda x: x is not None)

    if N:
        ds = ds.shuffle(42).select(range(min(N, len(ds))))

    OUT1.parent.mkdir(parents=True, exist_ok=True)
    with OUT1.open("w", encoding="utf-8") as f1, OUT2.open("w", encoding="utf-8") as f2:
        for i, ex in enumerate(ds):
            rid = f"r1-{i+1:06d}"
            f1.write(json.dumps({"id": rid, **ex}, ensure_ascii=False) + "\n")
            # 改为轮次列表格式，方便后续追加
            rounds = [
                {
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "correction": ex["correction"]
                }
            ]
            f2.write(json.dumps({"id": rid, "rounds": rounds}, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(ds)} samples to:")
    print(f"  - {OUT1}")
    print(f"  - {OUT2}")

if __name__ == "__main__":
    main()



