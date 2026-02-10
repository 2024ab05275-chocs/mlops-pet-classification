import argparse
import json
import os
from pathlib import Path

from src.utils.preprocess import clear_processed, split_dataset



def _data_signature(input_dir: Path, max_total: int, seed: int) -> dict:
    count = 0
    latest_mtime = 0.0
    for root, _, files in os.walk(input_dir):
        for name in files:
            p = Path(root) / name
            try:
                stat = p.stat()
            except FileNotFoundError:
                continue
            count += 1
            if stat.st_mtime > latest_mtime:
                latest_mtime = stat.st_mtime
    return {
        "file_count": count,
        "latest_mtime": latest_mtime,
        "max_total": max_total,
        "seed": seed,
    }


def _load_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total", type=int, default=10000)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print("[step 2.1] Input/Output")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("[step 2.2] Checking if preprocessing is needed")
    meta_path = output_dir / ".preprocess_meta.json"
    current = _data_signature(input_dir, args.max_total, args.seed)
    existing = _load_meta(meta_path)
    if existing == current and output_dir.exists() and any(output_dir.rglob("*")):
        print("[step 2.3] No changes detected; skipping preprocessing")
        return

    print(f"[step 2.4] Starting preprocessing (max_total={args.max_total})")
    clear_processed(output_dir)
    split_dataset(input_dir, output_dir, seed=args.seed, max_total=args.max_total)
    _write_meta(meta_path, current)
    print(f"Preprocessed data written to {output_dir}")


if __name__ == "__main__":
    main()
