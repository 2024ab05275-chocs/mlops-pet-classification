import os
import subprocess
from pathlib import Path


def main():
    dataset = os.environ.get("KAGGLE_DATASET")
    if not dataset:
        raise SystemExit("KAGGLE_DATASET env var is required, e.g. tunguz/cats-and-dogs")

    print("[step 1.1] Resolve dataset and output directory")
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Skip download if data already exists
    if any(out_dir.rglob("*")):
        print(f"[step 1.2] Data already exists in {out_dir}, skipping download")
        return

    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"]
    print("[step 1.3] Downloading from Kaggle")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[step 1.4] Download complete")


if __name__ == "__main__":
    main()
