import os
import subprocess
from pathlib import Path


def main():
    dataset = os.environ.get("KAGGLE_DATASET")
    if not dataset:
        raise SystemExit("KAGGLE_DATASET env var is required, e.g. tunguz/cats-and-dogs")

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Skip download if data already exists
    if any(out_dir.rglob("*")):
        print(f"Data already exists in {out_dir}, skipping download")
        return

    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
