import argparse
from pathlib import Path

from src.utils.preprocess import clear_processed, split_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("Starting preprocessing...")
    clear_processed(output_dir)
    split_dataset(input_dir, output_dir, seed=args.seed)
    print(f"Preprocessed data written to {output_dir}")


if __name__ == "__main__":
    main()
