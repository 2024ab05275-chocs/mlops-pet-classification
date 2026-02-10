import os
import random
import shutil
from pathlib import Path
from typing import Tuple

from PIL import Image


VALID_EXTS = {".jpg", ".jpeg", ".png"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS


def resize_and_save(src_path: Path, dst_path: Path, size: Tuple[int, int] = (224, 224)) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        img.save(dst_path)


def _collect_class_dirs(raw_dir: Path) -> Tuple[Path, Path]:
    # Accept either raw_dir/Cat and raw_dir/Dog or raw_dir/PetImages/Cat, etc.
    candidates = [raw_dir, raw_dir / "PetImages", raw_dir / "cats_and_dogs", raw_dir / "CatsAndDogs"]
    for base in candidates:
        cat_dir = base / "Cat"
        dog_dir = base / "Dog"
        if cat_dir.exists() and dog_dir.exists():
            return cat_dir, dog_dir
    raise FileNotFoundError("Could not find Cat/Dog folders in raw data directory")


def split_dataset(raw_dir: Path, output_dir: Path, seed: int = 42, splits=(0.8, 0.1, 0.1), max_total: int = 10000) -> None:
    random.seed(seed)
    cat_dir, dog_dir = _collect_class_dirs(raw_dir)

    def _split_class(class_dir: Path, class_name: str) -> None:
        images = [p for p in class_dir.iterdir() if p.is_file() and _is_image_file(p)]
        random.shuffle(images)
        per_class = max_total // 2 if max_total else None
        if per_class is not None and len(images) > per_class:
            images = images[:per_class]
        n = len(images)
        n_train = int(n * splits[0])
        n_val = int(n * splits[1])
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        print(f"{class_name}: total={n} train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for img_path in split_imgs:
                dst = output_dir / split_name / class_name / img_path.name
                resize_and_save(img_path, dst)

    _split_class(cat_dir, "Cat")
    _split_class(dog_dir, "Dog")


def clear_processed(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
