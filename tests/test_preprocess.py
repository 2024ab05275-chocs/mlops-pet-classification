from pathlib import Path
from PIL import Image

from src.utils.preprocess import split_dataset


def _make_image(path: Path):
    img = Image.new("RGB", (300, 300), color=(10, 20, 30))
    img.save(path)


def test_split_dataset_creates_224(tmp_path: Path):
    raw = tmp_path / "raw" / "PetImages"
    cat_dir = raw / "Cat"
    dog_dir = raw / "Dog"
    cat_dir.mkdir(parents=True)
    dog_dir.mkdir(parents=True)

    for i in range(5):
        _make_image(cat_dir / f"cat_{i}.jpg")
        _make_image(dog_dir / f"dog_{i}.jpg")

    out = tmp_path / "processed"
    split_dataset(raw.parent, out, seed=1)

    sample = next((out / "train" / "Cat").iterdir())
    img = Image.open(sample)
    assert img.size == (224, 224)
