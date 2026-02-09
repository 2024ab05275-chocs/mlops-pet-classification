from pathlib import Path
from PIL import Image


def main():
    out = Path("scripts/sample.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), color=(120, 180, 200))
    img.save(out)
    print(f"Created {out}")


if __name__ == "__main__":
    main()
