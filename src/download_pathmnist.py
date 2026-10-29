import argparse
from pathlib import Path
from medmnist import PathMNIST, INFO

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data")
    ap.add_argument("--size", type=int, default=64, choices=[28, 64, 128, 224])
    ap.add_argument("--per-split", type=int, default=2000)
    args = ap.parse_args()

    classes = INFO["pathmnist"]["label"]  # {"0": "adipose", ...}
    out = Path(args.output)

    for split in ["train", "val", "test"]:
        ds = PathMNIST(split=split, download=True, size=args.size)
        n = min(len(ds), args.per_split)
        for i in range(n):
            img, label = ds[i]
            name = classes[str(int(label[0]))].replace(" ", "_").replace("/", "-")
            d = out / split / name
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{split}_{i:06d}.png")
        print(f"{split}: {n} imagens salvas")

if __name__ == "__main__":
    main()