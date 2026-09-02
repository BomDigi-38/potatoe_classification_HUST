#!/usr/bin/env python3
"""
check_dataset_balance.py — Vérifie l'équilibre saine/malade du dataset avant entraînement.

Supporte les deux structures acceptées par train_classif_pdt.py :
    Mode A : data_dir/{train,val}/{saine,malade}/*.jpg
    Mode B : data_dir/{saine,malade}/*.jpg (pas encore splitté)

Usage :
    python check_dataset_balance.py --data_dir ./dataset/data
"""

import argparse
import sys
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
CLASSES = ("malade", "saine")


def count_images(class_dir: Path) -> int:
    # Récursif : comme ImageFolder (os.walk), les images peuvent être
    # nichées dans des sous-dossiers (ex. par sous-type de maladie).
    return sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def collect_counts(data_dir: Path) -> dict:
    """Retourne (counts, warnings) où counts = {split: {nom_dossier_classe: n}}.
    Les noms de dossiers de classe sont pris tels quels (pas de normalisation),
    pour pouvoir détecter une incohérence entre splits (ex. singulier/pluriel)."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    warnings = []

    if train_dir.is_dir() and val_dir.is_dir():
        counts = {}
        class_dir_names = {}
        for split_name, split_dir in (("train", train_dir), ("val", val_dir)):
            cls_dirs = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
            class_dir_names[split_name] = cls_dirs
            counts[split_name] = {cls: count_images(split_dir / cls) for cls in cls_dirs}

        if class_dir_names["train"] != class_dir_names["val"]:
            warnings.append(
                f"[warn] Les noms de dossiers de classe diffèrent entre train {class_dir_names['train']} "
                f"et val {class_dir_names['val']} ! torchvision.ImageFolder traite train et val comme deux "
                "jeux de classes indépendants — si l'ordre alphabétique diffère, les labels train/val ne "
                "correspondront plus (bug silencieux, aucune vérification n'existe actuellement sur val_ds.classes "
                "dans train_classif_pdt.py, contrairement à train_ds.classes)."
            )
        return counts, warnings

    class_dirs = {d.name: d for d in data_dir.iterdir() if d.is_dir() and d.name in CLASSES}
    if len(class_dirs) != 2:
        sys.exit(
            f"[erreur] Impossible de comprendre la structure de {data_dir}. "
            "Attendu : data_dir/train/{saine,malade} + data_dir/val/{saine,malade} "
            "OU data_dir/{saine,malade}."
        )
    return {"(non splitté)": {cls: count_images(d) for cls, d in class_dirs.items()}}, warnings


def print_report(counts: dict, warnings: list, warn_ratio: float):
    print("[info] Répartition des classes :\n")

    all_classes = sorted({cls for cls_counts in counts.values() for cls in cls_counts})
    header = f"{'split':<15}" + "".join(f"{cls:>14}" for cls in all_classes) + f"{'total':>12}"
    print(header)
    print("-" * len(header))

    totals = {cls: 0 for cls in all_classes}
    for split, cls_counts in counts.items():
        row_total = sum(cls_counts.get(cls, 0) for cls in all_classes)
        row = f"{split:<15}" + "".join(f"{cls_counts.get(cls, 0):>14}" for cls in all_classes) + f"{row_total:>12}"
        print(row)
        for cls in all_classes:
            totals[cls] += cls_counts.get(cls, 0)

    grand_total = sum(totals.values())
    print("-" * len(header))
    print(f"{'TOTAL':<15}" + "".join(f"{totals[cls]:>14}" for cls in all_classes) + f"{grand_total:>12}")

    print()
    for cls in all_classes:
        pct = 100 * totals[cls] / grand_total if grand_total else 0
        print(f"  - {cls}: {totals[cls]} images ({pct:.1f}%)")

    if warnings:
        print()
        for w in warnings:
            print(w)

    minority = min(totals.values())
    majority = max(totals.values())
    ratio = majority / minority if minority > 0 else float("inf")
    maj_cls = max(totals, key=totals.get)
    min_cls = min(totals, key=totals.get)

    print(f"\n[info] Ratio majorité/minorité : {ratio:.2f} ({maj_cls} vs {min_cls})")

    if ratio >= warn_ratio:
        print(
            f"[warn] Dataset déséquilibré (ratio >= {warn_ratio}). Pistes : pondération de classe "
            "dans CrossEntropyLoss, WeightedRandomSampler dans le DataLoader, ou augmentation "
            "de données ciblée sur la classe minoritaire."
        )
    else:
        print(f"[info] Dataset raisonnablement équilibré (ratio < {warn_ratio}).")


def main():
    parser = argparse.ArgumentParser(
        description="Vérifie l'équilibre saine/malade du dataset avant entraînement."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Dossier racine des images.")
    parser.add_argument(
        "--warn_ratio", type=float, default=2.0,
        help="Seuil de ratio majorité/minorité à partir duquel un déséquilibre est signalé (défaut: 2.0).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"[erreur] {data_dir} n'existe pas ou n'est pas un dossier.")

    counts, warnings = collect_counts(data_dir)
    print_report(counts, warnings, args.warn_ratio)


if __name__ == "__main__":
    main()
