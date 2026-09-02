#!/usr/bin/env python3
"""
analyse_arborescence.py — Affiche l'arborescence d'un dossier avec le nombre
d'images dans chaque sous-dossier.

Pour chaque dossier, deux compteurs sont affichés : le nombre d'images
directement dans ce dossier, et le nombre total d'images en comptant les
sous-dossiers (utile car ce dataset a des sous-dossiers imbriqués par type
de maladie, ex. dataset/data/val/malade/Common Scab - OK/).

Usage :
    python analyse_arborescence.py --root ./dataset
    python analyse_arborescence.py --root "./dataset - Copie" --max_depth 3
    python analyse_arborescence.py --root ./dataset/data --extensions .jpg,.png
"""

import argparse
import sys
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def count_images_direct(folder: Path, extensions: tuple) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def count_images_recursive(folder: Path, extensions: tuple) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def print_tree(folder: Path, extensions: tuple, max_depth, prefix: str = "", depth: int = 0):
    subdirs = sorted((d for d in folder.iterdir() if d.is_dir()), key=lambda d: d.name.lower())

    if max_depth is not None and depth >= max_depth:
        return

    for i, d in enumerate(subdirs):
        is_last = i == len(subdirs) - 1
        branch = "└── " if is_last else "├── "
        direct = count_images_direct(d, extensions)
        total = count_images_recursive(d, extensions)
        if direct == total:
            counts = f"{total} image(s)"
        else:
            counts = f"{direct} image(s) directe(s), {total} au total"
        print(f"{prefix}{branch}{d.name}/  [{counts}]")

        extension = "    " if is_last else "│   "
        print_tree(d, extensions, max_depth, prefix + extension, depth + 1)


def main():
    parser = argparse.ArgumentParser(
        description="Affiche l'arborescence d'un dossier avec le nombre d'images dans chaque sous-dossier."
    )
    parser.add_argument("--root", type=str, required=True, help="Dossier racine à analyser.")
    parser.add_argument(
        "--extensions", type=str, default=",".join(IMG_EXTENSIONS),
        help="Extensions d'image à compter, séparées par des virgules (défaut: .jpg,.jpeg,.png).",
    )
    parser.add_argument(
        "--max_depth", type=int, default=None,
        help="Profondeur maximale d'affichage (défaut: illimitée).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[erreur] {root} n'existe pas ou n'est pas un dossier.")

    extensions = tuple(e.strip().lower() for e in args.extensions.split(","))

    total = count_images_recursive(root, extensions)
    direct = count_images_direct(root, extensions)
    print(f"{root}/  [{direct} image(s) directe(s), {total} au total]")
    print_tree(root, extensions, args.max_depth)

    print(f"\n[info] Total : {total} image(s) sous {root} (extensions : {', '.join(extensions)}).")


if __name__ == "__main__":
    main()
