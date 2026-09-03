#!/usr/bin/env python3
"""
split_dataset.py — Découpe un dataset Mode B (<classe>/*.jpg, pas encore
splitté) en Mode A (train/<classe>, val/<classe>), une fois pour toutes, pour
que train_classif_pdt.py n'ait plus jamais à recalculer ce split à chaque
lancement (il détecte Mode A et l'utilise tel quel).

Split stratifié par classe (mélange seedé + coupe selon --val_split), écrit
en hard links (aucune duplication sur disque, aucun fichier original
déplacé/modifié). Tolère des sous-dossiers de sous-type sous chaque classe
(ex. malade/Common Scab - OK/), comme le reste du projet.

Usage :
    python split_dataset.py --data_dir ./dataset/mon_dataset --output_dir ./dataset/mon_dataset_split
    python train_classif_pdt.py --data_dir ./dataset/mon_dataset_split   # ensuite, Mode A détecté directement
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_class_images(data_dir: Path):
    """Retourne {classe: [chemins image]} (recherche récursive par classe,
    tolère des sous-dossiers de sous-type)."""
    by_class = {}
    for cls_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        imgs = [p for p in sorted(cls_dir.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
        by_class[cls_dir.name] = imgs
    return by_class


def split_and_link(by_class: dict, output_dir: Path, val_split: float, seed: int):
    rng = random.Random(seed)
    print()
    for cls, imgs in sorted(by_class.items()):
        imgs = imgs[:]
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_split))
        val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

        for split_name, split_imgs in (("train", train_imgs), ("val", val_imgs)):
            out_dir = output_dir / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            # Préfixe numérique : des sous-dossiers de sous-type différents
            # réutilisent souvent les mêmes noms de fichiers, un simple
            # img.name écraserait silencieusement des images en cas de collision.
            for i, img in enumerate(split_imgs):
                (out_dir / f"{i:06d}_{img.name}").hardlink_to(img.resolve())

        print(f"  - {cls}: {len(train_imgs)} train / {len(val_imgs)} val")


def main():
    parser = argparse.ArgumentParser(
        description="Découpe un dataset Mode B (<classe>/*.jpg) en Mode A (train/val) via hard links.")
    parser.add_argument("--data_dir", type=str, required=True, help="Dossier <classe>/*.jpg|png (Mode B).")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier de sortie (train/<classe>, val/<classe>) ; purgé et régénéré à chaque exécution.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction de chaque classe mise en val.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0 < args.val_split < 1):
        sys.exit(f"[erreur] --val_split doit être dans ]0, 1[, reçu {args.val_split}.")

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        sys.exit(f"[erreur] {data_dir} n'existe pas ou n'est pas un dossier.")

    output_dir = Path(args.output_dir).resolve()
    if output_dir == data_dir or output_dir in data_dir.parents or data_dir in output_dir.parents:
        sys.exit(f"[erreur] --output_dir ({output_dir}) chevauche --data_dir ({data_dir}) — "
                 f"choisissez un dossier de sortie totalement séparé.")

    by_class = list_class_images(data_dir)
    by_class = {c: imgs for c, imgs in by_class.items() if imgs}
    if len(by_class) < 2:
        sys.exit(f"[erreur] Au moins 2 classes non vides sont nécessaires (trouvé {list(by_class)}).")

    total = sum(len(imgs) for imgs in by_class.values())
    print(f"[info] {total} images trouvées dans {len(by_class)} classes : {sorted(by_class)}")

    if output_dir.exists():
        print(f"[info] {output_dir} existe déjà, purgé avant régénération (dossier dérivé, régénérable depuis {data_dir}).")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"[info] Écriture du split dans {output_dir} (val_split={args.val_split}, seed={args.seed})...")
    split_and_link(by_class, output_dir, args.val_split, args.seed)

    print(f"\n[info] Terminé. Utilisez ensuite : python train_classif_pdt.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
