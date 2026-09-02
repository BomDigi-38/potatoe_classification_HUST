#!/usr/bin/env python3
"""
prepare_dataset.py — Fusionne les ~10 datasets sources hétérogènes de
dataset/ (noms de sous-dossiers incohérents : langues, casse, suffixes
" - OK", " -", etc.) en une structure canonique <classe>/*.jpg utilisable par
train_classif_pdt.py (Mode B) et evaluate_classif_pdt.py.

Chaque sous-dossier "feuille" du dataset brut est associé à une classe
canonique dans MAPPING (ex. les 4 variantes de "pourriture sèche" à travers 4
datasets sources sont fusionnées sous malade_dry_rot ; toutes les sources
saines fusionnent sous "saine"). Les catégories "non classées" (Defectuoso,
D-potato-Output, Crackingtype — pas de nom de maladie précis, mais pas
confirmées saines) sont regroupées sous malade_indeterminee : ce ne sont pas
des saines, donc elles comptent comme malades non spécifiées, pas comme un
3e groupe à part.

Une classe canonique dont le total fusionné (toutes ses sources confondues)
est strictement inférieur à --min_images est exclue de la sortie (mais reste
affichée dans le résumé, avec la raison). Le seuil est appliqué dynamiquement
sur le total compté à l'exécution, pas sur une liste pré-triée en dur : si le
dataset source évolue, relancer le script avec un seuil différent suffit.

Pour chaque classe gardée, --test_split fraction des images est mise à part
(hardlinks, jamais dupliquées sur disque) dans --test_dir — un jeu de test
totalement exclu de l'entraînement, pour evaluate_classif_pdt.py. Le reste va
dans --output_dir en structure plate <classe>/*.jpg (Mode B), que
train_classif_pdt.py splittera lui-même en train/val.

Usage :
    python prepare_dataset.py --root ./dataset
    python prepare_dataset.py --root ./dataset --min_images 300 --test_split 0.15
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Chemin relatif (sous --root) -> classe canonique. Une classe canonique peut
# recevoir plusieurs chemins sources (fusion inter-datasets).
MAPPING = {
    "dataset_papas_hibrido/Buen estado": "saine",
    "dataset_papas_hibrido/Defectuoso": "malade_indeterminee",

    "Healthy Potato Image Dataset - OK/Healthy_Potato_Images": "saine",

    "Original Images/Healthy Potato": "saine",
    "Original Images/Potato Dry Rot Disease": "malade_dry_rot",
    "Original Images/Potato Soft Rot Disease": "malade_soft_rot",

    "Potato disease classification - OK/Potato Image/Potato Image/D-potato-Output": "malade_indeterminee",
    "Potato disease classification - OK/Potato Image/Potato Image/H-potato-Output": "saine",

    "Potato Disease Dataset  - miOK/Black Scurf  - OK": "malade_black_scurf",
    "Potato Disease Dataset  - miOK/Blackleg _ OK": "malade_blackleg",
    "Potato Disease Dataset  - miOK/Common Scab - OK": "malade_common_scab",
    "Potato Disease Dataset  - miOK/Dry Rot -": "malade_dry_rot",
    "Potato Disease Dataset  - miOK/Healthy Potatoes -": "saine",
    "Potato Disease Dataset  - miOK/Miscellaneous -": "malade_miscellaneous",
    "Potato Disease Dataset  - miOK/Pink Rot": "malade_pink_rot",
    "Potato Disease Dataset  - miOK/Soft Rot": "malade_soft_rot",

    "Potato Disease Recognition Dataset - OK/Augmented Images/Augmented Images/Augmented Blackspot Bruising Disease": "malade_blackspot_bruising",
    "Potato Disease Recognition Dataset - OK/Augmented Images/Augmented Images/Augmented Healthy Potato": "saine",
    "Potato Disease Recognition Dataset - OK/Augmented Images/Augmented Images/Augmented Potato Brown Rot Disease": "malade_brown_rot",
    "Potato Disease Recognition Dataset - OK/Augmented Images/Augmented Images/Augmented Potato Dry Rot Disease": "malade_dry_rot",
    "Potato Disease Recognition Dataset - OK/Augmented Images/Augmented Images/Augmented Potato Soft Rot Disease": "malade_soft_rot",
    "Potato Disease Recognition Dataset - OK/Original Images/Original Images/Blackspot Bruising Disease": "malade_blackspot_bruising",
    "Potato Disease Recognition Dataset - OK/Original Images/Original Images/Healthy Potato": "saine",
    "Potato Disease Recognition Dataset - OK/Original Images/Original Images/Potato Dry Rot Disease": "malade_dry_rot",
    "Potato Disease Recognition Dataset - OK/Original Images/Original Images/Potato Soft Rot Disease": "malade_soft_rot",

    "Potato Tuber/Crackingtype": "malade_indeterminee",
    "Potato Tuber/PSTVD": "malade_pstvd",
    "Potato Tuber/PVY tuber cracking": "malade_pvy",
}


def infer_group(class_name: str) -> str:
    """Contrat de nommage partagé avec train_classif_pdt.py : tout nom de
    classe commençant par "malade" appartient au groupe malade, le reste
    (saine) au groupe saine."""
    return "malade" if class_name.startswith("malade") else "saine"


def collect_images(root: Path) -> dict:
    """Retourne {classe_canonique: [chemins image]} en agrégeant toutes les
    sources MAPPING vers cette classe."""
    by_class = {}
    for rel_path, cls in MAPPING.items():
        src_dir = root / rel_path
        if not src_dir.is_dir():
            print(f"[warn] Source introuvable, ignorée : {src_dir}")
            continue
        imgs = [p for p in sorted(src_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
        by_class.setdefault(cls, []).extend(imgs)
    return by_class


def print_summary(by_class: dict, kept_classes: set, min_images: int):
    print(f"\n[info] Classes candidates (seuil --min_images={min_images}) :\n")
    for cls in sorted(by_class):
        n = len(by_class[cls])
        status = "gardée" if cls in kept_classes else "EXCLUE (<min_images)"
        print(f"  - {cls:<28} {n:>6} images   [{status}]")

    group_totals = {}
    for cls in kept_classes:
        g = infer_group(cls)
        group_totals[g] = group_totals.get(g, 0) + len(by_class[cls])

    print("\n[info] Répartition par groupe (classes gardées uniquement) :\n")
    for g, total in sorted(group_totals.items()):
        print(f"  - {g}: {total} images")

    if len(group_totals) == 2:
        majority = max(group_totals.values())
        minority = min(group_totals.values())
        ratio = majority / minority if minority else float("inf")
        print(f"\n[info] Ratio majorité/minorité entre groupes : {ratio:.2f}")


def stratified_split_and_link(by_class: dict, kept_classes: set, output_dir: Path, test_dir, test_split: float, seed: int):
    rng = random.Random(seed)
    print()
    for cls in sorted(kept_classes):
        imgs = by_class[cls][:]
        rng.shuffle(imgs)
        n_test = max(1, int(len(imgs) * test_split)) if test_dir else 0
        test_imgs, pool_imgs = imgs[:n_test], imgs[n_test:]

        pool_dir = output_dir / cls
        pool_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(pool_imgs):
            (pool_dir / f"{i:06d}_{img.name}").hardlink_to(img.resolve())

        if test_dir:
            test_cls_dir = test_dir / cls
            test_cls_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(test_imgs):
                (test_cls_dir / f"{i:06d}_{img.name}").hardlink_to(img.resolve())

        print(f"  - {cls}: {len(pool_imgs)} pool (train+val) / {len(test_imgs)} test")


def main():
    parser = argparse.ArgumentParser(
        description="Fusionne dataset/ (multi-sources hétérogène) en une structure canonique de classification."
    )
    parser.add_argument("--root", type=str, default="./dataset", help="Dossier racine du dataset brut.")
    parser.add_argument("--output_dir", type=str, default="./dataset/data_multiclass", help="Dossier de sortie (pool train+val, Mode B pour train_classif_pdt.py).")
    parser.add_argument("--test_dir", type=str, default="./dataset/data_multiclass_test", help="Dossier de test tenu à l'écart (pour evaluate_classif_pdt.py). Chaîne vide pour désactiver.")
    parser.add_argument("--min_images", type=int, default=500, help="Classe canonique exclue si son total fusionné est strictement inférieur à ce seuil.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction de chaque classe mise à l'écart dans --test_dir.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[erreur] {root} n'existe pas ou n'est pas un dossier.")

    by_class = collect_images(root)
    kept_classes = {cls for cls, imgs in by_class.items() if len(imgs) >= args.min_images}
    if not kept_classes:
        sys.exit(f"[erreur] Aucune classe n'atteint --min_images={args.min_images}.")

    print_summary(by_class, kept_classes, args.min_images)

    output_dir = Path(args.output_dir)
    test_dir = Path(args.test_dir) if args.test_dir else None

    for d in (output_dir, test_dir):
        if d is not None and d.exists():
            print(f"\n[info] {d} existe déjà, purgé avant régénération (dossier dérivé, régénérable depuis {root}).")
            shutil.rmtree(d)

    output_dir.mkdir(parents=True, exist_ok=True)
    if test_dir:
        test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[info] Écriture des hardlinks dans {output_dir}" + (f" et {test_dir}" if test_dir else "") + " ...")
    stratified_split_and_link(by_class, kept_classes, output_dir, test_dir, args.test_split, args.seed)

    print(f"\n[info] Terminé. {len(kept_classes)} classes écrites dans {output_dir}.")


if __name__ == "__main__":
    main()
