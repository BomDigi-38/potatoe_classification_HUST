#!/usr/bin/env python3
"""
build_dataset_from_manifest.py — Construit un dataset <classe>/*.jpg (Mode B
pour train_classif_pdt.py) à partir d'un manifeste JSON qui associe des noms
de classe à des chemins sources arbitraires (dossiers et/ou fichiers image),
sans jamais déplacer/copier/renommer les fichiers originaux : le dossier de
sortie est peuplé uniquement de hard links (même principe que
prepare_dataset.py).

Manifeste (--manifest, JSON) :
    {
        "saine":        ["C:/photos/lot1", "C:/photos/extra_saine.jpg"],
        "malade_pvy":   ["C:/photos/lot2/pvy"],
        "malade_pstvd": ["C:/photos/lot3"]
    }
Chaque valeur est une liste de chemins ; un chemin peut être un dossier
(scanné récursivement) ou un fichier image individuel. Un fichier revendiqué
par deux classes différentes dans le manifeste est exclu des deux (affiché en
avertissement), pour ne jamais risquer un étiquetage ambigu silencieux.

Usage :
    python build_dataset_from_manifest.py --manifest classes_manifest.json --output_dir ./dataset/mon_dataset
    python train_classif_pdt.py --data_dir ./dataset/mon_dataset   # ensuite
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def collect_images(manifest):
    """Retourne (by_class, conflicts).
    by_class: {classe: [(chemin_source, [Path image, ...]), ...]}
    conflicts: {chemin_résolu: {classes qui le revendiquent}} — ces fichiers
    sont exclus de by_class."""
    raw_by_class = {}
    claims = {}

    for cls, paths in manifest.items():
        for raw_path in paths:
            src = Path(raw_path)
            if src.is_dir():
                imgs = [p for p in sorted(src.rglob("*"))
                        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
            elif src.is_file() and src.suffix.lower() in IMG_EXTENSIONS:
                imgs = [src]
            elif src.is_file():
                print(f"[warn] Ignoré (extension non supportée) : {src}")
                continue
            else:
                print(f"[warn] Chemin introuvable, ignoré : {src}")
                continue

            raw_by_class.setdefault(cls, []).append((str(src), imgs))
            for img in imgs:
                claims.setdefault(img.resolve(), set()).add(cls)

    conflicts = {p: classes for p, classes in claims.items() if len(classes) > 1}

    by_class = {}
    for cls, sources in raw_by_class.items():
        filtered_sources = []
        seen_in_class = set()
        for src_str, imgs in sources:
            kept = []
            for img in imgs:
                resolved = img.resolve()
                if resolved in conflicts or resolved in seen_in_class:
                    continue
                seen_in_class.add(resolved)
                kept.append(img)
            filtered_sources.append((src_str, kept))
        by_class[cls] = filtered_sources

    return by_class, conflicts


def print_recap(by_class: dict, conflicts: dict):
    print("\n[info] Récapitulatif par classe :\n")
    grand_total = 0
    for cls in sorted(by_class):
        sources = by_class[cls]
        cls_total = sum(len(imgs) for _, imgs in sources)
        grand_total += cls_total
        print(f"  {cls} : {cls_total} images")
        for src, imgs in sources:
            print(f"      - {src} : {len(imgs)}")

    print(f"\n[info] Total : {grand_total} images, {len(by_class)} classes.")

    if conflicts:
        print(f"\n[warn] {len(conflicts)} fichier(s) revendiqué(s) par plusieurs classes du "
              f"manifeste — EXCLUS du dataset généré (étiquetage ambigu) :")
        for resolved, classes in conflicts.items():
            print(f"      - {resolved} : {sorted(classes)}")


def materialize(by_class: dict, output_dir: Path):
    for cls, sources in by_class.items():
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        i = 0
        for _, imgs in sources:
            for img in imgs:
                (cls_dir / f"{i:06d}_{img.name}").hardlink_to(img.resolve())
                i += 1


def main():
    parser = argparse.ArgumentParser(
        description="Construit un dataset <classe>/*.jpg (hard links) à partir d'un manifeste JSON classe -> chemins.")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Fichier JSON {classe: [chemins de dossiers et/ou fichiers image, ...]}.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Dossier de sortie (Mode B pour train_classif_pdt.py) ; purgé et régénéré à chaque exécution.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        sys.exit(f"[erreur] Manifeste introuvable : {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        sys.exit("[erreur] Le manifeste doit être un objet JSON non vide {classe: [chemins...]}.")

    output_dir = Path(args.output_dir).resolve()

    # Garde-fou : ne jamais purger un dossier qui contient/est contenu par une
    # source du manifeste (le --output_dir est un dossier dérivé, régénérable,
    # jamais un dossier de données originales).
    all_sources = {Path(p).resolve() for paths in manifest.values() for p in paths if Path(p).exists()}
    for src in all_sources:
        if output_dir == src or output_dir in src.parents or src in output_dir.parents:
            sys.exit(f"[erreur] --output_dir ({output_dir}) chevauche une source du manifeste ({src}) — "
                     f"choisissez un dossier de sortie totalement séparé.")

    by_class, conflicts = collect_images(manifest)
    print_recap(by_class, conflicts)

    if output_dir.exists():
        print(f"\n[info] {output_dir} existe déjà, purgé avant régénération "
              f"(dossier dérivé, régénérable depuis le manifeste).")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"\n[info] Écriture des hardlinks dans {output_dir} ...")
    materialize(by_class, output_dir)
    print(f"\n[info] Terminé. {len(by_class)} classes écrites dans {output_dir}.")
    print(f"[info] Utilisez ensuite : python train_classif_pdt.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
