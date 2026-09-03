#!/usr/bin/env python3
"""
generate_all_datasets.py — Génère et split automatiquement les 3 étages du pipeline :
- Étage 1 (Validation de crops) : Distingue 'valide' et 'invalide' en détectant
  les dossiers d'erreurs (contenant 'error') à partir de final_sort.
- Étage 2 (Binaire) : 'saine' vs 'malade' (à partir de data_03092026).
- Étage 3 (Multiclasse) : Dry rot, Soft rot, Blackspot, PSTVD (explicitement inclus),
  et un groupe 'malade_inconnu' pour le reste (à partir de data_03092026).
"""

import argparse
import shutil
import sys
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def create_hardlinks(src_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    for img in sorted(src_dir.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMG_EXTENSIONS:
            target = dest_dir / f"{i:06d}_{img.name}"
            target.hardlink_to(img.resolve())
            i += 1


def split_dataset(data_dir: Path, output_dir: Path, val_split: float = 0.2, seed: int = 42):
    import random
    rng = random.Random(seed)
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    for cls_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        imgs = [p for p in sorted(cls_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
        if not imgs:
            continue
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_split))
        val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

        for split_name, split_imgs in (("train", train_imgs), ("val", val_imgs)):
            out_dir = output_dir / split_name / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(split_imgs):
                (out_dir / f"{i:06d}_{img.name}").hardlink_to(img.resolve())
        print(f"  - {cls_dir.name}: {len(train_imgs)} train / {len(val_imgs)} val")


def main():
    parser = argparse.ArgumentParser(description="Génère et split les 3 étages du pipeline.")
    parser.add_argument("--source_dir", type=str, required=True, help="Dossier unifié Mode B (ex: dataset/data_03092026).")
    parser.add_argument("--stage1_dir", type=str, required=True, help="Dossier brut non-augmenté final_sort pour l'Étage 1.")
    parser.add_argument("--output_base", type=str, default="./dataset", help="Dossier racine pour stocker les datasets finaux.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    stage1_dir = Path(args.stage1_dir).resolve()
    output_base = Path(args.output_base).resolve()

    if not source_dir.is_dir():
        sys.exit(f"[erreur] Dossier source introuvable : {source_dir}")
    if not stage1_dir.is_dir():
        sys.exit(f"[erreur] Dossier Stage 1 introuvable : {stage1_dir}")

    # --- ÉTAGE 1 : Validation de Crop (Valide vs Invalide / Erreurs) ---
    print("=== 1. Génération de l'Étage 1 : Validation de Crop (Valide / Invalide) ===")
    etage1_raw = output_base / "etage1_crop_raw"
    if etage1_raw.exists(): shutil.rmtree(etage1_raw)
    
    valide_dir = etage1_raw / "valide"
    invalide_dir = etage1_raw / "invalide"
    valide_dir.mkdir(parents=True, exist_ok=True)
    invalide_dir.mkdir(parents=True, exist_ok=True)

    i_val, i_inv = 0, 0
    for p in sorted(stage1_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            # Si un des parents du fichier contient 'error' (ex: dis _error, errror), c'est une erreur
            is_error = any("error" in part.lower() for part in p.parts)
            if is_error:
                target = invalide_dir / f"{i_inv:06d}_{p.name}"
                target.hardlink_to(p.resolve())
                i_inv += 1
            else:
                target = valide_dir / f"{i_val:06d}_{p.name}"
                target.hardlink_to(p.resolve())
                i_val += 1

    print(f"Trouvé : {i_val} crops valides et {i_inv} crops invalides/erreurs.")
    print(f"Split de l'Étage 1 vers {output_base / 'etage1_crop_split'} ...")
    split_dataset(etage1_raw, output_base / "etage1_crop_split")

    # --- ÉTAGE 2 : Binaire (Sain vs Malade) ---
    print("\n=== 2. Génération de l'Étage 2 : Binaire (Sain vs Malade) ===")
    etage2_raw = output_base / "etage2_binaire_raw"
    if etage2_raw.exists(): shutil.rmtree(etage2_raw)
    
    for cls_dir in source_dir.iterdir():
        if not cls_dir.is_dir(): continue
        target_name = "saine" if cls_dir.name == "saine" else "malade"
        create_hardlinks(cls_dir, etage2_raw / target_name)
        
    print(f"Split de l'Étage 2 vers {output_base / 'etage2_binaire_split'} ...")
    split_dataset(etage2_raw, output_base / "etage2_binaire_split")

    # --- ÉTAGE 3 : Multiclasse (Maladies explicites + PSTVD + Inconnu) ---
    print("\n=== 3. Génération de l'Étage 3 : Multiclasse fin (PSTVD inclus) ===")
    etage3_raw = output_base / "etage3_multiclass_raw"
    if etage3_raw.exists(): shutil.rmtree(etage3_raw)
    
    for cls_dir in source_dir.iterdir():
        if not cls_dir.is_dir(): continue
        if cls_dir.name == "saine": continue  # Exclu pour l'étage 3
        
        target_name = cls_dir.name
        # On garde explicitement dry_rot, soft_rot, blackspot_bruising et pstvd
        if target_name not in ["malade_dry_rot", "malade_soft_rot", "malade_blackspot_bruising", "malade_pstvd"]:
            target_name = "malade_inconnu"  # Regroupement des autres (brown rot, indéterminée, etc.)
            
        create_hardlinks(cls_dir, etage3_raw / target_name)
        
    print(f"Split de l'Étage 3 vers {output_base / 'etage3_multiclass_split'} ...")
    split_dataset(etage3_raw, output_base / "etage3_multiclass_split")

    print("\n[info] Terminé avec succès ! Tous les datasets prêts pour l'entraînement sont dans :")
    print(f" - Étage 1 : {output_base / 'etage1_crop_split'}")
    print(f" - Étage 2 : {output_base / 'etage2_binaire_split'}")
    print(f" - Étage 3 : {output_base / 'etage3_multiclass_split'}")


if __name__ == "__main__":
    main()