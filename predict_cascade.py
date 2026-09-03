#!/usr/bin/env python3
"""
predict_cascade.py — Pipeline complet à 3 étages, à partir de photos brutes
(pas encore détourées) et de 3 modèles déjà entraînés (train_classif_pdt.py).
Ce script n'entraîne rien.

    segment_tubers.py : détoure chaque tubercule individuel de la photo brute
                         (réutilisé directement, mêmes réglages déjà calibrés
                         — voir segment_tubers.py, aucun dupliqué ici).

    Stage 1 (binaire, classes ["invalide", "valide"]) : le crop est-il bien
             une pomme de terre ? Les crops "invalide" s'arrêtent ici.

    Stage 2 (binaire, groupe "malade"/"saine" par préfixe de nom de classe) :
             le tubercule est-il malade ? Les crops "saine" s'arrêtent ici.

    Stage 3 (multiclasse, type de maladie précis) : appliqué uniquement aux
             crops jugés malades par le stage 2. Si la confiance maximale est
             sous --confidence_threshold, la prédiction est remplacée par
             --uncertain_label ("malade_indeterminee" par défaut) plutôt que
             de faire confiance à un choix peu sûr.

Sortie (dans --output_dir) :
    crops_segmentation/   sortie brute de segment_tubers.py (crops, debug si --debug)
    pipeline_resultats.csv   une ligne par crop détouré : prédiction de chaque
                             étage atteint + confiance + verdict final
    tri/<verdict_final>/     crops organisés par verdict (hard links, originaux intacts)

Usage :
    python predict_cascade.py --input photos/ --output_dir out_pipeline \
        --stage1_model_path Model_Etage1/.../best_model.pt \
        --stage2_model_path Model_Etage2/.../best_model.pt \
        --stage3_model_path Model_Etage3/.../best_model.pt \
        --confidence_threshold 0.5
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import segment_tubers
from evaluate_classif_pdt import build_eval_transform, build_model

IMG_EXTENSIONS = segment_tubers.IMG_EXT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_group(class_name: str) -> str:
    """Même contrat de nommage que train_classif_pdt.py/prepare_dataset.py :
    tout nom de classe commençant par "malade" appartient au groupe malade,
    le reste au groupe saine."""
    return "malade" if class_name.startswith("malade") else "saine"


class RowRecorder:
    """Remplace le csv.DictWriter réel attendu par segment_tubers.process_image
    : récupère en mémoire les lignes (crops) produites pour une image, sans
    jamais écrire de CSV segment_tubers séparé."""

    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(dict(row))


def collect_images(inputs, excludes):
    """Même logique de collecte que segment_tubers.main() (dédoublonnage,
    --exclude, rel_key préservant l'arborescence) — réutilise IMG_EXT et
    rel_key_for du module plutôt que de les redéfinir."""
    srcs = [Path(p) for p in inputs]
    exclude_paths = [Path(p).resolve() for p in excludes]
    seen = {}
    for src in srcs:
        candidates = [src] if src.is_file() else sorted(src.rglob("*"))
        for f in candidates:
            if not f.is_file() or f.suffix.lower() not in IMG_EXTENSIONS:
                continue
            resolved = f.resolve()
            if any(resolved == ex or ex in resolved.parents for ex in exclude_paths):
                continue
            if resolved in seen:
                continue
            seen[resolved] = (f, segment_tubers.rel_key_for(f, src))
    return sorted(seen.values(), key=lambda t: t[1])


def load_stage_model(path: Path, device, stage_name: str):
    ckpt = torch.load(path, map_location=device)
    for key in ("model_state", "arch", "classes"):
        if key not in ckpt:
            sys.exit(f"[erreur] Checkpoint {stage_name} invalide : clé '{key}' manquante dans {path}.")
    classes = ckpt["classes"]
    model = build_model(ckpt["arch"], num_classes=len(classes), dropout=ckpt.get("head_dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[info] {stage_name} chargé : arch={ckpt['arch']}, classes={classes}")
    return model, classes


def classify_one(model, img_tensor, device):
    """Une seule image (déjà transformée) -> (index prédit, liste de probas)."""
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(outputs, dim=1)[0]
    return int(probs.argmax().item()), probs.cpu().tolist()


def resolve_stage3_label(pred_name, confidence, uncertain_label, threshold):
    """Retourne uncertain_label si confidence est sous threshold — repli
    libre : uncertain_label n'a pas besoin d'être une classe réellement
    entraînée du stage 3 (beaucoup de checkpoints stage 3 réels n'ont pas de
    classe "indéterminée" dédiée). Sinon pred_name inchangé. threshold=0 ->
    no-op."""
    if threshold > 0 and confidence < threshold:
        return uncertain_label
    return pred_name


def classify_crop(crop_path, tf, device,
                   stage1_model, stage1_classes,
                   stage2_model, stage2_classes,
                   stage3_model, stage3_classes,
                   uncertain_label, confidence_threshold):
    """Fait passer un crop déjà détouré par les 3 étages (court-circuite dès
    qu'un étage tranche invalide/saine). Retourne un dict stage1/2/3_pred/conf
    (chaînes vides pour les étages non atteints) + verdict_final. Réutilisée
    par predict_cascade.main() et par evaluate_pipeline.py — logique de
    cascade centralisée à un seul endroit."""
    img = tf(Image.open(crop_path).convert("RGB"))
    row = {"stage1_pred": "", "stage1_conf": "", "stage2_pred": "", "stage2_conf": "",
           "stage3_pred": "", "stage3_conf": "", "verdict_final": ""}

    s1_idx, s1_probs = classify_one(stage1_model, img, device)
    s1_pred = stage1_classes[s1_idx]
    row["stage1_pred"], row["stage1_conf"] = s1_pred, f"{s1_probs[s1_idx]:.4f}"
    if s1_pred != "valide":
        row["verdict_final"] = "invalide"
        return row

    s2_idx, s2_probs = classify_one(stage2_model, img, device)
    s2_pred = stage2_classes[s2_idx]
    row["stage2_pred"], row["stage2_conf"] = s2_pred, f"{s2_probs[s2_idx]:.4f}"
    if infer_group(s2_pred) == "saine":
        row["verdict_final"] = s2_pred
        return row

    s3_idx, s3_probs = classify_one(stage3_model, img, device)
    s3_pred_name = stage3_classes[s3_idx]
    s3_conf = s3_probs[s3_idx]
    final_stage3 = resolve_stage3_label(s3_pred_name, s3_conf, uncertain_label, confidence_threshold)
    row["stage3_pred"], row["stage3_conf"] = final_stage3, f"{s3_conf:.4f}"
    row["verdict_final"] = final_stage3
    return row


CSV_FIELDS = ["image", "rel_key", "crop_path", "segment_quality",
              "stage1_pred", "stage1_conf",
              "stage2_pred", "stage2_conf",
              "stage3_pred", "stage3_conf",
              "verdict_final"]


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet (segment_tubers -> stage1 validité -> stage2 malade/saine "
                    "-> stage3 type de maladie) à partir de photos brutes et de 3 checkpoints déjà entraînés.")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Photo(s)/dossier(s) bruts à analyser.")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier de sortie.")
    parser.add_argument("--exclude", type=str, nargs="+", default=[], help="Fichier(s)/dossier(s) à exclure de --input.")
    parser.add_argument("--stage1_model_path", type=str, required=True, help="Checkpoint binaire validité (classes invalide/valide).")
    parser.add_argument("--stage2_model_path", type=str, required=True, help="Checkpoint binaire malade/saine.")
    parser.add_argument("--stage3_model_path", type=str, required=True, help="Checkpoint multiclasse type de maladie.")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Stage 3 : si la confiance max est sous ce seuil, la prédiction est "
                             "remplacée par --uncertain_label (0 = désactivé).")
    parser.add_argument("--uncertain_label", type=str, default="malade_indeterminee",
                        help="Nom de la classe stage 3 vers laquelle basculer une prédiction peu "
                             "confiante — doit être une classe réelle du checkpoint stage 3.")
    parser.add_argument("--img_size", type=int, default=224, help="Doit correspondre à la valeur utilisée à l'entraînement des 3 modèles.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Passe --debug à segment_tubers.py (overlays de contrôle).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"[info] Device: {device}")

    stage1_path = Path(args.stage1_model_path)
    stage2_path = Path(args.stage2_model_path)
    stage3_path = Path(args.stage3_model_path)
    for p, name in ((stage1_path, "--stage1_model_path"), (stage2_path, "--stage2_model_path"), (stage3_path, "--stage3_model_path")):
        if not p.is_file():
            sys.exit(f"[erreur] {name} : {p} n'existe pas.")

    stage1_model, stage1_classes = load_stage_model(stage1_path, device, "Stage 1 (validité)")
    stage2_model, stage2_classes = load_stage_model(stage2_path, device, "Stage 2 (malade/saine)")
    stage3_model, stage3_classes = load_stage_model(stage3_path, device, "Stage 3 (type de maladie)")

    if len(stage1_classes) != 2 or "valide" not in stage1_classes:
        sys.exit(f"[erreur] Le stage 1 doit être binaire avec une classe 'valide', trouvé : {stage1_classes}.")
    stage2_saine_candidates = [i for i, c in enumerate(stage2_classes) if infer_group(c) == "saine"]
    if len(stage2_classes) != 2 or len(stage2_saine_candidates) != 1:
        sys.exit(f"[erreur] Le stage 2 doit être binaire avec une classe du groupe 'saine', trouvé : {stage2_classes}.")

    output_dir = Path(args.output_dir)
    seg_out_root = output_dir / "crops_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_argv = ["--input", *args.input, "--output", str(seg_out_root)]
    if args.debug:
        seg_argv.append("--debug")
    seg_args = segment_tubers.parse_args(seg_argv)
    seg_args.device = device.type

    print(f"[info] Chargement du modèle de segmentation ({seg_args.backend})...")
    gen, ctx = segment_tubers.build_generator(seg_args, device.type)

    files = collect_images(args.input, args.exclude)
    if not files:
        sys.exit(f"[erreur] Aucune image trouvée dans {args.input}.")
    print(f"[info] {len(files)} photo(s) à analyser.")

    tf = build_eval_transform(args.img_size)
    tri_dir = output_dir / "tri"
    verdict_counts = {}
    csv_path = output_dir / "pipeline_resultats.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for k, (path, rel_key) in enumerate(files, 1):
            print(f"[{k}/{len(files)}] {rel_key}{path.suffix}")
            recorder = RowRecorder()
            segment_tubers.process_image(path, rel_key, gen, ctx, seg_args, recorder, seg_out_root)

            for crop_row in recorder.rows:
                crop_path = seg_out_root / crop_row["crop_path"]

                row = {"image": path.name, "rel_key": rel_key, "crop_path": str(crop_path),
                       "segment_quality": crop_row["quality"]}
                row.update(classify_crop(
                    crop_path, tf, device,
                    stage1_model, stage1_classes,
                    stage2_model, stage2_classes,
                    stage3_model, stage3_classes,
                    args.uncertain_label, args.confidence_threshold))

                writer.writerow(row)
                verdict = row["verdict_final"]
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

                cls_dir = tri_dir / verdict
                cls_dir.mkdir(parents=True, exist_ok=True)
                link_path = cls_dir / crop_path.name
                if not link_path.exists():
                    link_path.hardlink_to(crop_path.resolve())

    print(f"\n[info] Résultats par verdict final :")
    for verdict, n in sorted(verdict_counts.items()):
        print(f"  - {verdict}: {n}")
    print(f"[info] Détail par crop -> {csv_path}")
    print(f"[info] Crops organisés par verdict (hardlinks) -> {tri_dir}")


if __name__ == "__main__":
    main()
