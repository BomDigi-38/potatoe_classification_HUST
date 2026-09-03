#!/usr/bin/env python3
"""
evaluate_pipeline.py — Évalue les 3 étages du pipeline (segment_tubers.py +
Stage 1 validité + Stage 2 malade/saine + Stage 3 type de maladie) en une
seule commande, à partir d'un dossier classifié Mode B (<classe>/*.jpg,
ex. saine/, malade_dry_rot/, ..., malade_indeterminee/ — construit par
exemple via build_dataset_from_manifest.py).

Réutilise directement predict_cascade.py (RowRecorder, collect_images,
load_stage_model, classify_one, classify_crop, infer_group) : la cascade des
3 étages n'est écrite qu'à un seul endroit dans tout le projet.

Chaque image du dossier classifié est déjà un tubercule individuel détouré :
    - Étage 1 : le nombre attendu de tubercules "valide" y est donc toujours 1
      (contrairement à de vraies photos de tas en vrac). Le taux de "1 valide
      trouvé pile" mesure la qualité de segment_tubers.py + CNN de validité.
    - Étage 2 : le nom du dossier (via infer_group, "malade"/"saine") sert de
      vérité terrain.
    - Étage 3 : le nom exact du dossier sert de vérité terrain, évalué
      uniquement sur les images où la cascade a atteint cet étage (Étage 1 a
      trouvé exactement 1 crop valide ET l'Étage 2 a prédit "malade") — cf.
      l'ancien mode évaluation de predict_cascade.py, même logique.

Sortie (dans --output_dir) :
    evaluation_pipeline.json   3 sections (stage1/stage2/stage3) + détail par image
    detail_par_image.csv       une ligne par image
    matrice_confusion_stage2.png / _stage3.png, rapport_classification_stage2.txt / _stage3.txt
        (si scikit-learn/matplotlib disponibles)

Usage :
    python evaluate_pipeline.py --data_dir dataset_test_pipeline --output_dir eval_pipeline_out \
        --stage1_model_path Model_Etage1/.../best_model.pt \
        --stage2_model_path Model_Etage2/.../best_model.pt \
        --stage3_model_path Model_Etage3/.../best_model.pt \
        --confidence_threshold 0.5
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

import segment_tubers
from evaluate_classif_pdt import build_eval_transform
from predict_cascade import (RowRecorder, classify_crop, infer_group,
                              load_stage_model)

try:
    from sklearn.metrics import classification_report
    import matplotlib
    matplotlib.use("Agg")
    from evaluate_classif_pdt import plot_confusion_matrix
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def list_class_images(data_dir: Path):
    """Retourne [(chemin, classe), ...] pour toutes les images de data_dir/<classe>/."""
    items = []
    for cls_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.rglob("*")):
            if img.is_file() and img.suffix.lower() in segment_tubers.IMG_EXT:
                items.append((img, cls_dir.name))
    return items


DETAIL_FIELDS = ["image", "vraie_classe", "total_crops_bruts", "n_valide", "stage1_ok",
                 "stage2_pred", "stage2_ok", "stage3_pred", "stage3_ok"]


def write_report(output_dir, name, true_labels, pred_labels, classes):
    """true_labels/pred_labels : indices dans classes. Écrit rapport +
    matrice de confusion si scikit-learn/matplotlib disponibles."""
    if not HAS_PLOT or not true_labels:
        return None
    report_labels = list(range(len(classes)))
    report = classification_report(true_labels, pred_labels, labels=report_labels,
                                    target_names=classes, digits=3, zero_division=0)
    with open(output_dir / f"rapport_classification_{name}.txt", "w", encoding="utf-8") as f:
        f.write(report)
    plot_confusion_matrix(true_labels, pred_labels, classes, output_dir / f"matrice_confusion_{name}.png")
    return classification_report(true_labels, pred_labels, labels=report_labels,
                                  target_names=classes, digits=3, zero_division=0, output_dict=True)


def build_and_write_summary(output_dir, details, stage2_true, stage2_pred, stage3_true, stage3_pred,
                             stage2_classes_report, stage3_classes_report):
    """Recalcule le résumé à partir de ce qui a été traité jusqu'ici et
    l'écrit dans evaluation_pipeline.json — appelée périodiquement pendant la
    boucle (pas seulement à la fin) pour qu'un arrêt en cours de route laisse
    un résumé exploitable, pas juste le CSV détail brut."""
    n_stage1_ok = sum(d["stage1_ok"] for d in details)
    n_stage2_evalues = sum(1 for d in details if d["stage2_ok"] is not None)
    n_stage2_ok = sum(1 for d in details if d["stage2_ok"])
    n_stage3_evalues = sum(1 for d in details if d["stage3_ok"] is not None)
    n_stage3_ok = sum(1 for d in details if d["stage3_ok"])

    summary = {
        "n_images_traitees": len(details),
        "stage1": {
            "n_images": len(details),
            "n_exactement_1_valide": n_stage1_ok,
            "taux": round(n_stage1_ok / len(details), 4) if details else None,
        },
        "stage2": {
            "n_images_evaluees": n_stage2_evalues,
            "n_corrects": n_stage2_ok,
            "taux": round(n_stage2_ok / n_stage2_evalues, 4) if n_stage2_evalues else None,
            "rapport": write_report(output_dir, "stage2", stage2_true, stage2_pred, stage2_classes_report),
        },
        "stage3": {
            "n_images_evaluees": n_stage3_evalues,
            "n_corrects": n_stage3_ok,
            "taux": round(n_stage3_ok / n_stage3_evalues, 4) if n_stage3_evalues else None,
            "rapport": write_report(output_dir, "stage3", stage3_true, stage3_pred, stage3_classes_report),
        },
    }
    with open(output_dir / "evaluation_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Évalue les 3 étages du pipeline (segment_tubers + stage1/2/3) sur un dossier classifié.")
    parser.add_argument("--data_dir", type=str, required=True, help="Dossier <classe>/*.jpg (Mode B), vérité terrain.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage1_model_path", type=str, required=True)
    parser.add_argument("--stage2_model_path", type=str, required=True)
    parser.add_argument("--stage3_model_path", type=str, required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Stage 3 : voir predict_cascade.py --confidence_threshold.")
    parser.add_argument("--uncertain_label", type=str, default="malade_indeterminee")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--debug", action="store_true", help="Passe --debug à segment_tubers.py.")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Fraction du dataset à évaluer (0-1], tirage aléatoire stratifié par "
                             "classe (jamais moins d'1 image gardée par classe). 1.0 = tout le dataset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"[info] Device: {device}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"[erreur] {data_dir} n'existe pas ou n'est pas un dossier.")

    stage1_model, stage1_classes = load_stage_model(Path(args.stage1_model_path), device, "Stage 1 (validité)")
    stage2_model, stage2_classes = load_stage_model(Path(args.stage2_model_path), device, "Stage 2 (malade/saine)")
    stage3_model, stage3_classes = load_stage_model(Path(args.stage3_model_path), device, "Stage 3 (type de maladie)")

    if "valide" not in stage1_classes:
        sys.exit(f"[erreur] Le stage 1 doit avoir une classe 'valide', trouvé : {stage1_classes}.")

    items = list_class_images(data_dir)
    if not items:
        sys.exit(f"[erreur] Aucune image trouvée dans {data_dir}.")

    if not (0 < args.sample_frac <= 1.0):
        sys.exit(f"[erreur] --sample_frac doit être dans ]0, 1], reçu {args.sample_frac}.")
    if args.sample_frac < 1.0:
        rng = random.Random(args.seed)
        by_class = {}
        for img, cls in items:
            by_class.setdefault(cls, []).append(img)
        sampled = []
        for cls, imgs in by_class.items():
            imgs = imgs[:]
            rng.shuffle(imgs)
            n_keep = max(1, round(len(imgs) * args.sample_frac))
            sampled.extend((img, cls) for img in imgs[:n_keep])
        items = sampled
        print(f"[info] --sample_frac {args.sample_frac} : {len(items)} images retenues "
              f"(tirage aléatoire stratifié par classe, seed={args.seed}).")

    ground_truth_classes = sorted({c for _, c in items})
    print(f"[info] {len(items)} images dans {len(ground_truth_classes)} classes : {ground_truth_classes}")

    output_dir = Path(args.output_dir)
    seg_out_root = output_dir / "crops_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_argv = ["--input", str(data_dir), "--output", str(seg_out_root)]
    if args.debug:
        seg_argv.append("--debug")
    seg_args = segment_tubers.parse_args(seg_argv)
    seg_args.device = device.type
    print(f"[info] Chargement du modèle de segmentation ({seg_args.backend})...")
    gen, ctx = segment_tubers.build_generator(seg_args, device.type)

    tf = build_eval_transform(args.img_size)

    stage2_classes_report = ["malade", "saine"]
    stage3_classes_report = sorted(set(stage3_classes)
                                    | {c for c in ground_truth_classes if infer_group(c) == "malade"}
                                    | ({args.uncertain_label} if args.confidence_threshold > 0 else set()))

    details = []
    stage2_true, stage2_pred = [], []
    stage3_true, stage3_pred = [], []
    summary_path = output_dir / "evaluation_pipeline.json"
    csv_path = output_dir / "detail_par_image.csv"
    SUMMARY_REFRESH_EVERY = 50  # rafraîchit evaluation_pipeline.json toutes les N images, pas seulement à la fin

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_fh:
        csv_writer = csv.DictWriter(csv_fh, fieldnames=DETAIL_FIELDS)
        csv_writer.writeheader()

        for k, (path, true_class) in enumerate(items, 1):
            rel_key = f"{true_class}__{path.stem}"  # unique même si 2 classes partagent un nom de fichier
            print(f"[{k}/{len(items)}] {true_class}/{path.name}", end=" ")
            recorder = RowRecorder()
            segment_tubers.process_image(path, rel_key, gen, ctx, seg_args, recorder, seg_out_root)

            n_valide_rows = []
            for crop_row in recorder.rows:
                crop_path = seg_out_root / crop_row["crop_path"]
                s1_row = classify_crop(crop_path, tf, device, stage1_model, stage1_classes,
                                        stage2_model, stage2_classes, stage3_model, stage3_classes,
                                        args.uncertain_label, args.confidence_threshold)
                if s1_row["stage1_pred"] == "valide":
                    n_valide_rows.append(s1_row)

            stage1_ok = len(n_valide_rows) == 1
            detail = {"image": f"{true_class}/{path.name}", "vraie_classe": true_class,
                      "total_crops_bruts": len(recorder.rows), "n_valide": len(n_valide_rows),
                      "stage1_ok": stage1_ok, "stage2_pred": "", "stage2_ok": None,
                      "stage3_pred": "", "stage3_ok": None}

            if stage1_ok:
                row = n_valide_rows[0]
                true_group = infer_group(true_class)
                pred_group = infer_group(row["stage2_pred"])
                detail["stage2_pred"] = row["stage2_pred"]
                detail["stage2_ok"] = (pred_group == true_group)
                stage2_true.append(stage2_classes_report.index(true_group))
                stage2_pred.append(stage2_classes_report.index(pred_group))

                if true_group == "malade" and pred_group == "malade":
                    detail["stage3_pred"] = row["verdict_final"]
                    detail["stage3_ok"] = (row["verdict_final"] == true_class)
                    # true_class et verdict_final sont toujours dans stage3_classes_report par
                    # construction (union stage3_classes / classes malade du dossier / uncertain_label).
                    if true_class in stage3_classes_report and row["verdict_final"] in stage3_classes_report:
                        stage3_true.append(stage3_classes_report.index(true_class))
                        stage3_pred.append(stage3_classes_report.index(row["verdict_final"]))

            details.append(detail)
            csv_writer.writerow(detail)
            csv_fh.flush()  # force l'écriture sur disque immédiatement (survit à un arrêt brutal)
            print("OK" if stage1_ok else f"echec stage1 (n_valide={len(n_valide_rows)})")

            if k % SUMMARY_REFRESH_EVERY == 0:
                build_and_write_summary(output_dir, details, stage2_true, stage2_pred,
                                         stage3_true, stage3_pred, stage2_classes_report, stage3_classes_report)

    summary = build_and_write_summary(output_dir, details, stage2_true, stage2_pred,
                                       stage3_true, stage3_pred, stage2_classes_report, stage3_classes_report)

    print(f"\n[info] Étage 1 : {summary['stage1']['n_exactement_1_valide']}/{len(details)} images "
          f"avec exactement 1 crop valide ({summary['stage1']['taux'] * 100:.1f}%).")
    if summary["stage2"]["n_images_evaluees"]:
        print(f"[info] Étage 2 : {summary['stage2']['n_corrects']}/{summary['stage2']['n_images_evaluees']} "
              f"corrects ({summary['stage2']['taux'] * 100:.1f}%).")
    if summary["stage3"]["n_images_evaluees"]:
        print(f"[info] Étage 3 : {summary['stage3']['n_corrects']}/{summary['stage3']['n_images_evaluees']} "
              f"corrects ({summary['stage3']['taux'] * 100:.1f}%).")
    print(f"[info] Résumé complet -> {summary_path}")
    print(f"[info] Détail par image -> {csv_path}")


if __name__ == "__main__":
    main()
