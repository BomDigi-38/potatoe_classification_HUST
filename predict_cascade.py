#!/usr/bin/env python3
"""
predict_cascade.py — Classification à deux étages, à partir de deux modèles
déjà entraînés (train_classif_pdt.py). Ce script n'entraîne rien, ne modifie
aucun des scripts existants : il charge deux checkpoints .pt et enchaîne :

    Stage 1 (binaire, ex. V1_savemodels/.../best_model.pt, classes
             ["malades", "saines"]) : chaque image est classée saine/malade
             en argmax. Les images "saine" ressortent directement ainsi,
             SANS jamais passer par le stage 2.

    Stage 2 (multiclasse, ex. Model/<run>/best_model.pt, classes
             malade_dry_rot, ..., saine) : appliqué UNIQUEMENT aux images que
             le stage 1 a jugées malades, pour obtenir le nom précis de la
             maladie (ou malade_indeterminee).

Le stage 1 doit être un modèle binaire (2 classes) — c'est un filtre, pas un
classifieur fin. Le stage 2 peut être entraîné sur un sous-ensemble des
maladies seulement (ex. --exclude_classes utilisé à l'entraînement) : il n'a
pas besoin de connaître "saine" lui-même, cette sortie est ajoutée par ce
script pour les images court-circuitées au stage 1.

Deux modes, détectés automatiquement sur la structure de --data_dir :

    Mode évaluation (--data_dir a des sous-dossiers de classe, ex.
    dataset/data_multiclass_test/) : la taxonomie de référence pour le
    rapport est celle du DOSSIER DE TEST (pas forcément celle du stage 2) —
    le stage 2 doit juste couvrir un sous-ensemble de ces classes, plus
    "saine" doit être parmi les classes du dossier de test. Toute vraie
    classe que le stage 2 ne connaît pas (ex. une maladie explicitement
    exclue à l'entraînement) ne pourra jamais être prédite correctement :
    elle apparaît comme erreur systématique dans le rapport — c'est le
    comportement attendu, pas un bug, ça montre où la cascade "retombe"
    faute d'avoir appris cette classe.
        - accuracy binaire du stage 1 seul, accuracy de la cascade complète
        - rapport de classification, matrice de confusion (taxonomie du jeu de test)
        - images_mal_classees.csv, metrics.json

    Mode prédiction (--data_dir est un dossier plat d'images, pas de
    sous-dossiers — aucune vérité terrain) :
        - tri_images.csv : une ligne par image (prédictions des deux étages
          + prédiction finale + confiance)
        - tri/<classe_predite>/... : les images organisées par classe prédite
          via hardlinks (ne duplique ni ne déplace les fichiers d'origine)

Usage :
    python predict_cascade.py --stage1_model_path V1_savemodels/alldataset_v1/outputs_classif/best_model.pt --stage2_model_path Model/<run>/best_model.pt --data_dir ./dataset/data_multiclass_test
    python predict_cascade.py --stage1_model_path ... --stage2_model_path ... --data_dir ./nouvelles_images_sans_etiquette
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from evaluate_classif_pdt import (
    build_eval_transform,
    build_model,
    parse_class_map,
    plot_confidence_histogram,
    plot_confusion_matrix,
    write_misclassified_csv,
)

try:
    import matplotlib  # noqa: F401 (déjà configuré en mode "Agg" par evaluate_classif_pdt.py)
    from sklearn.metrics import classification_report
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_group(class_name: str) -> str:
    """Même contrat de nommage que train_classif_pdt.py/prepare_dataset.py :
    tout nom de classe commençant par "malade" (singulier ou pluriel, ex.
    "malades", "malade_dry_rot") appartient au groupe malade, le reste au
    groupe saine."""
    return "malade" if class_name.startswith("malade") else "saine"


class FlatImageDataset(Dataset):
    """Dataset minimal pour un dossier plat d'images sans sous-dossiers de
    classe (mode prédiction : pas de vérité terrain)."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


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


def predict_probs(model, loader, device):
    """Passe d'inférence pure, alignée sur l'ordre du DataLoader
    (shuffle=False) : retourne (all_preds, all_probs)."""
    all_preds, all_probs = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_preds, all_probs


def apply_uncertainty(pred_idx, probs, uncertain_idx, threshold):
    """Stage 2 (type de maladie) seulement : bascule vers uncertain_idx si la
    confiance max est sous threshold. threshold=0 ou uncertain_idx=None ->
    no-op (comportement identique à avant l'ajout de cette option)."""
    if threshold > 0 and uncertain_idx is not None and max(probs) < threshold:
        return uncertain_idx
    return pred_idx


def run_evaluation_mode(data_dir, tf, stage1_model, stage1_classes, stage1_saine_idx,
                         stage2_model, stage2_classes, class_map, batch_size, num_workers,
                         device, output_dir, no_plots, confidence_threshold, uncertain_idx):
    try:
        eval_ds = datasets.ImageFolder(data_dir, transform=tf)
    except FileNotFoundError as e:
        sys.exit(f"[erreur] Impossible de charger {data_dir} : {e}")

    if len(eval_ds.classes) < 2:
        sys.exit(f"[erreur] Attendu au moins 2 sous-dossiers de classe dans {data_dir}, trouvé : {eval_ds.classes}.")

    # La taxonomie de référence pour le rapport est celle du jeu de test
    # (pas celle du stage 2 : le stage 2 peut ne couvrir qu'un sous-ensemble
    # des maladies, ex. entraîné avec --exclude_classes). --class_map permet
    # juste de renommer des dossiers dont le nom diffère ; eval_ds.targets
    # reste valide tel quel (aucun réindexage nécessaire, contrairement à
    # reconcile_classes qui alignait sur l'ordre du checkpoint).
    classes = [class_map.get(c, c) for c in eval_ds.classes] if class_map else list(eval_ds.classes)

    missing_stage2_classes = set(stage2_classes) - set(classes)
    if missing_stage2_classes:
        sys.exit(f"[erreur] Le stage 2 a des classes absentes du jeu de test {data_dir} : "
                  f"{sorted(missing_stage2_classes)}. Classes du jeu de test : {classes}.")
    if "saine" not in classes:
        sys.exit(f"[erreur] 'saine' doit être une des classes du jeu de test {data_dir} "
                  f"(nécessaire pour représenter les images court-circuitées au stage 1). Trouvé : {classes}.")

    saine_idx = classes.index("saine")
    stage2_name_to_report_idx = {name: classes.index(name) for name in stage2_classes}
    known_stage2_classes = set(stage2_classes)
    unreachable = [c for c in classes if c != "saine" and c not in known_stage2_classes]
    if unreachable:
        print(f"[warn] Classes du jeu de test que le stage 2 ne connaît pas (jamais prédites correctement) : {unreachable}")

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"[info] Mode évaluation : {len(eval_ds)} images, classes={classes}")

    sample_paths = [p for p, _ in eval_ds.samples]
    true_labels = eval_ds.targets

    stage1_preds, stage1_probs = predict_probs(stage1_model, loader, device)
    stage1_groups = [infer_group(stage1_classes[p]) for p in stage1_preds]

    malade_indices = [i for i, g in enumerate(stage1_groups) if g == "malade"]
    stage2_preds_by_idx, stage2_probs_by_idx = {}, {}
    if malade_indices:
        sub_loader = DataLoader(Subset(eval_ds, malade_indices), batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
        sub_preds, sub_probs = predict_probs(stage2_model, sub_loader, device)
        for local_i, global_i in enumerate(malade_indices):
            stage2_preds_by_idx[global_i] = sub_preds[local_i]
            stage2_probs_by_idx[global_i] = sub_probs[local_i]

    final_preds, final_probs = [], []
    for i in range(len(eval_ds)):
        if stage1_groups[i] == "saine":
            final_preds.append(saine_idx)
            # Le stage 2 n'a jamais tourné sur cette image : on ne peut pas
            # reconstituer une vraie distribution sur les 7 classes, mais
            # write_misclassified_csv/plot_confidence_histogram ne lisent que
            # probs[pred_idx] ou max(probs) — on y place donc la confiance
            # réelle du stage 1 sur "saine", 0 ailleurs.
            probs_vec = [0.0] * len(classes)
            probs_vec[saine_idx] = stage1_probs[i][stage1_saine_idx]
            final_probs.append(probs_vec)
        else:
            s2_pred_local = stage2_preds_by_idx[i]
            s2_probs_local = stage2_probs_by_idx[i]
            s2_pred_local = apply_uncertainty(s2_pred_local, s2_probs_local, uncertain_idx, confidence_threshold)
            pred_name = stage2_classes[s2_pred_local]
            final_preds.append(stage2_name_to_report_idx[pred_name])
            # Replace chaque probabilité du stage 2 (indexée localement dans
            # stage2_classes) à sa position dans l'espace de classes du jeu
            # de test, potentiellement plus grand (ex. stage 2 à 4 classes,
            # jeu de test à 7). max(probs_vec) == max(s2_probs_local), donc
            # la confiance rapportée reste correcte.
            probs_vec = [0.0] * len(classes)
            for local_idx, name in enumerate(stage2_classes):
                probs_vec[stage2_name_to_report_idx[name]] = s2_probs_local[local_idx]
            final_probs.append(probs_vec)

    accuracy_cascade = sum(1 for t, p in zip(true_labels, final_preds) if t == p) / len(true_labels)
    true_groups = [infer_group(classes[t]) for t in true_labels]
    accuracy_stage1 = sum(1 for tg, sg in zip(true_groups, stage1_groups) if tg == sg) / len(true_groups)
    n_shortcut = len(eval_ds) - len(malade_indices)
    n_stage2 = len(malade_indices)

    print(f"[info] Accuracy stage 1 (binaire malade/saine) : {accuracy_stage1:.4f}")
    print(f"[info] Images court-circuitées à l'étage 1 (saine) : {n_shortcut} | envoyées à l'étage 2 : {n_stage2}")
    print(f"[info] Accuracy cascade complète ({len(classes)} classes) : {accuracy_cascade:.4f}")

    write_misclassified_csv(output_dir / "images_mal_classees.csv", sample_paths, true_labels, final_preds, final_probs, classes)

    metrics = {
        "accuracy_cascade": accuracy_cascade,
        "accuracy_stage1_binaire": accuracy_stage1,
        "n_samples": len(eval_ds),
        "n_shortcut_saine": n_shortcut,
        "n_envoyees_stage2": n_stage2,
        "classes": classes,
        "classification_report": None,
    }

    if HAS_PLOT and not no_plots:
        report_labels = list(range(len(classes)))
        report_txt = classification_report(true_labels, final_preds, labels=report_labels, target_names=classes, digits=3)
        print("\n[rapport de classification - cascade]\n" + report_txt)
        with open(output_dir / "rapport_classification.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)
        metrics["classification_report"] = classification_report(
            true_labels, final_preds, labels=report_labels, target_names=classes, digits=3, output_dict=True
        )
        plot_confusion_matrix(true_labels, final_preds, classes, output_dir / "matrice_confusion.png")
        plot_confidence_histogram(true_labels, final_preds, final_probs, output_dir / "histogramme_confiance.png")
    else:
        if no_plots:
            print("[info] --no_plots activé : seuls metrics.json et images_mal_classees.csv sont écrits.")
        else:
            print("[warn] matplotlib/scikit-learn absents : rapport et matrice de confusion non générés.")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[info] Terminé. Résultats écrits dans {output_dir}.")


def run_prediction_mode(data_dir, tf, stage1_model, stage1_classes, stage1_saine_idx,
                         stage2_model, stage2_classes, batch_size, num_workers, device, output_dir,
                         confidence_threshold, uncertain_idx):
    paths = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
    if not paths:
        sys.exit(f"[erreur] Aucune image trouvée directement dans {data_dir}.")
    print(f"[info] Mode prédiction : {len(paths)} images sans étiquette dans {data_dir}")

    loader = DataLoader(FlatImageDataset(paths, tf), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    stage1_preds, stage1_probs = predict_probs(stage1_model, loader, device)
    stage1_groups = [infer_group(stage1_classes[p]) for p in stage1_preds]

    malade_indices = [i for i, g in enumerate(stage1_groups) if g == "malade"]
    stage2_preds_by_idx, stage2_probs_by_idx = {}, {}
    if malade_indices:
        sub_paths = [paths[i] for i in malade_indices]
        sub_loader = DataLoader(FlatImageDataset(sub_paths, tf), batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
        sub_preds, sub_probs = predict_probs(stage2_model, sub_loader, device)
        for local_i, global_i in enumerate(malade_indices):
            stage2_preds_by_idx[global_i] = sub_preds[local_i]
            stage2_probs_by_idx[global_i] = sub_probs[local_i]

    csv_rows, final_pred_names = [], []
    for i, path in enumerate(paths):
        stage1_pred_name = stage1_classes[stage1_preds[i]]
        stage1_conf = stage1_probs[i][stage1_preds[i]]
        if stage1_groups[i] == "saine":
            stage2_pred_name, stage2_conf_str = "", ""
            final_pred_name = "saine"
            final_conf = stage1_probs[i][stage1_saine_idx]
        else:
            s2_probs = stage2_probs_by_idx[i]
            s2_pred = apply_uncertainty(stage2_preds_by_idx[i], s2_probs, uncertain_idx, confidence_threshold)
            stage2_pred_name = stage2_classes[s2_pred]
            stage2_conf = s2_probs[s2_pred]
            stage2_conf_str = f"{stage2_conf:.4f}"
            final_pred_name = stage2_pred_name
            final_conf = stage2_conf
        final_pred_names.append(final_pred_name)
        csv_rows.append([str(path), stage1_pred_name, f"{stage1_conf:.4f}", stage2_pred_name, stage2_conf_str, final_pred_name, f"{final_conf:.4f}"])

    csv_path = output_dir / "tri_images.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chemin", "stage1_pred", "stage1_confiance", "stage2_pred", "stage2_confiance", "prediction_finale", "confiance_finale"])
        writer.writerows(csv_rows)
    print(f"[info] Tri écrit -> {csv_path} ({len(paths)} images, {len(malade_indices)} envoyées à l'étage 2)")

    tri_dir = output_dir / "tri"
    for path, final_pred_name in zip(paths, final_pred_names):
        cls_dir = tri_dir / final_pred_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        (cls_dir / path.name).hardlink_to(path.resolve())
    print(f"[info] Images organisées par classe prédite (hardlinks, originaux intacts) -> {tri_dir}")
    print(f"[info] Terminé. Résultats écrits dans {output_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description="Classification à deux étages (stage 1 binaire -> stage 2 multiclasse) à partir de deux checkpoints déjà entraînés. N'entraîne rien."
    )
    parser.add_argument("--stage1_model_path", type=str, required=True, help="Checkpoint .pt binaire (saine/malade), utilisé comme filtre.")
    parser.add_argument("--stage2_model_path", type=str, required=True, help="Checkpoint .pt multiclasse, utilisé uniquement sur les images jugées malades par le stage 1. Peut couvrir un sous-ensemble des maladies (ex. --exclude_classes à l'entraînement) ; n'a pas besoin de connaître 'saine' lui-même.")
    parser.add_argument("--data_dir", type=str, required=True, help="Dossier à classer. Sous-dossiers de classe -> mode évaluation ; dossier plat d'images -> mode prédiction.")
    parser.add_argument("--output_dir", type=str, default=None, help="Dossier de sortie. Par défaut : sous-dossier horodaté cascade_<nom_data_dir>_<date>_<heure> créé à côté du checkpoint stage 2.")
    parser.add_argument("--img_size", type=int, default=224, help="Doit correspondre à la valeur utilisée à l'entraînement des deux modèles.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--class_map", type=str, default=None, help="Mode évaluation seulement : fait correspondre des noms de sous-dossiers de --data_dir aux classes du stage 2 quand les noms diffèrent.")
    parser.add_argument("--no_plots", action="store_true", help="Mode évaluation seulement : n'écrit que metrics.json et images_mal_classees.csv.")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Stage 2 (type de maladie) : si la confiance max est sous ce seuil, "
                             "la prédiction est remplacée par --uncertain_label (0 = désactivé).")
    parser.add_argument("--uncertain_label", type=str, default="malade_indeterminee",
                        help="Nom de la classe stage 2 vers laquelle basculer une prédiction peu "
                             "confiante — doit être une classe réelle du checkpoint stage 2.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"[info] Device: {device}")
    print(f"[warn] --img_size={args.img_size} : doit correspondre à la valeur utilisée pour entraîner les deux modèles.")

    stage1_path = Path(args.stage1_model_path)
    stage2_path = Path(args.stage2_model_path)
    data_dir = Path(args.data_dir)
    for p, name in ((stage1_path, "--stage1_model_path"), (stage2_path, "--stage2_model_path")):
        if not p.is_file():
            sys.exit(f"[erreur] {name} : {p} n'existe pas.")
    if not data_dir.is_dir():
        sys.exit(f"[erreur] --data_dir {data_dir} n'existe pas ou n'est pas un dossier.")

    stage1_model, stage1_classes = load_stage_model(stage1_path, device, "Stage 1")
    stage2_model, stage2_classes = load_stage_model(stage2_path, device, "Stage 2")

    if len(stage1_classes) != 2:
        sys.exit(f"[erreur] Le stage 1 doit être un modèle binaire (2 classes), trouvé {len(stage1_classes)} : {stage1_classes}.")
    stage1_saine_candidates = [i for i, c in enumerate(stage1_classes) if infer_group(c) == "saine"]
    if len(stage1_saine_candidates) != 1:
        sys.exit(f"[erreur] Impossible de déterminer la classe saine du stage 1 parmi {stage1_classes}.")
    stage1_saine_idx = stage1_saine_candidates[0]

    uncertain_idx = None
    if args.confidence_threshold > 0:
        if args.uncertain_label not in stage2_classes:
            sys.exit(f"[erreur] --uncertain_label '{args.uncertain_label}' absent des classes "
                     f"du stage 2 : {stage2_classes}.")
        uncertain_idx = stage2_classes.index(args.uncertain_label)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = stage2_path.parent / f"cascade_{data_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Résultats écrits dans : {output_dir}")

    tf = build_eval_transform(args.img_size)
    is_labeled = any(p.is_dir() for p in data_dir.iterdir())

    if is_labeled:
        class_map = parse_class_map(args.class_map) if args.class_map else None
        run_evaluation_mode(data_dir, tf, stage1_model, stage1_classes, stage1_saine_idx,
                             stage2_model, stage2_classes, class_map, args.batch_size, args.num_workers,
                             device, output_dir, args.no_plots, args.confidence_threshold, uncertain_idx)
    else:
        run_prediction_mode(data_dir, tf, stage1_model, stage1_classes, stage1_saine_idx,
                             stage2_model, stage2_classes, args.batch_size, args.num_workers, device, output_dir,
                             args.confidence_threshold, uncertain_idx)


if __name__ == "__main__":
    main()
