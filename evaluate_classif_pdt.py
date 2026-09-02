#!/usr/bin/env python3
"""
evaluate_classif_pdt.py — Évaluation approfondie d'un modèle déjà entraîné
(train_classif_pdt.py) sur un jeu d'images totalement exclu de l'entraînement.

Ce script ne fait AUCUN entraînement : il charge un checkpoint .pt, fait une
seule passe d'inférence (model.eval() + torch.no_grad()) sur un dossier de
test contenant au moins 2 sous-dossiers de classe (N quelconque), puis génère
un ensemble d'analyses pour juger si le modèle généralise bien :
    - métriques globales (loss, accuracy, AUC si 2 classes) -> metrics.json
    - liste des images mal classées (triée par confiance) -> images_mal_classees.csv
    - rapport de classification (precision/recall/F1) -> rapport_classification.txt
    - matrice de confusion, histogramme de confiance (bien classé vs mal
      classé) -> .png ; courbe ROC et précision-rappel seulement si le
      checkpoint a exactement 2 classes (binaires par nature)

Structure de données attendue pour --data_dir :
    data_dir/
        <classe1>/   *.jpg
        <classe2>/   *.jpg
        ...
(pas de train/val ici — c'est un jeu de test frais, à part du dataset
d'entraînement/validation utilisé par train_classif_pdt.py)

--confidence_threshold (défaut 0 = désactivé) : si > 0, toute prédiction dont
la probabilité maximale est sous ce seuil est reclassée en --uncertain_label
("incertain" par défaut) dans le rapport/la matrice de confusion/le CSV — le
modèle reste un classifieur à N classes inchangé (il n'existe pas d'images
étiquetées "incertain" pour l'entraîner), c'est une règle de décision
appliquée après coup sur ses probabilités déjà calculées.

Par défaut, --output_dir n'a pas besoin d'être précisé : les résultats sont
écrits dans un sous-dossier horodaté (eval_<nom_test>_<date>_<heure>) créé
juste à côté du checkpoint, dans son dossier de run (produit par
train_classif_pdt.py) — chaque évaluation garde ainsi son propre sous-dossier,
sans jamais écraser une évaluation précédente sur le même modèle.

Usage :
    python evaluate_classif_pdt.py --model_path Model/mon_run/best_model.pt --data_dir ./dataset/mon_dossier_exclu
    python evaluate_classif_pdt.py --model_path Model/mon_run/best_model.pt --data_dir ./dataset/test_exclu --no_plots
    python evaluate_classif_pdt.py --model_path Model/mon_run/best_model.pt --data_dir ./dataset/test_exclu --output_dir ./ailleurs   # forcer un autre emplacement

    # Si les noms de sous-dossiers de --data_dir diffèrent de ceux du checkpoint
    # (ex. "malades"/"saines" au pluriel vs "malade"/"saine" stockés dans le .pt) :
    python evaluate_classif_pdt.py --model_path Model/mon_run/best_model.pt --data_dir .\\dataset\\data\\val --class_map "malades=malade,saines=saine"
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
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_curve,
        roc_auc_score,
        precision_recall_curve,
    )
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(arch: str, num_classes: int = 2, dropout: float = 0.0) -> nn.Module:
    """dropout doit reproduire exactement la tête utilisée à l'entraînement
    (train_classif_pdt.py) : > 0 -> nn.Sequential(Dropout, Linear), sinon
    nn.Linear seul. Les noms de paramètres diffèrent entre les deux formes
    (ex. "fc.weight" vs "fc.1.weight") donc load_state_dict échoue si ça ne
    correspond pas au checkpoint — lire ckpt.get("head_dropout", 0.0)."""
    def make_head(in_features: int) -> nn.Module:
        linear = nn.Linear(in_features, num_classes)
        return nn.Sequential(nn.Dropout(dropout), linear) if dropout > 0 else linear

    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = make_head(model.fc.in_features)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = make_head(model.classifier[1].in_features)
    else:
        raise ValueError(f"Architecture inconnue: {arch}")
    return model


def run_inference(model, loader, dataset, criterion, device):
    """Analogue à run_epoch() (train_classif_pdt.py), en mode inference pure
    (model.eval() + torch.no_grad(), pas d'optimizer). Capture en plus les
    probabilités softmax et les chemins d'image, nécessaires pour
    l'histogramme de confiance et l'export des images mal classées."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    sample_paths = [p for p, _ in dataset.samples]

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels, all_probs, sample_paths


def parse_class_map(spec: str) -> dict:
    """Parse --class_map "dossier1=classe_ckpt1,dossier2=classe_ckpt2" en dict
    {nom_dossier: nom_classe_checkpoint}."""
    class_map = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            sys.exit(f"[erreur] --class_map mal formé (attendu dossier=classe) : '{pair}'.")
        folder_name, ckpt_name = pair.split("=", 1)
        class_map[folder_name.strip()] = ckpt_name.strip()
    return class_map


def reconcile_classes(ckpt_classes, eval_ds, class_map: dict = None):
    """ImageFolder assigne les indices de classe par ordre alphabétique des
    noms de sous-dossiers de data_dir. Ces noms peuvent différer de ceux
    utilisés à l'entraînement (stockés dans le checkpoint) — ex. singulier vs
    pluriel, langue différente. --class_map permet de faire correspondre
    explicitement un nom de dossier à une classe du checkpoint ; sans lui,
    seule une correspondance exacte des noms est acceptée (sinon les
    métriques seraient silencieusement fausses)."""
    eval_classes = eval_ds.classes
    mapped_classes = [class_map.get(c, c) for c in eval_classes] if class_map else eval_classes

    if sorted(ckpt_classes) != sorted(mapped_classes):
        detail = f" (mappé vers {mapped_classes} via --class_map)" if class_map else ""
        sys.exit(
            f"[erreur] Classes du dataset {eval_classes}{detail} incompatibles avec celles du "
            f"checkpoint {ckpt_classes}. Vérifie --data_dir ou utilise --class_map pour faire "
            "correspondre les noms de dossiers aux classes du checkpoint."
        )

    if ckpt_classes != mapped_classes:
        print(
            f"[warn] Noms/ordre de classes différents entre dataset ({eval_classes}) et "
            f"checkpoint ({ckpt_classes}). Remapping des labels appliqué."
        )
        remap = {i: ckpt_classes.index(name) for i, name in enumerate(mapped_classes)}
        eval_ds.samples = [(p, remap[c]) for p, c in eval_ds.samples]
        eval_ds.targets = [remap[c] for c in eval_ds.targets]


def write_misclassified_csv(path, sample_paths, all_labels, all_preds, all_probs, classes):
    """pred_idx peut désigner une pseudo-classe absente de probs (ex.
    "incertain" ajoutée par --confidence_threshold, qui n'a pas d'entrée
    dans le vecteur softmax à len(classes d'origine) sorties) — dans ce cas
    on rapporte max(probs) : c'est justement cette confiance maximale, sous
    le seuil, qui a déclenché le classement en incertain."""
    rows = []
    for p, true_idx, pred_idx, probs in zip(sample_paths, all_labels, all_preds, all_probs):
        if true_idx != pred_idx:
            conf = probs[pred_idx] if pred_idx < len(probs) else max(probs)
            rows.append((p, classes[true_idx], classes[pred_idx], conf))

    rows.sort(key=lambda r: r[3], reverse=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chemin", "vrai_label", "label_predit", "confiance"])
        for chemin, vrai, predit, conf in rows:
            writer.writerow([chemin, vrai, predit, f"{conf:.4f}"])

    if rows:
        print(f"[info] {len(rows)} image(s) mal classée(s) -> {path}")
    else:
        print("[info] Aucune image mal classée.")


def plot_confusion_matrix(all_labels, all_preds, classes, path):
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    size = max(6, len(classes) * 1.3)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true_pos, y_score_pos, path):
    fpr, tpr, _ = roc_curve(y_true_pos, y_score_pos)
    auc = roc_auc_score(y_true_pos, y_score_pos)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return auc


def plot_precision_recall_curve(y_true_pos, y_score_pos, path):
    precision, recall, _ = precision_recall_curve(y_true_pos, y_score_pos)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe précision-rappel")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confidence_histogram(all_labels, all_preds, all_probs, path):
    confidences = [max(p) for p in all_probs]
    correct = [c for c, t, pr in zip(confidences, all_labels, all_preds) if t == pr]
    incorrect = [c for c, t, pr in zip(confidences, all_labels, all_preds) if t != pr]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct, bins=bins, alpha=0.6, label="bien classé", color="tab:green")
    ax.hist(incorrect, bins=bins, alpha=0.6, label="mal classé", color="tab:red")
    ax.set_xlabel("Confiance (proba de la classe prédite)")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution de la confiance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Évalue un modèle entraîné (checkpoint .pt) sur un jeu d'images exclu de l'entraînement."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Chemin du checkpoint .pt (produit par train_classif_pdt.py).")
    parser.add_argument("--data_dir", type=str, required=True, help="Dossier avec exactement 2 sous-dossiers de classe (jeu de test exclu de l'entraînement).")
    parser.add_argument("--output_dir", type=str, default=None, help="Dossier de sortie. Par défaut : sous-dossier horodaté eval_<nom_test>_<date>_<heure> créé dans le dossier du modèle (parent de --model_path), pour garder les résultats groupés avec le run d'entraînement.")
    parser.add_argument("--img_size", type=int, default=224, help="Doit correspondre à celui utilisé à l'entraînement (non stocké dans le checkpoint).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--positive_class", type=str, default=None, help="Classe traitée comme positive pour ROC/precision-rappel (défaut: 2e classe de l'ordre du checkpoint).")
    parser.add_argument("--class_map", type=str, default=None, help="Fait correspondre des noms de sous-dossiers de --data_dir à des classes du checkpoint quand les noms diffèrent (ex. 'malades=malade,saines=saine').")
    parser.add_argument("--no_plots", action="store_true", help="N'écrit que metrics.json et images_mal_classees.csv, saute matplotlib/sklearn.")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Si > 0 : toute prédiction dont la probabilité maximale est sous ce seuil est reclassée en --uncertain_label au lieu du nom de la classe la plus probable (le modèle reste un classifieur à N classes inchangé, c'est une règle de décision appliquée après coup sur ses probabilités). Défaut 0 = désactivé.")
    parser.add_argument("--uncertain_label", type=str, default="incertain", help="Nom affiché pour les prédictions sous --confidence_threshold.")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[info] Device: {device}")
    print(f"[warn] --img_size={args.img_size} : doit correspondre à la valeur utilisée à l'entraînement.")

    model_path = Path(args.model_path)
    if not model_path.is_file():
        sys.exit(f"[erreur] {model_path} n'existe pas.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_path.parent / f"eval_{Path(args.data_dir).name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Résultats écrits dans : {output_dir}")

    ckpt = torch.load(model_path, map_location=device)
    for key in ("model_state", "arch", "classes"):
        if key not in ckpt:
            sys.exit(f"[erreur] Checkpoint invalide : clé '{key}' manquante dans {model_path}.")
    ckpt_classes = ckpt["classes"]

    model = build_model(ckpt["arch"], num_classes=len(ckpt_classes), dropout=ckpt.get("head_dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[info] Modèle chargé : arch={ckpt['arch']}, classes={ckpt_classes}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"[erreur] {data_dir} n'existe pas ou n'est pas un dossier.")

    eval_tf = build_eval_transform(args.img_size)
    try:
        eval_ds = datasets.ImageFolder(data_dir, transform=eval_tf)
    except FileNotFoundError as e:
        sys.exit(f"[erreur] Impossible de charger {data_dir} : {e}")

    if len(eval_ds.classes) < 2:
        sys.exit(
            f"[erreur] Attendu au moins 2 sous-dossiers de classe dans {data_dir}, "
            f"trouvé : {eval_ds.classes}."
        )

    class_map = parse_class_map(args.class_map) if args.class_map else None
    reconcile_classes(ckpt_classes, eval_ds, class_map)
    classes = ckpt_classes

    is_binary = len(classes) == 2
    positive_class, positive_idx = None, None
    if is_binary:
        if args.positive_class is None:
            positive_class = classes[1]
        elif args.positive_class in classes:
            positive_class = args.positive_class
        else:
            sys.exit(f"[erreur] --positive_class '{args.positive_class}' introuvable dans {classes}.")
        positive_idx = classes.index(positive_class)
        print(f"[info] Classe positive (ROC/precision-rappel) : {positive_class}")
    elif args.positive_class is not None:
        print(f"[warn] --positive_class ignoré : {len(classes)} classes détectées, "
              "ROC/precision-rappel (binaires) ne s'appliquent qu'à 2 classes.")

    val_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[info] Dataset d'évaluation : {len(eval_ds)} images, classes={classes}")

    criterion = nn.CrossEntropyLoss()
    loss, acc, all_preds, all_labels, all_probs, sample_paths = run_inference(
        model, val_loader, eval_ds, criterion, device
    )
    print(f"[info] Loss={loss:.4f} Accuracy={acc:.4f} N={len(eval_ds)}")

    per_class_counts = {cls: eval_ds.targets.count(i) for i, cls in enumerate(classes)}

    # Seuil de confiance en post-traitement : le modèle reste un classifieur
    # à len(classes) sorties inchangé (il n'existe pas d'images étiquetées
    # --uncertain_label) — on remplace juste, pour le rapport, toute
    # prédiction dont argmax(probs) < seuil par une pseudo-classe ajoutée en
    # bout de liste. Les vraies étiquettes (all_labels, per_class_counts)
    # restent sur les classes d'origine : aucune image n'est réellement
    # "incertaine".
    if args.confidence_threshold > 0:
        classes_for_report = classes + [args.uncertain_label]
        uncertain_idx = len(classes)
        final_preds = [uncertain_idx if max(probs) < args.confidence_threshold else pred
                       for probs, pred in zip(all_probs, all_preds)]
        n_uncertain = sum(1 for p in final_preds if p == uncertain_idx)
        pct_uncertain = 100 * n_uncertain / len(final_preds) if final_preds else 0
        print(f"[info] {n_uncertain} image(s) ({pct_uncertain:.1f}%) sous le seuil de confiance "
              f"{args.confidence_threshold} -> classées '{args.uncertain_label}'.")
    else:
        classes_for_report = classes
        final_preds = all_preds
        n_uncertain = 0

    write_misclassified_csv(
        output_dir / "images_mal_classees.csv", sample_paths, all_labels, final_preds, all_probs, classes_for_report
    )

    metrics = {
        "loss": loss,
        "accuracy": acc,
        "n_samples": len(eval_ds),
        "per_class_counts": per_class_counts,
        "positive_class": positive_class,
        "auc": None,
        "classification_report": None,
        "confidence_threshold": args.confidence_threshold,
        "n_incertain": n_uncertain,
        "pct_incertain": round(100 * n_uncertain / len(eval_ds), 2) if len(eval_ds) else 0,
    }

    if HAS_PLOT and not args.no_plots:
        report_labels = list(range(len(classes_for_report)))
        report_txt = classification_report(all_labels, final_preds, labels=report_labels, target_names=classes_for_report, digits=3)
        print("\n[rapport de classification]\n" + report_txt)
        with open(output_dir / "rapport_classification.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)
        metrics["classification_report"] = classification_report(
            all_labels, final_preds, labels=report_labels, target_names=classes_for_report, digits=3, output_dict=True
        )

        plot_confusion_matrix(all_labels, final_preds, classes_for_report, output_dir / "matrice_confusion.png")
        plot_confidence_histogram(all_labels, final_preds, all_probs, output_dir / "histogramme_confiance.png")

        if is_binary:
            y_true_pos = [1 if lbl == positive_idx else 0 for lbl in all_labels]
            y_score_pos = [p[positive_idx] for p in all_probs]
            auc = plot_roc_curve(y_true_pos, y_score_pos, output_dir / "roc_curve.png")
            metrics["auc"] = auc
            plot_precision_recall_curve(y_true_pos, y_score_pos, output_dir / "precision_recall_curve.png")
            print(f"[info] AUC = {auc:.4f}")
        else:
            print(f"[info] ROC/precision-rappel ignorées ({len(classes)} classes, binaire uniquement).")
    else:
        if args.no_plots:
            print("[info] --no_plots activé : seuls metrics.json et images_mal_classees.csv sont écrits.")
        else:
            print("[warn] matplotlib/scikit-learn absents : rapport, matrice de confusion, courbes ROC/PR et "
                  "histogramme de confiance non générés (pip install matplotlib scikit-learn --break-system-packages).")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[info] Terminé. Résultats écrits dans {output_dir}.")


if __name__ == "__main__":
    main()
