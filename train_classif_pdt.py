#!/usr/bin/env python3
"""
train_classif_pdt.py — Classification saine / malade de tubercules de pomme de terre.

Etape C du pipeline (voir discussion) : classifieur CNN par transfer learning
(ResNet18 ou EfficientNet-B0) sur images déjà recadrées/individuelles de tubercules.

Structure de données attendue (deux modes supportés). Les noms des dossiers
de classe peuvent être n'importe quoi (ex. malade/saine, malades/saines, ou
les classes fines produites par prepare_dataset.py) : seul le nombre de
sous-dossiers (au moins 2) est vérifié — N classes sont supportées.

Équilibrage par groupe (WeightedRandomSampler sur le train set, val non
affecté) : tout nom de classe commençant par "malade" est rattaché au groupe
malade, le reste au groupe saine (contrat de nommage avec prepare_dataset.py).
Le sampler donne à chaque groupe une masse totale égale et la répartit
également entre ses sous-classes, quels que soient leurs effectifs bruts.

Mode A — déjà splitté :
    data_dir/
        train/
            <classe1>/   *.jpg
            <classe2>/   *.jpg
        val/
            <classe1>/   *.jpg
            <classe2>/   *.jpg

Mode B — pas encore splitté (juste tes dossiers de classe actuels) :
    data_dir/
        <classe1>/   *.jpg
        <classe2>/   *.jpg
    -> le script fait un split stratifié train/val automatiquement (--val_split).

Forcer un nouveau split (--force_split) :
    Même si data_dir/train et data_dir/val existent déjà (Mode A), on peut
    forcer un nouveau split stratifié : les images des deux sont fusionnées
    puis redistribuées selon --val_split. Utile pour rééquilibrer un split
    train/val existant devenu trop déséquilibré. Les images ne sont jamais
    dupliquées sur le disque (hard links).

Pendant l'exécution, appuie sur P à tout moment pour mettre l'entraînement en
pause (ré-appuie sur P pour reprendre) — utile pour mettre le PC en veille
sans crainte. Une estimation de la durée totale (basée sur quelques batches
réels) est affichée juste avant le début de la première époque.

Usage :
    python train_classif_pdt.py --data_dir ./dataset/data --epochs 30
    python train_classif_pdt.py --data_dir ./dataset/data --quick   # smoke test rapide
    python train_classif_pdt.py --data_dir ./dataset/data --force_split --val_split 0.15   # re-split forcé à 85/15
    python train_classif_pdt.py --data_dir ./dataset/data --reduce_dataset_to 80   # entraîne sur 80% des images (tirage aléatoire stratifié)
    python train_classif_pdt.py --data_dir ./dataset/data --exclude_classes malade_pstvd,malade_pvy   # exclut des classes en mémoire, sans toucher au disque

Sélection du meilleur modèle (--selection_metric, défaut val_f1_macro) :
val_acc seul peut masquer un modèle qui overfit les classes minoritaires (peu
d'images, ex. malade_pstvd/malade_pvy) sans que ça se voie sur l'accuracy
globale dominée par les classes majoritaires. val_f1_macro (moyenne non
pondérée du F1 par classe) est donc le critère par défaut désormais — c'est
un changement de comportement par défaut par rapport aux versions
précédentes de ce script (qui utilisaient val_acc en dur) ; --selection_metric
val_acc restaure l'ancien comportement. Nécessite scikit-learn.

Régularisation contre l'overfitting sur les classes minoritaires :
--weight_decay (Adam, défaut 1e-4) et --dropout (défaut 0.3, juste avant la
couche de classification finale — 0 désactive, tête = nn.Linear seul comme
avant).

Sortie : chaque run crée son propre sous-dossier horodaté dans --output_dir
(défaut ./Model), nommé <dataset>_epoch<N>[_frac<pct>]_<date>_<heure> :
    - run_config.json (tous les arguments CLI, pour reproduire le run)
    - contenu_dataset.txt (train/val réellement utilisés : classes, comptes, noms de fichiers)
    - best_model.pt, history.json
    - courbes_entrainement.png, matrice_confusion.png (matplotlib)
    - rapport_classification.txt (precision/recall/F1)
Une ligne récapitulative est aussi ajoutée à --output_dir/runs_index.csv à la
fin de chaque run (pour comparer tous les runs sans ouvrir chaque dossier).
evaluate_classif_pdt.py écrit ses résultats dans ce même sous-dossier par défaut.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import msvcrt
except ImportError:
    msvcrt = None

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# HAS_PLOT gate le bloc courbes/matrice de confusion/rapport en fin de run
# (nécessite les deux). HAS_SKLEARN seul suffit pour val_f1_macro par époque
# (sélection du meilleur modèle) : ça ne doit pas dépendre de matplotlib.
HAS_PLOT = HAS_SKLEARN and HAS_MATPLOTLIB

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomOcclusion:
    """Occlusion aléatoire (rectangle ou cercle) pour simuler un tubercule
    partiellement caché par un voisin. Rectangle = bord coupé net (cadre/tas),
    cercle = silhouette d'un autre tubercule qui en recouvre un coin."""

    def __init__(self, p=0.3, scale=(0.02, 0.15), value=0.0):
        self.p, self.scale, self.value = p, scale, value

    def __call__(self, img):
        if random.random() > self.p:
            return img
        _, h, w = img.shape
        erase_area = random.uniform(*self.scale) * h * w

        if random.random() < 0.5:
            aspect = random.uniform(0.3, 3.3)
            eh = min(h, max(1, int(round((erase_area * aspect) ** 0.5))))
            ew = min(w, max(1, int(round((erase_area / aspect) ** 0.5))))
            top = random.randint(0, h - eh)
            left = random.randint(0, w - ew)
            img[:, top:top + eh, left:left + ew] = self.value
        else:
            radius = max(1, int(round((erase_area / math.pi) ** 0.5)))
            radius = min(radius, h // 2, w // 2)
            cy = random.randint(radius, h - radius) if h > 2 * radius else h // 2
            cx = random.randint(radius, w - radius) if w > 2 * radius else w // 2
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            img[:, mask] = self.value
        return img


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        # jitter modéré : on évite de trop distordre les couleurs, car la
        # décoloration/tâches sont souvent le signal même de la maladie
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        RandomOcclusion(p=0.3, scale=(0.15, 0.40)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def prepare_data_dir(data_dir: Path, val_split: float, seed: int, force_split: bool = False) -> Path:
    """Si data_dir contient déjà train/ et val/ (et que force_split est False),
    on ne touche à rien. Sinon (Mode B : data_dir/<classe1>, data_dir/<classe2>,
    ou force_split=True), on fusionne toutes les images trouvées et on crée un
    split stratifié dans un dossier temporaire à côté, sans dupliquer les
    images sur le disque (hard links : contrairement aux symlinks, ne demandent
    aucun privilège particulier sous Windows)."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    already_split = train_dir.is_dir() and val_dir.is_dir()

    if already_split and not force_split:
        print(f"[info] Mode A détecté : {train_dir} et {val_dir} existent déjà.")
        return data_dir

    if already_split:
        print(f"[info] --force_split activé : fusion de {train_dir} et {val_dir} avant nouveau split.")
        source_dirs = [train_dir, val_dir]
    else:
        source_dirs = [data_dir]

    class_dirs = [d for src in source_dirs for d in src.iterdir() if d.is_dir()]
    found_classes = sorted({d.name for d in class_dirs})
    if len(found_classes) < 2:
        sys.exit(
            f"[erreur] Impossible de comprendre la structure de {data_dir} (classes trouvées : {found_classes}). "
            "Attendu : au moins 2 sous-dossiers de classe, soit dans data_dir/train + data_dir/val, "
            "soit directement dans data_dir."
        )

    print("[info] Split automatique stratifié en cours...")
    split_root = data_dir.parent / (data_dir.name + "_split")
    if split_root.exists():
        shutil.rmtree(split_root)

    # Structure plate {chemin image: label} — source de vérité pour tout le split.
    # Récursif (rglob) : les images peuvent être nichées dans des sous-dossiers
    # (ex. par sous-type de maladie), notamment quand on fusionne train/ et val/.
    image_labels = {}
    for cls_dir in class_dirs:
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_labels[p] = cls_dir.name

    rng = random.Random(seed)
    by_class = {}
    for img, label in image_labels.items():
        by_class.setdefault(label, []).append(img)

    for label, images in by_class.items():
        rng.shuffle(images)
        n_val = max(1, int(len(images) * val_split))
        val_imgs, train_imgs = images[:n_val], images[n_val:]

        for split_name, imgs in (("train", train_imgs), ("val", val_imgs)):
            out_dir = split_root / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            # Préfixe numérique : des sous-dossiers de maladie différents réutilisent
            # souvent les mêmes noms de fichiers (1.jpg, 2.jpg, ...), un simple
            # img.name écraserait silencieusement des images en cas de collision.
            for i, img in enumerate(imgs):
                (out_dir / f"{i:06d}_{img.name}").hardlink_to(img.resolve())

        print(f"  - {label}: {len(train_imgs)} train / {len(val_imgs)} val")

    return split_root


def reduce_dataset(ds, pct: float, seed: int):
    """Sous-échantillonne ds à pct% des images, stratifié par classe
    (tirage aléatoire reproductible via seed)."""
    rng = random.Random(seed)
    by_class = {}
    for idx, target in enumerate(ds.targets):
        by_class.setdefault(target, []).append(idx)

    keep_indices = []
    for indices in by_class.values():
        indices = indices[:]
        rng.shuffle(indices)
        n_keep = max(1, round(len(indices) * pct / 100))
        keep_indices.extend(indices[:n_keep])

    keep_indices.sort()
    return Subset(ds, keep_indices)


def reduce_to_max_per_class(ds, max_per_class: int, seed: int):
    """Sous-échantillonne ds à au plus max_per_class images par classe
    (tirage aléatoire stratifié). Contrairement à un simple Subset(range(n)),
    garantit qu'au moins une classe de chaque type présent est gardée même
    avec beaucoup de classes — utilisé par --quick, où une simple tranche
    contiguë des N premiers échantillons (triés par classe par ImageFolder)
    ne contiendrait souvent qu'une seule classe."""
    rng = random.Random(seed)
    by_class = {}
    for idx, target in enumerate(get_effective_targets(ds)):
        by_class.setdefault(target, []).append(idx)

    keep_indices = []
    for indices in by_class.values():
        indices = indices[:]
        rng.shuffle(indices)
        keep_indices.extend(indices[:max_per_class])

    keep_indices.sort()
    return Subset(ds, keep_indices)


def infer_group(class_name: str) -> str:
    """Contrat de nommage partagé avec prepare_dataset.py : tout nom de
    classe commençant par "malade" appartient au groupe malade (ex.
    malade_dry_rot, malade_indeterminee), le reste (saine) au groupe saine.
    Rétro-compatible avec l'usage binaire existant (dossiers "malade"/"saine")."""
    return "malade" if class_name.startswith("malade") else "saine"


def get_effective_targets(ds) -> list:
    """Résout les labels alignés sur l'indexation propre de ds (0..len(ds)-1),
    même si ds est un Subset (éventuellement imbriqué, ex. --reduce_dataset_to
    suivi de --quick empilent deux Subset)."""
    if isinstance(ds, Subset):
        base_targets = get_effective_targets(ds.dataset)
        return [base_targets[i] for i in ds.indices]
    return list(ds.targets)


def exclude_classes_from_dataset(ds, excluded_names: set) -> None:
    """Retire en mémoire les échantillons des classes listées d'un
    ImageFolder déjà chargé (aucune écriture disque), et réindexe les
    classes restantes en 0..N-1 contigu. Modifie ds en place."""
    kept_classes = [c for c in ds.classes if c not in excluded_names]
    old_to_new = {ds.class_to_idx[c]: i for i, c in enumerate(kept_classes)}
    new_samples = [(p, old_to_new[t]) for p, t in ds.samples if t in old_to_new]
    ds.samples = new_samples
    ds.imgs = new_samples
    ds.targets = [t for _, t in new_samples]
    ds.classes = kept_classes
    ds.class_to_idx = {c: i for i, c in enumerate(kept_classes)}


def get_effective_samples(ds) -> list:
    """Même principe que get_effective_targets, mais retourne les tuples
    (chemin, label) de ImageFolder.samples plutôt que les seuls labels —
    nécessaire pour lister les noms de fichiers réellement utilisés dans le
    manifeste du run."""
    if isinstance(ds, Subset):
        base_samples = get_effective_samples(ds.dataset)
        return [base_samples[i] for i in ds.indices]
    return list(ds.samples)


def build_run_name(dataset_name: str, epochs: int, reduce_dataset_to) -> str:
    """<dataset>_epoch<N>[_frac<pct>]_<date>_<heure> — epochs est la valeur
    demandée en CLI (pas le nombre réel si early stopping arrête avant), car
    le nom doit être connu avant que l'entraînement ne commence."""
    frac_part = f"_frac{int(reduce_dataset_to)}" if reduce_dataset_to is not None else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_name}_epoch{epochs}{frac_part}_{timestamp}"


def write_dataset_manifest(path: Path, classes, split_samples: dict) -> None:
    """Énumère, pour chaque split (train/val) réellement utilisé dans ce run,
    le nombre d'images et les noms de fichiers de chaque classe."""
    with open(path, "w", encoding="utf-8") as f:
        for split_name, samples in split_samples.items():
            by_class = {}
            for p, t in samples:
                by_class.setdefault(classes[t], []).append(Path(p).name)

            f.write(f"=== {split_name} ({len(samples)} images) ===\n")
            for cls in classes:
                names = sorted(by_class.get(cls, []))
                f.write(f"\n--- {cls} ({len(names)} images) ---\n")
                for name in names:
                    f.write(f"{name}\n")
            f.write("\n")


def append_run_index(base_output_dir: Path, row: dict) -> None:
    """Ajoute une ligne à base_output_dir/runs_index.csv (créé avec en-tête
    si absent) — permet de comparer tous les runs sans ouvrir chaque dossier."""
    index_path = base_output_dir / "runs_index.csv"
    is_new = not index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def print_group_balance(targets: list, classes) -> None:
    """Affiche la répartition par classe et par groupe (malade/saine) du
    dataset d'entraînement, avant pondération."""
    counts = {cls: 0 for cls in classes}
    for t in targets:
        counts[classes[t]] += 1

    group_totals = {}
    print("[info] Répartition du train set par classe :")
    for cls in classes:
        g = infer_group(cls)
        group_totals[g] = group_totals.get(g, 0) + counts[cls]
        print(f"  - {cls} ({g}): {counts[cls]} images")

    print("[info] Répartition par groupe :")
    for g, total in sorted(group_totals.items()):
        print(f"  - {g}: {total} images")
    if len(group_totals) == 2:
        majority, minority = max(group_totals.values()), min(group_totals.values())
        ratio = majority / minority if minority else float("inf")
        print(f"[info] Ratio brut malade/saine (avant pondération) : {ratio:.2f}")


def build_group_balanced_weights(targets: list, classes) -> list:
    """Poids par échantillon pour WeightedRandomSampler : donne à chaque
    groupe (malade/saine) une masse totale égale, répartie également entre
    ses sous-classes (évite qu'une sous-classe volumineuse comme
    malade_indeterminee écrase les autres maladies à l'intérieur du groupe
    malade). poids = 1 / (nb_classes_du_groupe * effectif_de_la_classe)."""
    class_counts = {}
    for t in targets:
        class_counts[classes[t]] = class_counts.get(classes[t], 0) + 1

    n_classes_per_group = {}
    for cls, n in class_counts.items():
        g = infer_group(cls)
        n_classes_per_group[g] = n_classes_per_group.get(g, 0) + 1

    class_weight = {
        cls: 1.0 / (n_classes_per_group[infer_group(cls)] * n)
        for cls, n in class_counts.items()
    }
    return [class_weight[classes[t]] for t in targets]


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def estimate_duration(model, train_loader, val_loader, criterion, optimizer, device, epochs: int,
                       n_probe_batches: int = 5, n_warmup_batches: int = 2):
    """Chronomètre quelques batches réels (train + val) pour estimer la durée
    totale de l'entraînement avant de lancer les époques.

    Sur GPU, les tout premiers batches incluent des coûts fixes ponctuels
    (initialisation du contexte CUDA, autotuning cuDNN, compilation des
    premiers kernels) qui n'ont rien à voir avec le régime de croisière — les
    inclure dans la mesure fausserait l'estimation d'un facteur ~10x. On les
    absorbe donc via un échauffement non chronométré avant de mesurer, et on
    synchronise le GPU pour que le chrono reflète un travail réellement
    terminé (les appels CUDA sont asynchrones sinon)."""
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def time_batches(loader, train: bool):
        it = iter(loader)
        n_avail = len(loader)
        n_warmup = min(n_warmup_batches, n_avail)
        n_timed = min(n_probe_batches, max(n_avail - n_warmup, 0))

        def run_one(imgs, labels):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
            else:
                model(imgs)

        for _ in range(n_warmup):
            run_one(*next(it))
        sync()

        t0 = time.time()
        n = 0
        for _ in range(n_timed):
            run_one(*next(it))
            n += 1
        sync()
        return (time.time() - t0) / max(n, 1)

    model.train()
    train_batch_time = time_batches(train_loader, train=True)

    model.eval()
    with torch.no_grad():
        val_batch_time = time_batches(val_loader, train=False)

    epoch_time = train_batch_time * len(train_loader) + val_batch_time * len(val_loader)
    return epoch_time, epoch_time * epochs


def build_model(arch: str, num_classes: int = 2, dropout: float = 0.0) -> nn.Module:
    """dropout > 0 insère un nn.Dropout juste avant la couche de
    classification finale (tête = nn.Sequential(Dropout, Linear) au lieu
    d'un nn.Linear seul) — régularisation contre l'overfitting, utile
    surtout sur les classes minoritaires peu représentées. dropout=0.0
    (défaut) garde l'architecture d'origine à l'identique."""
    def make_head(in_features: int) -> nn.Module:
        linear = nn.Linear(in_features, num_classes)
        return nn.Sequential(nn.Dropout(dropout), linear) if dropout > 0 else linear

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = make_head(model.fc.in_features)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = make_head(model.classifier[1].in_features)
    else:
        raise ValueError(f"Architecture inconnue: {arch}")
    return model


def check_pause():
    """Non bloquant : si P/p a été pressé, affiche un message et attend qu'on
    ré-appuie sur P/p pour reprendre. Pendant l'attente, le PC peut être mis
    en veille sans risque (aucun calcul ni écriture disque en cours)."""
    if msvcrt is None or not msvcrt.kbhit():
        return
    if msvcrt.getch().lower() != b"p":
        return
    print("\n[pause] Entraînement en pause — appuie sur P pour reprendre "
          "(tu peux mettre le PC en veille maintenant sans risque).")
    while True:
        time.sleep(0.3)
        if msvcrt.kbhit() and msvcrt.getch().lower() == b"p":
            print("[pause] Reprise de l'entraînement.\n")
            return


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    torch.set_grad_enabled(train)
    for imgs, labels in loader:
        check_pause()
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Entraîne un CNN saine/malade pour tubercules de pomme de terre.")
    parser.add_argument("--data_dir", type=str, required=True, help="Dossier racine des images.")
    parser.add_argument("--output_dir", type=str, default="./Model", help="Dossier racine : un sous-dossier horodaté est créé pour ce run (modèle, courbes, rapport, manifeste).")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4, help="Processus de chargement/augmentation des images en parallèle (CPU) ; augmentez si le GPU attend souvent après les données (cf. nvidia-smi).")
    parser.add_argument("--multi_gpu", action="store_true", help="Utilise tous les GPU CUDA disponibles (torch.nn.DataParallel) si 2 ou plus sont détectés ; ignoré sinon.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Régularisation L2 de l'optimiseur Adam (0 = désactivée).")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout juste avant la couche de classification finale (0 = désactivé, tête = nn.Linear seul comme avant).")
    parser.add_argument("--selection_metric", type=str, default="val_f1_macro", choices=["val_acc", "val_f1_macro"], help="Critère de sélection de best_model.pt et du scheduler ReduceLROnPlateau. Défaut changé à val_f1_macro (val_acc peut masquer un modèle qui overfit les classes minoritaires) ; val_acc restaure l'ancien comportement. val_f1_macro nécessite scikit-learn.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Utilisé seulement en Mode B (pas de train/val existants) ou avec --force_split.")
    parser.add_argument("--force_split", action="store_true", help="Force un nouveau split stratifié même si data_dir/train et data_dir/val existent déjà (fusionne les images des deux avant de re-split).")
    parser.add_argument("--freeze_backbone", action="store_true", help="Gèle les couches convolutives, n'entraîne que la tête (rapide, utile si peu de données).")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping: nb d'époques sans amélioration avant arrêt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Smoke test rapide: sous-échantillonne les données, 2 époques, pour vérifier que le pipeline tourne de bout en bout.")
    parser.add_argument("--reduce_dataset_to", type=float, default=None, help="Réduit le dataset (train et val) à ce pourcentage (0-100), sélection aléatoire stratifiée par classe. Ex: 80 = 80%% des images.")
    parser.add_argument("--exclude_classes", type=str, default=None, help="Classes à exclure de l'entraînement, séparées par des virgules (ex. 'malade_pstvd,malade_pvy'). Filtrage en mémoire uniquement, aucun dossier créé/supprimé sur le disque.")
    args = parser.parse_args()

    if args.reduce_dataset_to is not None and not (0 < args.reduce_dataset_to <= 100):
        sys.exit(f"[erreur] --reduce_dataset_to doit être dans ]0, 100], reçu {args.reduce_dataset_to}.")

    if args.selection_metric == "val_f1_macro" and not HAS_SKLEARN:
        sys.exit("[erreur] --selection_metric val_f1_macro nécessite scikit-learn (absent). "
                  "Utilise --selection_metric val_acc, ou installe scikit-learn (pip install scikit-learn).")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")
    if msvcrt is not None:
        print("[info] Appuie sur P à tout moment pour mettre l'entraînement en pause (ré-appuie sur P pour reprendre). "
              "La fenêtre de ce terminal doit avoir le focus pour que la touche soit détectée.")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    run_name = build_run_name(data_dir.name, args.epochs, args.reduce_dataset_to)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Dossier de ce run : {run_dir}")

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "run_name": run_name, "device": str(device)}, f, indent=2)

    effective_data_dir = prepare_data_dir(data_dir, args.val_split, args.seed, args.force_split)

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = datasets.ImageFolder(effective_data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(effective_data_dir / "val", transform=val_tf)

    assert train_ds.classes == val_ds.classes, (
        f"[erreur] Classes différentes entre train {train_ds.classes} et val {val_ds.classes}. "
        "Les dossiers de classe de train/ et val/ doivent porter les mêmes noms."
    )
    print(f"[info] Classes détectées : {train_ds.classes}")

    if args.exclude_classes:
        excluded = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
        unknown = excluded - set(train_ds.classes)
        if unknown:
            sys.exit(f"[erreur] --exclude_classes contient des noms introuvables : {sorted(unknown)}. "
                      f"Classes disponibles : {train_ds.classes}.")
        exclude_classes_from_dataset(train_ds, excluded)
        exclude_classes_from_dataset(val_ds, excluded)
        print(f"[info] Classes exclues : {sorted(excluded)}")
        print(f"[info] Classes restantes : {train_ds.classes}")

    classes = train_ds.classes

    if args.reduce_dataset_to is not None:
        train_ds = reduce_dataset(train_ds, args.reduce_dataset_to, args.seed)
        val_ds = reduce_dataset(val_ds, args.reduce_dataset_to, args.seed)
        print(f"[info] --reduce_dataset_to {args.reduce_dataset_to}% : Train {len(train_ds)} images | Val {len(val_ds)} images")

    epochs = args.epochs
    if args.quick:
        print("[info] --quick activé : sous-échantillonnage stratifié + 2 époques pour smoke test.")
        epochs = 2
        train_ds = reduce_to_max_per_class(train_ds, max_per_class=8, seed=args.seed)
        val_ds = reduce_to_max_per_class(val_ds, max_per_class=4, seed=args.seed)

    write_dataset_manifest(
        run_dir / "contenu_dataset.txt", classes,
        {"train": get_effective_samples(train_ds), "val": get_effective_samples(val_ds)},
    )

    train_targets = get_effective_targets(train_ds)
    print_group_balance(train_targets, classes)
    train_weights = build_group_balanced_weights(train_targets, classes)
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_targets), replacement=True)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=False, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    print(f"[info] Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # raw_model reste la référence pour tout ce qui touche aux poids
    # (state_dict/checkpoints) : identique que --multi_gpu soit actif ou non,
    # pour garder le format de best_model.pt compatible avec l'existant
    # (DataParallel préfixerait les clés de "model.state_dict()" par "module.").
    raw_model = build_model(args.arch, num_classes=len(classes), dropout=args.dropout).to(device)
    if args.freeze_backbone:
        for name, param in raw_model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print("[info] Backbone gelé, seule la tête de classification est entraînée.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, raw_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    model = raw_model
    if args.multi_gpu:
        if device.type != "cuda":
            print("[warn] --multi_gpu ignoré : aucun GPU CUDA disponible.")
        elif torch.cuda.device_count() < 2:
            print(f"[warn] --multi_gpu ignoré : {torch.cuda.device_count()} GPU CUDA détecté (2+ requis).")
        else:
            model = nn.DataParallel(raw_model)
            print(f"[info] DataParallel actif sur {torch.cuda.device_count()} GPU "
                  f"(batch_size={args.batch_size} réparti entre eux, ~{args.batch_size // torch.cuda.device_count()} par GPU — "
                  f"envisagez d'augmenter --batch_size pour garder une taille de batch par GPU raisonnable).")

    epoch_time, total_time = estimate_duration(model, train_loader, val_loader, criterion, optimizer, device, epochs)
    print(f"[info] Estimation grossière (avant le 1er epoch réel, à prendre comme ordre de grandeur) : "
          f"~{format_duration(epoch_time)}/époque, ~{format_duration(total_time)} pour {epochs} époques. "
          "Une estimation fiable s'affichera après le premier epoch.")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1_macro": []}
    best_metric, best_state, epochs_no_improve = 0.0, None, 0
    report_labels = list(range(len(classes)))

    loop_t0 = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        val_f1_macro = f1_score(val_labels, val_preds, average="macro", labels=report_labels, zero_division=0) if HAS_SKLEARN else None
        selection_value = val_f1_macro if args.selection_metric == "val_f1_macro" else val_acc
        scheduler.step(selection_value)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)

        dt = time.time() - t0
        elapsed = time.time() - loop_t0
        avg_epoch = elapsed / epoch
        eta = avg_epoch * (epochs - epoch)  # borne haute : ignore un early stopping qui arrêterait avant
        f1_str = f" val_f1_macro={val_f1_macro:.4f}" if val_f1_macro is not None else ""
        print(f"[epoch {epoch:03d}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{f1_str} ({dt:.1f}s)")
        print(f"    -> moyenne {avg_epoch:.1f}s/époque | restant estimé (si pas d'early stopping) : {format_duration(eta)}")

        if selection_value > best_metric:
            best_metric = selection_value
            best_state = copy.deepcopy(raw_model.state_dict())
            epochs_no_improve = 0
            torch.save({"model_state": best_state, "arch": args.arch, "classes": classes, "head_dropout": args.dropout},
                       run_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[info] Early stopping à l'époque {epoch} (patience={args.patience}).")
                break

    epochs_ran = epoch
    print(f"[info] Meilleur {args.selection_metric}: {best_metric:.4f} — modèle sauvegardé dans {run_dir / 'best_model.pt'}")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if best_state is not None:
        raw_model.load_state_dict(best_state)
    _, final_val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

    final_val_f1_macro, final_val_f1_weighted = None, None
    if HAS_SKLEARN:
        final_val_f1_macro = f1_score(val_labels, val_preds, average="macro", labels=report_labels, zero_division=0)
        final_val_f1_weighted = f1_score(val_labels, val_preds, average="weighted", labels=report_labels, zero_division=0)
        print(f"[info] F1 macro final : {final_val_f1_macro:.4f} | F1 pondéré final : {final_val_f1_weighted:.4f}")

    if HAS_PLOT and len(val_labels) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("époque")
        axes[0].legend()

        axes[1].plot(history["train_acc"], label="train")
        axes[1].plot(history["val_acc"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("époque")
        axes[1].legend()

        axes[2].plot(history["val_f1_macro"], label="val", color="tab:orange")
        axes[2].set_title("F1 macro")
        axes[2].set_xlabel("époque")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(run_dir / "courbes_entrainement.png", dpi=150)
        plt.close(fig)

        cm = confusion_matrix(val_labels, val_preds, labels=report_labels)
        size = max(6, len(classes) * 1.3)
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(cm, cmap="Blues")
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
        fig.savefig(run_dir / "matrice_confusion.png", dpi=150)
        plt.close(fig)

        report = classification_report(val_labels, val_preds, labels=report_labels, target_names=classes, digits=3)
        print("\n[rapport de classification]\n" + report)
        with open(run_dir / "rapport_classification.txt", "w") as f:
            f.write(report)
    else:
        print("[warn] matplotlib/scikit-learn absents ou set de val vide : "
              "courbes et matrice de confusion non générées "
              "(pip install matplotlib scikit-learn --break-system-packages).")

    append_run_index(output_dir, {
        "run_name": run_name,
        "date_heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": data_dir.name,
        "epochs_demandes": args.epochs,
        "epochs_effectifs": epochs_ran,
        "reduce_dataset_to": args.reduce_dataset_to,
        "arch": args.arch,
        "selection_metric": args.selection_metric,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "val_acc_finale": round(final_val_acc, 4),
        "val_f1_macro_finale": round(final_val_f1_macro, 4) if final_val_f1_macro is not None else None,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_classes": len(classes),
    })

    print(f"[info] Terminé. val_acc finale (meilleur modèle) = {final_val_acc:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Sur Windows, les workers du DataLoader (--num_workers > 0, des
        # processus séparés) peuvent bloquer indéfiniment la fermeture propre
        # de l'interpréteur après un Ctrl+C. os._exit() coupe immédiatement,
        # sans attendre de les rejoindre — le dernier best_model.pt (déjà
        # sauvegardé à chaque amélioration) n'est pas perdu.
        print("\n[info] Interruption clavier (Ctrl+C) détectée — arrêt immédiat.")
        sys.stdout.flush()
        os._exit(1)
