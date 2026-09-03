#!/usr/bin/env python3
"""
Résumé
extend_dataset_overlay.py créé et testé (compilation + test fonctionnel + inspection visuelle de 2 composites) :

Estime le masque de chaque crop par seuillage sur le fond blanc (pas de canal alpha disponible dans vos crops actuels).
Pour chaque image, tire une autre pomme de terre au hasard (toutes classes confondues) et la pose en périphérie (un des 4 côtés, avec un léger décalage aléatoire) avec un bord adouci.
Écrit dans --output_dir : l'original (hard link, jamais modifié) + le composite _occ.png, pour chaque classe.
occlusion_manifest.csv trace chaque composite (image de base, occultant utilisé, côté, échelle).
Utilisation sur votre vrai dataset :


python extend_dataset_overlay.py --data_dir "chemin\vers\data_multiclass_split\train" --output_dir "chemin\vers\data_multiclass_split_occ"
(pointez bien sur le dossier train/ si votre dataset est déjà splitté train/val, pour ne jamais injecter de synthétique dans la validation)

Puis entraînez avec --data_dir sur ce nouveau dossier. Avant de lancer sur tout le dataset, je vous recommande d'inspecter visuellement quelques composites générés sur vos vraies images (pas juste les disques de test) — la qualité du masque par seuillage dépend de la propreté du fond blanc réel de vos crops, moins parfait qu'un aplat de test synthétique.

"""


"""
extend_dataset_overlay.py — Étend un dataset de classification (crops
individuels de tubercules) avec des occlusions synthétiques réalistes : pour
chaque image, une autre pomme de terre (silhouette extraite par seuillage sur
le fond) est posée en périphérie, pour simuler un tubercule voisin qui en
cache un autre dans un tas en vrac.

Contrairement à RandomOcclusion (train_classif_pdt.py — rectangle/cercle
synthétique appliqué à la volée pendant l'entraînement), ce script compose de
vraies silhouettes de pommes de terre et écrit le résultat une fois sur
disque, dans un nouveau dossier : le dataset de sortie double de taille
(chaque image d'origine + sa version occluse). Aucune donnée source n'est
modifiée (originaux repris par hard link, jamais copiés/déplacés).

Les crops n'ayant pas de canal alpha (fond --bg white uni de
segment_tubers.py), le masque de chaque image est estimé par seuillage sur le
fond clair, pas lu directement.

Structure attendue de --data_dir (Mode B) :
    data_dir/
        <classe1>/   *.png ou *.jpg
        <classe2>/   *.png ou *.jpg
Si votre dataset est déjà splitté train/val, pointez --data_dir sur le
dossier train/ pour ne jamais générer d'occlusion synthétique dans la
validation.

Usage :
    python extend_dataset_overlay.py --data_dir ./dataset/data_multiclass_split/train --output_dir ./dataset/data_multiclass_split_occ
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_class_images(data_dir: Path):
    """Retourne [(chemin, classe), ...] pour toutes les images de data_dir/<classe>/
    (recherche récursive par classe, tolère des sous-dossiers de sous-type)."""
    items = []
    for cls_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.rglob("*")):
            if img.is_file() and img.suffix.lower() in IMG_EXTENSIONS:
                items.append((img, cls_dir.name))
    return items


def extract_mask(img_bgr, bg_threshold=235):
    """Masque booléen du tubercule, estimé par seuillage sur un fond clair
    unicolore (ex. --bg white de segment_tubers.py). Ne garde que le plus
    grand contour externe, rempli plein, pour éviter qu'une zone claire
    interne (reflet, lésion pâle) ne troue le masque. Repli : aucun contour
    trouvé -> masque = image entière (dégradé mais sans crash)."""
    bg = np.all(img_bgr >= bg_threshold, axis=2)
    fg = (~bg).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(img_bgr.shape[:2], dtype=bool)
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 1, -1)
    return mask.astype(bool)


def composite_occlusion(base_bgr, base_mask, occ_bgr, occ_mask, scale_range, feather, rng):
    """Pose occ_bgr/occ_mask en périphérie de base_bgr. Retourne
    (image composée, côté choisi, échelle utilisée), ou None si le
    placement est impossible (bbox dégénérée d'un côté ou de l'autre)."""
    bx, by, bw, bh = cv2.boundingRect(base_mask.astype(np.uint8))
    if bw < 4 or bh < 4:
        return None

    ox, oy, ow, oh = cv2.boundingRect(occ_mask.astype(np.uint8))
    if ow < 2 or oh < 2:
        return None
    occ_crop = occ_bgr[oy:oy + oh, ox:ox + ow]
    occ_m = occ_mask[oy:oy + oh, ox:ox + ow].astype(np.uint8)

    scale = rng.uniform(*scale_range)
    target = max(1, int(scale * max(bw, bh)))
    ratio = target / max(ow, oh)
    new_w, new_h = max(1, int(ow * ratio)), max(1, int(oh * ratio))
    occ_crop = cv2.resize(occ_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    occ_m = cv2.resize(occ_m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    side = rng.choice(["gauche", "droite", "haut", "bas"])
    jitter = rng.uniform(-0.2, 0.2)
    if side == "gauche":
        cx, cy = bx, by + bh * (0.5 + jitter)
    elif side == "droite":
        cx, cy = bx + bw, by + bh * (0.5 + jitter)
    elif side == "haut":
        cx, cy = bx + bw * (0.5 + jitter), by
    else:
        cx, cy = bx + bw * (0.5 + jitter), by + bh

    px, py = int(cx - new_w / 2), int(cy - new_h / 2)

    H, W = base_bgr.shape[:2]
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(W, px + new_w), min(H, py + new_h)
    if x1 <= x0 or y1 <= y0:
        return None
    sx0, sy0 = x0 - px, y0 - py
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)

    alpha = occ_m[sy0:sy1, sx0:sx1].astype(np.float32)
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha * 255, (k, k), 0) / 255.0
    alpha = alpha[..., None]

    out = base_bgr.copy()
    region = out[y0:y1, x0:x1].astype(np.float32)
    occ_region = occ_crop[sy0:sy1, sx0:sx1].astype(np.float32)
    out[y0:y1, x0:x1] = (occ_region * alpha + region * (1 - alpha)).astype(np.uint8)

    return out, side, scale


def main():
    parser = argparse.ArgumentParser(
        description="Étend un dataset de classification avec des occlusions "
                     "synthétiques réalistes (une pomme de terre posée devant une autre).")
    parser.add_argument("--data_dir", type=str, required=True, help="Dossier <classe>/*.png|jpg (Mode B).")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier de sortie (créé si absent).")
    parser.add_argument("--bg_threshold", type=int, default=235, help="Seuil (0-255) : un pixel dont les 3 canaux dépassent ce seuil est considéré comme fond.")
    parser.add_argument("--occ_scale_min", type=float, default=0.25, help="Taille mini de l'occultant, en fraction de la plus grande dimension de la pomme de terre de base.")
    parser.add_argument("--occ_scale_max", type=float, default=0.5)
    parser.add_argument("--feather", type=int, default=3, help="Adoucissement du bord de l'occultant (px).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.occ_scale_min <= 0 or args.occ_scale_max < args.occ_scale_min:
        sys.exit("[erreur] --occ_scale_min doit être > 0 et <= --occ_scale_max.")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"[erreur] {data_dir} n'existe pas ou n'est pas un dossier.")

    pool = list_class_images(data_dir)
    if len(pool) < 2:
        sys.exit(f"[erreur] Au moins 2 images sont nécessaires (trouvé {len(pool)}).")
    print(f"[info] {len(pool)} images trouvées dans {len({c for _, c in pool})} classes.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest_path = output_dir / "occlusion_manifest.csv"
    n_ok, n_skip = 0, 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "base_image", "classe", "occluder_image", "occluder_classe",
            "cote", "echelle", "composite_path"])
        writer.writeheader()

        for i, (base_path, cls) in enumerate(pool):
            cls_out_dir = output_dir / cls
            cls_out_dir.mkdir(parents=True, exist_ok=True)

            link_path = cls_out_dir / base_path.name
            if not link_path.exists():
                link_path.hardlink_to(base_path.resolve())

            base_bgr = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
            if base_bgr is None:
                print(f"  !! illisible, ignorée : {base_path}")
                n_skip += 1
                continue
            base_mask = extract_mask(base_bgr, args.bg_threshold)

            occ_idx = rng.randrange(len(pool) - 1)
            if occ_idx >= i:
                occ_idx += 1
            occ_path, occ_cls = pool[occ_idx]
            occ_bgr = cv2.imread(str(occ_path), cv2.IMREAD_COLOR)
            if occ_bgr is None:
                print(f"  !! occultant illisible, ignorée : {occ_path}")
                n_skip += 1
                continue
            occ_mask = extract_mask(occ_bgr, args.bg_threshold)

            result = composite_occlusion(
                base_bgr, base_mask, occ_bgr, occ_mask,
                (args.occ_scale_min, args.occ_scale_max), args.feather, rng)
            if result is None:
                print(f"  !! composite impossible, ignorée : {base_path.name}")
                n_skip += 1
                continue
            composite, side, scale = result

            out_path = cls_out_dir / f"{base_path.stem}_occ.png"
            cv2.imwrite(str(out_path), composite)

            writer.writerow({
                "base_image": str(base_path), "classe": cls,
                "occluder_image": str(occ_path), "occluder_classe": occ_cls,
                "cote": side, "echelle": round(scale, 3),
                "composite_path": str(out_path),
            })
            n_ok += 1

            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{len(pool)}] ...")

    print(f"\n[info] {n_ok} composites générés, {n_skip} ignorée(s) sur {len(pool)} images sources.")
    print(f"[info] Dataset étendu : {output_dir}")
    print(f"[info] Manifeste : {manifest_path}")


if __name__ == "__main__":
    main()
