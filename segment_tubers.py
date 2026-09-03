#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étage 1 — Détourage individuel de tubercules dans des images de tas en vrac.

Pipeline (aucune annotation requise) :
    SAM2 automatic mask generation (zero-shot)
      -> métriques géométriques par masque
      -> filtrage hiérarchique (suppression du tas et des fragments)
      -> NMS sur IoU
      -> filtrage géométrique (aire, solidité, élongation, contact bord)
      -> export des crops détourés + CSV de métadonnées

Le CSV porte un flag `quality` (ok / suspect) destiné à conditionner la
confiance accordée à la sortie de l'étage 2 : un tubercule occulté peut
cacher sa lésion, le classifieur renverra un "sain" faussement confiant.

Usage :
    python segment_tubers.py --input imgs/ --output out/ --debug
    python segment_tubers.py --input imgs/ --output out/ --backend sam1 \
        --checkpoint mobile_sam.pt --model-type vit_t
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ===========================================================================
# 1. Backend de segmentation
# ===========================================================================

def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def build_generator(args, device: str):
    """Retourne (generator, autocast_ctx_factory)."""
    if args.backend == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        if args.checkpoint:
            sam = build_sam2(args.model_cfg, args.checkpoint,
                             device=device, apply_postprocessing=False)
            gen = SAM2AutomaticMaskGenerator(
                model=sam,
                points_per_side=args.points_per_side,
                points_per_batch=64,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_thresh,
                stability_score_offset=0.7,
                crop_n_layers=args.crop_n_layers,
                crop_n_points_downscale_factor=2,
                box_nms_thresh=0.7,
                min_mask_region_area=args.min_mask_region_area,
                use_m2m=True,
            )
        else:
            # Téléchargement automatique depuis le Hub HuggingFace.
            gen = SAM2AutomaticMaskGenerator.from_pretrained(
                args.hf_model,
                device=device,
                points_per_side=args.points_per_side,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_thresh,
                crop_n_layers=args.crop_n_layers,
                min_mask_region_area=args.min_mask_region_area,
            )

        def ctx():
            import torch
            if device == "cuda":
                return torch.autocast("cuda", dtype=torch.bfloat16)
            return nullcontext()

        return gen, ctx

    # --- SAM 1 / MobileSAM (même API, mais paquets pip distincts) ----------
    if args.model_type == "vit_t":
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    else:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    sam.eval()
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=2,
        box_nms_thresh=0.7,
        min_mask_region_area=args.min_mask_region_area,
    )
    return gen, nullcontext


# ===========================================================================
# 2. Métriques géométriques
# ===========================================================================

def mask_metrics(mask: np.ndarray) -> dict | None:
    """Métriques de forme d'un masque booléen. None si dégénéré."""
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 10.0:
        return None

    # Un masque multi-composantes est un regroupement de tubercules disjoints,
    # pas un tubercule : les métriques de contour ne décriraient que la plus
    # grosse composante alors que px_area couvre l'ensemble.
    total_px = float(np.count_nonzero(mask))
    main_px = float(np.count_nonzero(
        cv2.drawContours(np.zeros_like(m8), [cnt], -1, 1, -1)))
    if total_px > 0 and main_px / total_px < 0.90:
        return None

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    perim = float(cv2.arcLength(cnt, True))
    circularity = 4.0 * np.pi * area / (perim ** 2) if perim > 0 else 0.0

    (_, _), (rw, rh), angle = cv2.minAreaRect(cnt)
    major, minor = max(rw, rh), min(rw, rh)
    elongation = major / minor if minor > 1e-6 else 999.0
    # Taux de remplissage du rectangle orienté : discrimine les formes
    # concaves/composites d'un tubercule convexe.
    rect_fill = area / (rw * rh) if rw * rh > 0 else 0.0

    # Sommets du contour simplifié : peu de sommets + rect_fill élevé =
    # forme anguleuse/rectangulaire (damier, objet manufacturé), pas un tubercule.
    n_vertices = len(cv2.approxPolyDP(cnt, 0.02 * perim, True))

    x, y, w, h = cv2.boundingRect(cnt)
    return {
        "px_area": float(np.count_nonzero(mask)),
        "contour_area": area,
        "solidity": solidity,
        "circularity": circularity,
        "elongation": elongation,
        "rect_fill": rect_fill,
        "n_vertices": n_vertices,
        "major_axis": major,
        "minor_axis": minor,
        "angle": angle,
        "bbox": (x, y, w, h),
    }


def touches_border(bbox, shape, tol: int = 3) -> int:
    x, y, w, h = bbox
    H, W = shape[:2]
    return int(x <= tol or y <= tol or x + w >= W - tol or y + h >= H - tol)


# ===========================================================================
# 3. Filtrage
# ===========================================================================

def _pair_intersection(mi, mj, bi, bj) -> int:
    """Intersection de deux masques, calculée sur le recouvrement des bboxes."""
    x0 = max(bi[0], bj[0])
    y0 = max(bi[1], bj[1])
    x1 = min(bi[0] + bi[2], bj[0] + bj[2])
    y1 = min(bi[1] + bi[3], bj[1] + bj[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(mi[y0:y1, x0:x1] & mj[y0:y1, x0:x1]))


def geometric_reject(r, args, img_area):
    """Rejets Filtre 0 bon marché (forme du masque uniquement, aucun accès
    aux pixels de l'image) — appliqués dans process_image() avant le calcul
    coûteux flou/saturation, pour ne pas le payer sur des candidats de toute
    façon condamnés (cas dominant observé : des centaines de micro-fragments
    par image). Retourne la clé de stats du rejet, ou None si le candidat
    passe ce premier tri."""
    a = r["px_area"]
    if a > args.max_area_frac * img_area:
        return "fond/tas"
    if a < args.min_area_frac * img_area:
        return "trop petit"
    if r["elongation"] > args.max_elongation:
        return "trop allonge"
    if r["solidity"] < args.min_solidity_hard:
        return "non convexe"
    if r["n_vertices"] <= args.sq_max_vertices and r["rect_fill"] > args.sq_min_rect_fill:
        return "anguleux (carré)"
    return None


def filter_masks(records, args, stats):
    """Cascade de filtres. Retourne la liste des masques conservés.
    Les candidats reçus ont déjà passé geometric_reject() + les seuils
    flou/saturation (appliqués en amont dans process_image)."""
    kept = records
    if not kept:
        return []

    # --- Précalcul des intersections deux à deux --------------------------
    n = len(kept)
    inter = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            v = _pair_intersection(kept[i]["mask"], kept[j]["mask"],
                                   kept[i]["bbox"], kept[j]["bbox"])
            inter[i, j] = inter[j, i] = v
    areas = np.array([r["px_area"] for r in kept], dtype=np.int64)

    # --- Filtre 1 : parents (tas, groupes de 2-3 tubercules) --------------
    # Un masque contenant >= 2 autres masques nettement plus petits est un
    # regroupement, pas un tubercule individuel. Mais une forme contenue ne
    # compte comme "sous-tubercule" que si elle a elle-même une forme plausible
    # de tubercule (ronde/convexe) : un germe ou de la terre contenus dans une
    # grosse patate ne doivent pas faire rejeter la patate comme "tas" — ils
    # seront retirés ensuite par le Filtre 2 (fragment) sans tuer leur parent.
    alive = np.ones(n, dtype=bool)
    contain = inter / np.maximum(areas[None, :], 1)   # contain[i,j] = part de j dans i
    for i in range(n):
        candidates = np.where((contain[i] > args.containment_thresh)
                              & (areas < 0.65 * areas[i]))[0]
        candidates = candidates[candidates != i]
        children = [j for j in candidates
                    if kept[j]["solidity"] >= args.group_child_min_solidity
                    and kept[j]["elongation"] <= args.group_child_max_elongation]
        if len(children) >= 2:
            alive[i] = False
            stats["parent (groupe)"] += 1

    # --- Filtre 2 : fragments (sous-parties d'un tubercule vivant) --------
    for j in range(n):
        if not alive[j]:
            continue
        parents = np.where(alive & (contain[:, j] > args.containment_thresh)
                           & (areas > 1.8 * areas[j]))[0]
        parents = parents[parents != j]
        if len(parents) > 0:
            alive[j] = False
            stats["fragment"] += 1

    # --- Filtre 3 : NMS sur IoU -------------------------------------------
    order = np.argsort(-areas)
    for idx_a in range(n):
        i = order[idx_a]
        if not alive[i]:
            continue
        for idx_b in range(idx_a + 1, n):
            j = order[idx_b]
            if not alive[j]:
                continue
            union = areas[i] + areas[j] - inter[i, j]
            if union > 0 and inter[i, j] / union > args.nms_iou:
                # On garde celui dont la géométrie est la plus "tubercule".
                stats["doublon (NMS)"] += 1
                if kept[i]["solidity"] >= kept[j]["solidity"]:
                    alive[j] = False
                else:
                    alive[i] = False
                    break

    survivors = [kept[i] for i in range(n) if alive[i]]
    if not survivors:
        return []

    # --- Filtre 4 : cohérence de taille (aire relative à la médiane) ------
    med = float(np.median([r["px_area"] for r in survivors]))
    final = []
    for r in survivors:
        r["area_rel"] = r["px_area"] / med if med > 0 else 0.0
        if not (args.rel_area_min <= r["area_rel"] <= args.rel_area_max):
            stats["aire incoherente"] += 1
            continue
        final.append(r)

    # --- Recouvrement résiduel avec les voisins (proxy d'occlusion) -------
    for a in final:
        ov = 0.0
        for b in final:
            if b is a:
                continue
            v = _pair_intersection(a["mask"], b["mask"], a["bbox"], b["bbox"])
            ov = max(ov, v / max(a["px_area"], 1))
        a["neighbor_overlap"] = ov

    return final


def grade(r, args) -> str:
    """ok / suspect — conditionne la confiance de l'étage 2."""
    if r["border"]:
        return "suspect"
    if r["solidity"] < args.min_solidity_soft:
        return "suspect"
    if r["neighbor_overlap"] > 0.08:
        return "suspect"
    if not (0.6 <= r["area_rel"] <= 1.8):
        return "suspect"
    return "ok"


# ===========================================================================
# 4. Découpe
# ===========================================================================

BG_FILL = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128)}
REMAINDER_COLOR = {"purple": (128, 0, 128), "white": (255, 255, 255), "blue": (255, 0, 0)}


def crop_tuber(img_bgr, mask, bbox, args):
    """Retourne (crop_bgr, alpha) détouré, éventuellement réaligné et carré."""
    H, W = img_bgr.shape[:2]
    x, y, w, h = bbox
    pad = int(round(args.pad * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    # Garde-fou : un bbox remis à l'échelle native peut déborder/coller au bord
    # et donner x1<=x0 ou y1<=y0 après clamp -> crop vide, cv2 plante dessus.
    x0, y0 = min(x0, W - 1), min(y0, H - 1)
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)

    sub = img_bgr[y0:y1, x0:x1].copy()
    sub_m = mask[y0:y1, x0:x1].astype(np.uint8)

    if args.feather > 0:
        k = args.feather * 2 + 1
        alpha = cv2.GaussianBlur(sub_m * 255, (k, k), 0)
    else:
        alpha = sub_m * 255

    if args.bg != "keep":
        if args.bg == "median":
            fill = np.median(sub[sub_m.astype(bool)], axis=0) if sub_m.any() else (255, 255, 255)
        else:
            fill = BG_FILL[args.bg]
        a = (alpha.astype(np.float32) / 255.0)[..., None]
        sub = (sub.astype(np.float32) * a
               + np.array(fill, dtype=np.float32) * (1.0 - a)).astype(np.uint8)

    if args.align:
        cnts, _ = cv2.findContours(sub_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
            if rw < rh:                       # aligner le grand axe sur l'horizontale
                ang += 90.0
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
            border = cv2.BORDER_REPLICATE if args.bg == "keep" else cv2.BORDER_CONSTANT
            bval = (0, 0, 0) if args.bg == "keep" else BG_FILL.get(args.bg, (255, 255, 255))
            sub = cv2.warpAffine(sub, M, (sub.shape[1], sub.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=border,
                                 borderValue=bval)
            alpha = cv2.warpAffine(alpha, M, (alpha.shape[1], alpha.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
            ys, xs = np.nonzero(alpha > 127)
            if len(xs) > 0:
                sub = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    if args.square:
        hh, ww = sub.shape[:2]
        side = max(hh, ww)
        top, left = (side - hh) // 2, (side - ww) // 2
        fill = BG_FILL.get(args.bg, (255, 255, 255))
        sub = cv2.copyMakeBorder(sub, top, side - hh - top, left, side - ww - left,
                                 cv2.BORDER_CONSTANT, value=fill)
        alpha = cv2.copyMakeBorder(alpha, top, side - hh - top, left, side - ww - left,
                                   cv2.BORDER_CONSTANT, value=0)

    if args.out_size:
        interp = cv2.INTER_AREA if sub.shape[0] > args.out_size else cv2.INTER_CUBIC
        sub = cv2.resize(sub, (args.out_size, args.out_size), interpolation=interp)
        alpha = cv2.resize(alpha, (args.out_size, args.out_size), interpolation=interp)

    return sub, alpha


def draw_overlay(img_bgr, kept):
    ov = img_bgr.copy()
    rng = np.random.default_rng(0)
    for i, r in enumerate(kept):
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        ov[r["mask"]] = (0.55 * ov[r["mask"]] + 0.45 * np.array(color)).astype(np.uint8)
        x, y, w, h = r["bbox"]
        col = (0, 200, 0) if r["quality"] == "ok" else (0, 140, 255)
        cv2.rectangle(ov, (x, y), (x + w, y + h), col, 2)
        cv2.putText(ov, f"{i}", (x + 3, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)
    return ov


def draw_remainder(img_bgr, kept, color):
    """Peint les tubercules déjà extraits en couleur unie ; le reste garde
    ses pixels d'origine pour repérer visuellement des tubercules manqués."""
    rem = img_bgr.copy()
    covered = np.zeros(img_bgr.shape[:2], dtype=bool)
    for r in kept:
        covered |= r["mask"]
    rem[covered] = color
    return rem


# ===========================================================================
# 5. Traitement d'une image
# ===========================================================================

def process_image(path: Path, rel_key: str, gen, ctx_factory, args, writer, out_root):
    img0 = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img0 is None:
        print(f"  !! illisible : {rel_key}{path.suffix}")
        return 0, 0, 0.0, 0.0, 0.0

    H0, W0 = img0.shape[:2]
    scale = 1.0
    img = img0
    if args.max_side and max(H0, W0) > args.max_side:
        scale = args.max_side / max(H0, W0)
        img = cv2.resize(img0, (int(W0 * scale), int(H0 * scale)),
                         interpolation=cv2.INTER_AREA)

    t0_gen = time.time()
    with ctx_factory():
        raw = gen.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t_gen = time.time() - t0_gen

    t0_filt = time.time()
    img_area = img.shape[0] * img.shape[1]
    stats = {k: 0 for k in ["fond/tas", "trop petit", "trop allonge", "non convexe",
                            "anguleux (carré)", "flou", "caillou (couleur)",
                            "parent (groupe)", "fragment", "doublon (NMS)",
                            "aire incoherente"]}

    records = []
    for m in raw:
        seg = np.ascontiguousarray(m["segmentation"], dtype=bool)
        met = mask_metrics(seg)
        if met is None:
            continue
        met["mask"] = seg
        met["border"] = touches_border(met["bbox"], img.shape)
        met["stability"] = float(m.get("stability_score", 0.0))
        met["pred_iou"] = float(m.get("predicted_iou", 0.0))

        # Rejets bon marché (forme seule) AVANT le calcul flou/saturation :
        # évite de payer cette étape coûteuse sur les candidats de toute
        # façon condamnés (cas dominant observé : micro-fragments par dizaines).
        reason = geometric_reject(met, args, img_area)
        if reason is not None:
            stats[reason] += 1
            continue

        x, y, w, h = met["bbox"]
        mask_crop = seg[y:y + h, x:x + w]
        sub_img = img[y:y + h, x:x + w]
        if mask_crop.any():
            # HSV une seule fois : le canal V sert de proxy niveaux de gris
            # pour le flou, évite une 2e conversion couleur (BGR2GRAY).
            hsv = cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)
            lap = cv2.Laplacian(hsv[..., 2], cv2.CV_64F)
            met["sharpness"] = float(lap[mask_crop].var())
            met["mean_saturation"] = float(hsv[..., 1][mask_crop].mean())
        else:
            met["sharpness"] = 0.0
            met["mean_saturation"] = 0.0

        if met["sharpness"] < args.min_sharpness:
            stats["flou"] += 1
            continue
        if met["mean_saturation"] < args.min_saturation:
            stats["caillou (couleur)"] += 1
            continue

        records.append(met)

    kept = filter_masks(records, args, stats)
    kept.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    for r in kept:
        r["quality"] = grade(r, args)
    t_filt = time.time() - t0_filt

    t0_crop = time.time()
    crop_dir = out_root / "crops" / rel_key
    crop_dir.mkdir(parents=True, exist_ok=True)
    base = Path(rel_key).name

    n_ok = 0
    for i, r in enumerate(kept):
        try:
            # Remontée à la résolution native pour la découpe.
            if scale != 1.0:
                full_mask = cv2.resize(r["mask"].astype(np.uint8), (W0, H0),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                x, y, w, h = r["bbox"]
                fb = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            else:
                full_mask, fb = r["mask"], r["bbox"]

            crop, alpha = crop_tuber(img0, full_mask, fb, args)
            name = f"{base}_t{i:03d}.png"
            if args.rgba:
                cv2.imwrite(str(crop_dir / name), np.dstack([crop, alpha]))
            else:
                cv2.imwrite(str(crop_dir / name), crop)

            writer.writerow({
                "image": f"{rel_key}{path.suffix}", "tuber_id": i,
                "x": fb[0], "y": fb[1], "w": fb[2], "h": fb[3],
                "area_px": int(r["px_area"] / (scale ** 2)),
                "area_rel": round(r["area_rel"], 3),
                "solidity": round(r["solidity"], 3),
                "circularity": round(r["circularity"], 3),
                "elongation": round(r["elongation"], 3),
                "rect_fill": round(r["rect_fill"], 3),
                "sharpness": round(r["sharpness"], 1),
                "saturation": round(r["mean_saturation"], 1),
                "neighbor_overlap": round(r["neighbor_overlap"], 3),
                "border_contact": r["border"],
                "stability": round(r["stability"], 3),
                "pred_iou": round(r["pred_iou"], 3),
                "quality": r["quality"],
                "crop_path": str(Path("crops") / rel_key / name),
            })
            n_ok += (r["quality"] == "ok")
        except Exception as e:
            print(f"  !! tubercule {i} de {rel_key}{path.suffix} ignoré (erreur : {e})")
            continue

    if args.debug:
        overlay_path = out_root / "debug" / "overlay" / f"{rel_key}_overlay.jpg"
        remainder_path = out_root / "debug" / "remainder" / f"{rel_key}_remainder.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        remainder_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), draw_overlay(img, kept),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(remainder_path),
                    draw_remainder(img, kept, REMAINDER_COLOR[args.remainder_color]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    t_crop = time.time() - t0_crop

    rej = ", ".join(f"{k}={v}" for k, v in stats.items() if v)
    print(f"  {rel_key}{path.suffix}: {len(raw)} masques bruts -> {len(kept)} tubercules "
          f"({n_ok} ok, {len(kept) - n_ok} suspects)"
          + (f"  [rejets: {rej}]" if rej else ""))
    print(f"    [timing] SAM2={t_gen:.2f}s  filtrage={t_filt:.2f}s  export={t_crop:.2f}s")
    return len(kept), n_ok, t_gen, t_filt, t_crop


# ===========================================================================
# 6. CLI
# ===========================================================================

CSV_FIELDS = ["image", "tuber_id", "x", "y", "w", "h", "area_px", "area_rel",
              "solidity", "circularity", "elongation", "rect_fill",
              "sharpness", "saturation",
              "neighbor_overlap", "border_contact", "stability", "pred_iou",
              "quality", "crop_path"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Détourage zero-shot de tubercules (SAM2 AMG + filtrage géométrique).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io = p.add_argument_group("E/S")
    io.add_argument("--input", required=True, nargs="+",
                    help="Image(s) et/ou dossier(s) d'images (plusieurs chemins possibles)")
    io.add_argument("--output", default="out", help="Dossier de sortie")
    io.add_argument("--debug", action="store_true", help="Overlays de contrôle")
    io.add_argument("--limit", type=int, default=0,
                    help="Ne traiter que les N premières images (0 = toutes)")
    io.add_argument("--run-name", default=None,
                    help="Nom du run (sous-dossier de --output) ; horodatage auto si omis")
    io.add_argument("--remainder-color", choices=list(REMAINDER_COLOR), default="purple",
                    help="Couleur des tubercules déjà extraits dans la capture remainder "
                         "(le reste garde ses pixels d'origine pour repérer les manqués)")
    io.add_argument("--exclude", nargs="+", default=[],
                    help="Fichier(s)/dossier(s) à exclure de --input")
    io.add_argument("--resume", action="store_true",
                    help="Reprend un run interrompu dans --run-name (saute les images déjà traitées)")

    mo = p.add_argument_group("Modèle")
    mo.add_argument("--backend", choices=["sam2", "sam1"], default="sam2")
    mo.add_argument("--checkpoint", default=None, help="Chemin du .pt (sinon HF Hub)")
    mo.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    mo.add_argument("--hf-model", default="facebook/sam2.1-hiera-small",
                    help="Variante SAM2.1 (Hub HF) : hiera-tiny/small/base-plus/large, "
                         "du plus rapide au plus précis")
    mo.add_argument("--model-type", default="vit_h", help="SAM1 : vit_h/vit_l/vit_b/vit_t")
    mo.add_argument("--device", default="auto")
    mo.add_argument("--max-side", type=int, default=1024,
                    help="Redimensionnement avant AMG (0 = désactivé)")

    am = p.add_argument_group("AMG")
    am.add_argument("--points-per-side", type=int, default=24)
    am.add_argument("--pred-iou-thresh", type=float, default=0.80)
    am.add_argument("--stability-thresh", type=float, default=0.92)
    am.add_argument("--crop-n-layers", type=int, default=0)
    am.add_argument("--min-mask-region-area", type=float, default=25.0,
                    help="Aire minimale (en pixels bruts) filtrée à l'intérieur du générateur "
                         "SAM, avant le calcul des métriques Python — monter cette valeur (ex. "
                         "300-800) réduit le nombre de micro-masques à traiter quand un backend "
                         "(ex. MobileSAM) en propose beaucoup, sans changer --min-area-frac")

    fi = p.add_argument_group("Filtrage")
    fi.add_argument("--min-area-frac", type=float, default=0.025,
                    help="Aire minimale d'un tubercule, en fraction de l'aire image (1/40 par défaut)")
    fi.add_argument("--max-area-frac", type=float, default=0.5)
    fi.add_argument("--max-elongation", type=float, default=7.0)
    fi.add_argument("--min-solidity-hard", type=float, default=0.3)
    fi.add_argument("--min-solidity-soft", type=float, default=0.35)
    fi.add_argument("--containment-thresh", type=float, default=0.40,
                    help="Fraction de recouvrement (même partiel/périphérique) au-delà de "
                         "laquelle une petite forme est absorbée par la grosse qui la touche "
                         "(fragment retiré, la grosse forme est gardée) — abaissé pour capter "
                         "aussi les germes en périphérie, pas seulement totalement contenus")
    fi.add_argument("--nms-iou", type=float, default=0.55)
    fi.add_argument("--rel-area-min", type=float, default=0.30)
    fi.add_argument("--rel-area-max", type=float, default=6.00)
    fi.add_argument("--group-child-min-solidity", type=float, default=0.75,
                    help="Une forme contenue ne compte comme sous-tubercule (Filtre parent) "
                         "que si sa propre solidité dépasse ce seuil ; sinon (germe/terre) "
                         "le parent est gardé et elle sera retirée comme fragment")
    fi.add_argument("--group-child-max-elongation", type=float, default=2.5,
                    help="Idem group-child-min-solidity, sur l'élongation max de l'enfant")
    fi.add_argument("--sq-max-vertices", type=int, default=6,
                    help="Rejette les masques ~polygonaux (damier, objets carrés)")
    fi.add_argument("--sq-min-rect-fill", type=float, default=0.92,
                    help="Seuil de rect_fill combiné à sq-max-vertices pour le rejet 'carré'")
    fi.add_argument("--min-sharpness", type=float, default=0.0,
                    help="Variance du Laplacien mini (0 = désactivé) ; rejette le flou "
                         "(arrière-plan hors mise au point)")
    fi.add_argument("--min-saturation", type=float, default=0.0,
                    help="Saturation HSV moyenne mini (0 = désactivé) ; rejette les objets "
                         "ternes (cailloux)")

    cr = p.add_argument_group("Découpe")
    cr.add_argument("--bg", choices=["white", "black", "gray", "median", "keep"],
                    default="white", help="Fond hors masque ('keep' = pixels d'origine)")
    cr.add_argument("--pad", type=float, default=0.06, help="Marge (fraction du grand côté)")
    cr.add_argument("--feather", type=int, default=2, help="Adoucissement du bord (px)")
    cr.add_argument("--square", action="store_true", help="Padding carré avant resize")
    cr.add_argument("--out-size", type=int, default=0, help="Taille finale (0 = natif)")
    cr.add_argument("--align", action="store_true", help="Aligner le grand axe à l'horizontale")
    cr.add_argument("--rgba", action="store_true", help="PNG avec canal alpha")

    return p.parse_args(argv)


def rel_key_for(f: Path, src: Path) -> str:
    """Chemin relatif de f à sa racine d'entrée src (sans extension), pour que
    la sortie reflète l'arborescence du dossier d'entrée au lieu d'aplatir
    tous les fichiers par leur seul nom (risque de collision entre
    sous-dossiers réutilisant les mêmes noms, ex. 1.jpg dans chaque session)."""
    if src.is_file():
        return f.stem
    return str(f.relative_to(src).with_suffix("")).replace("\\", "/")


def main(argv=None):
    args = parse_args(argv)
    srcs = [Path(p) for p in args.input]
    excludes = [Path(p).resolve() for p in args.exclude]

    seen = {}
    for src in srcs:
        candidates = [src] if src.is_file() else sorted(src.rglob("*"))
        for f in candidates:
            if not f.is_file() or f.suffix.lower() not in IMG_EXT:
                continue
            resolved = f.resolve()
            if any(resolved == ex or ex in resolved.parents for ex in excludes):
                continue
            if resolved in seen:
                continue
            seen[resolved] = (f, rel_key_for(f, src))
    files = sorted(seen.values(), key=lambda t: t[1])

    if not files:
        sys.exit(f"Aucune image trouvée dans {srcs}")

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(args.output) / run_name

    if args.resume:
        if not out_root.exists():
            sys.exit(f"--resume demandé mais le dossier de run n'existe pas : {out_root}")
    else:
        if out_root.exists():
            sys.exit(f"Le dossier de run existe déjà : {out_root}\n"
                     f"Choisissez un autre --run-name, supprimez-le, ou passez --resume pour continuer.")
        out_root.mkdir(parents=True)

    processed_log = out_root / "processed.log"
    already_done = set()
    if args.resume and processed_log.is_file():
        already_done = set(processed_log.read_text(encoding="utf-8").splitlines())
    if already_done:
        before = len(files)
        files = [(f, rk) for f, rk in files if rk not in already_done]
        print(f"[resume] {before - len(files)} image(s) déjà traitée(s) ignorée(s), "
              f"{len(files)} restante(s).")

    if args.limit:
        files = files[:args.limit]

    params_name = f"params_resume_{time.strftime('%Y%m%d_%H%M%S')}.txt" if args.resume else "params.txt"
    with open(out_root / params_name, "w", encoding="utf-8") as fh:
        fh.write(f"run_name = {run_name}\n")
        for k, v in vars(args).items():
            fh.write(f"{k} = {v}\n")

    device = pick_device(args.device)
    print(f"Run={run_name}  Backend={args.backend}  device={device}  images={len(files)}")
    t0 = time.time()
    gen, ctx = build_generator(args, device)
    print(f"Modèle chargé en {time.time() - t0:.1f}s\n")

    total, total_ok = 0, 0
    loop_t0 = time.time()
    csv_path = out_root / "tubers.csv"
    write_header = not (args.resume and csv_path.is_file())
    with open(csv_path, "a" if args.resume else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        total_gen, total_filt, total_crop = 0.0, 0.0, 0.0
        for k, (f, rel_key) in enumerate(files, 1):
            print(f"[{k}/{len(files)}]", end=" ")
            try:
                n, n_ok, t_gen, t_filt, t_crop = process_image(
                    f, rel_key, gen, ctx, args, writer, out_root)
            except Exception as e:
                print(f"  !! image ignorée (erreur : {e})")
                n, n_ok, t_gen, t_filt, t_crop = 0, 0, 0.0, 0.0, 0.0
            total += n
            total_ok += n_ok
            total_gen += t_gen
            total_filt += t_filt
            total_crop += t_crop
            with open(processed_log, "a", encoding="utf-8") as plog:
                plog.write(rel_key + "\n")
            elapsed = time.time() - loop_t0
            avg = elapsed / k
            eta = avg * (len(files) - k)
            print(f"    -> ecoule {elapsed / 60:.1f} min | restant estime {eta / 60:.1f} min "
                  f"({avg:.1f} s/image en moyenne)")

    total_measured = total_gen + total_filt + total_crop
    print(f"\n{total} tubercules extraits ({total_ok} 'ok', {total - total_ok} 'suspect') "
          f"en {time.time() - t0:.1f}s")
    if total_measured > 0:
        print(f"Repartition du temps : SAM2={total_gen:.1f}s ({100*total_gen/total_measured:.0f}%)  "
              f"filtrage={total_filt:.1f}s ({100*total_filt/total_measured:.0f}%)  "
              f"export={total_crop:.1f}s ({100*total_crop/total_measured:.0f}%)")
    print(f"-> {out_root / 'tubers.csv'}")
    print(f"-> {out_root / 'crops'}")


if __name__ == "__main__":
    main()
