#!/usr/bin/env python3
"""
Compute CIDEr scores per image (skip SPICE) with all paths/data hard-coded
and save results to results3.json.

All inputs are set as constants in this file:
  - ANNOTATION_FILE
  - RESULTS_FILE
  - OUTPUT_FILE

Run:
  python compute_cider_per_image_hardcoded.py
"""
import json
import sys

# Hard-coded paths / data (wszystko na sztywno)
ANNOTATION_FILE = "annotations/captions_val2014.json"
RESULTS_FILE = "results2.json"
OUTPUT_FILE = "results3.json"

try:
    from pycocotools.coco import COCO
except Exception as e:
    print("Brak biblioteki pycocotools lub problem z importem:", e, file=sys.stderr)
    raise

try:
    # Import only CIDEr scorer to avoid SPICE computation
    from pycocoevalcap.cider.cider import Cider
except Exception as e:
    print("Brak pycocoevalcap.cider lub problem z importem:", e, file=sys.stderr)
    raise

def build_gt_dict(coco, img_ids):
    gts = {}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        gts[img_id] = [ann["caption"] for ann in anns]
    return gts

def build_res_dict(coco_res, img_ids):
    res = {}
    ann_ids = coco_res.getAnnIds(imgIds=img_ids)
    anns = coco_res.loadAnns(ann_ids)
    for ann in anns:
        # ann['image_id'] is typically int
        res.setdefault(ann["image_id"], []).append(ann["caption"])
    return res

def main():
    print("Używam sztywnych ścieżek:")
    print("  ANNOTATION_FILE =", ANNOTATION_FILE)
    print("  RESULTS_FILE    =", RESULTS_FILE)
    print("  OUTPUT_FILE     =", OUTPUT_FILE)

    coco = COCO(ANNOTATION_FILE)
    coco_res = coco.loadRes(RESULTS_FILE)

    img_ids = coco_res.getImgIds()
    if not img_ids:
        raise RuntimeError("Brak image_ids w pliku wyników.")

    # Ensure deterministic order (use returned order from getImgIds)
    common_img_ids = []
    # Build ground truth and predictions
    gts_all = build_gt_dict(coco, img_ids)
    res_all = build_res_dict(coco_res, img_ids)
    for img_id in img_ids:
        if img_id in gts_all and img_id in res_all:
            common_img_ids.append(img_id)

    if not common_img_ids:
        raise RuntimeError("Brak wspólnych image_id między adnotacjami a wynikami.")

    # Prepare dicts preserving the same insertion order as common_img_ids
    gts = {img_id: gts_all[img_id] for img_id in common_img_ids}
    res = {img_id: res_all[img_id] for img_id in common_img_ids}

    print(f"Licząc CIDEr dla {len(common_img_ids)} obrazów (bez SPICE)...")
    cider = Cider()
    overall_score, per_image_scores = cider.compute_score(gts, res)

    # Build output
    per_image_list = []
    for img_id, score in zip(common_img_ids, per_image_scores):
        per_image_list.append({
            "image_id": int(img_id),
            "cider": float(round(float(score), 6))
        })

    result_obj = {
        "overall_cider": float(round(float(overall_score), 6)),
        "per_image": per_image_list
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result_obj, f, ensure_ascii=False, indent=2)

    print(f"Zapisano wyniki do: {OUTPUT_FILE}")
    print("Overall CIDEr:", result_obj["overall_cider"])

if __name__ == "__main__":
    main()
