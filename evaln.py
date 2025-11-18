# rom pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
#
# annotation_file = 'annotations/captions_val2014.json'
# results_file = 'results.json'
#
# # create coco object and coco_result object
# coco = COCO(annotation_file)
# coco_result = coco.loadRes(results_file)
#
# # create coco_eval object by taking coco and coco_result
# coco_eval = COCOEvalCap(coco, coco_result)
#
# # evaluate on a subset of images by setting
# # coco_eval.params['image_id'] = coco_result.getImgIds()
# # please remove this line when evaluating the full validation set
# coco_eval.params['image_id'] = coco_result.getImgIds()
#
# # evaluate results
# # SPICE will take a few minutes the first time, but speeds up due to caching
# coco_eval.evaluate()
#
# # print output evaluation scores
# for metric, score in coco_eval.eval.items():
#     print(f'{metric}: {score:.3f}')
#!/usr/bin/env python3
"""
extract_captions.py

Użycie:
    python extract_captions.py input_file.json output_captions.json

Co robi:
- wczytuje plik (najczęściej JSON z polami "overall" i "imgToEval")
- wyciąga dla każdego wpisu w "imgToEval" pole image_id i caption
- zapisuje listę obiektów JSON: [{"image_id": 184613, "caption": "..."}, ...]
- próbuje najpierw json.loads, a jeśli plik zawiera NaN lub inne nie-JSONowe wartości,
  robi prostą pre-propagację (zamiana NaN->null) i ponawia próbę.
- jeśli nadal nie można sparsować, używa wyrażeń regularnych aby wyciągnąć
  bloki "image" i pole "caption".
"""

import argparse
import json
import re
import sys
from typing import List, Dict

def try_load_json(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Spróbuj naprawić typowe problemy: NaN, Infinity, -Infinity -> null
        fixed = re.sub(r'\bNaN\b', 'null', content, flags=re.IGNORECASE)
        fixed = re.sub(r'\bInfinity\b', 'null', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'\b-Infinity\b', 'null', fixed, flags=re.IGNORECASE)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

def extract_from_parsed(data) -> List[Dict]:
    results = []
    if not isinstance(data, dict):
        return results
    # Typowe miejsce, gdzie są dane: data["imgToEval"]
    imgmap = None
    if "imgToEval" in data and isinstance(data["imgToEval"], dict):
        imgmap = data["imgToEval"]
    else:
        # czasem plik może zawierać bezpośrednio mapę obrazów
        # sprawdź wszystkie pola, szukając podobnej struktury
        for k, v in data.items():
            if isinstance(v, dict):
                # heurystyka: jeśli w środku jest "caption" lub "image_id"
                sample_vals = list(v.values())
                if any(isinstance(x, dict) and ("caption" in x or "image_id" in x) for x in sample_vals):
                    imgmap = v
                    break
    if imgmap is None:
        return results

    for key, entry in imgmap.items():
        if isinstance(entry, dict):
            image_id = entry.get("image_id")
            # jeżeli brak, spróbuj użyć klucza jako id
            if image_id is None:
                try:
                    image_id = int(key)
                except Exception:
                    image_id = None
            caption = entry.get("caption")
            if caption is not None and image_id is not None:
                results.append({"image_id": int(image_id), "caption": caption})
    return results

def extract_with_regex(content: str) -> List[Dict]:
    results = []
    # znajdź bloki "12345": { ... "caption": "..." ... }
    # używamy DOTALL aby objąć wieloliniowe bloki
    pattern = re.compile(r'"(\d+)"\s*:\s*\{(.*?)\}', re.DOTALL)
    for m in pattern.finditer(content):
        key = m.group(1)
        block = m.group(2)
        # spróbuj znaleźć caption w bloku
        cap_m = re.search(r'"caption"\s*:\s*"((?:\\.|[^"\\])*)"', block)
        if cap_m:
            caption = cap_m.group(1).encode('utf-8').decode('unicode_escape')
            try:
                image_id = int(key)
                results.append({"image_id": image_id, "caption": caption})
            except Exception:
                continue
        else:
            # czasem image_id jest polem w bloku i caption może być dalej
            cap_m2 = re.search(r"caption\s*:\s*'([^']*)'", block)
            if cap_m2:
                caption = cap_m2.group(1)
                try:
                    image_id = int(key)
                    results.append({"image_id": image_id, "caption": caption})
                except Exception:
                    continue
    return results

def main():
    parser = argparse.ArgumentParser(description="Wyodrębnij image_id i caption z pliku oceny i zapisz do JSON.")
    parser.add_argument("input", requred=False, default="results.json", help="Plik wejściowy (zawierający 'imgToEval')")
    parser.add_argument("output", requred=False, default="result2.json", help="Plik wyjściowy JSON (lista obiektów)")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Nie można wczytać pliku wejściowego: {e}", file=sys.stderr)
        sys.exit(2)

    parsed = try_load_json(content)
    results = []
    if parsed is not None:
        results = extract_from_parsed(parsed)

    if not results:
        # fallback: regex
        results = extract_with_regex(content)

    if not results:
        print("Nie udało się wyodrębnić żadnych par image_id/caption. Sprawdź format pliku.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Nie można zapisać pliku wyjściowego: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"Zapisano {len(results)} wpisów do {args.output}")

if __name__ == "__main__":
    main()
