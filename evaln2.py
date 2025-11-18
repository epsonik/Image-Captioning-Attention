#!/usr/bin/env python3
"""
extract_captions.py

Wersja z ustalonymi plikami:
- Wejście:  results.json
- Wyjście:  results2.json

Skrypt:
- wczytuje results.json (próbuje naprawić NaN/Infinity),
- wyciąga image_id i caption z pola "imgToEval",
- fallback: wyrażenia regularne jeśli plik nie jest poprawnym JSON-em,
- zapisuje listę obiektów [{"image_id": ..., "caption": "..."}, ...] do results2.json.
"""

import json
import re
import sys
from typing import List, Dict

INPUT_FILE = "results.json"
OUTPUT_FILE = "results2.json"

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
    imgmap = None
    if "imgToEval" in data and isinstance(data["imgToEval"], dict):
        imgmap = data["imgToEval"]
    else:
        for k, v in data.items():
            if isinstance(v, dict):
                sample_vals = list(v.values())
                if any(isinstance(x, dict) and ("caption" in x or "image_id" in x) for x in sample_vals):
                    imgmap = v
                    break
    if imgmap is None:
        return results

    for key, entry in imgmap.items():
        if isinstance(entry, dict):
            image_id = entry.get("image_id")
            if image_id is None:
                try:
                    image_id = int(key)
                except Exception:
                    image_id = None
            caption = entry.get("caption")
            if caption is not None and image_id is not None:
                try:
                    image_id = int(image_id)
                except Exception:
                    pass
                results.append({"image_id": image_id, "caption": caption})
    return results

def extract_with_regex(content: str) -> List[Dict]:
    results = []
    # znajdź bloki "12345": { ... "caption": "..." ... }
    pattern = re.compile(r'"(\d+)"\s*:\s*\{(.*?)\}', re.DOTALL)
    for m in pattern.finditer(content):
        key = m.group(1)
        block = m.group(2)
        cap_m = re.search(r'"caption"\s*:\s*"((?:\\.|[^"\\])*)"', block)
        if cap_m:
            caption = cap_m.group(1).encode('utf-8').decode('unicode_escape')
            try:
                image_id = int(key)
                results.append({"image_id": image_id, "caption": caption})
            except Exception:
                continue
        else:
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
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Nie można wczytać pliku wejściowego '{INPUT_FILE}': {e}", file=sys.stderr)
        sys.exit(2)

    parsed = try_load_json(content)
    results = []
    if parsed is not None:
        results = extract_from_parsed(parsed)

    if not results:
        results = extract_with_regex(content)

    if not results:
        print(f"Nie udało się wyodrębnić żadnych par image_id/caption z '{INPUT_FILE}'. Sprawdź format pliku.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Nie można zapisać pliku wyjściowego '{OUTPUT_FILE}': {e}", file=sys.stderr)
        sys.exit(3)

    print(f"Zapisano {len(results)} wpisów do {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
