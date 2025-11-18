#!/usr/bin/env python3
"""
extract_captions.py

Wersja z ustalonymi plikami:
- Wejście:  results.json
- Wyjście:  results2.json

Skrypt:
- wczytuje results.json (próbuje naprawić NaN/Infinity),
- wyciąga image_id i caption z pola "imgToEval" (normalizuje image_id jeśli jest ścieżką),
- fallback: wyrażenia regularne jeśli plik nie jest poprawnym JSON-em,
- zapisuje listę obiektów [{"image_id": ..., "caption": "..."}, ...] do results2.json.

Zmiany:
- Dodano normalizację image_id: jeśli image_id jest ścieżką/plikiem, skrypt wyciąga ostatnią grupę cyfr
  (np. "/.../COCO_val2014_000000544444.jpg" -> 544444).
"""

import json
import re
import sys
from typing import List, Dict, Optional, Any

INPUT_FILE = "results.json"
OUTPUT_FILE = "results2.json"


def try_load_json(content: str) -> Optional[Any]:
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


def normalize_image_id(raw) -> Optional[int]:
    """
    Próbuj wyciągnąć int z różnych formatów:
    - jeśli już int -> zwróć
    - jeśli str zawiera tylko cyfry -> int
    - jeśli str to ścieżka/nazwa pliku: znajdź ostatnią grupę cyfr i zwróć ją jako int
    - w przeciwnym razie -> None
    """
    if raw is None:
        return None
    # Jeśli już int
    if isinstance(raw, int):
        return raw
    # Jeśli typ float, spróbuj przekonwertować jeśli to całkowite
    if isinstance(raw, float):
        try:
            return int(raw)
        except Exception:
            return None
    # Jeśli string
    if isinstance(raw, str):
        s = raw.strip()
        # bezpośrednie cyfry
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
        # znajdź wszystkie grupy cyfr i wybierz ostatnią (np. w nazwie pliku)
        digits = re.findall(r'\d+', s)
        if digits:
            try:
                return int(digits[-1])
            except Exception:
                return None
    return None


def extract_from_parsed(data) -> List[Dict]:
    results = []
    if not isinstance(data, dict):
        return results
    imgmap = None
    if "imgToEval" in data and isinstance(data["imgToEval"], dict):
        imgmap = data["imgToEval"]
    else:
        # Szukamy słownika, który wygląda jak mapa image->entry
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
            # Preferuj pole "image_id" w samym entry, fallback na klucz
            raw_image_id = entry.get("image_id", None)
            image_id = normalize_image_id(raw_image_id)
            if image_id is None:
                # spróbuj użyć klucza (może to być ścieżka lub string z cyframi)
                image_id = normalize_image_id(key)
            caption = entry.get("caption")
            if caption is None:
                # czasem caption może być pod inną właściwością lub w tablicy - ale zostawiamy prosty przypadek
                continue
            if image_id is not None:
                results.append({"image_id": image_id, "caption": caption})
    return results


def extract_with_regex(content: str) -> List[Dict]:
    """
    Regex fallback:
    - dopasowuje pary "some_key": { ... "caption": "..." ... }
    - key może być ścieżką, nazwą pliku lub liczbą; próbujemy znormalizować image_id z key lub z pola "image_id" wewnątrz bloku
    """
    results = []
    # znajdź bloki "<any_key>": { ... }
    pattern = re.compile(r'"([^"]+)"\s*:\s*\{(.*?)\}', re.DOTALL)
    for m in pattern.finditer(content):
        key = m.group(1)
        block = m.group(2)

        # najpierw spróbuj znaleźć pole "image_id" wewnątrz bloku
        imgid_m = re.search(r'"image_id"\s*:\s*"((?:\\.|[^"\\])*)"', block)
        raw_image_id = None
        if imgid_m:
            raw_image_id = imgid_m.group(1).encode('utf-8').decode('unicode_escape')
        else:
            imgid_m2 = re.search(r'"image_id"\s*:\s*([0-9]+)', block)
            if imgid_m2:
                raw_image_id = imgid_m2.group(1)

        # caption: obsługujemy zarówno "caption": "..." jak i 'caption': '...'
        cap_m = re.search(r'"caption"\s*:\s*"((?:\\.|[^"\\])*)"', block)
        caption = None
        if cap_m:
            caption = cap_m.group(1).encode('utf-8').decode('unicode_escape')
        else:
            cap_m2 = re.search(r"'caption'\s*:\s*'([^']*)'", block)
            if cap_m2:
                caption = cap_m2.group(1)
            else:
                # luźniejszy dopasowanie bez cudzysłowów wokół klucza
                cap_m3 = re.search(r'caption\s*:\s*"((?:\\.|[^"\\])*)"', block)
                if cap_m3:
                    caption = cap_m3.group(1).encode('utf-8').decode('unicode_escape')
                else:
                    cap_m4 = re.search(r"caption\s*:\s*'([^']*)'", block)
                    if cap_m4:
                        caption = cap_m4.group(1)

        # jeśli nie ma caption, pomiń
        if caption is None:
            continue

        # normalizuj image_id: najpierw z pola image_id, potem z klucza
        image_id = normalize_image_id(raw_image_id) if raw_image_id is not None else None
        if image_id is None:
            image_id = normalize_image_id(key)

        if image_id is not None:
            results.append({"image_id": image_id, "caption": caption})
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
