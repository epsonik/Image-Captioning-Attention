#!/usr/bin/env python3
"""
Skrypt do:
- uruchamiania oceny COCO (jeśli podasz plik z adnotacjami i plik z wynikami w formacie COCO)
  ORAZ/ LUB
- wczytywania gotowych wyników ewaluacji (plik JSON zawierający "overall" i "imgToEval"),
  i ładnego ich wypisania.

Przykłady użycia:
  - Ocena od podstaw:
      python evaluate_coco_results.py --annotations annotations/captions_val2014.json --results results.json
  - Wczytanie gotowej ewaluacji (np. wynik skryptu COCOEvalCap zapisany do pliku):
      python evaluate_coco_results.py --eval eval_output.json
  - Wczytanie pliku results.json bez adnotacji (tylko podgląd):
      python evaluate_coco_results.py --results results.json --preview

Skrypt automatycznie odróżnia:
- "eval JSON" (zawiera klucze "overall" i "imgToEval") — wtedy tylko wypisze metryki,
- albo "COCO results" (lista predykcji) — wtedy, jeśli podasz annotations, wykona ewaluację;
  jeśli adnotacje nie są podane, tylko pokaże podgląd wyników.

Wymagane biblioteki (jeśli chcesz uruchamiać COCOEvalCap):
  pip install pycocotools
  oraz pakiet pycocoevalcap (z repozytorium pycocoevalcap)

"""
import argparse
import json
import math
import sys
from typing import Any

# próbujemy zaimportować COCO i COCOEvalCap tylko wtedy, gdy będą potrzebne
try:
    from pycocotools.coco import COCO
except Exception:
    COCO = None

try:
    # pycocoevalcap może nie być dostępny w środowisku
    from pycocoevalcap.eval import COCOEvalCap
except Exception:
    COCOEvalCap = None


def is_eval_json(obj: Any) -> bool:
    """Rozpoznaje gotowy plik ewaluacji zawierający 'overall' i 'imgToEval'."""
    return isinstance(obj, dict) and 'overall' in obj and 'imgToEval' in obj


def fmt_number(v):
    if v is None:
        return "None"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return str(v)
        return f"{v:.3f}"
    return str(v)


def print_overall(overall: dict):
    print("Overall metrics:")
    # zachowujemy kolejność kluczy jeśli to OrderedDict, w p.p. sortujemy alfabetycznie
    for k in overall:
        v = overall[k]
        # jeśli v to dict -> pretty print (rzadkie w sekcji overall, ale zabezpieczamy)
        if isinstance(v, dict):
            print(f"  {k}:")
            for subk, subv in v.items():
                print(f"    {subk}: {fmt_number(subv)}")
        else:
            print(f"  {k}: {fmt_number(v)}")
    print()


def pretty_print_spice(spice_obj: dict, indent="    "):
    # SPICE dla pojedynczego obrazka jest zagnieżdżonym słownikiem.
    for facet, stats in spice_obj.items():
        print(f"{indent}{facet}:")
        if isinstance(stats, dict):
            for sk, sv in stats.items():
                print(f"{indent}  {sk}: {fmt_number(sv)}")
        else:
            print(f"{indent}  {fmt_number(stats)}")


def print_imgtoeval(imgToEval: dict, max_images: int = 10):
    print(f"Per-image evaluation (pokazano maksymalnie {max_images} obrazków):")
    count = 0
    # imgToEval może mieć klucze jako string id obrazka
    for img_id, metrics in imgToEval.items():
        if count >= max_images:
            break
        count += 1
        print(f"- image_id: {img_id}")
        # wypisz podstawowe metryki w sensownym porządku
        # Sortujemy klucze by readability, ale zachowujemy ich zawartość
        for k in sorted(metrics.keys()):
            v = metrics[k]
            if k == "SPICE" and isinstance(v, dict):
                print("  SPICE:")
                pretty_print_spice(v, indent="    ")
            elif k == "caption":
                print(f"  caption: {v}")
            elif k == "image_id":
                # już wypisane
                continue
            else:
                # liczby
                print(f"  {k}: {fmt_number(v)}")
        print()
    if count == 0:
        print("  brak wpisów w imgToEval\n")


def run_coco_eval(annotation_file: str, results_file: str, image_ids_all: bool = True):
    if COCO is None:
        print("Błąd: biblioteka pycocotools nie jest zainstalowana lub nie można jej zaimportować.", file=sys.stderr)
        sys.exit(2)
    if COCOEvalCap is None:
        print("Błąd: pycocoevalcap (COCOEvalCap) nie jest zainstalowany lub nie można go zaimportować.", file=sys.stderr)
        sys.exit(2)

    print(f"Wczytuję adnotacje z: {annotation_file}")
    coco = COCO(annotation_file)
    print(f"Wczytuję wyniki (COCO results) z: {results_file}")
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)

    if image_ids_all:
        try:
            coco_eval.params['image_id'] = coco_result.getImgIds()
        except Exception:
            # zabezpieczenie: getImgIds może nie istnieć — wtedy spróbujemy pobrać z coco_result directly
            try:
                coco_eval.params['image_id'] = [r['image_id'] for r in coco_result]
            except Exception:
                pass

    print("Trwa ewaluacja... (może chwilę potrwać, SPICE może cache'ować wyniki)...")
    coco_eval.evaluate()

    # coco_eval.eval jest zwykle dict z metrykami
    if hasattr(coco_eval, "eval") and isinstance(coco_eval.eval, dict):
        print_overall(coco_eval.eval)
    else:
        print("Brak atrybutu coco_eval.eval lub nie jest to dict", file=sys.stderr)

    # zapisz wynik ewaluacji do pliku helper (opcjonalnie)
    # mozna dopisać zapis do pliku tutaj jeśli potrzeba


def preview_results_json(results_file: str, max_preview: int = 10):
    """Pokaż podgląd pliku results.json (lista predykcji), bez ewaluacji."""
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"Plik {results_file} zawiera listę wyników (liczba wpisów: {len(data)}). Pokazuję pierwsze {max_preview}:")
        for i, item in enumerate(data[:max_preview]):
            print(f"#{i+1}: image_id={item.get('image_id')}  caption={item.get('caption')}")
    else:
        print(f"Plik {results_file} nie jest listą. Typ obiektu: {type(data)}")
        # ewentualnie jeśli jest to dict z evaluate -> pokaż jeśli można
        if is_eval_json(data):
            print("To wygląda jak gotowy plik ewaluacji.")
            print_overall(data.get("overall", {}))
            print_imgtoeval(data.get("imgToEval", {}), max_preview)


def main():
    p = argparse.ArgumentParser(description="Ewaluacja/wyświetlenie wyników COCO/COCOEvalCap")
    p.add_argument("--annotations", "-a", default="annotations/captions_val2014.json",help="Plik z adnotacjami COCO (np. annotations/captions_val2014.json)")
    p.add_argument("--results", "-r", default="results.json", help="Plik z wynikami modelu w formacie COCO (lista dict z 'image_id' i 'caption')")
    p.add_argument("--eval", "-e", help="Plik z gotową ewaluacją (zawiera 'overall' i 'imgToEval')")
    p.add_argument("--preview", action="store_true", help="Jeśli podano --results i brak adnotacji, pokaż tylko podgląd results.json")
    p.add_argument("--per-image", type=int, default=5, help="Ile przykładów per-image pokazać dla eval JSON (domyślnie 5)")
    args = p.parse_args()

    if args.eval:
        # wczytujemy gotową ewaluację i wypisujemy
        with open(args.eval, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not is_eval_json(data):
            print(f"Plik {args.eval} nie wygląda na wynik ewaluacji (brakuje kluczy 'overall' i 'imgToEval').", file=sys.stderr)
            sys.exit(1)
        print_overall(data.get("overall", {}))
        print_imgtoeval(data.get("imgToEval", {}), max_images=args.per_image)
        return

    if args.results and args.annotations:
        # wykonaj pełną ewaluację
        run_coco_eval(args.annotations, args.results)
        return

    if args.results and args.preview:
        preview_results_json(args.results)
        return

    # brak odpowiednich argumentów
    p.print_help()
    print("\nPrzykład: python evaluate_coco_results.py --annotations annotations/captions_val2014.json --results results.json", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
