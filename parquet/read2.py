import pyarrow.parquet as pq
import pandas as pd
import pyarrow.compute as pc


def filtruj_pyarrow(sciezka_pliku):
    try:
        szukany_caption = "a man riding skis on top of a body of water"

        # 1. Czytanie z filtrowaniem na poziomie PyArrow
        # To jest kluczowa zaleta pyarrow - czytamy tylko to, co potrzebne.
        tabela = pq.read_table(
            sciezka_pliku,
            filters=[('caption', '==', szukany_caption), ('image_id', '==', 227220)]
        )

        # Konwersja do Pandas dla łatwiejszej manipulacji kolumnami
        df = tabela.to_pandas()

        if df.empty:
            print(f"Nie znaleziono wierszy dla: '{szukany_caption}'")
            return

        # 2. Wybór kolumn (plik_zrodlowy + metryki)
        kolumny_do_pokazania = []

        # Sprawdzenie czy istnieje kolumna 'plik_zrodlowy'
        if 'plik_zrodlowy' in df.columns:
            kolumny_do_pokazania.append('plik_zrodlowy')

        # Wykrywanie metryk (wszystkie kolumny liczbowe)
        metryki = df.select_dtypes(include=['number']).columns.tolist()

        # Dodajemy metryki do listy (unikając duplikatów)
        for m in metryki:
            if m not in kolumny_do_pokazania:
                kolumny_do_pokazania.append(m)

        # 3. Wyświetlenie wyników
        wynik_koncowy = df[kolumny_do_pokazania]

        print(f"--- Znaleziono {len(wynik_koncowy)} wiersz(y) ---")
        print(wynik_koncowy)

        return wynik_koncowy

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


# Uruchomienie
if __name__ == "__main__":
    filtruj_pyarrow('wyniki_z_cnn.parquet')
