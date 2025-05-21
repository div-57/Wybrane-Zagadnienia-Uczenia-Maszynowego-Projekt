# Wybrane-Zagadnienia-Uczenia-Maszynowego-Projekt

## Opis projektu
Celem projektu jest predykcja, którzy zawodnicy otrzymają nominacje do drużyn All-NBA oraz All-Rookie w sezonie 2023/24. Wykorzystano historyczne dane z sezonów 2018/19–2022/23 oraz klasyfikator ```RandomForestClassifier```, aby na podstawie statystyk meczowych wygenerować ranking i przypisać pięciu najlepszym zawodnikom miejsce w odpowiedniej drużynie.

## Wymagania
- Python 3.8 lub nowszy,
- Biblioteki wymienione w pliku ```requirements.txt```.

Instalacja zależności:
```bash
pip install -r requirements.txt
```

## Struktura repozytorium
```graphql
.
├── Wozniak_Dawid.json       # Wynikowe predykcje (All-NBA i All-Rookie Teams)
├── requirements.txt         # Lista zależności
├── main.py                  # Skrypt główny
├── processing/              # Moduły pomocnicze
│   ├── allnba.csv           # Dane historyczne All-NBA & All-Rookie
│   ├── predict.py           # Logika predykcji
│   ├── test_data.py         # Pobieranie i przygotowanie danych NBA API
│   └── train.py             # Trenowanie i serializacja modeli
└── models/                  # Wczytywane pretrenowane modele
    ├── model_allnba_2018-19.pkl
    ├── …                   
    └── model_allrookie_2022-23.pkl
```

## Uruchomienie
Skrypt main.py wykonuje wszystkie etapy:
1. Pobiera dane meczowe z API NBA.
2. Łączy je z historycznymi nominacjami z pliku ```allnba.csv```.
3. Trenuje nowe modele lub ładuje pretrenowane.
4. Predykuje nominacje i zapisuje wynik w pliku JSON.

Przykład wywołania:
```bash
python main.py <ścieżka_do_pliku_wyników.json>
```

## Opis najważniejszych plików
1. ```processing/test_data.py``` - pobiera dane z ```nba_api``` dla zawodników spełniających kryteria:
   - min. 65 rozegranych meczów,
   - średni czas gry ≥ 20 minut.

Zwraca Pandas DataFrame z kluczowymi statystykami.

2. ```processing/train.py``` - wczytuje dane historyczne nominacji z ```allnba.csv```, koduje klasy (```0``` – brak nominacji, ```1```…```5``` – odpowiednio 1st, 2nd, 3rd All-NBA, 1st i 2nd All-Rookie), trenuje ```RandomForestClassifier``` dla każdego sezonu i zapisuje modele w ```models/```.

3. ```processing/predict.py``` - ładuje odpowiedni model (```models/model_allnba_<sezon>.pkl``` lub ```model_allrookie_<sezon>.pkl```), dla każdego zawodnika oblicza prawdopodobieństwo przynależności i wybiera top 5 zawodników na każdą drużynę, dbając, by jeden gracz nie wystąpił w więcej niż jednym zestawieniu.

## Przykładowe wyniki
Plik ```Wozniak_Dawid.json``` przechowuje wyniki predykcji, np.:
```bash
{
    "first all-nba team": [
        "Luka Doncic",
        "Nikola Jokic",
        "Giannis Antetokounmpo",
        "Shai Gilgeous-Alexander",
        "Jayson Tatum"
    ],
    "second all-nba team": [
        ...
    ],
    "third all-nba team": [
        ...
    ],
    "first rookie all-nba team": [
        ...
    ],
    "second rookie all-nba team": [
        ...
    ]
}
```

*Autor: Woźniak Dawid*