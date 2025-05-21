import json
import argparse
from pathlib import Path
from processing.train import train_nba_models
from processing.test_data import prepare_test_data_all_nba, prepare_test_data_all_rookie
from processing.predict import make_prediction


def main():
    parser = argparse.ArgumentParser(description="Predict All-NBA and All-Rookie Teams.")
    parser.add_argument('results_file', type=str, help="Path to the output results file.")
    args = parser.parse_args()

    results_file = Path(args.results_file)

    # Trenowanie modeli (należy odkomentować linię poniżej)
    # train_nba_models()

    # Przygotowanie danych testowych
    X_test_allnba, player_names_allnba = prepare_test_data_all_nba()
    X_test_allrookie, player_names_allrookie = prepare_test_data_all_rookie()

    # Predykcja
    predictions = make_prediction(X_test_allnba, player_names_allnba, X_test_allrookie, player_names_allrookie)

    # Zapis predykcji do pliku
    with results_file.open('w') as output_file:
        json.dump(predictions, output_file, indent=4)


if __name__ == '__main__':
    main()
