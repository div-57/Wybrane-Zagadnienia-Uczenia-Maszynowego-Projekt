import pickle
import numpy as np
import pandas as pd


def make_prediction(X_test_allnba, player_names_allnba, X_test_allrookie, player_names_allrookie):
    """
    Funkcja dokonująca predykcji dla zawodników mogących być nominowanym do drużyn All-NBA i All-Rookie,
    na podstawie danych testowych i wytrenowanych modeli.

    :param X_test_allnba: Dane testowe dla nominowanych do drużyny All-NBA.
    :param player_names_allnba: Dane zawodników mogących być nominowanym do drużyny All-NBA.
    :param X_test_allrookie: Dane testowe dla nominowanych do drużyny All-Rookie.
    :param player_names_allrookie: Dane zawodników mogących być nominowanym do drużyny All-Rookie.
    :return: Słownik zawierający przewidziane drużyny.
    """
    seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    weights = [0.1, 0.1, 0.2, 0.3, 0.3]

    # Inicjalizacja tablic do przechowywania przewidywanych prawdopodobieństw
    pred_prob_allnba = np.zeros((X_test_allnba.shape[0], 4))
    pred_prob_allrookie = np.zeros((X_test_allrookie.shape[0], 3))

    for season, weight in zip(seasons, weights):
        # Wczytanie wytrenowanych modeli dla All-NBA
        with open(f'models/model_allnba_{season}.pkl', 'rb') as file:
            model_allnba = pickle.load(file)
        # Predykcja prawdopodobieństw dla All-NBA
        season_pred_allnba = model_allnba.predict_proba(X_test_allnba)
        if season_pred_allnba.shape[1] > 4:
            season_pred_allnba = season_pred_allnba[:, :4]
        pred_prob_allnba += weight * season_pred_allnba

        # Wczytanie wytrenowanych modeli dla All-Rookie
        with open(f'models/model_allrookie_{season}.pkl', 'rb') as file:
            model_allrookie = pickle.load(file)
        # Predykcja prawdopodobieństw dla All-Rookie
        season_pred_allrookie = model_allrookie.predict_proba(X_test_allrookie)
        if season_pred_allrookie.shape[1] > 3:
            season_pred_allrookie = season_pred_allrookie[:, :3]
        pred_prob_allrookie += weight * season_pred_allrookie

    # Utworzenie ramki danych dla przewidywań All-NBA
    prob_df_allnba = pd.DataFrame(pred_prob_allnba, columns=[0, 1, 2, 3])
    prob_df_allnba['PLAYER_NAME'] = player_names_allnba

    # Utworzenie ramki danych dla przewidywań All-Rookie
    prob_df_allrookie = pd.DataFrame(pred_prob_allrookie, columns=[0, 1, 2])
    prob_df_allrookie['PLAYER_NAME'] = player_names_allrookie

    # Inicjalizacja tablic do przechowywania przewidywanych drużyn
    allnba_teams = []
    all_teams = []
    team_count = 5

    # Predykcja drużyn All-NBA na podstawie prawdopodobieństw
    for team_num in range(1, 4):
        team = []
        for _ in range(team_count):
            candidate = prob_df_allnba.loc[
                ~prob_df_allnba['PLAYER_NAME'].isin(all_teams),
                [team_num, 'PLAYER_NAME']
            ].sort_values(by=team_num, ascending=False).iloc[0]
            team.append(candidate['PLAYER_NAME'])
            all_teams.append(candidate['PLAYER_NAME'])
        allnba_teams.append(team)

    # Predykcja drużyn All-Rookie na podstawie prawdopodobieństw
    all_rookie_teams = []
    all_teams = []
    for team_num in range(1, 3):
        team = []
        for _ in range(team_count):
            candidate = prob_df_allrookie.loc[
                ~prob_df_allrookie['PLAYER_NAME'].isin(all_teams),
                [team_num, 'PLAYER_NAME']
            ].sort_values(by=team_num, ascending=False).iloc[0]
            team.append(candidate['PLAYER_NAME'])
            all_teams.append(candidate['PLAYER_NAME'])
        all_rookie_teams.append(team)

    # Zwrócenie słownika zawierającego przewidziane drużyny
    return {
        "first all-nba team": allnba_teams[0],
        "second all-nba team": allnba_teams[1],
        "third all-nba team": allnba_teams[2],
        "first rookie all-nba team": all_rookie_teams[0],
        "second rookie all-nba team": all_rookie_teams[1]
    }
