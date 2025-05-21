import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from nba_api.stats import endpoints


def train_nba_models():
    """
    Funkcja trenująca modele do typowania zawodników nominowanych do All-NBA Team i All-Rookie Team,
    na podstawie wczytanych danych z pliku zawierającego informacje o nominowanych w poprzednich sezonach.

    :return: Brak zwracanej wartości.
    """
    # Wczytywanie danych z pliku informujących o nominacjach w poprzednich sezonach.
    all_nba_data = pd.read_csv('processing/allnba.csv', delimiter=';', header=None)
    # Mapowanie graczy do drużyn w oparciu o sezon i dane personalne.
    player_to_team_mapping = {}
    for index, row in all_nba_data.iterrows():
        season = row[0]
        for team in range(1, 6):
            players = row[(team-1)*5+1:team*5+1]
            for player in players:
                player_to_team_mapping[(season, player)] = team

    # Lista sezonów, dla których modele są trenowane.
    seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']

    for season in seasons:
        # Pobieranie statystyk wszystkich zawoników NBA dla danego sezonu.
        all_nba_stats = endpoints.leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, season_type_all_star='Regular Season')
        df_all_nba = all_nba_stats.league_dash_player_stats.get_data_frame()

        # Przygotowanie danych do trenowania modelu All-NBA.
        df_all_nba['All_NBA_Team'] = df_all_nba.apply(
            lambda row: player_to_team_mapping.get((season, row['PLAYER_NAME']), 0), axis=1)
        df_all_nba = df_all_nba[[
            'AGE', 'GP', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
            'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
            'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3', 'All_NBA_Team']]

        X_train = df_all_nba.drop(columns=['All_NBA_Team'])
        y_train = df_all_nba['All_NBA_Team']

        # Trenowanie modelu All-NBA.
        model_all_nba = RandomForestClassifier(random_state=42)
        model_all_nba.fit(X_train, y_train)

        # Zapis wytrenowanego modelu do pliku.
        model_file_all_nba = f'models/model_allnba_{season}.pkl'
        with open(model_file_all_nba, 'wb') as file:
            pickle.dump(model_all_nba, file)

        # Pobieranie statystyk debiutantów w NBA dla danego sezonu.
        all_rookie_stats = endpoints.leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, season_type_all_star='Regular Season', player_experience_nullable='Rookie')
        df_all_rookie = all_rookie_stats.league_dash_player_stats.get_data_frame()

        # Przygotowanie danych do trenowania modelu All-Rookie.
        df_all_rookie['All_NBA_Team'] = df_all_rookie.apply(
            lambda row: player_to_team_mapping.get((season, row['PLAYER_NAME']), 0), axis=1)
        df_all_rookie = df_all_rookie[[
            'AGE', 'GP', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
            'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
            'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3', 'All_NBA_Team']]

        X_train = df_all_rookie.drop(columns=['All_NBA_Team'])
        y_train = df_all_rookie['All_NBA_Team']

        # Trenowanie modelu All-Rookie.
        model_all_rookie = RandomForestClassifier(random_state=42)
        model_all_rookie.fit(X_train, y_train)

        # Zapis wytrenowanego model do pliku.
        model_file_all_rookie = f'models/model_allrookie_{season}.pkl'
        with open(model_file_all_rookie, 'wb') as file:
            pickle.dump(model_all_rookie, file)
