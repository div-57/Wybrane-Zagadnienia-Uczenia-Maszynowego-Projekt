import pandas as pd
from nba_api.stats import endpoints


def prepare_test_data_all_nba():
    """
    Funkcja przygotowująca dane testowe dla zawodników nominowanych do drużyny All-NBA
    w sezonie 2023-24 na podstawie danych API NBA.

    :return: Krotka zawierająca dane testowe X oraz nazwy zawodników.
    """
    # Pobranie danych z API NBA dla drużyny All-NBA w sezonie 2023-24
    actual_data_all_nba = endpoints.leaguedashplayerstats.LeagueDashPlayerStats(
        season='2023-24', season_type_all_star='Regular Season'
    )
    # Przetworzenie danych do postaci DataFrame
    df_all_nba = actual_data_all_nba.league_dash_player_stats.get_data_frame()
    # Filtrowanie zawodników, którzy grali minimum 65 gier oraz średnio minimum 20 minut na grę
    df_all_nba = df_all_nba[df_all_nba['GP'] >= 65]
    df_all_nba = df_all_nba[df_all_nba['MIN'] / df_all_nba['GP'] >= 20]
    # Wybór odpowiednich kolumn dla danych testowych
    X_test_all_nba = df_all_nba[[
        'AGE', 'GP', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
        'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3'
    ]]
    # Wydobycie danych osobowych zawodników
    player_names_all_nba = df_all_nba['PLAYER_NAME'].reset_index(drop=True)
    return X_test_all_nba, player_names_all_nba


def prepare_test_data_all_rookie():
    """
    Funkcja przygotowująca dane testowe dla zawodników nominowanych do drużyny All-Rookie
    w sezonie 2023-24 na podstawie danych API NBA.

    :return: Krotka zawierająca dane testowe X oraz nazwy zawodników.
    """
    # Pobranie danych z API NBA dla drużyny All-Rookie w sezonie 2023-24
    actual_data_all_rookie = endpoints.leaguedashplayerstats.LeagueDashPlayerStats(
        season='2023-24', season_type_all_star='Regular Season', player_experience_nullable='Rookie'
    )
    # Przetworzenie danych do postaci DataFrame
    df_all_rookie = actual_data_all_rookie.league_dash_player_stats.get_data_frame()
    # Wybór odpowiednich kolumn dla danych testowych
    X_test_all_rookie = df_all_rookie[[
        'AGE', 'GP', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
        'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3'
    ]]
    # Wydobycie danych osobowych zawodników
    player_names_all_rookie = df_all_rookie['PLAYER_NAME'].reset_index(drop=True)
    return X_test_all_rookie, player_names_all_rookie
