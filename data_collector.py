import yfinance as yf
import numpy as np
from config import assets
import pandas as pd

# Téléchargement des prix via Yahoo Finance
def download_data(start_date: str, end_date: str) -> pd.DataFrame:
    all_tickers = [ticker for sublist in assets.values() for ticker in sublist]
    
    data = (
        yf.download(all_tickers, start=start_date, end=end_date)['Close']
        .interpolate(method='linear', limit_direction='forward')  # Remplissage des valeurs manquantes
        .replace([np.inf, -np.inf], np.nan)  # Suppression des valeurs infinies
        .dropna()  # Suppression des lignes restantes avec NaN
    )
    return data

# Calcul des rendements log
def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(data / data.shift(1)).dropna()
    return returns

# Sauvegarde des données en CSV
def save_data_to_csv(data: pd.DataFrame, returns: pd.DataFrame) -> None:
    data.to_csv("data/market_prices.csv")
    returns.to_csv("data/market_returns.csv")
    print("Données enregistrées localement !")

