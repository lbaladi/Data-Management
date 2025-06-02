import pandas as pd
import numpy as np
import yfinance as yf
from base_builder import Client, Portfolio
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score




def calculate_RSI(prices: pd.Series, window: int=14):
    """
    Calcule le RSI (Relative Strength Index) pour une série de prix.
    :param prices: Série de prix
    :param window: Fenêtre pour le calcul (par défaut 14 jours)
    :return: RSI calculé
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI.iloc[-1]  # Retourner la valeur RSI du dernier jour


def compute_metrics_LR(data_df: pd.DataFrame, returns_df: pd.DataFrame, LR_assets: list):
    """
    Calcule les indicateurs pour tous les actifs de LR_assets sur chaque lundi du dataset
    et ajoute une colonne "Target" qui correspond à la volatilité de la semaine suivante.
    
    :param data_df: DataFrame contenant les prix des actifs.
    :param returns_df: DataFrame contenant les rendements des actifs.
    :param LR_assets: Liste des tickers des actifs du portefeuille low risk.
    :return: DataFrame avec les métriques et la cible.
    """

    # Identifier tous les lundis
    mondays = data_df.index[data_df.index.weekday == 0]

    results = []

    for asset in LR_assets:
        for i, date in enumerate(mondays):
            if date not in returns_df.index:
                continue  # Si la date n'est pas présente dans les rendements, on saute
            
            # Extraire les prix et rendements jusqu'à la date actuelle
            prices = data_df[asset].loc[:date]
            returns = returns_df[asset].loc[:date]

            if len(returns) < 20:  # Vérifier qu'on a assez de données pour les calculs
                continue

            # Calcul des volatilités
            volat_5j = returns[-5:].std() * np.sqrt(252 / 5)
            volat_10j = returns[-10:].std() * np.sqrt(252 / 10)
            volat_1m = returns[-20:].std() * np.sqrt(252 / 20)

            # Calcul des rendements
            return_5j = (prices.iloc[-1] / prices.iloc[-6]) - 1 if len(prices) >= 6 else np.nan
            return_10j = (prices.iloc[-1] / prices.iloc[-11]) - 1 if len(prices) >= 11 else np.nan

            # SMA 10 jours
            SMA_10 = prices[-10:].mean()

            RSI_14 = calculate_RSI(prices)

            # Volatilité annualisée sur la semaine
            volat_weekly = returns[-5:].std() * np.sqrt(52)

            # Ajouter les indicateurs supplémentaires
            daily_ret = prices.pct_change().iloc[-1] if len(prices) > 1 else np.nan
            daily_ret = daily_ret if not np.isnan(daily_ret) else prices.pct_change().ffill().iloc[-1]  # ffill si NaN
            SMA_50 = prices[-50:].mean() if len(prices) >= 50 else np.nan
            SMA_200 = prices[-200:].mean() if len(prices) >= 200 else np.nan
            close_price = prices.iloc[-1]  # Dernier prix (close) dans la série de données

            # Stocker les résultats
            results.append([date, asset, volat_5j, volat_10j, volat_1m, return_5j, return_10j, SMA_10, RSI_14, volat_weekly, daily_ret, SMA_50, SMA_200, close_price])

    # Transformer en DataFrame
    columns = ["Date", "Ticker", "Volat_5j", "Volat_10j", "Volat_1m", "Return_5j", "Return_10j", "SMA_10", "RSI_14", "VOLAT_weekly", "Daily_Return", "SMA_50", "SMA_200", "Close"]
    df_metrics = pd.DataFrame(results, columns=columns)

    # Trier les résultats chronologiquement
    df_metrics.sort_values(by=["Date", "Ticker"], inplace=True)

    # Ajouter la colonne volat_next_week avec shift pour décaler la volatilité d'une semaine
    df_metrics["volat_next_week"] = df_metrics.groupby("Ticker")["VOLAT_weekly"].shift(-1)

    # Ajouter la colonne Target avec conditions sur la volatilité et le rendement
    df_metrics["Target"] = 0  # Initialisation de la cible à 0 (ne rien faire)
    # Condition d'achat : volatilité faible et rendement positif
    df_metrics.loc[(df_metrics["volat_next_week"] <= 0.15) & (df_metrics["Return_5j"] > 0), "Target"] = 1
    # Condition de vente : volatilité élevée et rendement négatif
    df_metrics.loc[(df_metrics["volat_next_week"] > 0.15) & (df_metrics["Return_5j"] < 0), "Target"] = -1

    # Convertir la colonne Date en index et nettoyer les NaN
    df_metrics['Date'] = pd.to_datetime(df_metrics['Date'])
    df_metrics.set_index('Date', inplace=True)
    
    df_metrics = df_metrics.dropna(subset=['volat_next_week'])

    return df_metrics


def load_market_data(start_date: str, end_date: str, df_metrics: pd.DataFrame):
    """Télécharge les données macroéconomiques et fusionne avec df_metrics."""
    
    try:
        # === Données Macro ===
        vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
        bond_yield = yf.download('^TNX', start=start_date, end=end_date)['Close']

        if vix.empty or bond_yield.empty:
            raise ValueError("Les données VIX ou Bond Yield n'ont pas été téléchargées correctement.")

        # === Données CPI via l'API FRED ===
        API_KEY = "c98e768f0aaece65a16d054b36285f2c" 
        fred = Fred(api_key=API_KEY)

        cpi_data = fred.get_series('CPIAUCSL', start_date, end_date)

        if cpi_data is None:
            raise ValueError("Les données CPI n'ont pas été récupérées correctement depuis FRED.")

        cpi_data = pd.DataFrame(cpi_data, columns=['CPI'])

        # Convertir en variations mensuelles, puis interpoler
        cpi_data['CPI_Pct_Change'] = cpi_data['CPI'].diff() / cpi_data['CPI'].shift(1)
        cpi_data = cpi_data.reindex(vix.index, method='ffill')  # Assurer un index aligné

        # Création du DataFrame macroéconomique
        macro_data = pd.DataFrame(index=vix.index)
        macro_data['VIX'] = vix
        macro_data['Bond_Yield'] = bond_yield
        macro_data['CPI'] = cpi_data['CPI_Pct_Change'] 

        # Normalisation des variations journalières (sauf CPI déjà en %)
        macro_data[['VIX', 'Bond_Yield']] = macro_data[['VIX', 'Bond_Yield']].pct_change(fill_method=None)

        # Assurez-vous que les dates dans df_metrics et macro_data sont compatibles
        df_metrics = df_metrics.reset_index()
        df_metrics['Date'] = pd.to_datetime(df_metrics['Date'])  # Assurez-vous que les dates sont au format datetime

        # Fusionner avec df_metrics en utilisant uniquement 'Date' comme clé
        df_metrics = df_metrics.merge(macro_data, on='Date', how='left')

        # Re-établir l'index multi-niveau (Date, Ticker)
        df_metrics.set_index(['Date', 'Ticker'], inplace=True)

        # Supprimer les doublons (au cas où)
        df_metrics = df_metrics.drop_duplicates()

        return df_metrics.dropna()

    except Exception as e:
        print(f"Erreur lors du téléchargement des données: {e}")
        return None


def compute_metrics_LT(data_df: pd.DataFrame, returns_df: pd.DataFrame, LT_assets: list, max_transactions_per_month: int=2, rebalance_frequency: str='M'):
    """
    Calcule les indicateurs et la cible pour une stratégie Low Turnover (2 transactions max par mois).
    
    :param data_df: DataFrame avec les prix des actifs.
    :param returns_df: DataFrame avec les rendements des actifs.
    :param LT_assets: Liste des tickers des actifs du portefeuille low turnover.
    :param max_transactions_per_month: Nombre maximal de transactions autorisées par mois (par défaut 2).
    :param rebalance_frequency: Fréquence de rééquilibrage (par défaut mensuelle).
    :return: DataFrame avec les métriques et la colonne "Target" pour la stratégie Low Turnover.
    """

    # Filtrer pour ne garder que les lundis
    mondays = data_df.index[data_df.index.weekday == 0]
    rebalance_dates = mondays.to_series().groupby(mondays.to_period('M')).head(2)  # Sélectionner deux lundis ouvrés par mois

    results = []
    transaction_count = 0  # Nombre de transactions effectuées dans le mois

    for asset in LT_assets:
        for date in rebalance_dates:
            transaction_count = 0  # Réinitialiser à chaque nouvelle date de rééquilibrage
            if date not in returns_df.index:
                print(f"Date ignorée (non trouvée dans returns_df): {date}")  # Debugging
                continue  # Si la date n'est pas présente dans les rendements, on saute
            
            # Extraire les prix et rendements jusqu'à la date actuelle
            prices = data_df[asset].loc[:date]
            returns = returns_df[asset].loc[:date]

            if len(returns) < 20:  # Vérifier qu'on a assez de données pour les calculs
                continue

            # Calcul des indicateurs pour l'actif
            volat_5j = returns[-5:].std() * np.sqrt(252 / 5)
            volat_10j = returns[-10:].std() * np.sqrt(252 / 10)
            volat_1m = returns[-20:].std() * np.sqrt(252 / 20)

            return_5j = (prices.iloc[-1] / prices.iloc[-6]) - 1 if len(prices) >= 6 else np.nan
            return_10j = (prices.iloc[-1] / prices.iloc[-11]) - 1 if len(prices) >= 11 else np.nan
            
            SMA_10 = prices[-10:].mean()
            RSI_14 = calculate_RSI(prices)
            daily_ret = prices.pct_change().iloc[-1] if len(prices) > 1 else np.nan
            daily_ret = daily_ret if not np.isnan(daily_ret) else prices.pct_change().ffill().iloc[-1]  # ffill si NaN

            close_price = prices.iloc[-1]  # Dernier prix (close) dans la série de données

            results.append([date, asset, volat_5j, volat_10j, volat_1m, return_5j, return_10j, SMA_10, RSI_14, daily_ret, close_price])

    # Transformer en DataFrame
    columns = ["Date", "Ticker", "Volat_5j", "Volat_10j", "Volat_1m", "Return_5j", "Return_10j", "SMA_10", "RSI_14", "Daily_Return", "Close"]
    df_metrics = pd.DataFrame(results, columns=columns)

    # Trier les résultats chronologiquement
    df_metrics.sort_values(by=["Date", "Ticker"], inplace=True)
    
    df_metrics['Sharpe'] = df_metrics['Daily_Return'] / df_metrics['Daily_Return'].std()


    # Maintenant, ajouter les données macroéconomiques
    start_date = df_metrics['Date'].min()
    end_date = df_metrics['Date'].max()
    
    # Charger les données macro et fusionner avec df_metrics
    df_metrics = load_market_data(start_date, end_date, df_metrics)

    # Calculer la cible (target) après avoir ajouté les données macro
    df_metrics['Target'] = 0  # Par défaut, ne pas agir
    
    for idx, row in df_metrics.iterrows():
        volat_5j = row['Volat_5j']
        volat_10j = row['Volat_10j']
        return_10j = row['Return_10j']
        RSI_14 = row['RSI_14']
        
        if volat_5j < volat_10j and return_10j > 0 or RSI_14 < 30:
            df_metrics.at[idx, 'Target'] = 1  # Acheter
        elif volat_5j > volat_10j and return_10j < 0 or RSI_14 > 80:
            df_metrics.at[idx, 'Target'] = -1  # Vendre
    
    df_metrics = df_metrics.reset_index()  # Transformer l'index en colonnes
    df_metrics = df_metrics.set_index("Date")  # Garder uniquement la date en index
    df_metrics.index = pd.to_datetime(df_metrics.index)

    return df_metrics


def get_volume_data(tickers: list, start_date: str, end_date: str):
    """
    Récupère les données de volume de trading pour une liste de tickers sur une période spécifiée.
    Les données sont téléchargées via Yahoo Finance et renvoyées sous forme de DataFrame avec les volumes renommés par ticker.

    Args:
        tickers (list): Liste des tickers des actifs à analyser
        start_date (str): Date de début de la période (format: 'YYYY-MM-DD')
        end_date (str): Date de fin de la période (format: 'YYYY-MM-DD')

    Returns:
        pandas.DataFrame: DataFrame contenant les volumes de chaque actif
    """

    volume_data = []

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if 'Volume' in data.columns:
            volume_data.append(data[['Volume']].rename(columns={'Volume': ticker}))  # Renomme avec le ticker
        else:
            print(f"Aucune donnée de volume pour {ticker}")

    if volume_data:
        volume_df = pd.concat(volume_data, axis=1)
        volume_df.columns.name = None  # Supprime le nom des colonnes pour éviter la ligne "Ticker"
    else:
        volume_df = pd.DataFrame()
    
    
    # Vérifie si c'est un MultiIndex et le simplifie
    if isinstance(volume_df.columns, pd.MultiIndex):
        volume_df.columns = volume_df.columns.get_level_values(1)  # Prend le bon niveau des colonnes
    

    return volume_df


def compute_metrics_HY(data_df: pd.DataFrame, returns_df: pd.DataFrame, HY_assets: list, volume_df: pd.DataFrame):
    """
    Calcule une série de métriques financières pour des actifs à haut rendement (High Yield) sur la base de leurs prix et volumes.
    Inclut des mesures telles que la volatilité, le rendement, le RSI, les moyennes mobiles, et le ratio de Sharpe.
    Retourne un DataFrame avec des informations sur les performances et des signaux d'achat/vente basés sur le rendement hebdomadaire.

    Args:
        data_df (pandas.DataFrame): DataFrame des prix historiques des actifs
        returns_df (pandas.DataFrame): DataFrame des rendements historiques des actifs
        HY_assets (list): Liste des actifs à haut rendement à analyser
        volume_df (pandas.DataFrame): DataFrame des volumes de trading

    Returns:
        pandas.DataFrame: DataFrame contenant les métriques calculées et les signaux de trading
    """
    
    mondays = data_df.index[data_df.index.weekday == 0]
    results = []
    
    for asset in HY_assets:
        for i, date in enumerate(mondays):
            if date not in returns_df.index:
                continue  
            
            prices = data_df[asset].loc[:date]
            returns = returns_df[asset].loc[:date]
            volume = volume_df[asset].loc[:date] if asset in volume_df else pd.Series(np.nan, index=prices.index)
            
            if len(returns) < 20:
                continue
            
            volat_5j = returns[-5:].std() * np.sqrt(252 / 5)
            volat_10j = returns[-10:].std() * np.sqrt(252 / 10)

            return_5j = (prices.iloc[-1] / prices.iloc[-6]) - 1 if len(prices) >= 6 else np.nan
            return_10j = (prices.iloc[-1] / prices.iloc[-11]) - 1 if len(prices) >= 11 else np.nan
            return_1y = (prices.iloc[-1] / prices.iloc[-252]) - 1 if len(prices) >= 252 else np.nan  # Rendement sur 1 an

            RSI_14 = calculate_RSI(prices)

            daily_ret = prices.pct_change().iloc[-1] if len(prices) > 1 else np.nan
            daily_ret = daily_ret if not np.isnan(daily_ret) else prices.pct_change().ffill().iloc[-1]  # ffill si NaN

            weekly_ret = (prices.iloc[-1] / prices.iloc[-5]) - 1 if len(prices) >= 5 else np.nan
            
            close_price = prices.iloc[-1]

            SMA_50 = prices[-50:].mean() if len(prices) >= 50 else np.nan
            SMA_200 = prices[-200:].mean() if len(prices) >= 200 else np.nan

            avg_volume_50d = volume[-50:].mean() if len(volume) >= 50 else np.nan
            
            # Calcul du drawdown max
            peak = prices.cummax()
            drawdown = (prices - peak) / peak
            max_drawdown = drawdown.min() if not drawdown.empty else np.nan
            
            # Calcul du ratio de Sharpe
            mean_return = returns.mean()
            risk_free_rate = 0.01  # Assumer un taux sans risque de 1%
            std_dev = returns.std()
            sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else np.nan
            
            results.append([date, asset, volat_5j, volat_10j, return_5j, return_10j, return_1y, 
                            RSI_14, daily_ret, weekly_ret, SMA_50, SMA_200, close_price,
                            avg_volume_50d, max_drawdown, sharpe_ratio
                            ])
    
    columns = ["Date", "Ticker", "Volat_5j", "Volat_10j", "Return_5j", "Return_10j", "Return_1y", 
            "RSI_14", "Daily_Return", "Weekly_Return","SMA_50", "SMA_200", "Close",
               "Avg_Volume_50d", "Max_Drawdown", "Sharpe_Ratio"
              ]
    
    df_metrics = pd.DataFrame(results, columns=columns)
    df_metrics.sort_values(by=["Date", "Ticker"], inplace=True)
    
    df_metrics['Date'] = pd.to_datetime(df_metrics['Date'])
    df_metrics.set_index('Date', inplace=True)
    
    threshold_buy = df_metrics["Weekly_Return"].quantile(0.75)  # Seulement le top 25% des gains
    threshold_sell = df_metrics["Weekly_Return"].quantile(0.25)  # Seulement le bottom 25% des pertes

    df_metrics['Target'] = 0  # Par défaut, neutre
    df_metrics.loc[df_metrics["Weekly_Return"] > threshold_buy, 'Target'] = 1  # Acheter
    df_metrics.loc[df_metrics["Weekly_Return"] < threshold_sell, 'Target'] = -1  # Vendre

    return df_metrics


def train_and_evaluate_model(data: pd.DataFrame, predictors: list, target_column: str, train_end_date: str='2022-12-31', test_start_date='2023-01-01'):
    """
    Entraîne et évalue un modèle de classification (RandomForest) pour prédire les cibles 
    à partir des données et des features données.

    :param data: DataFrame contenant les données avec les features et la cible
    :param predictors: Liste des noms de colonnes à utiliser comme features pour l'entraînement
    :param target_column: Nom de la colonne cible (Target) dans les données
    :param train_end_date: Date de fin pour les données d'entraînement
    :param test_start_date: Date de début pour les données de test
    :return: None, mais imprime les résultats du modèle et retourne le modèle pour des prédictions futures
    """

    # Séparation des données train et test selon les dates
    train_data = data[data.index <= train_end_date]
    test_data = data[data.index > test_start_date]

    # Initialisation du modèle
    model = RandomForestClassifier(n_estimators=200, min_samples_split=10, max_depth=3, random_state=42, class_weight='balanced')

    # Découper en 80/20 pour l'entraînement et le test dans les données train
    split_index = int(0.8 * len(train_data))
    train = train_data.iloc[:split_index]
    test = train_data.iloc[split_index:]

    # Entraîner le modèle
    model.fit(train[predictors], train[target_column])

    # Prédiction sur le test set (80/20 des données)
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)

    # Évaluation du modèle sur l'échantillon test
    print("Precision Score:", precision_score(test[target_column], preds, average='macro'))
    print("Classification Report:\n", classification_report(test[target_column], preds))
    print("Confusion Matrix:\n", confusion_matrix(test[target_column], preds))
    print("Accuracy Score:", accuracy_score(test[target_column], preds))

    # Compter le nombre de 0, 1, et -1 dans les prédictions et cibles réelles
    target_counts = test[target_column].value_counts()
    pred_counts = pd.Series(preds).value_counts()

    print(f"Vraies cibles (Target) :\n{target_counts}")
    print(f"Prédictions (Predictions) :\n{pred_counts}")

    # Visualisation des résultats sur test set
    combined = pd.concat([test[target_column], preds], axis=1)
    combined.columns = ['Actual', 'Predicted']
    combined.plot(title="Résultats sur l'échantillon test", figsize=(10, 5))

    # Retourner le modèle entraîné pour prédictions futures
    return model


def train_and_predict(df_metrics: pd.DataFrame, predictors: list, target_column: str, start_date: str='2022-12-31'):
    """
    Fonction générique pour entraîner un modèle et faire des prédictions, tout en affichant les résultats comparés à la réalité.
    
    :param df_metrics: DataFrame contenant les données à utiliser pour l'entraînement.
    :param predictors: Liste des caractéristiques utilisées pour l'entraînement.
    :param target_column: Le nom de la colonne cible à prédire (ex. 'Target').
    :param start_date: Date de début pour filtrer les données de prédiction (par défaut '2022-12-31').
    :return: DataFrame avec les prédictions.
    """
    
    # Entraînement et évaluation du modèle
    model = train_and_evaluate_model(df_metrics, predictors, target_column)

    # Prédiction sur les données après start_date
    predict_data = df_metrics[df_metrics.index > start_date]
    predict_data['Predictions'] = model.predict(predict_data[predictors])

    # Affichage des résultats
    print(predict_data[['Target', 'Predictions']])  # Comparaison des vraies valeurs avec les prévisions
    predict_data[['Target', 'Predictions']].plot(title=f"Prédictions vs Réalité ({start_date} - 2024)", figsize=(10, 5))

    return predict_data


def update_portfolio_with_balance(predict_data: pd.DataFrame, old_allocations: dict, assets_ptf: list, current_date: str, client_id: int, db_session):
    """
    Met à jour le portefeuille en fonction des prédictions, des ventes et des achats, et met à jour la balance du client dans la base de données.

    :param predict_data: DataFrame contenant les prédictions, prix et volatilités des actifs.
    :param old_allocations: Dictionnaire contenant l'allocation actuelle des actifs en valeur (€).
    :param assets_ptf: Liste des actifs du portefeuille.
    :param current_date: Date d'actualisation sous forme de string (YYYY-MM-DD).
    :param client_id: L'ID du client dans la base de données.
    :param db_session: La session de base de données pour effectuer les mises à jour sur la balance du client.
    :return: Nouveau portefeuille avec transactions de vente et d'achat, nouvelles allocations et pondérations.
    """
    # Convertir la date en format Timestamp
    current_date = pd.to_datetime(current_date)

    # Récupérer les prix des actifs pour le jour actuel
    prices_today = predict_data.loc[current_date, ['Ticker', 'Close']]

    # Sélectionner les actifs à vendre (Predictions == -1)
    predictions_today = predict_data.loc[current_date]
    
    assets_to_sell = predictions_today.loc[predictions_today['Predictions'] == -1, 'Ticker'].tolist()

    cash_generated = 0
    sell_transactions = {}

    # Vente des actifs
    for asset in assets_to_sell:
        price_asset_sell = prices_today.loc[prices_today['Ticker'] == asset, 'Close'].values[0]
        quantity_sold = 0.5 * old_allocations[asset] / price_asset_sell
        cash_generated += quantity_sold * price_asset_sell
        sell_transactions[asset] = quantity_sold
        old_allocations[asset] -= quantity_sold * price_asset_sell  # Mise à jour des allocations après vente

    # Mettre à jour la balance du client (id = client_id) dans la base de données
    client_balance = db_session.query(Client).filter(Client.id == client_id).first()
    client_balance.balance += cash_generated
    db_session.commit()  # Sauvegarde de la nouvelle balance

    # Sélectionner les actifs à acheter (Predictions == 1)
    assets_to_buy = predictions_today.loc[predictions_today['Predictions'] == 1, 'Ticker'].tolist()

    # Achat des actifs de façon équitable
    buy_transactions = {}
    balance_client = client_balance.balance
    if assets_to_buy and balance_client > 0:
        cash_per_asset = balance_client / len(assets_to_buy)
        for asset in assets_to_buy:
            price_asset_buy = prices_today.loc[prices_today['Ticker'] == asset, 'Close'].values[0]
            quantity_bought = cash_per_asset / price_asset_buy
            buy_transactions[asset] = quantity_bought
            old_allocations[asset] += quantity_bought * price_asset_buy  # Mise à jour des allocations après achat

        # Mise à jour de la balance après l'achat
        client_balance.balance -= balance_client
        db_session.commit()  # Sauvegarde de la nouvelle balance


    # Calculer la nouvelle valeur totale du portefeuille
    portfolio_value = sum(old_allocations[asset] * prices_today.loc[prices_today['Ticker'] == asset, 'Close'].values[0] for asset in assets_ptf)

    # Calculer les nouvelles pondérations
    weights = {asset: old_allocations[asset] / sum(old_allocations.values()) for asset in old_allocations}

    # Retirer les transactions de vente pour lesquelles le poids est nul
    sell_transactions = {ticker: qty for ticker, qty in sell_transactions.items() if weights.get(ticker, 0.0) != 0.0}

    return sell_transactions, buy_transactions, old_allocations, weights, portfolio_value, prices_today


