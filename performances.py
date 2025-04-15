import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from base_builder import Manager, Client, Deal, Product, Portfolio
import seaborn as sns
from sqlalchemy import text
from fpdf import FPDF

def get_portfolio_weights(db_path: str, client_id: int):
    """
    Récupère les poids des actifs du portefeuille d'un client spécifique et applique les traitements nécessaires.

    :param db_path: Chemin vers la base de données SQLite.
    :param client_id: Identifiant du client pour lequel récupérer les poids.
    :return: DataFrame structuré avec les poids des actifs pour le client donné.
    """
    with sqlite3.connect(db_path) as conn:
        query_weights = """
        SELECT p.date, pr.name AS asset_name, p.weight
        FROM portfolios p
        JOIN products pr ON p.product_id = pr.id
        WHERE p.client_id = ?
        """
        df_weights = pd.read_sql_query(query_weights, conn, params=(client_id,))
        df_weights['date'] = pd.to_datetime(df_weights['date'])
        df_weights = df_weights.pivot(index='date', columns='asset_name', values='weight').fillna(0)
        df_weights = df_weights.fillna(method='ffill')  # on utilise les memes poids jusqu'au prochain rebalancement
    return df_weights


def compute_portfolio_returns(db_path: str, client_ids: list[int], benchmark_name: str="SPY"):
    """
    Récupère la performance des portefeuilles et du benchmark, en affichant les rendements
    sous forme de colonnes séparées par client.

    :param db_path: Chemin vers la base de données SQLite.
    :param client_ids: Liste des IDs des clients.
    :param benchmark_name: Nom du benchmark (ex: "SPY").
    :return: DataFrame structuré avec les colonnes des rendements pour chaque client et le benchmark.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Récupérer l'ID du benchmark
        cursor.execute("SELECT id FROM products WHERE name = ?", (benchmark_name,))
        benchmark_id = cursor.fetchone()
        if benchmark_id is None:
            print(f"Benchmark {benchmark_name} non trouvé dans la base.")
            return None
        benchmark_id = benchmark_id[0]

        # Récupérer les rendements des actifs
        query_returns = """
        SELECT r.date, pr.name AS asset_name, r.return_value
        FROM returns r
        JOIN products pr ON r.product_id = pr.id
        WHERE r.date >= '2023-01-01'
        """
        df_returns = pd.read_sql_query(query_returns, conn)
        df_returns['date'] = pd.to_datetime(df_returns['date'])
        df_returns = df_returns.pivot(index='date', columns='asset_name', values='return_value')

        # Récupérer les rendements du benchmark
        query_benchmark = """
        SELECT date, return_value AS benchmark_return
        FROM returns
        WHERE product_id = ? AND date >= '2023-01-01'
        """
        df_benchmark = pd.read_sql_query(query_benchmark, conn, params=[benchmark_id])
        df_benchmark['date'] = pd.to_datetime(df_benchmark['date'])
        df_benchmark.set_index("date", inplace=True)

    # Initialiser le DataFrame final avec le benchmark
    df_final = df_benchmark.copy()

    # Calcul des rendements des portefeuilles
    for client_id in client_ids:
        df_weights = get_portfolio_weights(db_path, client_id)

        # Vérifier que le client possède bien un portefeuille
        if df_weights.empty:
            print(f"Aucun portefeuille trouvé pour le client {client_id}.")
            continue

        # Assurer la correspondance des dates
        df_weights = df_weights.reindex(df_returns.index, method='ffill').fillna(0)

        # Calcul des rendements du portefeuille : somme pondérée des rendements des actifs
        df_final[f"portfolio_return_client{client_id}"] = (df_weights * df_returns).sum(axis=1)

    return df_final


def plot_cumulative_returns(df: pd.DataFrame, save_path: str):
    """
    Trace le rendement cumulé des portefeuilles et du benchmark sur la période donnée.
    Le rendement cumulé est calculé comme le produit cumulé de (1 + rendement) pour chaque actif.
    """

    cumulative_returns = (1 + df).cumprod()
    portfolio_names = {
        'portfolio_return_client1': "Portfolio 'Low risk'",
        'portfolio_return_client2': "Portfolio 'Low turnover'",
        'portfolio_return_client3': "Portfolio 'High yield equity only'",
        'benchmark_return': "Benchmark (S&P 500)"
    }
    plt.figure(figsize=(12, 5))
    for col in df.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=portfolio_names.get(col, col))
    plt.title("Rendement Cumulé des Portefeuilles et du Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Rendement Cumulé")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def plot_rolling_volatility(df: pd.DataFrame, save_path: str, rolling_window: int=30,):
    """
    Trace la volatilité roulante (calculée sur une fenêtre mobile de 30 jours par défaut) pour chaque portefeuille.
    La volatilité est annualisée en multipliant l'écart type par la racine carrée de 252 (nombre de jours de trading dans une année).
    """
    volatility_rolling = df.rolling(window=rolling_window).std() * np.sqrt(252)
    portfolio_names = {
        'portfolio_return_client1': "Portfolio 'Low risk'",
        'portfolio_return_client2': "Portfolio 'Low turnover'",
        'portfolio_return_client3': "Portfolio 'High yield equity only'",
        'benchmark_return': "Benchmark (S&P 500)"
    }
    plt.figure(figsize=(12, 5))
    for col in df.columns:
        plt.plot(volatility_rolling.index, volatility_rolling[col], label=portfolio_names.get(col, col))
    plt.title("Volatilité roulante (30 jours)")
    plt.xlabel("Date")
    plt.ylabel("Volatilité annualisée")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def plot_drawdown(df: pd.DataFrame, save_path: str):
    """
    Trace le drawdown maximum (perte par rapport au pic précédent) pour chaque portefeuille.
    Le drawdown est calculé comme la différence entre le rendement cumulé et son maximum roulant, divisé par ce maximum.
    """
    cumulative_returns = (1 + df).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    portfolio_names = {
        'portfolio_return_client1': "Portfolio 'Low risk'",
        'portfolio_return_client2': "Portfolio 'Low turnover'",
        'portfolio_return_client3': "Portfolio 'High yield equity only'",
        'benchmark_return': "Benchmark (S&P 500)"
    }
    plt.figure(figsize=(12, 5))
    for col in df.columns:
        plt.plot(drawdown.index, drawdown[col], label=portfolio_names.get(col, col))  
    plt.title("Drawdown maximum")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()



def save_table_as_png(df: pd.DataFrame, save_path: str):
    """
    Enregistre un DataFrame en tant qu'image PNG pour une insertion facile dans un rapport PDF.
    """
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1))  # Ajustement dynamique de la hauteur
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Ajuste la taille de la table

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def calculate_portfolio_statistics(df: pd.DataFrame, benchmark_returns: pd.Series, risk_free_rate: float=0.01, save_path: str = None):
    """
    Calcule diverses statistiques de performance pour chaque portefeuille, y compris le rendement annualisé,
    la volatilité, le ratio de Sharpe, le drawdown maximum, ainsi que l'alpha, le beta et le R² par rapport au benchmark.
    """
    portfolio_returns = df.mean(axis=0)
    annualized_returns = (1 + portfolio_returns) ** 252 - 1
    portfolio_volatility = df.std(axis=0) * np.sqrt(252)
    sharpe_ratio = (annualized_returns - risk_free_rate) / portfolio_volatility
    cumulative_returns = (1 + df).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min(axis=0)
    
    X = sm.add_constant(benchmark_returns)
    beta, alpha, r_squared = [], [], []
    for col in df.columns:
        y = df[col]
        model = sm.OLS(y, X).fit()
        alpha.append(model.params[0])
        beta.append(model.params[1])
        r_squared.append(model.rsquared)
    
    metrics = pd.DataFrame({
        'Mean Annualized Return': annualized_returns,
        'Volatility': portfolio_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Alpha': alpha,
        'Beta': beta,
        'R²': r_squared
    })
    
    portfolio_names = {
        'portfolio_return_client1': "Portfolio 'Low risk'",
        'portfolio_return_client2': "Portfolio 'Low turnover'",
        'portfolio_return_client3': "Portfolio 'High yield equity only'",
        'benchmark_return': "Benchmark (S&P 500)"
    }
    metrics.index = [portfolio_names.get(col, col) for col in metrics.index]
    if save_path:
        save_table_as_png(metrics, save_path)
    
    return metrics


def rank_managers(session, df: pd.DataFrame, benchmark_returns: pd.Series, risk_free_rate: float=0.01, save_path: str = None):
    """
    Calcule les performances des portefeuilles des clients, assigne un score et classe les managers.
    """
    metrics = calculate_portfolio_statistics(df, benchmark_returns, risk_free_rate)
    
    # Calcul du score de performance pour chaque portefeuille
    metrics['Score'] = (
        0.4 * metrics['Sharpe Ratio'] +
        0.3 * metrics['Alpha'] -
        0.2 * metrics['Max Drawdown'] +
        0.1 * metrics['R²']
    )
    
    # Associer chaque portefeuille à un client et son manager
    client_risk_profiles = {
        "Portfolio 'Low risk'": "Low risk",
        "Portfolio 'Low turnover'": "Low turnover",
        "Portfolio 'High yield equity only'": "High yield equity only"
    }
    
    rankings = []
    
    for portfolio_name, risk_profile in client_risk_profiles.items():
        # Récupérer le client correspondant
        client = session.query(Client).filter_by(risk_profile=risk_profile).first()
        
        if client:
            manager = session.query(Manager).filter_by(client_id=client.id).first()
            
            if manager:
                rankings.append({
                    'Manager': manager.name,
                    'Risk Profile': risk_profile,
                    'Score': metrics.loc[portfolio_name, 'Score']
                })
    
    # Trier du meilleur au moins bon
    rankings_df = pd.DataFrame(rankings).sort_values(by="Score", ascending=False)
    
    if save_path:
        save_table_as_png(rankings_df, save_path) 
    
    return rankings_df


def plot_deals_per_monday(engine, portfolio_id: int, save_path: str):
    """
    Affiche un graphique en barres du nombre de deals par lundi pour un portefeuille donné.

    Args:
        engine: Connexion SQLAlchemy à la base de données.
        portfolio_id (int): ID du portefeuille à analyser (1, 2 ou 3).
    """
    query = text("""
        SELECT date, COUNT(*) as num_deals
        FROM deals
        WHERE portfolio_id = :portfolio_id
        GROUP BY date
        ORDER BY date
    """)

    # Exécution de la requête
    with engine.connect() as conn:
        result = conn.execute(query, {"portfolio_id": portfolio_id})
        data = result.fetchall()

    # Vérifier si des données ont été récupérées
    if not data:
        print("Aucune donnée trouvée pour ce portefeuille.")
        return

    # Convertir les résultats en DataFrame
    df = pd.DataFrame(data, columns=["date", "num_deals"])
    df["date"] = pd.to_datetime(df["date"])

    # Création du graphique
    plt.figure(figsize=(10, 5))
    plt.bar(df["date"], df["num_deals"], color='royalblue')
    plt.xlabel("Date")
    plt.ylabel("Nombre de Deals")
    plt.title(f"Nombre de Deals par Lundi - Portefeuille {portfolio_id}")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.show()


def top_5_products(session, save_path: str):
    """Récupère les 5 produits les plus achetés/vendus tous clients confondus avec leurs noms et les affiche sous forme d'histogramme."""

    # Récupérer toutes les transactions avec les noms des produits
    deals = session.query(Deal.product_id_sold, Deal.product_id_bought, Deal.quantity, Product.name) \
        .join(Product, Deal.product_id_sold == Product.id) \
        .all()

    # Convertir en DataFrame
    deals_df = pd.DataFrame(deals, columns=['product_id_sold', 'product_id_bought', 'quantity', 'name'])

    # Comptabiliser les ventes (product_id_sold) et les achats (product_id_bought)
    product_sales = deals_df.groupby('name')['quantity'].sum().reset_index(name='total_trades')

    # Trier par volume total des transactions
    top_5 = product_sales.sort_values(by='total_trades', ascending=False).head(5)

    # Affichage des résultats sous forme d'histogramme
    plt.figure(figsize=(10, 6))
    plt.bar(top_5['name'], top_5['total_trades'], color='royalblue')
    plt.xlabel("Produits")
    plt.ylabel("Quantité échangée")
    plt.title("Top 5 des produits les plus tradés")
    plt.xticks(rotation=45)  # Rotation des noms de produits pour lisibilité
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.show()

    return top_5


def plot_pie_chart(session, client_id: int, date: str, save_path: str):
    """Affiche un camembert de la répartition des produits à une date donnée dans le portefeuille avec les noms des produits."""
    
    # Récupérer les transactions à cette date avec les noms des produits
    produits_ptf = session.query(Portfolio.product_id, Portfolio.qty, Product.name) \
        .join(Product, Portfolio.product_id == Product.id) \
        .filter(Portfolio.client_id == client_id) \
        .filter(Portfolio.date == date).all()

    # Convertir en DataFrame
    ptf_df = pd.DataFrame(produits_ptf, columns=['product_id', 'qty', 'name'])
    
    # Calculer la somme des quantités par produit
    product_distribution = ptf_df.groupby('name')['qty'].sum()

    # Créer un camembert
    plt.figure(figsize=(8, 6))
    plt.pie(product_distribution, labels=product_distribution.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"Répartition des produits pour le portefeuille {client_id} à la date {date}")
    plt.axis('equal')  # Assure que le camembert est bien circulaire
    plt.savefig(save_path)
    plt.show()


def plot_product_distribution(session, client_id: int, save_path: str):
    """Affiche l'évolution des parts des produits dans le portefeuille au fil du temps."""
    
    # Récupérer toutes les transactions du portefeuille
    portfolios = session.query(Portfolio.date, Portfolio.product_id, Portfolio.qty, Product.name) \
        .join(Product, Portfolio.product_id == Product.id) \
        .filter(Portfolio.client_id == client_id).all()

    # Convertir en DataFrame
    portfolios_df = pd.DataFrame(portfolios, columns=['date', 'product_id', 'qty', 'product_name'])
    portfolios_df['date'] = pd.to_datetime(portfolios_df['date'])

    # Calculer les poids de chaque produit à chaque date
    total_qty_by_date = portfolios_df.groupby('date')['qty'].transform('sum')
    portfolios_df['weight'] = portfolios_df['qty'] / total_qty_by_date

    # Pivot pour format Stacked Area Chart
    pivot_df = portfolios_df.pivot_table(index='date', columns='product_name', values='weight', aggfunc='sum')

    # Graphique en aires empilées
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind='area', stacked=True, colormap='tab10', alpha=0.7, figsize=(12, 6))

    plt.title(f"Évolution de la répartition des produits dans le portefeuille {client_id}")
    plt.xlabel("Date")
    plt.ylabel("Poids des produits")
    plt.legend(title="Produits")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.show()


def plot_dominant_product_step(session, client_id: int, save_path: str):
    """Affiche les périodes où chaque produit a été dominant dans le portefeuille."""

    portfolios = session.query(Portfolio.date, Portfolio.product_id, Portfolio.qty, Product.name) \
        .join(Product, Portfolio.product_id == Product.id) \
        .filter(Portfolio.client_id == client_id).all()

    portfolios_df = pd.DataFrame(portfolios, columns=['date', 'product_id', 'qty', 'product_name'])
    portfolios_df['date'] = pd.to_datetime(portfolios_df['date'])

    # Calculer le produit dominant à chaque date
    total_qty_by_date = portfolios_df.groupby('date')['qty'].transform('sum')
    portfolios_df['weight'] = portfolios_df['qty'] / total_qty_by_date
    dominant_product = portfolios_df.groupby(['date', 'product_name'])['weight'].sum().reset_index()
    dominant_product = dominant_product.loc[dominant_product.groupby('date')['weight'].idxmax()]

    # Création du step plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=dominant_product['date'], y=dominant_product['weight'], color='g', drawstyle='steps-post', marker='o')

    # Ajouter le nom du produit lorsqu'il change
    prev_product = None
    for i in range(len(dominant_product)):
        current_product = dominant_product['product_name'].iloc[i]
        if current_product != prev_product:  # Afficher seulement si changement
            plt.text(dominant_product['date'].iloc[i], dominant_product['weight'].iloc[i],
                     current_product, fontsize=10, ha='right', va='bottom')
        prev_product = current_product

    plt.title(f"Changement du produit dominant dans le portefeuille {client_id}")
    plt.xlabel("Date")
    plt.ylabel("Poids du produit dominant")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.show()


def calculate_hhi(weights):
    """Calcule l'indice de Herfindahl-Hirschman (HHI) pour mesurer la diversification."""
    return sum(w**2 for w in weights)


def plot_portfolio_diversification_hhi(session, client_id: int, save_path: str):
    """Affiche l'évolution de la diversification du portefeuille au fil du temps via l'HHI."""
    
    # Récupérer les transactions du portefeuille
    portfolios = session.query(Portfolio.date, Portfolio.product_id, Portfolio.qty) \
        .filter(Portfolio.client_id == client_id).all()

    # Convertir en DataFrame
    portfolios_df = pd.DataFrame(portfolios, columns=['date', 'product_id', 'qty'])
    portfolios_df['date'] = pd.to_datetime(portfolios_df['date'])

    # Calculer l'HHI pour chaque date
    hhi_values = []
    unique_dates = portfolios_df['date'].unique()

    for date in sorted(unique_dates):
        daily_data = portfolios_df[portfolios_df['date'] == date]
        total_quantity = daily_data['qty'].sum()
        
        if total_quantity > 0:
            weights = (daily_data['qty'] / total_quantity).values
            hhi = calculate_hhi(weights)
        else:
            hhi = None  # Pas de transaction ce jour-là

        hhi_values.append((date, hhi))

    # Convertir en DataFrame
    hhi_df = pd.DataFrame(hhi_values, columns=['date', 'HHI'])
    hhi_df['HHI_smooth'] = hhi_df['HHI'].ewm(span=5, adjust=False).mean()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(hhi_df['date'], hhi_df['HHI_smooth'], linestyle='-', label='HHI (Lissé)')
    plt.axhline(y=1/len(portfolios_df['product_id'].unique()), color='r', linestyle='--', label='Diversification optimale')
    plt.title(f'Évolution de la Diversification (HHI) - Portefeuille {client_id}')
    plt.xlabel('Date')
    plt.ylabel('HHI (concentration)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.show()


from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
engine = create_engine("sqlite:///Fund.db")
# Création de la session
Session = sessionmaker(bind=engine)  # Créer une classe de session liée à l'engine
session = Session()  # Créer une instance de session


# Fonction qui lance tous les graphiques
def generate_charts():
    """Fonction pour générer et enregistrer les graphiques nécessaires pour le rapport."""
    
    # Récupérer les résultats des portefeuilles
    df_results = compute_portfolio_returns("Fund.db", [1, 2, 3])

    # Générer les graphiques de performance globale
    plot_cumulative_returns(df_results, "reports/cumulative_returns.png")
    plot_rolling_volatility(df_results, "reports/rolling_volat.png")
    plot_drawdown(df_results, "reports/max_drawdowns.png")

    # Générer les graphiques des transactions
    plot_deals_per_monday(engine, portfolio_id=1, save_path="reports/deal_nb_1.png")
    plot_deals_per_monday(engine, portfolio_id=2, save_path="reports/deal_nb_2.png")
    plot_deals_per_monday(engine, portfolio_id=3, save_path="reports/deal_nb_3.png")
    
    top_5 = top_5_products(session, save_path="reports/top5_assets.png")
    
    # Générer les graphiques de répartition des actifs
    plot_pie_chart(session, client_id=1, date='2024-12-23', save_path="reports/asset_weights_1.png")
    plot_pie_chart(session, client_id=2, date='2024-12-02', save_path="reports/asset_weights_2.png")
    plot_pie_chart(session, client_id=3, date='2024-12-30', save_path="reports/asset_weights_3.png")

    # Générer les graphiques de distribution des produits
    plot_product_distribution(session, client_id=1, save_path="reports/evolution_weight_1.png")
    plot_product_distribution(session, client_id=2, save_path="reports/evolution_weight_2.png")
    plot_product_distribution(session, client_id=3, save_path="reports/evolution_weight_3.png")

    plot_dominant_product_step(session, client_id=1, save_path="reports/evolution_highest_weight_1.png")
    plot_dominant_product_step(session, client_id=2, save_path="reports/evolution_highest_weight_2.png")
    plot_dominant_product_step(session, client_id=3, save_path="reports/evolution_highest_weight_3.png")

    # Générer les graphiques de diversification du portefeuille
    plot_portfolio_diversification_hhi(session, client_id=1, save_path="reports/hhi_1.png") 
    plot_portfolio_diversification_hhi(session, client_id=2, save_path="reports/hhi_2.png")
    plot_portfolio_diversification_hhi(session, client_id=3, save_path="reports/hhi_3.png")

    # Calcul des statistiques et du classement des managers
    benchmark_returns = df_results['benchmark_return']
    df_portfolio_returns = df_results.drop(columns='benchmark_return')
    calculate_portfolio_statistics(df_portfolio_returns, benchmark_returns, save_path="reports/portfolio_stats.png")
    rank_managers(session, df_portfolio_returns, benchmark_returns, save_path="reports/managers_ranking.png")


################################
##### enregistrement pdf #######
################################

def add_images_row(pdf, image_paths, width=85, height=65):
    """
    Affiche deux images côte à côte sur une même ligne.
    """
    x_start = pdf.get_x()
    for i, img in enumerate(image_paths):
        pdf.image(img, x=x_start + i * (width + 5), y=pdf.get_y(), w=width, h=height)
    pdf.ln(height + 5)  # Passe à la ligne suivante

def add_centered_image(pdf, image_path, width=120, height=90):
    """
    Affiche une image centrée sur la page.
    """
    page_width = pdf.w
    x_position = (page_width - width) / 2
    pdf.image(image_path, x=x_position, y=pdf.get_y(), w=width, h=height)
    pdf.ln(height + 10)

def generate_pdf_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='', size=12)
    
    # Introduction
    pdf.cell(0, 10, "Analyse de Performance des Portefeuilles", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 8, "Ce rapport présente une analyse détaillée des performances de trois portefeuilles d'investissement, chacun ayant un profil de risque distinct. Nous étudierons leur rendement, volatilité, ratio de Sharpe et d'autres indicateurs clés.")
    pdf.ln(5)
    
    # Ajout des graphiques initiaux (Rendements cumulés, Volatilité, Drawdown)
    pdf.cell(0, 10, "1. Performance Globale", ln=True)
    add_images_row(pdf, ["reports/cumulative_returns.png", "reports/rolling_volat.png"])
    add_centered_image(pdf, "reports/max_drawdowns.png")
    
    # Passage à la page suivante pour le prochain point
    pdf.add_page()
    pdf.cell(0, 10, "2. Évolution des Transactions", ln=True)
    add_images_row(pdf, ["reports/deal_nb_1.png", "reports/deal_nb_2.png"])
    add_centered_image(pdf, "reports/deal_nb_3.png")
    
    # Passage à la page suivante pour le prochain point
    pdf.add_page()
    pdf.cell(0, 10, "3. Répartition des Actifs", ln=True)
    add_images_row(pdf, ["reports/asset_weights_1.png", "reports/asset_weights_2.png"])
    add_centered_image(pdf, "reports/asset_weights_3.png")
    
    # Passage à la page suivante pour le prochain point
    pdf.add_page()
    pdf.cell(0, 10, "4. Distribution des Produits", ln=True)
    add_images_row(pdf, ["reports/evolution_weight_1.png", "reports/evolution_weight_2.png"])
    add_centered_image(pdf, "reports/evolution_weight_3.png")
    
    # Passage à la page suivante pour le prochain point
    pdf.add_page()
    pdf.cell(0, 10, "5. Diversification du Portefeuille (HHI)", ln=True)
    add_images_row(pdf, ["reports/hhi_1.png", "reports/hhi_2.png"])
    add_centered_image(pdf, "reports/hhi_3.png")
    
    # Section des métriques - Nouvelle page
    pdf.add_page()
    pdf.cell(0, 10, "6. Statistiques de Performance", ln=True)
    pdf.image("reports/portfolio_stats.png", x=10, w=180)
    pdf.ln(5)
    
    # Explication du classement des managers
    pdf.cell(0, 10, "7. Classement des Managers", ln=True)
    pdf.multi_cell(0, 8, "Le classement des managers est basé sur un score composite calculé ainsi :")
    pdf.ln(5)
    pdf.multi_cell(0, 8, "Score = (0.4 * Sharpe Ratio) + (0.3 * Alpha) - (0.2 * Max Drawdown) + (0.1 * R²)")
    pdf.ln(5)
    pdf.image("reports/managers_ranking.png", x=10, w=180)
    
    pdf.image("reports/top5_assets.png", x=10, w=160)
    pdf.ln(5)

    # Conclusion
    pdf.cell(0, 10, "Conclusion", ln=True)
    pdf.multi_cell(0, 8, "L'analyse de performance montre des différences marquées entre les trois portefeuilles. Le portefeuille 'Low risk' est le plus stable avec un bon ratio de Sharpe. 'Low turnover' semble moins efficace, tandis que 'High yield equity only' offre des rendements élevés mais au prix d'une volatilité accrue. ")
    
    # Sauvegarde du rapport
    pdf.output("reports/Performance_Report.pdf")
    print("Rapport PDF généré avec succès : reports/Performance_Report.pdf")
