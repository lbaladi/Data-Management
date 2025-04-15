import numpy as np
from scipy.optimize import minimize
from base_builder import Base, Product, Portfolio, Client
from datetime import datetime
import pandas as pd
from typing import Callable

# Fonction d'optimisation Low risk avec contrainte de volatilité
def lowrisk_optimization(returns_df: pd.DataFrame, target_volatility: float = 0.1, max_weight: float = 0.6) -> np.ndarray:
    """
    Optimise un portefeuille pour une stratégie à faible risque avec une contrainte de volatilité.
    L'objectif est de minimiser le risque du portefeuille tout en respectant une cible de volatilité annualisée.
    Les poids des actifs sont ajustés de manière à ce que la volatilité du portefeuille soit comprise dans un intervalle autour de la volatilité cible.

    Args:
        returns_df (pandas.DataFrame): DataFrame des rendements historiques des actifs
        target_volatility (float): Volatilité cible annualisée pour le portefeuille (défaut: 0.10 ou 10%)
        max_weight (float): Poids maximum pour un actif individuel (défaut: 0.60 ou 60%)

    Returns:
        numpy.ndarray: Poids optimisés pour chaque actif du portefeuille
    """

    expected_returns = returns_df.mean()*252
    cov_matrix = returns_df.cov()*252 
    num_assets = len(expected_returns)
    
    # Objectif: minimiser le risque
    def objective(weights):
        return -np.sum(weights * expected_returns)
    
    # Contrainte de volatilité comme une inégalité (≤ au lieu de =)
    def volatility_constraint(weights):
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return target_volatility*1.05 - portfolio_vol  # Doit être ≥ 0
    
    def volatility_constraint_lower(weights):
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        # On veut que la volatilité soit au-dessus de 95% de la cible
        return portfolio_vol - (target_volatility * 0.95)  # Doit être ≥ 0 (plancher)
    
    # Initialisation aléatoire pour éviter de rester bloqué dans un minimum local
    np.random.seed(42)  # Pour reproductibilité
    initial_weights = np.random.random(num_assets)
    initial_weights = initial_weights / np.sum(initial_weights)  # Normalisation

    # Bornes pour les poids
    bounds = [(0, max_weight) for _ in range(num_assets)]


    # Contraintes
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Somme = 1
        {'type': 'ineq', 'fun': volatility_constraint},  # Volatilité ≤ cible
        {'type': 'ineq', 'fun': volatility_constraint_lower}  # Volatilité ≥ cible
    ]

    # Exécuter l'optimisation
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-9, 'disp': True}
    )

    # Si l'optimisation réussit
    if result.success:
        # Sélectionner les 10 actifs avec les poids les plus élevés
        weights = result.x
        top_indices = np.argsort(weights)[-10:]
        final_weights = np.zeros(num_assets)
        final_weights[top_indices] = weights[top_indices]
        
        # Renormaliser pour que la somme soit 1
        final_weights = final_weights / np.sum(final_weights)
        
        # Vérifier la volatilité
        final_vol = np.sqrt(final_weights.T @ cov_matrix @ final_weights)
        print(f"Volatilité annualisée du portefeuille : {final_vol:.4f}")
        
        return final_weights
    else:
        print(f"Échec de l'optimisation: {result.message}")
        
        # Essayer une approche de secours - utiliser des poids égaux pour les actifs à faible volatilité
        vols = np.sqrt(np.diag(cov_matrix))
        low_vol_indices = np.argsort(vols)[:10]
        backup_weights = np.zeros(num_assets)
        backup_weights[low_vol_indices] = 1/10
        
        print("Utilisation de l'approche de secours: allocation sur les 10 actifs les moins volatils")
        backup_vol = np.sqrt(backup_weights.T @ cov_matrix @ backup_weights)
        print(f"Volatilité de l'approche de secours: {backup_vol:.4f}")
        
        return backup_weights


def lowturnover_optimization(returns_df: pd.DataFrame, risk_free_rate: float = 0.02, target_volatility: float = 0.10, max_weight: float = 0.15, min_weight: float = 0.01) -> np.ndarray:
    """
    Optimise un portefeuille pour une stratégie Low Turnover qui limite les transactions.
    Utilise une approche de maximisation du ratio de Sharpe avec contrainte de volatilité cible
    et favorise les actifs les plus stables.
    
    Args:
        returns_df (pandas.DataFrame): DataFrame des rendements historiques
        risk_free_rate (float): Taux sans risque annualisé (défaut: 0.02 ou 2%)
        target_volatility (float): Volatilité cible annualisée pour le portefeuille (défaut: 0.10 ou 10%)
        max_weight (float): Poids maximum pour un actif individuel (défaut: 0.15 ou 15%)
        min_weight (float): Poids minimum pour les actifs sélectionnés (défaut: 0.01 ou 1%)
        
    Returns:
        numpy.ndarray: Poids optimisés pour chaque actif
    """
    # Annualiser les rendements et la covariance
    expected_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    num_assets = len(expected_returns)
    asset_names = returns_df.columns
    
    # Calculer les volatilités individuelles et les ratios de Sharpe individuels
    vols = np.sqrt(np.diag(cov_matrix))
    individual_sharpe = (expected_returns - risk_free_rate) / vols
    
    # Calculer la stabilité des rendements par actif (moins le rendement varie, plus il est stable)
    # Utilisons le coefficient de variation comme mesure de stabilité
    rolling_returns = returns_df.rolling(window=20).mean()  # Moyennes mobiles sur 20 jours
    stability_scores = rolling_returns.std() / rolling_returns.mean().abs()
    stability_scores = stability_scores.fillna(1)  # Gérer les NaN
    
    # Plus le score est bas, plus l'actif est stable
    stability_ranks = stability_scores.rank()
    
    # Définir la fonction objectif: maximiser le ratio de Sharpe ajusté à la stabilité
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Si la volatilité est trop éloignée de la cible, pénaliser fortement
        vol_penalty = abs(portfolio_vol - target_volatility) * 10
        
        # Calculer le ratio de Sharpe
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        
        # Pénalité pour les actifs instables (pondération par la stabilité)
        stability_penalty = np.sum(weights * stability_scores)
        
        # Retourner le négatif car minimize() minimise la fonction
        return -(sharpe - vol_penalty - 0.2 * stability_penalty)
    
    # Identifier les actifs les plus prometteurs (meilleur compromis rendement/risque/stabilité)
    # Nous allons pré-sélectionner les actifs à utiliser dans l'optimisation
    combined_score = individual_sharpe - 0.2 * stability_scores
    
    # Sélectionner les 15 meilleurs actifs 
    top_assets_indices = combined_score.nlargest(15).index
    
    # Créer un masque pour les actifs sélectionnés
    selected_mask = np.zeros(num_assets, dtype=bool)
    for asset in top_assets_indices:
        idx = returns_df.columns.get_loc(asset)
        selected_mask[idx] = True
    
    # Initialisation avec des poids égaux pour les actifs sélectionnés
    initial_weights = np.zeros(num_assets)
    initial_weights[selected_mask] = 1.0 / selected_mask.sum()
    
    # Bornes pour les poids: 0 pour les actifs non sélectionnés, min_weight à max_weight pour les sélectionnés
    bounds = []
    for i in range(num_assets):
        if selected_mask[i]:
            bounds.append((min_weight, max_weight))
        else:
            bounds.append((0, 0))
    
    # Contraintes: la somme des poids doit être égale à 1
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    ]
    
    # Exécuter l'optimisation
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-9, 'disp': True}
    )
    
    if result.success:
        optimized_weights = result.x
        
        # Calculer les métriques finales
        portfolio_return = np.dot(optimized_weights, expected_returns)
        portfolio_vol = np.sqrt(optimized_weights.T @ cov_matrix @ optimized_weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
        
        # Calculer le ratio de diversification (approche de TOBAM)
        weighted_vols = optimized_weights * vols
        div_ratio = np.sum(weighted_vols) / portfolio_vol
        
        print(f"\nOptimisation Low Turnover améliorée réussie:")
        print(f"Rendement annualisé attendu: {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
        print(f"Ratio de Diversification: {div_ratio:.4f}")
        
        # Calculer la concentration du portefeuille (mesure d'Herfindahl)
        herfindahl = np.sum(optimized_weights**2)
        print(f"Indice de concentration (Herfindahl): {herfindahl:.4f} (plus bas = plus diversifié)")
        print(f"Nombre d'actifs sélectionnés: {np.sum(optimized_weights > 0.005)}")
        
        # Afficher les allocations individuelles
        print("\nAllocations:")
        allocations = [(asset, weight) for asset, weight in zip(asset_names, optimized_weights) if weight > 0.005]
        for asset, weight in sorted(allocations, key=lambda x: x[1], reverse=True):
            print(f"  {asset}: {weight:.4f}")
        
        return optimized_weights
    else:
        print(f"Échec de l'optimisation: {result.message}")
        
        # Solution de secours: Portefeuille à risque minimum avec les actifs pré-sélectionnés
        def min_vol_objective(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)
        
        backup_result = minimize(
            min_vol_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if backup_result.success:
            backup_weights = backup_result.x
            backup_return = np.dot(backup_weights, expected_returns)
            backup_vol = np.sqrt(backup_weights.T @ cov_matrix @ backup_weights)
            backup_sharpe = (backup_return - risk_free_rate) / backup_vol
            
            print("Utilisation de la stratégie de secours: Portefeuille à risque minimum")
            print(f"Rendement de secours: {backup_return:.4f} ({backup_return*100:.2f}%)")
            
            return backup_weights
        else:
            # Si tout échoue, répartition équipondérée sur les actifs pré-sélectionnés
            equal_weights = np.zeros(num_assets)
            equal_weights[selected_mask] = 1.0 / selected_mask.sum()
            return equal_weights


def highyield_optimization(returns_df: pd.DataFrame, max_weight: float = 0.20) -> np.ndarray:
    """
    Optimise un portefeuille d'actions pour maximiser le rendement attendu (High Yield).
    
    Args:
        returns_df (pandas.DataFrame): DataFrame des rendements historiques des actions
        max_weight (float): Poids maximum pour un actif individuel (défaut: 0.6 ou 60%)
        
    Returns:
        numpy.ndarray: Poids optimisés pour chaque actif
    """
    # Annualiser les rendements attendus (252 jours de trading par an)
    expected_returns = returns_df.mean() * 252
    num_assets = len(expected_returns)
    
    # Fonction objectif: maximiser le rendement (minimiser le négatif du rendement)
    def objective(weights):
        return -np.dot(weights, expected_returns)
    
    # Initialisation avec des poids aléatoires pour éviter les minima locaux
    np.random.seed(42)  # Pour reproductibilité
    initial_weights = np.random.random(num_assets)
    initial_weights = initial_weights / np.sum(initial_weights)
    
    # Bornes pour les poids (entre 0 et max_weight)
    bounds = [(0, max_weight) for _ in range(num_assets)]
    
    # Contraintes: la somme des poids doit être égale à 1
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    ]
    
    # Exécuter l'optimisation
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-9, 'disp': True}
    )
    
    if result.success:
        # Calculer le rendement attendu du portefeuille optimisé
        optimized_weights = result.x
        portfolio_return = np.dot(optimized_weights, expected_returns)
        
        # Calculer la volatilité du portefeuille (uniquement à titre informatif)
        cov_matrix = returns_df.cov() * 252
        portfolio_vol = np.sqrt(optimized_weights.T @ cov_matrix @ optimized_weights)
        
        print(f"Optimisation High Yield réussie:")
        print(f"Rendement annualisé attendu: {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
        print(f"Volatilité annualisée: {portfolio_vol:.4f} ({portfolio_vol*100:.2f}%)")
        print(f"Ratio Sharpe (sans risque-free): {portfolio_return/portfolio_vol:.4f}")
        
        # Afficher les allocations individuelles
        asset_names = returns_df.columns
        for asset, weight in zip(asset_names, optimized_weights):
            if weight > 0.005:  # Afficher uniquement les allocations significatives
                print(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)")
        
        return optimized_weights
    else:
        print(f"Échec de l'optimisation: {result.message}")
        
        # Solution de secours: investir dans les actions ayant les meilleurs rendements historiques
        top_assets = expected_returns.argsort()[::-1]  # Tri décroissant
        
        # Répartir le capital entre les 3 meilleures actions
        backup_weights = np.zeros(num_assets)
        top_3 = top_assets[:3]
        backup_weights[top_3] = 1/3
        
        print("Utilisation de la stratégie de secours: allocation sur les 3 actions avec les meilleurs rendements")
        backup_return = np.dot(backup_weights, expected_returns)
        cov_matrix = returns_df.cov() * 252
        backup_vol = np.sqrt(backup_weights.T @ cov_matrix @ backup_weights)
        print(f"Rendement de secours: {backup_return:.4f} ({backup_return*100:.2f}%)")
        print(f"Volatilité de secours: {backup_vol:.4f} ({backup_vol*100:.2f}%)")
        
        return backup_weights




def allocate_portfolio(risk_profile: str, 
                       optimized_weights: np.ndarray, 
                       all_assets: list[str], 
                       observation_date: datetime.date, 
                       session):

    """
    Alloue le portefeuille pour une stratégie donnée et stocke les allocations dans la base de données.
    Gère également la mise à jour des allocations existantes et l'enregistrement des transactions.
    :param risk_profile: Profil de risque du client ("Low risk", "High yield")
    :param optimized_weights: Numpy array des poids optimisés des actifs
    :param all_assets: Liste des noms des actifs correspondants
    :param observation_date: Date d'observation pour les allocations
    :param session: Session SQLAlchemy active pour les requêtes
    """
    # Récupération du client correspondant
    client = session.query(Client).filter_by(risk_profile=risk_profile).first()
    if not client:
        raise ValueError(f"Aucun client avec le profil '{risk_profile}' trouvé.")

    client_balance = client.balance
    
    # Récupération du portefeuille actuel du client
    current_portfolio = session.query(Portfolio).filter_by(
        client_id=client.id
    ).filter(Portfolio.date < observation_date).all()
    
    # Dictionnaire pour stocker les allocations actuelles (product_id -> Portfolio)
    current_allocations = {p.product_id: p for p in current_portfolio}
    
    # Dictionnaire pour les nouvelles allocations
    new_allocations = {}
    
    # Affichage des résultats
    print(f"Optimisation pour la stratégie {risk_profile}:")
    
    
    # Parcourir tous les actifs et poids optimisés
    for asset_name, weight in zip(all_assets, optimized_weights):
        if weight > 0.001:  # Ignorer les poids négligeables
            print(f"Actif: {asset_name}, Poids optimisé: {weight:.4f}")
            
            # Récupérer le produit correspondant
            product = session.query(Product).filter_by(name=asset_name).first()
            if product:
                new_allocations[product.id] = {
                    'product': product,
                    'weight': weight,
                    'value': client_balance * weight
                }
    
    # Transactions à enregistrer
    transactions = []


    # 1. Créer ou mettre à jour les allocations
    for product_id, allocation_info in new_allocations.items():
        product = allocation_info['product']
        weight = allocation_info['weight']
        value = allocation_info['value']
        
        existing_allocation = session.query(Portfolio).filter_by(
            date=observation_date, client_id=client.id, product_id=product_id
        ).first()
        
        if existing_allocation:
            # Mise à jour d'une allocation existante
            old_value = existing_allocation.value
            existing_allocation.weight = weight
            existing_allocation.value = value
            
            # Enregistrer une transaction si la valeur a changé
            if old_value != value:
                transactions.append({
                    'product_id': product_id,
                    'change': value - old_value,
                    'type': 'update'
                })
        else:
            # Création d'une nouvelle allocation
            portfolio = Portfolio(
                date=observation_date,
                client_id=client.id,
                product_id=product_id,
                weight=weight,
                value=value
            )
            session.add(portfolio)
            
            # Enregistrer une transaction d'achat
            transactions.append({
                'product_id': product_id,
                'change': value,
                'type': 'buy'
            })
    
    # 2. Supprimer les allocations qui ne font plus partie du portefeuille
    for product_id, old_allocation in current_allocations.items():
        if product_id not in new_allocations:
            # Enregistrer une transaction de vente
            transactions.append({
                'product_id': product_id,
                'change': -old_allocation.value,
                'type': 'sell'
            })
    
    
    client.balance = 0  # Mettre le solde à zéro après l'allocation

    session.commit()
    print(f"Allocation pour la stratégie {risk_profile} sauvegardée dans la base de données.")
    print(f"Nombre de transactions: {len(transactions)}")


def optimize_and_allocate(assets: list[str], 
                          returns_df: pd.DataFrame, 
                          strategy_name: str, 
                          optimization_func: Callable[[pd.DataFrame], np.ndarray], 
                          session):

    """
    Applique une optimisation sur un sous-ensemble d'actifs et alloue le portefeuille.

    Arguments:
    - strategy_name : Nom de la stratégie (ex: "Low risk", "High yield equity only", "Low turnover")
    - filter_condition : Fonction de filtrage des actifs
    - optimization_func : Fonction d'optimisation associée
    - session : Session SQLAlchemy active
    """

    # Optimisation
    optimized_weights = optimization_func(returns_df[assets])

    # Allocation du portefeuille
    allocate_portfolio(strategy_name, optimized_weights, assets, datetime(2023, 1, 3).date(), session)



def get_last_portfolio_allocation(session, client_id: int):
    """
    Récupère la dernière allocation du portefeuille spécifié (par exemple, 'low_risk') pour un client donné.
    :param session: session SQLAlchemy
    :param client_id: ID du client dont on veut récupérer l'allocation (par défaut 1)
    :param portfolio_type: Type du portefeuille ('low_risk', par exemple)
    :return: Dictionnaire avec le ticker comme clé et la quantité détenue comme valeur
    """
    # Récupérer la dernière date du portefeuille pour ce client et ce type de portefeuille
    last_portfolio = session.query(Portfolio).filter_by(client_id=client_id).order_by(Portfolio.date.desc()).first()

    if not last_portfolio:
        print(f"Aucune donnée trouvée pour le portefeuille du client {client_id}.")
        return {}

    # Récupérer toutes les quantités des actifs à la dernière date
    portfolio_qties = {}
    portfolios = session.query(Portfolio).filter_by(client_id=client_id, date=last_portfolio.date).all()

    for portfolio in portfolios:
        # On suppose que chaque actif du portefeuille a un ticker (le nom de l'actif)
        portfolio_qties[portfolio.product.name] = portfolio.qty

    return portfolio_qties


def normalize_allocations(assets: list, allocations: dict):
    """
    Ajoute les actifs manquants avec une allocation de 0 et normalise les poids.
    
    :param assets: Liste des actifs à considérer
    :param allocations: Dictionnaire des allocations existantes {actif: poids}
    :return: Dictionnaire des poids normalisés {actif: poids}
    """
    # Ajouter les actifs manquants avec allocation = 0
    for asset in assets:
        if asset not in allocations:
            allocations[asset] = 0

    # Réordonner selon assets
    allocations = {asset: allocations[asset] for asset in assets}
    
    # Normaliser les poids (évite division par zéro)
    total_weight = sum(allocations.values())
    if total_weight == 0:
        weights = {asset: 0 for asset in allocations}
    else:
        weights = {asset: allocations[asset] / total_weight for asset in allocations}

    return allocations, weights
