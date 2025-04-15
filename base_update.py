from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from base_builder import Base, Product, Return, Client, Manager, Portfolio, Deal
from config import assets
from datetime import datetime
from sqlalchemy import func
import random
import pandas as pd
from strategies import update_portfolio_with_balance
from typing import Dict, List, Any

# Connexion à la base de données
def create_session():
    engine = create_engine("sqlite:///Fund.db")
    Session = sessionmaker(bind=engine)
    return Session()


# Remplissage de la table `products` avec le nom des actifs et leur classe
def insert_products_table(session):
    """ Remplie la table products avec les actifs et leurs profils de risque 
    en vérifiant si les produits existent déjà dans la base. """
    for asset_class, tickers_dict in assets.items():
        for ticker, risk_profiles in tickers_dict.items():
            existing_product = session.query(Product).filter_by(name=ticker).first()
            if not existing_product:
                risk_profile_str = ",".join(risk_profiles)  # Convertit la liste en string
                product = Product(name=ticker, type=asset_class, risk_profile=risk_profile_str)
                session.add(product)
    session.commit()
    print("Table 'products' remplie avec profils de risque !")


# Insertion des rendements dans la table `returns`
def insert_returns_into_db(session, returns: pd.DataFrame):
    """ Insère les rendements des actifs dans la table returns en associant chaque rendement à son produit correspondant. """
    for date, row in returns.iterrows():
        for ticker, ret in row.items():
            product = session.query(Product).filter_by(name=ticker).first()
            if product:
                session.add(Return(product_id=product.id, date=date, return_value=ret))
    session.commit()
    print("Table 'returns' remplie !")


# Fonction pour ajouter des clients
def add_clients(session, clients_data: List[Dict[str, Any]]):
    """ Ajoute de nouveaux clients à la base de données uniquement s'ils n'existent pas déjà, en vérifiant par leur nom. """
    # Vérifier si les clients existent déjà dans la base
    existing_clients = {c.name for c in session.query(Client.name).all()}

    # Ajout des clients uniquement s'ils n'existent pas déjà
    for data in clients_data:
        if data["name"] not in existing_clients:
            client = Client(**data)
            session.add(client)

    session.commit()


def generate_random_client_data():
    """Génère des noms, adresses, codes postaux, e-mails et numéros aléatoires pour les clients."""
    
    first_names = ["Léa", "Gabriel", "Emma", "Hugo", "Chloé", "Lucas", "Manon", "Noah", "Camille", "Nathan"]
    last_names = ["Durand", "Moreau", "Lefevre", "Rousseau", "Fournier", "Dubois", "Marchand", "Bernard", "Gauthier", "Robert"]

    addresses_with_postcodes = [
        ("12 Rue des Érables, Paris", "75001"),
        ("34 Avenue des Champs, Lyon", "69002"),
        ("56 Boulevard Haussmann, Marseille", "13008"),
        ("78 Impasse des Lilas, Toulouse", "31000"),
        ("90 Place de la République, Bordeaux", "33000")
    ]

    risk_profiles = ["Low risk", "Low turnover", "High yield equity only"]
    
    clients_data = []
    used_addresses = set()

    for i in range(3):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        
        email = f"{first_name.lower()}.{last_name.lower()}@fondsmultiasset.com"
        phone_number = f"+33 {random.randint(6, 7)}{random.randint(0, 9)} {random.randint(10, 99)} {random.randint(10, 99)} {random.randint(10, 99)}"
        
        # Sélectionner une adresse non encore utilisée
        available_addresses = [a for a in addresses_with_postcodes if a not in used_addresses]
        if not available_addresses:
            raise ValueError("Pas assez d'adresses uniques pour assigner à tous les clients.")
        
        address, postcode = random.choice(available_addresses)
        used_addresses.add((address, postcode))  # Marquer comme utilisée
        
        client = {
            "name": full_name,
            "risk_profile": risk_profiles[i],  # Assure que les trois profils sont bien assignés
            "adresse": address,
            "code_postal": postcode,
            "mail": email,
            "numero_tel": phone_number
        }
        
        clients_data.append(client)

    return clients_data


# Fonction pour ajouter des managers
def add_managers(session, managers_data: List[Dict[str, Any]]):
    """ Ajoute des managers à la base de données s'ils n'ont pas encore été associés à un client spécifique. """
    for manager in managers_data:
        existing_manager = session.query(Manager).filter_by(client_id=manager['client_id']).first()
        if not existing_manager:
            session.add(Manager(**manager))

    session.commit()


def update_asset_price_for_specific_date(session, data: pd.DataFrame, target_date):
    """
    Met à jour la colonne `asset_price` de la table `portfolios` pour un actif donné,
    en récupérant son prix à la date spécifique (ou la dernière date valide avant).
    :param session: session SQLAlchemy
    :param data: DataFrame contenant les prix des actifs (colonnes = tickers, index = dates)
    :param target_date: Date cible pour récupérer le prix des actifs (par défaut '2022-12-30')
    """
    # Vérification que la date cible existe dans l'index de data
    if target_date not in data.index:
        # Recherche de la dernière date valide avant la cible
        closest_date = data.index[data.index <= target_date].max()
        if pd.isna(closest_date):
            print(f"Aucune date valide avant {target_date}.")
            return  # Si aucune date valide n'est trouvée, on arrête la fonction
        print(f"Date cible non trouvée. Utilisation de la dernière date valide : {closest_date}")
    else:
        closest_date = target_date

    # Mise à jour du prix dans la table portfolios pour chaque product_id
    for ticker in data.columns:
        price_at_closest_date = data.loc[closest_date, ticker]

        # Récupération du produit (actif) dans la table `products` basé sur le ticker
        product = session.query(Product).filter_by(name=ticker).first()
        if product:
            # Récupération de tous les portefeuilles associés à cet actif (product_id)
            portfolios = session.query(Portfolio).filter_by(product_id=product.id).all()

            for portfolio in portfolios:
                portfolio.asset_price = price_at_closest_date  # Mise à jour du prix de l'actif
                if portfolio.value is not None and price_at_closest_date > 0:
                    portfolio.qty = portfolio.value / price_at_closest_date  # Mise à jour de la quantité

    session.commit()
    print("Table 'portfolios' mise à jour avec les prix et quantités !")


def update_first_amount_held(session, target_date):
    """
    Calcule la somme de `asset_price * qty` pour chaque `client_id` à la date spécifiée 
    et met à jour la colonne `amount_held` dans la table `clients`.
    :param session: session SQLAlchemy
    :param target_date: Date cible)
    """
    # Vérification si la date cible existe dans la table portfolios
    portfolios_at_date = session.query(Portfolio).filter_by(date=target_date).all()
    if not portfolios_at_date:
        print(f"Aucune donnée trouvée pour la date {target_date}.")
        return
    
    # Calcul de la somme des produits asset_price * qty pour chaque client_id
    client_amounts = {}
    
    for portfolio in portfolios_at_date:
        client_id = portfolio.client_id
        asset_price = portfolio.asset_price
        qty = portfolio.qty
        
        if asset_price is not None and qty is not None:
            amount = asset_price * qty
            
            # On additionne les montants pour chaque client
            if client_id in client_amounts:
                client_amounts[client_id] += amount
            else:
                client_amounts[client_id] = amount

    # Mise à jour de la colonne `amount_held` pour chaque client dans la table `clients`
    for client_id, amount_held in client_amounts.items():
        client = session.query(Client).filter_by(id=client_id).first()
        if client:
            client.amount_held = amount_held
            manager = session.query(Manager).filter_by(client_id=client.id).first()
        if not manager:
            print(f"Aucun manager trouvé pour le client {client.name}.")
            return
        # Mettre à jour la colonne 'total_value' avec la nouvelle valeur du portefeuille
        manager.total_value = amount_held

    session.commit()
    print("Tables 'clients' et 'manager' mises à jour avec les montants détenus !")


def update_weight_in_fund():
    """Met à jour 'weight_in_fund' pour chaque manager en fonction du total des portefeuilles sous gestion."""
    engine = create_engine("sqlite:///Fund.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Calculer la somme totale de tous les portefeuilles sous gestion
    total_portfolio_value = session.query(func.sum(Manager.total_value)).scalar() or 1  # Évite division par zéro
    
    # Récupérer tous les managers
    managers = session.query(Manager).all()
    
    # Mettre à jour 'weight_in_fund' pour chaque manager
    for manager in managers:
        manager.weight_in_fund = manager.total_value / total_portfolio_value

    # Commiter la mise à jour
    session.commit()
    # Fermer la session
    session.close()
    print("Mise à jour des poids terminée.")


def update_portfolio_in_db(session, weights: Dict[str, float], allocations: Dict[str, float], prices_today: pd.DataFrame, date_str: str, risk_pfl: str):
    """Met à jour la table Portfolios avec les nouvelles allocations et poids."""
    date = datetime.strptime(date_str, "%Y-%m-%d").date()

    # Récupérer dynamiquement le client correspondant au profil considéré
    client = session.query(Client).filter_by(risk_profile=risk_pfl).first()
    if not client:
        print("Aucun client trouvé avec le profil {risk_pfl}.")
        return
    
    for ticker, qty in allocations.items():
        if weights.get(ticker, 0) == 0:  # Ignore les poids nuls
            continue
        product = session.query(Product).filter_by(name=ticker).first()
        if product:
            asset_price = prices_today.loc[prices_today['Ticker'] == ticker, 'Close'].values[0]
            

            portfolio_entry = session.query(Portfolio).filter_by(date=date, product_id=product.id).first()
            
            if portfolio_entry:
                portfolio_entry.qty = qty
                portfolio_entry.weight = weights[ticker]
                portfolio_entry.value = asset_price * qty
            else:
                session.add(Portfolio(
                    date=date,
                    product_id=product.id,
                    client_id=client.id,
                    weight=weights[ticker],
                    qty=qty,
                    asset_price=asset_price,
                    value=asset_price * qty
                ))

    session.commit()


def update_client_amount_held(session, risk_pfl : str, portfolio_value: float):
    """Met à jour la colonne 'amount_held' du client avec la valeur du portefeuille."""
    # Récupérer dynamiquement le client correspondant au profil 'Low risk'
    client = session.query(Client).filter_by(risk_profile=risk_pfl).first()
    
    if not client:
        print("Aucun client trouvé avec le profil {risk_pfl}.")
        return
    
    # Mettre à jour la colonne 'amount_held' avec la nouvelle valeur du portefeuille
    client.amount_held = portfolio_value

    manager = session.query(Manager).filter_by(client_id=client.id).first()
    if not manager:
        print(f"Aucun manager trouvé pour le client {client.name}.")
        return
    # Mettre à jour la colonne 'total_value' avec la nouvelle valeur du portefeuille
    manager.total_value = portfolio_value
    # Commiter la transaction
    session.commit()


def insert_deals_into_db(session, sell_transactions: Dict[str, float], buy_transactions: Dict[str, float], idclient: int, date_str: str):
    """Ajoute les transactions de vente et d'achat dans la table Deals."""
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    for ticker, qty in sell_transactions.items():
        if qty != 0:  # Vérifier que la quantité est différente de 0
            product = session.query(Product).filter_by(name=ticker).first()
            if product:
                session.add(Deal(
                    portfolio_id=idclient,
                    date=date,
                    action="SELL",
                    product_id_sold=product.id,
                    quantity=qty
                ))

    for ticker, qty in buy_transactions.items():
        if qty != 0:  # Vérifier que la quantité est différente de 0
            product = session.query(Product).filter_by(name=ticker).first()
            if product:
                session.add(Deal(
                    portfolio_id=idclient,
                    date=date,
                    action="BUY",
                    product_id_bought=product.id,
                    quantity=qty
                ))

    session.commit()


# Fonction générique pour traiter le portefeuille d'un client
def update_client_portfolio(predict_data: pd.DataFrame, allocations: Dict[str, float], assets: List[str], client_id: int, client_name: str):
    """ Met à jour le portefeuille d'un client en fonction des prévisions de données et des allocations, 
    et enregistre les transactions et les valeurs dans la base de données. """
    # Connexion à la base de données
    engine = create_engine("sqlite:///Fund.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Récupérer toutes les dates disponibles dans predict_data sans doublons
    all_dates = sorted(set(predict_data.index))

    for current_date in all_dates:
        current_date_str = current_date.strftime('%Y-%m-%d')  # Conversion en string

        # Appeler la fonction pour mettre à jour le portefeuille et la balance du client
        sell_transactions, buy_transactions, new_allocations, weights, portfolio_value, prices_today = update_portfolio_with_balance(
            predict_data, allocations, assets, current_date_str, client_id, session)

        allocations = new_allocations.copy()

        # Mettre à jour la base de données
        update_portfolio_in_db(session, weights, new_allocations, prices_today, current_date_str, client_name)
        update_client_amount_held(session, client_name, portfolio_value)
        insert_deals_into_db(session, sell_transactions, buy_transactions, client_id, current_date_str)

    # Fermer la session après la boucle
    session.close()


