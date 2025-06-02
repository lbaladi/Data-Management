from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, inspect
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Table Clients
class Client(Base):
    __tablename__ = 'clients'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    risk_profile = Column(String, nullable=False)  # Ex: "Low risk", "High yield equity only", "Low turnover"
    balance = Column(Float, default=1000000) # Solde du client. On l'initialise à $1m.
    amount_held = Column(Float, default=0.0)  # Valeur du portefeuille détenu par le client
    
    
    # Nouvelles colonnes
    adresse = Column(String, nullable=False)
    code_postal = Column(String, nullable=False)
    mail = Column(String, unique=True, nullable=False)
    numero_tel = Column(String, unique=True, nullable=False)

    portfolios = relationship("Portfolio", back_populates="client")
    manager = relationship("Manager", back_populates="client", uselist=False)    


# Table Produits
class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String)  # "Equity", "Bond", "Commodity"...
    risk_profile = Column(String)  # Ex: "low_vol,high_yield"
    
    returns = relationship("Return", back_populates="product")

# Table Returns
class Return(Base):
    __tablename__ = 'returns'
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'))
    date = Column(Date, nullable=False)
    return_value = Column(Float)
    
    product = relationship("Product", back_populates="returns")

# Table Portfolios
class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)  # Date de l'allocation
    client_id = Column(Integer, ForeignKey('clients.id'))  # ID du client
    product_id = Column(Integer, ForeignKey('products.id'))  # ID de l'actif
    weight = Column(Float, nullable=False)  # Poids de l'actif dans le portefeuille
    value = Column(Float, nullable=False)  # Valeur de l'actif dans le portefeuille
    asset_price = Column(Float, nullable=True)  # Prix de l'actif
    qty = Column(Float, nullable=True)  # Quantité d'actifs détenue
    
    client = relationship("Client", back_populates="portfolios")
    product = relationship("Product")
    deals = relationship("Deal", back_populates="portfolio")

# Table Managers
class Manager(Base):
    __tablename__ = 'managers'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    client_id = Column(Integer, ForeignKey('clients.id'), unique=True)  # Un seul manager par client
    total_value = Column(Float, default=0.0)  # Valeur totale du portefeuille du client
    weight_in_fund = Column(Float, default=0.0)  # Poids du client dans le fonds global

    client = relationship("Client", back_populates="manager")

# Table Deals
class Deal(Base):
    __tablename__ = 'deals'
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    date = Column(Date, nullable=False)
    action = Column(String, nullable=False)  # "BUY" ou "SELL"
    product_id_sold = Column(Integer, ForeignKey('products.id'), nullable=True)  # Produits vendus
    product_id_bought = Column(Integer, ForeignKey('products.id'), nullable=True)  # Produits achetés
    #product_id = Column(Integer, ForeignKey('products.id'), nullable=True)
    quantity = Column(Integer)

    portfolio = relationship("Portfolio", back_populates="deals")
    #product = relationship("Product", foreign_keys=[product_id])
    product_sold = relationship("Product", foreign_keys=[product_id_sold])  # Relation pour produit vendu
    product_bought = relationship("Product", foreign_keys=[product_id_bought])  # Relation pour produit acheté


# Fonction pour créer la base de données et les tables si elles n'existent pas
def create_database():
    engine = create_engine('sqlite:///Fund.db')
    Base.metadata.create_all(engine)

    # Inspect the database to check if tables exist
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # Create tables only if they do not exist
    if 'clients' not in tables:
        Base.metadata.create_all(engine)
