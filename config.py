


assets = {
    "Equity": {
        "JNJ": ["low_risk", "low_turnover"],
        "PG": ["low_risk"],
        "KO": ["low_risk"],
        "WMT": ["low_risk"],
        "MCD": ["low_risk"],
        "MSFT":["low_turnover"],
        "AWK": ["low_turnover"],
        "O": ["high_yield"],
        "T": ["high_yield"],
        "VZ": ["high_yield"],
        "PFE": ["high_yield"],
        "NVDA": ["high_yield"],
        "NRG": ["high_yield"],
        "MO": ["high_yield"],
        "XOM": ["high_yield"],
        "JNJ": ["high_yield"],
        "PG": ["high_yield"]
    },
    "ETF": {
        "USMV": ["low_risk"],
        "URTH": ["low_risk"],
        "SPLV": ["low_risk"],
        "XLP": ["low_risk", "low_turnover"],
        "SPY" : ["low_turnover"],
        "VIG": ["low_risk", "low_turnover"],
        "GLD" : ["low_turnover"],
        "VNQ": ["low_turnover"],
        "VYM": ["low_turnover"],
        "SCHD": ["low_turnover"],
        "XLV": ["low_turnover"]
    },
    "Bond": {
        "TIP": ["low_risk"],
        "BNDX": ["low_risk"]
    }
}


LR_assets = [
    ticker for category in assets.values()  # Parcours les catégories (Equity, ETF, Bond)
    for ticker, tags in category.items()   # Parcours les actifs et leurs tags
    if "low_risk" in tags                  # Garde ceux qui ont "low_risk"
]

HY_assets = [
    ticker for category in assets.values()  # Parcours les catégories (Equity, ETF, Bond)
    for ticker, tags in category.items()   # Parcours les actifs et leurs tags
    if "high_yield" in tags                  # Garde ceux qui ont "high_yield"
]

LT_assets = [
    ticker for category in assets.values()  # Parcours les catégories (Equity, ETF, Bond)
    for ticker, tags in category.items()   # Parcours les actifs et leurs tags
    if "low_turnover" in tags                  # Garde ceux qui ont "low_turnover"
]

