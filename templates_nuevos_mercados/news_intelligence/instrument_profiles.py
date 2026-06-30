"""
Perfiles de instrumentos para el sistema de news intelligence del portfolio.
Define keywords de búsqueda, drivers relevantes y umbrales de alerta por instrumento.
"""

INSTRUMENT_PROFILES = {
    "BTCUSD": {
        "name": "Bitcoin",
        "keywords": [
            "bitcoin", "BTC", "crypto", "cryptocurrency", "blockchain",
            "SEC crypto", "ETF bitcoin", "crypto regulation", "coinbase",
            "binance", "stablecoin", "USDT", "tether", "crypto exchange",
            "digital asset", "crypto ban", "El Salvador bitcoin",
        ],
        "drivers": [
            "regulation", "macro_usd", "institutional_adoption",
            "on_chain", "exchange_flows", "geopolitics", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en criptomonedas (Bitcoin/BTCUSD). "
            "Evaluá si esta noticia impacta el precio de BTC: regulación cripto, "
            "adopción institucional, movimientos de ballenas, correlación con riesgo macro, "
            "aprobaciones ETF, sanciones o prohibiciones."
        ),
        "magnitude_threshold": 3,   # solo alertar si magnitud >= este valor
        "confidence_threshold": 0.5,
    },
    "ETHUSD": {
        "name": "Ethereum",
        "keywords": [
            "ethereum", "ETH", "ether", "defi", "NFT", "smart contract",
            "EIP", "ethereum upgrade", "layer 2", "staking ethereum",
            "crypto regulation", "SEC ethereum", "altcoin",
        ],
        "drivers": [
            "regulation", "defi", "upgrade", "macro_usd",
            "institutional_adoption", "geopolitics", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en Ethereum (ETHUSD). "
            "Evaluá el impacto en ETH: upgrades de red, DeFi, regulación, "
            "adopción institucional, correlación con BTC, upgrades EIP."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.5,
    },
    "XAUUSD": {
        "name": "Gold / Oro",
        "keywords": [
            "gold price", "gold rally", "precious metals", "XAU", "bullion",
            "Fed interest rates", "inflation", "CPI", "USD weakness",
            "safe haven", "geopolitical risk", "central bank gold",
            "guerra", "war", "conflict", "tension", "Iran", "Russia",
            "treasury yields", "real yields", "gold ETF",
        ],
        "drivers": [
            "macro_usd", "inflation", "geopolitics", "central_bank",
            "real_yields", "safe_haven", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en oro (XAUUSD). "
            "Evaluá el impacto en el precio del oro: datos de inflación (CPI/PPI), "
            "política Fed (tasas reales), tensiones geopolíticas, compras de bancos centrales, "
            "debilidad del USD, demanda de refugio seguro."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.55,
    },
    "BRENT_N6": {
        "name": "Brent Crude Oil",
        "keywords": [
            "brent crude", "oil price", "OPEC", "OPEC+", "crude oil",
            "Iran sanctions", "Strait of Hormuz", "Saudi Arabia oil",
            "Russia oil", "oil supply", "IEA", "EIA crude",
            "oil demand", "refinery", "petroleum", "energy crisis",
        ],
        "drivers": [
            "opec", "geopolitics", "supply_disruption", "macro_demand",
            "sanctions", "inventory", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en petróleo Brent (BRENT). "
            "Evaluá el impacto en Brent: decisiones OPEC+, tensiones Medio Oriente, "
            "sanciones a Rusia/Irán, datos de inventarios EIA, demanda China, "
            "cierre de rutas marítimas."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.55,
    },
    "WTI_N6": {
        "name": "WTI Crude Oil",
        "keywords": [
            "WTI crude", "oil price", "OPEC", "US crude", "shale oil",
            "EIA crude inventory", "oil demand", "petroleum", "refinery",
            "crude oil supply", "energy", "oil sanctions",
        ],
        "drivers": [
            "opec", "geopolitics", "us_inventory", "macro_demand",
            "shale_production", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en petróleo WTI (US crude). "
            "Evaluá el impacto: inventarios EIA, producción shale, OPEC+, "
            "demanda industrial USA, tensiones geopolíticas."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.55,
    },
    "UK100": {
        "name": "FTSE 100",
        "keywords": [
            "FTSE 100", "UK economy", "Bank of England", "BOE",
            "UK inflation", "UK GDP", "British pound", "GBP",
            "UK interest rates", "UK stocks", "London market",
            "Brexit", "UK trade", "UK recession", "FTSE",
            "UK budget", "chancellor", "UK employment",
        ],
        "drivers": [
            "boe_policy", "uk_macro", "gbp", "global_risk",
            "energy_prices", "geopolitics", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en el FTSE 100 (UK100). "
            "Evaluá el impacto: política del Bank of England, datos macro UK (GDP, CPI, empleo), "
            "movimiento del GBP, riesgo global que afecta Londres, precios energéticos, "
            "ganancias corporativas de grandes compañías FTSE."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.5,
    },
    "US30": {
        "name": "Dow Jones Industrial Average",
        "keywords": [
            "Dow Jones", "DJIA", "US stocks", "Fed rate", "Federal Reserve",
            "US GDP", "US inflation", "CPI", "jobs report", "NFP",
            "earnings season", "US economy", "recession", "S&P 500",
            "wall street", "US trade war", "tariff",
        ],
        "drivers": [
            "fed_policy", "us_macro", "earnings", "trade_policy",
            "global_risk", "geopolitics", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en el Dow Jones (US30). "
            "Evaluá el impacto: política Fed (tasas, QE), datos macro USA (NFP, CPI, GDP), "
            "temporada de ganancias corporativas, aranceles y guerra comercial, "
            "riesgo geopolítico global."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.5,
    },
    "US500": {
        "name": "S&P 500",
        "keywords": [
            "S&P 500", "SPX", "US stocks", "Fed rate", "Federal Reserve",
            "US GDP", "CPI", "jobs report", "NFP", "earnings",
            "tech stocks", "US economy", "recession", "risk-off",
            "wall street", "interest rates", "quantitative easing",
        ],
        "drivers": [
            "fed_policy", "us_macro", "earnings", "tech_sector",
            "global_risk", "geopolitics", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en el S&P 500 (US500). "
            "Evaluá el impacto: política Fed, datos macro USA, ganancias tech, "
            "CPI/NFP, movimientos de riesgo global."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.5,
    },
    "USTEC": {
        "name": "Nasdaq 100",
        "keywords": [
            "Nasdaq", "tech stocks", "AI", "artificial intelligence",
            "Apple", "Microsoft", "Google", "Amazon", "Meta", "Nvidia",
            "Fed rate", "interest rates", "tech earnings",
            "semiconductor", "chip", "big tech", "growth stocks",
        ],
        "drivers": [
            "fed_policy", "tech_earnings", "ai_sector", "semiconductor",
            "us_macro", "geopolitics", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en el Nasdaq 100 (USTEC). "
            "Evaluá el impacto: ganancias de grandes tech (Apple/MS/Google/Nvidia), "
            "política Fed (tasas afectan más a growth), regulación tech, avances AI, "
            "chip bans o restricciones de semiconductores."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.5,
    },
    "HK50": {
        "name": "Hang Seng 50",
        "keywords": [
            "Hang Seng", "Hong Kong stocks", "China economy", "PBOC",
            "China GDP", "China stimulus", "Alibaba", "Tencent",
            "China tech regulation", "Hong Kong", "yuan", "CNY",
            "US-China trade", "Taiwan tension", "China property",
            "Evergrande", "China exports",
        ],
        "drivers": [
            "china_macro", "pboc_policy", "us_china_tension",
            "china_tech_regulation", "hk_political", "geopolitics", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en el Hang Seng (HK50). "
            "Evaluá el impacto: política PBOC, datos macro China (GDP, PMI, exportaciones), "
            "regulación tech china (Alibaba, Tencent), tensiones US-China/Taiwan, "
            "sector inmobiliario chino, estímulos fiscales."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.5,
    },
    "Corn_N6": {
        "name": "Corn / Maíz CBOT",
        "keywords": [
            "corn price", "maiz", "CBOT corn", "corn crop", "USDA corn",
            "corn demand", "ethanol", "corn export", "Ukraine corn",
            "Brazil corn", "Argentina corn", "corn weather",
            "corn planting", "corn harvest", "grain",
        ],
        "drivers": [
            "usda_report", "weather_us", "weather_br", "weather_ar",
            "ethanol_demand", "china_demand", "supply_global", "geopolitics", "other"
        ],
        "sentiment_prompt": (
            "Sos un analista especializado en maíz CBOT (Corn). "
            "Evaluá el impacto: reportes USDA (oferta/demanda), clima en el Corn Belt USA, "
            "producción Brazil/Argentina, demanda China, demanda de etanol, "
            "exportaciones desde Ucrania."
        ),
        "magnitude_threshold": 3,
        "confidence_threshold": 0.5,
    },
}
