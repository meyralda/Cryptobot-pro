import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # EMA Periyotları
    ema_short = int(os.getenv("EMA_SHORT", 9))
    ema_mid   = int(os.getenv("EMA_MID", 21))
    ema_long  = int(os.getenv("EMA_LONG", 50))

    # RSI Ayarları
    rsi_period    = int(os.getenv("RSI_PERIOD", 14))
    rsi_oversold  = float(os.getenv("RSI_OVERSOLD", 30))
    rsi_low       = float(os.getenv("RSI_LOW", 40))
    rsi_high      = float(os.getenv("RSI_HIGH", 60))
    rsi_overbought = float(os.getenv("RSI_OVERBOUGHT", 70))

    # Hacim Ayarları
    volume_window = int(os.getenv("VOLUME_WINDOW", 20))
    volume_surge_factor = float(os.getenv("VOLUME_SURGE_FACTOR", 2.0))

    # ATR Ayarları
    atr_period = int(os.getenv("ATR_PERIOD", 14))
    atr_low_pct = float(os.getenv("ATR_LOW_PCT", 0.5))
    atr_mid_pct = float(os.getenv("ATR_MID_PCT", 1.0))
    atr_high_pct = float(os.getenv("ATR_HIGH_PCT", 2.0))
    atr_max_pct = float(os.getenv("ATR_MAX_PCT", 3.0))

    # Fiyat Akışı (Price Action)
    price_action_window = int(os.getenv("PRICE_ACTION_WINDOW", 20))

    # Ağırlıklar (Toplamı 1.0 olmalı)
    weights = {
        "trend_strength": 0.20,
        "momentum": 0.15,
        "volume_pressure": 0.15,
        "order_book_depth": 0.10,
        "volatility_risk": 0.10,
        "price_action": 0.15,
        "market_sentiment": 0.15
    }

cfg = Config()
