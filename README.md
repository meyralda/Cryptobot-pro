"""
7-Parameter Scoring System
==========================
Her parametre [0, 100] aralığında bir puan döndürür.
Nihai skor, tüm yedi parametrenin ağırlıklı ortalamasıdır.

Tüm ağırlıklar ve eşik değerleri bot/core/config.py üzerinden yönetilir.
bot/.env dosyasına değerler ekleyerek her şeyi özelleştirebilirsiniz.

Parametreler
------------
1. Trend Gücü       — EMA hizalaması (kısa/orta/uzun vade)
2. Momentum         — RSI aşırı alım/satım konumu
3. Hacim Baskısı    — Anlık hacim vs. hareketli ortalama hacim
4. Emir Defteri     — Bid/ask USDT derinliği oranı
5. Volatilite Riski — ATR yüzde fiyat hareketi (düşük = güvenli)
6. Fiyat Akışı      — Destek/direnç yakınlığı
7. Piyasa Duyarlılığı — Fear & Greed Index (alternative.me)
"""

import numpy as np
import logging
from typing import Optional
from bot.core.config import cfg

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  1. Trend Gücü
# ────────────────────────────────────────────────────────────────────

def score_trend_strength(closes: list) -> float:
    """
    EMA kısa/orta/uzun hizalaması.
    Tüm EMA'lar yükseliş düzenindeyse → 100; düşüş düzenindeyse → 0.
    EMA periyotları: EMA_SHORT, EMA_MID, EMA_LONG (bot/.env)
    """
    period_long = cfg.ema_long
    if len(closes) < period_long:
        return 50.0
    c = np.array(closes, dtype=float)
    e_short = _ema(c, cfg.ema_short)[-1]
    e_mid   = _ema(c, cfg.ema_mid)[-1]
    e_long  = _ema(c, cfg.ema_long)[-1]
    price   = c[-1]
    score = 0
    if price   > e_short: score += 25
    if e_short > e_mid:   score += 25
    if e_mid   > e_long:  score += 25
    if price   > e_long:  score += 25
    return float(score)


# ────────────────────────────────────────────────────────────────────
#  2. Momentum (RSI)
# ────────────────────────────────────────────────────────────────────

def score_momentum(closes: list) -> float:
    """
    RSI(period). Aşırı satım → yüksek puan, aşırı alım → düşük puan.
    Eşikler: RSI_OVERSOLD, RSI_LOW, RSI_HIGH, RSI_OVERBOUGHT (bot/.env)
    """
    period = cfg.rsi_period
    if len(closes) < period + 1:
        return 50.0
    rsi = _rsi(np.array(closes, dtype=float), period)
    if rsi < cfg.rsi_oversold:
        return 90.0
    elif rsi < cfg.rsi_low:
        return 75.0
    elif rsi < cfg.rsi_high:
        return 50.0
    elif rsi < cfg.rsi_overbought:
        return 35.0
    else:
        return 10.0


# ────────────────────────────────────────────────────────────────────
#  3. Hacim Baskısı
# ────────────────────────────────────────────────────────────────────

def score_volume_pressure(volumes: list) -> float:
    """
    Anlık hacim / son N-bar ortalaması. Yüksek oran → güçlü ilgi → yüksek puan.
    Pencere: VOLUME_WINDOW; çarpan: VOLUME_SURGE_FACTOR (bot/.env)
    """
    window = cfg.volume_window
    if len(volumes) < window + 1:
        return 50.0
    vols = np.array(volumes, dtype=float)
    avg_vol = vols[-window - 1:-1].mean()
    if avg_vol == 0:
        return 50.0
    ratio = vols[-1] / avg_vol
    score = min(100.0, (ratio / cfg.volume_surge_factor) * 100)
    return float(score)


# ────────────────────────────────────────────────────────────────────
#  4. Emir Defteri Derinliği
# ────────────────────────────────────────────────────────────────────

def score_order_book_depth(order_book: dict) -> float:
    """
    Bid-side USDT derinliği / (bid + ask) toplam derinliği.
    Bid ağırlığı yüksekse alıcılar baskın → yüksek puan.
    Ağırlık: WEIGHT_ORDER_BOOK_DEPTH (bot/.env)
    """
    try:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        bid_depth = sum(p * q for p, q in bids)
        ask_depth = sum(p * q for p, q in asks)
        total = bid_depth + ask_depth
        if total == 0:
            return 50.0
        return float((bid_depth / total) * 100)
    except Exception:
        return 50.0


# ────────────────────────────────────────────────────────────────────
#  5. Volatilite Riski (ATR)
# ────────────────────────────────────────────────────────────────────

def score_volatility_risk(highs: list, lows: list, closes: list) -> float:
    """
    ATR(period) fiyatın yüzdesi olarak hesaplanır.
    Düşük ATR% → güvenli giriş → yüksek puan.
    Bantlar: ATR_LOW_PCT / ATR_MID_PCT / ATR_HIGH_PCT / ATR_MAX_PCT (bot/.env)
    """
    period = cfg.atr_period
    if len(closes) < period + 1:
        return 50.0
    h  = np.array(highs,  dtype=float)
    lo = np.array(lows,   dtype=float)
    c  = np.array(closes, dtype=float)
    tr = np.maximum(h[1:] - lo[1:],
         np.maximum(abs(h[1:] - c[:-1]),
                    abs(lo[1:] - c[:-1])))
    atr     = tr[-period:].mean()
    atr_pct = (atr / c[-1]) * 100
    if atr_pct < cfg.atr_low_pct:  return 90.0
    elif atr_pct < cfg.atr_mid_pct:  return 70.0
    elif atr_pct < cfg.atr_high_pct: return 50.0
    elif atr_pct < cfg.atr_max_pct:  return 30.0
    else:                            return 10.0


# ────────────────────────────────────────────────────────────────────
#  6. Fiyat Akışı (Destek / Direnç)
# ────────────────────────────────────────────────────────────────────

def score_price_action(closes: list) -> float:
    """
    Son N bar içindeki destek (min) ve direnç (max) seviyelerine yakınlık.
    Fiyat desteğe yakınsa → iyi giriş fırsatı → yüksek puan.
    Pencere: PRICE_ACTION_WINDOW (bot/.env)
    """
    window = cfg.price_action_window
    if len(closes) < window:
        return 50.0
    c = np.array(closes, dtype=float)
    recent    = c[-window:]
    support   = recent.min()
    resistance = recent.max()
    price     = c[-1]
    rang      = resistance - support
    if rang == 0:
        return 50.0
    proximity_to_support = 1 - (price - support) / rang
    return float(proximity_to_support * 100)


# ────────────────────────────────────────────────────────────────────
#  7. Piyasa Duyarlılığı (Fear & Greed Index)
# ────────────────────────────────────────────────────────────────────

def score_market_sentiment(symbol: str = "", timestamp_ms: Optional[int] = None) -> float:
    """
    alternative.me Fear & Greed Index — 0-100 arası duyarlılık endeksi.

    Canlı mod  (timestamp_ms=None):
        Günün endeks değeri API'den çekilir, 1 saat bellekte önbelleğe alınır.

    Backtest modu (timestamp_ms verilmişse):
        O UTC tarihine ait tarihsel F&G değeri kullanılır (ileriye bakış yok).

    Zıt strateji (varsayılan, SENTIMENT_CONTRARIAN=true):
        Aşırı Korku (0–24)  → 80–100 puan  (alım fırsatı)
        Aşırı Açgözlülük (75+) → 0–25 puan (dikkat)

    Trend takip (SENTIMENT_CONTRARIAN=false):
        Doğrudan eşleme — yüksek endeks → yüksek puan.

    Ağırlık: WEIGHT_MARKET_SENTIMENT (bot/.env)
    """
    try:
        from bot.core.sentiment import get_live_fng, get_historical_fng, fng_to_score
        raw = get_historical_fng(timestamp_ms) if timestamp_ms is not None else get_live_fng()
        return fng_to_score(raw)
    except Exception as e:
        logger.warning(f"Duyarlılık skoru hesaplanamadı: {e} — nötr değer (50) kullanılıyor")
        return 50.0


# ────────────────────────────────────────────────────────────────────
#  Bileşik Skor
# ────────────────────────────────────────────────────────────────────

class Scorer:
    """
    7 parametreli bileşik skor hesaplayıcı.
    Tüm parametreler [0, 100] aralığına normalleştirilir, ardından ağırlıklandırılır.
    Ağırlıklar ve eşikler bot/core/config.py üzerinden gelir.
    """

    def score(
        self,
        symbol: str,
        ohlcv: list,
        order_book: Optional[dict] = None,
        timestamp_ms: Optional[int] = None,
    ) -> dict:
        """
        ohlcv        : [timestamp, open, high, low, close, volume] listesi
        timestamp_ms : Backtest için UTC zaman damgası. None ise canlı F&G kullanılır.

        Dönüş: bireysel skorlar, ağırlıklar ve bileşik skoru içeren dict.
        """
        if not ohlcv:
            return {"composite": 0.0, "error": "OHLCV verisi yok"}

        highs   = [row[2] for row in ohlcv]
        lows    = [row[3] for row in ohlcv]
        closes  = [row[4] for row in ohlcv]
        volumes = [row[5] for row in ohlcv]

        scores = {
            "trend_strength":   score_trend_strength(closes),
            "momentum":         score_momentum(closes),
            "volume_pressure":  score_volume_pressure(volumes),
            "order_book_depth": score_order_book_depth(order_book or {}),
            "volatility_risk":  score_volatility_risk(highs, lows, closes),
            "price_action":     score_price_action(closes),
            "market_sentiment": score_market_sentiment(symbol, timestamp_ms=timestamp_ms),
        }

        weights   = cfg.weights
        composite = sum(scores[k] * weights[k] for k in scores)

        return {
            "symbol":     symbol,
            "composite":  round(composite, 2),
            "parameters": {k: round(v, 2) for k, v in scores.items()},
            "weights":    weights,
        }


# ────────────────────────────────────────────────────────────────────
#  Teknik gösterge yardımcıları
# ────────────────────────────────────────────────────────────────────

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    k   = 2 / (period + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = data[i] * k + ema[i - 1] * (1 - k)
    return ema


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0)
    losses   = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))
