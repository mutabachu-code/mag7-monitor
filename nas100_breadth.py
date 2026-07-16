"""
nas100_breadth.py
-----------------
NAS100 Internal Breadth Engine.

Fetches all ~100 NAS100 component stocks in ONE yfinance batch call,
caches the result for 15 minutes, then computes institutional-grade
breadth metrics used to build the Harmonized Final Signal.

Rate-limit strategy:
  - yf.download(all_tickers, period='30d', interval='1d') = 1 HTTP request
  - 15-minute cache means max 4 fetches/hour regardless of refresh rate
  - Graceful partial-data handling: any ticker that fails is skipped silently

Metrics computed:
  1. % Above SMA50        — classic breadth (>60% = healthy, <40% = weak)
  2. % Above SMA20        — short-term momentum breadth
  3. Advance/Decline      — % with positive 5-day return
  4. Volume Confirmation  — % with above-average volume today
  5. RSI Distribution     — overbought / healthy / oversold breakdown
  6. Sector Rotation      — which NAS100 sectors are leading vs lagging
  7. Leaders / Laggards   — top 5 and worst 5 by 5-day return
  8. Harmonized Score     — 0-100 composite breadth conviction score

The Harmonized Final Signal combines this with:
  - Master signal conviction score (from master_signal.py)
  - Regime state
  - Options Intelligence (GEX, expected move)
  - Macro risk score
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ── NAS100 COMPONENTS ─────────────────────────────────────────────────────────
# Current NAS100 top components by weight (as of 2025)
# Grouped by sector for rotation analysis
NAS100_COMPONENTS = {
    "Mega Cap Tech": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    ],
    "Software": [
        "ADBE", "CRM", "ORCL", "PANW", "CRWD", "SNOW", "DDOG", "ZS", "TEAM",
        "WDAY", "FTNT", "ANSS", "CDNS", "MNST",
    ],
    "Semiconductors": [
        "AMD", "QCOM", "INTC", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON",
        "MCHP", "TXN", "NXPI", "MPWR",
    ],
    "Consumer Internet": [
        "NFLX", "ABNB", "BKNG", "EBAY", "MTCH", "ZM", "DOCU",
    ],
    "Biotech / Health": [
        "AMGN", "GILD", "VRTX", "REGN", "IDXX", "DXCM", "ILMN", "MRNA",
        "BIIB", "SGEN",
    ],
    "Consumer / Retail": [
        "COST", "SBUX", "MDLZ", "PEP", "MNST", "ODFL", "PCAR",
    ],
    "EV / Auto": [
        "TSLA",  # already in Mega Cap but also sector rep
    ],
    "Cloud / Data": [
        "MSFT",  # Azure, duplicate intentional for sector signal
        "AMZN",  # AWS
        "GOOGL", # GCP
        "NET", "AKAM", "FANG", "FAST",
    ],
    "Financial Tech": [
        "PYPL", "PAYX", "FISV", "CINF",
    ],
    "Hardware / Other": [
        "AAPL", "HPQ", "STX", "WDC", "NTAP",
    ],
}

# Flat unique list — deduplicated
NAS100_ALL = sorted(set(
    t for tickers in NAS100_COMPONENTS.values() for t in tickers
))

# Sector map for each ticker (first sector wins for deduplication)
_SECTOR_MAP: Dict[str, str] = {}
for sector, tickers in NAS100_COMPONENTS.items():
    for t in tickers:
        if t not in _SECTOR_MAP:
            _SECTOR_MAP[t] = sector


CACHE_TTL = 900   # 15 minutes


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class ComponentData:
    ticker: str
    sector: str
    price: float
    ret_1d: float      # 1-day return %
    ret_5d: float      # 5-day return %
    ret_20d: float     # 20-day return %
    above_sma20: bool
    above_sma50: bool
    rsi: float
    vol_ratio: float   # today vol / 20d avg vol
    atr_pct: float     # daily ATR as % of price


@dataclass
class SectorBreadth:
    sector: str
    ticker_count: int
    pct_above_sma50: float
    avg_5d_return: float
    avg_vol_ratio: float
    signal: str        # "LEADING" | "NEUTRAL" | "LAGGING"
    signal_color: str


@dataclass
class NAS100BreadthReport:
    # Core breadth metrics
    tickers_analyzed: int
    tickers_failed: int

    pct_above_sma50: float      # % of components above 50-day SMA
    pct_above_sma20: float      # % of components above 20-day SMA
    advance_decline: float      # % with positive 5d return
    vol_confirmation_pct: float # % with above-average volume
    pct_rsi_overbought: float   # % with RSI > 70
    pct_rsi_healthy: float      # % with RSI 40-60
    pct_rsi_oversold: float     # % with RSI < 30

    # Breadth assessment
    breadth_score: int          # 0-100 composite
    breadth_label: str          # "STRONG" | "HEALTHY" | "MIXED" | "WEAK" | "BEARISH"
    breadth_color: str

    # Leaders & laggards
    leaders: List[ComponentData]    # top 5 by 5d return
    laggards: List[ComponentData]   # worst 5 by 5d return
    most_active: List[ComponentData]# top 5 by volume ratio

    # Sector rotation
    sector_data: List[SectorBreadth]
    leading_sector: str
    lagging_sector: str

    # Harmonized signal inputs
    breadth_trend_bias: str     # "BULLISH" | "NEUTRAL" | "BEARISH"
    sell_off_confirmed: bool    # True if >70% declining on elevated volume
    rally_confirmed: bool       # True if >70% advancing on elevated volume
    divergence_warning: bool    # Index up but breadth weak

    # Meta
    data_as_of: str             # timestamp of last fetch
    cache_age_mins: float


@dataclass
class HarmonizedSignal:
    """
    The single final signal combining ALL dashboard layers + NAS100 breadth.
    Resolves conflicts between layers and outputs one clear actionable view.
    """
    # Final verdict
    direction: str          # "STRONG LONG" | "LONG" | "NEUTRAL" | "SHORT" | "STRONG SHORT" | "AVOID"
    conviction_pct: int     # 0-100
    action: str             # "BUY NOW" | "WAIT FOR DIP" | "HOLD" | "SELL NOW" | "WAIT FOR BOUNCE" | "STAY OUT"

    # Score breakdown
    master_score: int           # from master_signal (-100 to +100)
    breadth_score: int          # from nas100_breadth (0-100, centered at 50)
    conflict_detected: bool     # True if master signal and breadth disagree
    conflict_reason: str

    # Price guidance
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Context
    key_risks: List[str]
    key_confirmations: List[str]
    summary: str
    lot_multiplier: float


# ── BATCH FETCH ───────────────────────────────────────────────────────────────

def _fetch_components_batch() -> Optional[pd.DataFrame]:
    """
    Single yf.download() call for all NAS100 components.
    Returns MultiIndex DataFrame: columns = (ticker, field).
    """
    try:
        print(f"[nas100_breadth] Fetching {len(NAS100_ALL)} components (batch)...")
        t0  = time.time()
        df  = yf.download(
            NAS100_ALL,
            period="30d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=25,
        )
        elapsed = time.time() - t0
        print(f"[nas100_breadth] Batch complete in {elapsed:.1f}s, shape={df.shape}")
        return df if not df.empty else None
    except Exception as e:
        print(f"[nas100_breadth] Batch fetch error: {e}")
        return None


def _extract_ticker(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Safely extract single-ticker DataFrame from batch MultiIndex result."""
    try:
        if ticker in df.columns:
            sub = df[ticker].copy()
        elif (ticker, "Close") in df.columns:
            sub = df.xs(ticker, axis=1, level=0).copy()
        else:
            return None
        sub = sub.dropna(subset=["Close"])
        return sub if len(sub) >= 10 else None
    except Exception:
        return None


# ── COMPONENT ANALYSIS ────────────────────────────────────────────────────────

def _rsi14(series: pd.Series) -> float:
    try:
        delta = series.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 1e-10)
        return float((100 - 100 / (1 + rs)).iloc[-1])
    except Exception:
        return 50.0


def _analyse_component(df: pd.DataFrame, ticker: str) -> Optional[ComponentData]:
    """Compute all metrics for one component from its daily OHLCV data."""
    try:
        closes = df["Close"].dropna()
        vols   = df["Volume"].dropna() if "Volume" in df.columns else None

        if len(closes) < 21:
            return None

        price   = float(closes.iloc[-1])
        sma20   = float(closes.rolling(20).mean().iloc[-1])
        sma50   = float(closes.rolling(min(50, len(closes))).mean().iloc[-1])

        ret_1d  = float(closes.pct_change(1).iloc[-1])  * 100
        ret_5d  = float(closes.pct_change(5).iloc[-1])  * 100 if len(closes) >= 6  else 0.0
        ret_20d = float(closes.pct_change(20).iloc[-1]) * 100 if len(closes) >= 21 else 0.0

        rsi = _rsi14(closes)

        # Volume ratio
        vol_ratio = 1.0
        if vols is not None and len(vols) >= 6:
            vol_avg   = float(vols.rolling(20).mean().iloc[-1])
            vol_today = float(vols.iloc[-1])
            vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1.0

        # ATR%
        atr_pct = 0.01
        if "High" in df.columns and "Low" in df.columns:
            highs  = df["High"].dropna().values[-5:]
            lows   = df["Low"].dropna().values[-5:]
            ranges = highs - lows
            atr    = float(np.mean(ranges)) if len(ranges) > 0 else price * 0.01
            atr_pct = atr / price if price > 0 else 0.01

        return ComponentData(
            ticker=ticker,
            sector=_SECTOR_MAP.get(ticker, "Other"),
            price=round(price, 2),
            ret_1d=round(ret_1d, 2),
            ret_5d=round(ret_5d, 2),
            ret_20d=round(ret_20d, 2),
            above_sma20=(price > sma20),
            above_sma50=(price > sma50),
            rsi=round(rsi, 1),
            vol_ratio=round(vol_ratio, 2),
            atr_pct=round(atr_pct * 100, 2),
        )
    except Exception as e:
        print(f"[nas100_breadth] {ticker} analysis error: {e}")
        return None


# ── SECTOR AGGREGATION ────────────────────────────────────────────────────────

def _compute_sector_breadth(components: List[ComponentData]) -> List[SectorBreadth]:
    sectors: Dict[str, List[ComponentData]] = {}
    for c in components:
        sectors.setdefault(c.sector, []).append(c)

    result = []
    for sector, comps in sectors.items():
        if not comps:
            continue
        pct_above = sum(1 for c in comps if c.above_sma50) / len(comps) * 100
        avg_ret   = float(np.mean([c.ret_5d for c in comps]))
        avg_vol   = float(np.mean([c.vol_ratio for c in comps]))

        if pct_above >= 65 and avg_ret > 1.0:
            signal = "LEADING"
            color  = "#2d9e2d"
        elif pct_above <= 35 or avg_ret < -1.0:
            signal = "LAGGING"
            color  = "#c9302c"
        else:
            signal = "NEUTRAL"
            color  = "#e6a817"

        result.append(SectorBreadth(
            sector=sector,
            ticker_count=len(comps),
            pct_above_sma50=round(pct_above, 1),
            avg_5d_return=round(avg_ret, 2),
            avg_vol_ratio=round(avg_vol, 2),
            signal=signal,
            signal_color=color,
        ))

    return sorted(result, key=lambda s: s.avg_5d_return, reverse=True)


# ── BREADTH SCORE ─────────────────────────────────────────────────────────────

def _compute_breadth_score(
    pct_sma50: float, pct_sma20: float, adv_dec: float,
    vol_conf: float, pct_ob: float, pct_os: float,
) -> Tuple[int, str, str]:
    """Composite breadth score 0-100."""
    score = 0

    # SMA50 (35 pts)
    if pct_sma50 >= 70:   score += 35
    elif pct_sma50 >= 55: score += 25
    elif pct_sma50 >= 45: score += 15
    elif pct_sma50 >= 30: score += 5
    else:                  score += 0

    # Advance/Decline (30 pts)
    if adv_dec >= 70:   score += 30
    elif adv_dec >= 55: score += 22
    elif adv_dec >= 45: score += 12
    elif adv_dec >= 30: score += 5
    else:               score += 0

    # Volume confirmation (20 pts)
    if vol_conf >= 60:   score += 20
    elif vol_conf >= 45: score += 13
    elif vol_conf >= 30: score += 6
    else:                score += 0

    # RSI health (15 pts) — overbought = fragile, oversold = bearish
    if pct_ob >= 40:     score += 2    # overbought = fragile
    elif pct_os >= 40:   score += 3    # oversold = possible bounce but bearish
    elif pct_ob < 20 and pct_os < 20:
        score += 15                    # healthy RSI distribution

    score = max(0, min(100, score))

    if score >= 72:
        return score, "STRONG", "#2d9e2d"
    elif score >= 58:
        return score, "HEALTHY", "#5cb85c"
    elif score >= 42:
        return score, "MIXED", "#e6a817"
    elif score >= 28:
        return score, "WEAK", "#c9302c"
    else:
        return score, "BEARISH", "#8b0000"


# ── MAIN BREADTH REPORT ───────────────────────────────────────────────────────

def get_nas100_breadth() -> Optional[NAS100BreadthReport]:
    """
    Main entry point. Returns cached report if fresh, else fetches and computes.
    Safe to call on every 60s refresh — fetches at most every 15 minutes.
    """
    # Check cache
    cache_ts = st.session_state.get("nas100_breadth_ts", 0)
    age      = time.time() - cache_ts
    if age < CACHE_TTL:
        cached = st.session_state.get("nas100_breadth")
        if cached:
            cached.cache_age_mins = round(age / 60, 1)
            return cached

    # Fetch
    batch_df = _fetch_components_batch()
    if batch_df is None or batch_df.empty:
        # Return stale cache rather than nothing
        stale = st.session_state.get("nas100_breadth")
        if stale:
            stale.cache_age_mins = round(age / 60, 1)
            return stale
        return None

    # Analyse each component
    components: List[ComponentData] = []
    failed = 0
    for ticker in NAS100_ALL:
        sub = _extract_ticker(batch_df, ticker)
        if sub is None:
            failed += 1
            continue
        comp = _analyse_component(sub, ticker)
        if comp:
            components.append(comp)
        else:
            failed += 1

    if len(components) < 10:
        print(f"[nas100_breadth] Too few components ({len(components)}) — aborting")
        return st.session_state.get("nas100_breadth")

    n = len(components)

    # ── BREADTH METRICS ───────────────────────────────────────────────────────
    pct_sma50  = sum(1 for c in components if c.above_sma50) / n * 100
    pct_sma20  = sum(1 for c in components if c.above_sma20) / n * 100
    adv_dec    = sum(1 for c in components if c.ret_5d > 0)  / n * 100
    vol_conf   = sum(1 for c in components if c.vol_ratio > 1.1) / n * 100
    pct_ob     = sum(1 for c in components if c.rsi > 70)  / n * 100
    pct_os     = sum(1 for c in components if c.rsi < 30)  / n * 100
    pct_hlth   = sum(1 for c in components if 40 <= c.rsi <= 60) / n * 100

    breadth_score, breadth_label, breadth_color = _compute_breadth_score(
        pct_sma50, pct_sma20, adv_dec, vol_conf, pct_ob, pct_os
    )

    # ── LEADERS / LAGGARDS ───────────────────────────────────────────────────
    by_ret  = sorted(components, key=lambda c: c.ret_5d, reverse=True)
    leaders  = by_ret[:5]
    laggards = by_ret[-5:]
    most_active = sorted(components, key=lambda c: c.vol_ratio, reverse=True)[:5]

    # ── SECTOR ROTATION ───────────────────────────────────────────────────────
    sector_data   = _compute_sector_breadth(components)
    leading_sec   = sector_data[0].sector  if sector_data else "N/A"
    lagging_sec   = sector_data[-1].sector if sector_data else "N/A"

    # ── TREND BIAS ────────────────────────────────────────────────────────────
    if pct_sma50 >= 60 and adv_dec >= 60:
        trend_bias = "BULLISH"
    elif pct_sma50 <= 40 or adv_dec <= 35:
        trend_bias = "BEARISH"
    else:
        trend_bias = "NEUTRAL"

    # ── SELL-OFF / RALLY CONFIRMATION ─────────────────────────────────────────
    sell_off_confirmed = (adv_dec <= 25 and vol_conf >= 50)
    rally_confirmed    = (adv_dec >= 70 and vol_conf >= 45)

    # ── DIVERGENCE WARNING ────────────────────────────────────────────────────
    # Index at high but <45% of components above SMA50 = narrow leadership
    divergence_warning = (pct_sma50 < 45 and adv_dec < 45)

    report = NAS100BreadthReport(
        tickers_analyzed=n,
        tickers_failed=failed,
        pct_above_sma50=round(pct_sma50, 1),
        pct_above_sma20=round(pct_sma20, 1),
        advance_decline=round(adv_dec, 1),
        vol_confirmation_pct=round(vol_conf, 1),
        pct_rsi_overbought=round(pct_ob, 1),
        pct_rsi_healthy=round(pct_hlth, 1),
        pct_rsi_oversold=round(pct_os, 1),
        breadth_score=breadth_score,
        breadth_label=breadth_label,
        breadth_color=breadth_color,
        leaders=leaders,
        laggards=laggards,
        most_active=most_active,
        sector_data=sector_data,
        leading_sector=leading_sec,
        lagging_sector=lagging_sec,
        breadth_trend_bias=trend_bias,
        sell_off_confirmed=sell_off_confirmed,
        rally_confirmed=rally_confirmed,
        divergence_warning=divergence_warning,
        data_as_of=pd.Timestamp.now().strftime("%H:%M:%S"),
        cache_age_mins=0.0,
    )

    st.session_state["nas100_breadth"]    = report
    st.session_state["nas100_breadth_ts"] = time.time()
    return report


# ── HARMONIZED FINAL SIGNAL ───────────────────────────────────────────────────

def compute_harmonized_signal(
    master_sig,         # MasterSignal from master_signal.py
    breadth: Optional[NAS100BreadthReport],
    regime,             # RegimeState
    macro_snap,
    gex,
    expected_move,
) -> Optional[HarmonizedSignal]:
    """
    Resolve all dashboard layers into ONE final actionable signal.
    Specifically designed to catch conflicts (e.g. master says BUY but
    breadth shows 80% of NAS100 declining = don't trust the signal).
    """
    if master_sig is None:
        return None

    confirmations = []
    risks         = []
    conflict      = False
    conflict_reason = ""

    # ── MASTER SIGNAL BASE ────────────────────────────────────────────────────
    master_score   = master_sig.conviction_score   # -100 to +100
    master_dir_bull = master_sig.direction == "LONG"
    master_dir_bear = master_sig.direction == "SHORT"

    # ── BREADTH ADJUSTMENT ────────────────────────────────────────────────────
    breadth_adj   = 0   # adjustment to master score
    breadth_score = 50  # neutral default

    if breadth:
        breadth_score = breadth.breadth_score  # 0-100

        # Convert 0-100 breadth score to -50/+50 adjustment
        breadth_adj = int((breadth_score - 50) * 0.6)  # max ±30 pts adjustment

        # Conflict detection
        if master_dir_bull and breadth.breadth_trend_bias == "BEARISH":
            conflict = True
            conflict_reason = (
                f"Master signal LONG but only {breadth.pct_above_sma50:.0f}% "
                f"of NAS100 above SMA50 — narrow leadership, don't chase"
            )
            risks.append(conflict_reason)
            breadth_adj = min(breadth_adj, -15)  # force downgrade

        elif master_dir_bear and breadth.breadth_trend_bias == "BULLISH":
            conflict = True
            conflict_reason = (
                f"Master signal SHORT but {breadth.pct_above_sma50:.0f}% "
                f"of NAS100 above SMA50 — broad strength, shorting into tide"
            )
            risks.append(conflict_reason)
            breadth_adj = max(breadth_adj, +15)  # force downgrade

        # Sell-off confirmation amplifies SHORT
        if breadth.sell_off_confirmed:
            if master_dir_bear:
                breadth_adj -= 10
                confirmations.append(
                    f"Sell-off confirmed: {breadth.advance_decline:.0f}% declining "
                    f"on elevated volume across NAS100"
                )
            else:
                risks.append("Sell-off confirmed across NAS100 — caution on longs")

        # Rally confirmation amplifies LONG
        if breadth.rally_confirmed:
            if master_dir_bull:
                breadth_adj += 10
                confirmations.append(
                    f"Rally confirmed: {breadth.advance_decline:.0f}% advancing "
                    f"on elevated volume across NAS100"
                )
            else:
                risks.append("Broad rally in NAS100 — shorting into strength")

        if breadth.divergence_warning:
            risks.append(
                f"Divergence: index moving but only {breadth.pct_above_sma50:.0f}% "
                f"of components above SMA50 — fragile rally"
            )

        # Leader/laggard context
        if breadth.leaders:
            leader_names = ", ".join(c.ticker for c in breadth.leaders[:3])
            confirmations.append(f"Leaders: {leader_names}")
        if breadth.laggards:
            laggard_names = ", ".join(c.ticker for c in breadth.laggards[:3])
            risks.append(f"Laggards: {laggard_names}")

    # ── REGIME CHECK ─────────────────────────────────────────────────────────
    if regime:
        if regime.state == 2:
            return HarmonizedSignal(
                direction="AVOID", conviction_pct=0,
                action="STAY OUT",
                master_score=master_score, breadth_score=breadth_score,
                conflict_detected=False, conflict_reason="Crisis regime",
                entry_zone_low=master_sig.entry_zone.low,
                entry_zone_high=master_sig.entry_zone.high,
                stop_loss=master_sig.stop_loss,
                take_profit_1=master_sig.take_profit_1,
                take_profit_2=master_sig.take_profit_2,
                key_risks=["CRISIS regime: no new entries"],
                key_confirmations=[],
                summary="CRISIS regime active. All trading halted. Protect capital.",
                lot_multiplier=0.0,
            )
        if regime.state == 1:
            risks.append("CHOP regime: mean-reversion only, 50% lot size")

    # ── EXPECTED MOVE RISK ────────────────────────────────────────────────────
    if expected_move and expected_move.exhaustion_pct >= 90:
        risks.append(
            f"Expected move {expected_move.exhaustion_pct:.0f}% consumed — "
            f"only {expected_move.expected_move_remaining_pts:.0f} pts remain"
        )

    # ── GEX CONTEXT ──────────────────────────────────────────────────────────
    if gex:
        if gex.gamma_regime == "NEGATIVE":
            confirmations.append(
                f"Negative GEX: dealers amplify moves "
                f"(flip zone {gex.gamma_flip_price:,.0f})"
            )
        else:
            confirmations.append("Positive GEX: mean-reversion favored")

    # ── MACRO ─────────────────────────────────────────────────────────────────
    if macro_snap:
        if macro_snap.risk_score >= 40:
            risks.append(f"Macro risk {macro_snap.risk_score}/100: {macro_snap.risk_level}")
        else:
            confirmations.append(f"Macro clear ({macro_snap.risk_score}/100)")

    # ── FINAL SCORE ───────────────────────────────────────────────────────────
    final_score  = max(-100, min(100, master_score + breadth_adj))
    conviction   = min(100, int(abs(final_score)))
    lot_mult     = master_sig.lot_adjustment

    if breadth and breadth.breadth_score < 35:
        lot_mult = round(lot_mult * 0.6, 2)   # reduce size on weak breadth
    if conflict:
        lot_mult = round(lot_mult * 0.7, 2)   # reduce size on conflict

    # ── DIRECTION LABEL ───────────────────────────────────────────────────────
    if final_score >= 60:
        direction = "STRONG LONG"
        action    = "BUY NOW" if not conflict else "BUY — verify breadth first"
    elif final_score >= 25:
        direction = "LONG"
        action    = "BUY ON PULLBACK" if not conflict else "WAIT — conflict present"
    elif final_score <= -60:
        direction = "STRONG SHORT"
        action    = "SELL NOW" if not conflict else "SELL — verify breadth first"
    elif final_score <= -25:
        direction = "SHORT"
        action    = "SELL ON BOUNCE" if not conflict else "WAIT — conflict present"
    else:
        direction = "NEUTRAL"
        action    = "HOLD / WAIT"
        lot_mult  = 0.0

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    b_str = (
        f"NAS100 breadth {breadth.breadth_label} "
        f"({breadth.pct_above_sma50:.0f}% above SMA50, "
        f"{breadth.advance_decline:.0f}% advancing). "
        if breadth else ""
    )
    c_str = f"Conflict: {conflict_reason}. " if conflict else ""
    summary = (
        f"{direction} | {action}. "
        f"Harmonized score: {final_score:+d}. "
        f"{b_str}{c_str}"
        f"Lot multiplier: {lot_mult}x."
    )

    return HarmonizedSignal(
        direction=direction,
        conviction_pct=conviction,
        action=action,
        master_score=master_score,
        breadth_score=breadth_score,
        conflict_detected=conflict,
        conflict_reason=conflict_reason,
        entry_zone_low=master_sig.entry_zone.low,
        entry_zone_high=master_sig.entry_zone.high,
        stop_loss=master_sig.stop_loss,
        take_profit_1=master_sig.take_profit_1,
        take_profit_2=master_sig.take_profit_2,
        key_risks=risks[:5],
        key_confirmations=confirmations[:5],
        summary=summary,
        lot_multiplier=max(0.0, min(1.0, lot_mult)),
    )


# ── RENDER ────────────────────────────────────────────────────────────────────

def render_nas100_breadth(report: Optional[NAS100BreadthReport]):
    """Render the NAS100 internal breadth panel."""
    import streamlit as st

    st.subheader("🌐 NAS100 Internal Breadth — Component Analysis")

    if report is None:
        st.warning(
            "NAS100 component data loading... "
            "First fetch takes ~10s. Refreshes every 15 minutes."
        )
        return

    # Header metrics
    st.caption(
        f"📊 {report.tickers_analyzed} components analyzed "
        f"({report.tickers_failed} failed) | "
        f"Data: {report.data_as_of} | "
        f"Cache age: {report.cache_age_mins:.0f} min"
    )

    # Breadth score banner
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:8px;"
        f"background:{report.breadth_color}22;"
        f"border:2px solid {report.breadth_color};margin-bottom:10px'>"
        f"<span style='font-size:1.15em;font-weight:bold;color:{report.breadth_color}'>"
        f"NAS100 Breadth: {report.breadth_label} ({report.breadth_score}/100)"
        f"</span>"
        f"{'&nbsp;&nbsp;🔴 SELL-OFF CONFIRMED' if report.sell_off_confirmed else ''}"
        f"{'&nbsp;&nbsp;🟢 RALLY CONFIRMED'    if report.rally_confirmed    else ''}"
        f"{'&nbsp;&nbsp;⚠️ DIVERGENCE WARNING' if report.divergence_warning else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Core metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Above SMA50",    f"{report.pct_above_sma50:.0f}%",
              delta="Bullish" if report.pct_above_sma50 > 55 else "Bearish")
    m2.metric("Above SMA20",    f"{report.pct_above_sma20:.0f}%")
    m3.metric("Adv/Dec",        f"{report.advance_decline:.0f}%",
              delta=f"{report.advance_decline-50:+.0f}% vs neutral")
    m4.metric("Vol Confirmed",  f"{report.vol_confirmation_pct:.0f}%")
    m5.metric("RSI Healthy",    f"{report.pct_rsi_healthy:.0f}%",
              delta=f"OB:{report.pct_rsi_overbought:.0f}% OS:{report.pct_rsi_oversold:.0f}%")

    # RSI distribution bar
    ob_w  = int(report.pct_rsi_overbought)
    hl_w  = int(report.pct_rsi_healthy)
    os_w  = int(report.pct_rsi_oversold)
    neu_w = max(0, 100 - ob_w - hl_w - os_w)
    st.markdown(
        f"<div style='margin:6px 0 2px;font-size:0.8em;color:#aaa'>RSI Distribution</div>"
        f"<div style='display:flex;height:10px;border-radius:5px;overflow:hidden'>"
        f"<div style='width:{ob_w}%;background:#c9302c' title='Overbought {ob_w}%'></div>"
        f"<div style='width:{hl_w}%;background:#2d9e2d' title='Healthy {hl_w}%'></div>"
        f"<div style='width:{neu_w}%;background:#888'   title='Neutral {neu_w}%'></div>"
        f"<div style='width:{os_w}%;background:#8b0000' title='Oversold {os_w}%'></div>"
        f"</div>"
        f"<div style='display:flex;gap:16px;margin-top:3px;font-size:0.75em;color:#888'>"
        f"<span>🔴 OB: {ob_w}%</span>"
        f"<span>🟢 Healthy: {hl_w}%</span>"
        f"<span>⚫ Neutral: {neu_w}%</span>"
        f"<span>🟤 OS: {os_w}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Leaders / Laggards / Active
    col_l, col_lg, col_a = st.columns(3)

    with col_l:
        st.markdown("**🏆 Top Leaders (5d)**")
        for c in report.leaders:
            vol_str = f" 🔊{c.vol_ratio:.1f}x" if c.vol_ratio > 1.5 else ""
            st.markdown(
                f"<span style='color:#2d9e2d;font-weight:bold'>{c.ticker}</span> "
                f"<span style='color:#5cb85c'>{c.ret_5d:+.1f}%</span>"
                f"<span style='color:#888;font-size:0.85em'>{vol_str} RSI:{c.rsi:.0f}</span>",
                unsafe_allow_html=True,
            )

    with col_lg:
        st.markdown("**💀 Worst Laggards (5d)**")
        for c in report.laggards:
            vol_str = f" 🔊{c.vol_ratio:.1f}x" if c.vol_ratio > 1.5 else ""
            st.markdown(
                f"<span style='color:#c9302c;font-weight:bold'>{c.ticker}</span> "
                f"<span style='color:#d9534f'>{c.ret_5d:+.1f}%</span>"
                f"<span style='color:#888;font-size:0.85em'>{vol_str} RSI:{c.rsi:.0f}</span>",
                unsafe_allow_html=True,
            )

    with col_a:
        st.markdown("**⚡ Most Active (Vol)**")
        for c in report.most_active:
            ret_col = "#2d9e2d" if c.ret_1d >= 0 else "#c9302c"
            st.markdown(
                f"<span style='color:#e6a817;font-weight:bold'>{c.ticker}</span> "
                f"<span style='color:{ret_col}'>{c.ret_1d:+.1f}%</span> "
                f"<span style='color:#888;font-size:0.85em'>{c.vol_ratio:.1f}x vol</span>",
                unsafe_allow_html=True,
            )

    # Sector rotation
    with st.expander("📊 Sector Rotation", expanded=False):
        for s in report.sector_data:
            bar_w = int(s.pct_above_sma50)
            st.markdown(
                f"<div style='margin:4px 0'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:2px'>"
                f"<span style='color:#ddd;font-size:0.9em'>{s.sector} "
                f"<span style='color:#888'>({s.ticker_count})</span></span>"
                f"<span style='color:{s.signal_color};font-size:0.85em;font-weight:bold'>"
                f"{s.signal} | {s.avg_5d_return:+.1f}% | {s.pct_above_sma50:.0f}% SMA50"
                f"</span></div>"
                f"<div style='background:#333;border-radius:3px;height:5px'>"
                f"<div style='background:{s.signal_color};width:{bar_w}%;height:5px;"
                f"border-radius:3px'></div></div></div>",
                unsafe_allow_html=True,
            )


def render_harmonized_signal(hs: Optional[HarmonizedSignal], risk_config=None):
    """Render the final Harmonized Signal — the single source of truth."""
    import streamlit as st

    st.subheader("⚡ Harmonized Final Signal")

    if hs is None:
        st.warning("Harmonized signal unavailable — waiting for all data layers.")
        return

    dir_config = {
        "STRONG LONG":  ("#0d4a0d", "#2d9e2d", "📈"),
        "LONG":         ("#1a3a1a", "#5cb85c", "📈"),
        "NEUTRAL":      ("#2a2a2a", "#888888", "⏸️"),
        "SHORT":        ("#3a1a1a", "#d9534f", "📉"),
        "STRONG SHORT": ("#4a0d0d", "#c9302c", "📉"),
        "AVOID":        ("#2a0d0d", "#8b0000", "🚫"),
    }
    bg, border, icon = dir_config.get(hs.direction, dir_config["NEUTRAL"])

    conflict_html = ""
    if hs.conflict_detected:
        conflict_html = (
            f"<div style='background:#5a3a00;padding:6px 10px;border-radius:5px;"
            f"margin-top:8px;font-size:0.9em;color:#e6a817'>"
            f"⚡ CONFLICT DETECTED: {hs.conflict_reason}</div>"
        )

    st.markdown(
        f"""<div style='padding:16px;border-radius:10px;background:{bg};
                    border:2px solid {border};margin-bottom:12px'>
          <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
            <span style='font-size:1.8em;font-weight:900;color:{border}'>{icon} {hs.direction}</span>
            <span style='font-size:1.2em;color:#ddd'>{hs.action}</span>
            <span style='color:#aaa;font-size:0.9em'>
              Conviction: {hs.conviction_pct}% &nbsp;|&nbsp;
              Master: {hs.master_score:+d} &nbsp;|&nbsp;
              Breadth: {hs.breadth_score}/100
            </span>
          </div>
          {conflict_html}
          <div style='color:#ccc;margin-top:10px;font-size:0.9em'>{hs.summary}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Trade levels
    if hs.direction not in ("NEUTRAL", "AVOID"):
        st.markdown("**📐 Harmonized Trade Levels**")
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.markdown(
            f"<div style='background:#1a2a1a;padding:10px;border-radius:8px;"
            f"border:1px solid #2d9e2d;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.8em'>ENTRY ZONE</div>"
            f"<div style='color:#2d9e2d;font-weight:bold;font-size:1.1em'>"
            f"{hs.entry_zone_low:,.0f}–{hs.entry_zone_high:,.0f}</div></div>",
            unsafe_allow_html=True,
        )
        lc2.markdown(
            f"<div style='background:#2a1a1a;padding:10px;border-radius:8px;"
            f"border:1px solid #c9302c;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.8em'>STOP LOSS</div>"
            f"<div style='color:#c9302c;font-weight:bold;font-size:1.1em'>"
            f"{hs.stop_loss:,.0f}</div></div>",
            unsafe_allow_html=True,
        )
        lc3.markdown(
            f"<div style='background:#2a2a1a;padding:10px;border-radius:8px;"
            f"border:1px solid #e6a817;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.8em'>TP1</div>"
            f"<div style='color:#e6a817;font-weight:bold;font-size:1.1em'>"
            f"{hs.take_profit_1:,.0f}</div></div>",
            unsafe_allow_html=True,
        )
        lc4.markdown(
            f"<div style='background:#1a2a1a;padding:10px;border-radius:8px;"
            f"border:1px solid #5cb85c;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.8em'>TP2</div>"
            f"<div style='color:#5cb85c;font-weight:bold;font-size:1.1em'>"
            f"{hs.take_profit_2:,.0f}</div></div>",
            unsafe_allow_html=True,
        )

        # Lot size
        lot_base = risk_config.lot_size if risk_config else 0.02
        lot_rec  = max(0.01, round(lot_base * hs.lot_multiplier, 2))
        lot_col  = "#2d9e2d" if hs.lot_multiplier >= 0.8 else (
                   "#e6a817" if hs.lot_multiplier >= 0.5 else "#c9302c")
        st.markdown(
            f"**Lot:** <span style='color:{lot_col};font-weight:bold'>{lot_rec}</span> "
            f"<span style='color:#888'>(base {lot_base} × {hs.lot_multiplier:.2f})</span>",
            unsafe_allow_html=True,
        )

    # Confirmations & risks
    rc, rk = st.columns(2)
    with rc:
        if hs.key_confirmations:
            st.markdown("**✅ Confirmations**")
            for c in hs.key_confirmations:
                st.markdown(f"<span style='color:#5cb85c'>• {c}</span>",
                            unsafe_allow_html=True)
    with rk:
        if hs.key_risks:
            st.markdown("**⚠️ Risks**")
            for r in hs.key_risks:
                st.markdown(f"<span style='color:#d9534f'>• {r}</span>",
                            unsafe_allow_html=True)
