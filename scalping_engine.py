"""
scalping_engine.py
------------------
Sniper scalping detection for Forex + NAS100.
Reads from existing session_state cache — zero extra yfinance calls.

FOREX detects:
  1. Order Blocks (OB)       — last opposing candle before impulse move
  2. Fair Value Gaps (FVG)   — price imbalances / liquidity voids
  3. Liquidity Sweeps        — stop hunts above/below swing highs/lows
  4. Asian Range levels      — London breakout targets
  5. Session Open levels     — London/NY open price as key reference

NAS100 detects:
  1. VWAP deviation scalp    — price extended from VWAP = snap-back trade
  2. Gap fill                — overnight gap identification + fill target
  3. Key level bounces       — round numbers, daily high/low, prev close
  4. Open Drive              — first 15min direction bias

ENHANCED LIQUIDITY SWEEP DETECTION (v2):
  A. Stop raids above/below equal highs/lows with RVOL spike confirmation
  B. Overnight high/low sweeps
  C. Previous day level sweeps
  D. Fade setup signal after confirmed sweep
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timezone, timedelta


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class ScalpSetup:
    """A single sniper scalping opportunity."""
    pair: str
    setup_type: str        # "ORDER_BLOCK" | "FVG" | "LIQ_SWEEP" | "ASIAN_RANGE" | "VWAP_DEV" | "GAP_FILL" | "KEY_LEVEL"
    direction: str         # "BUY" | "SELL"
    entry_zone_high: float
    entry_zone_low: float
    target: float
    invalidation: float    # price that kills the setup
    strength: str          # "STRONG" | "MEDIUM" | "WEAK"
    description: str
    pips_to_target: float  # for forex
    risk_pips: float


@dataclass
class LiquiditySweepDetail:
    """Enhanced liquidity sweep with RVOL confirmation."""
    sweep_type: str          # "EQUAL_HIGHS" | "EQUAL_LOWS" | "OVERNIGHT_HIGH" | "OVERNIGHT_LOW" | "PREV_DAY_HIGH" | "PREV_DAY_LOW"
    swept_level: float       # the level that was swept
    sweep_size_pts: float    # how far price went beyond the level
    rvol_spike: bool         # was there a volume spike during sweep?
    rvol_ratio: float        # current vol / avg vol
    rejection_confirmed: bool  # did price close back inside?
    fade_direction: str      # "BUY" (after sweep low) | "SELL" (after sweep high)
    signal_text: str         # e.g. "Liquidity sweep detected → fade setup"


@dataclass
class ScalpReport:
    """Full scalping analysis for one instrument."""
    ticker: str
    current_price: float
    session: str           # "ASIAN" | "LONDON" | "NY" | "OVERLAP"

    # Forex specific
    order_blocks: List[ScalpSetup] = field(default_factory=list)
    fvgs: List[ScalpSetup] = field(default_factory=list)
    liq_sweeps: List[ScalpSetup] = field(default_factory=list)
    asian_range: Optional[dict] = None

    # NAS100 specific
    vwap: Optional[float] = None
    vwap_deviation_pct: Optional[float] = None
    vwap_setup: Optional[ScalpSetup] = None
    gap_fill_setup: Optional[ScalpSetup] = None
    key_levels: List[float] = field(default_factory=list)
    key_level_setup: Optional[ScalpSetup] = None
    open_drive: Optional[str] = None  # "BULLISH" | "BEARISH" | "CHOPPY"

    # Enhanced liquidity sweeps (NAS100)
    liquidity_sweeps: List[LiquiditySweepDetail] = field(default_factory=list)
    active_fade_setup: Optional[LiquiditySweepDetail] = None

    # Best setup (highest strength)
    best_setup: Optional[ScalpSetup] = None


# ── SESSION DETECTION ─────────────────────────────────────────────────────────

def _get_session() -> str:
    h = datetime.now(timezone.utc).hour
    if 22 <= h or h < 7:  return "ASIAN"
    if 7 <= h < 12:        return "LONDON"
    if 12 <= h < 17:       return "OVERLAP"
    return "NY"


def _get_pip(pair: str) -> float:
    return 0.01 if 'JPY' in pair else 0.0001


# ── FOREX SCALPING DETECTORS ──────────────────────────────────────────────────

def _detect_order_blocks(df: pd.DataFrame, pair: str,
                          current_price: float) -> List[ScalpSetup]:
    setups = []
    pip    = _get_pip(pair)

    if df is None or len(df) < 20:
        return setups

    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]

    highs  = df['High'].values
    lows   = df['Low'].values
    closes = df['Close'].values
    opens  = df['Open'].values

    lookback = min(50, len(df) - 5)

    for i in range(5, lookback):
        is_bearish_candle = closes[i] < opens[i]
        if is_bearish_candle:
            if (i + 3 < len(df) and
                closes[i+1] > opens[i+1] and
                closes[i+2] > opens[i+2] and
                closes[i+3] > closes[i]):

                ob_high = opens[i]
                ob_low  = closes[i]
                ob_range = ob_high - ob_low
                if ob_low - ob_range * 0.5 <= current_price <= ob_high + ob_range * 0.3:
                    target      = float(highs[i+3])
                    invalidation= ob_low - ob_range * 0.5
                    pips_target = (target - current_price) / pip
                    risk_pips   = (current_price - invalidation) / pip

                    if pips_target > 0 and risk_pips > 0 and pips_target / risk_pips > 1.5:
                        setups.append(ScalpSetup(
                            pair=pair, setup_type="ORDER_BLOCK", direction="BUY",
                            entry_zone_high=ob_high, entry_zone_low=ob_low,
                            target=target, invalidation=invalidation,
                            strength="STRONG" if pips_target / risk_pips > 2.5 else "MEDIUM",
                            description=f"Bullish OB at {ob_low:.5f}-{ob_high:.5f}. "
                                        f"Price returning to institutional demand zone.",
                            pips_to_target=round(pips_target, 1),
                            risk_pips=round(risk_pips, 1),
                        ))

        is_bullish_candle = closes[i] > opens[i]
        if is_bullish_candle:
            if (i + 3 < len(df) and
                closes[i+1] < opens[i+1] and
                closes[i+2] < opens[i+2] and
                closes[i+3] < closes[i]):

                ob_high = closes[i]
                ob_low  = opens[i]
                ob_range = ob_high - ob_low
                if ob_low - ob_range * 0.3 <= current_price <= ob_high + ob_range * 0.5:
                    target      = float(lows[i+3])
                    invalidation= ob_high + ob_range * 0.5
                    pips_target = (current_price - target) / pip
                    risk_pips   = (invalidation - current_price) / pip

                    if pips_target > 0 and risk_pips > 0 and pips_target / risk_pips > 1.5:
                        setups.append(ScalpSetup(
                            pair=pair, setup_type="ORDER_BLOCK", direction="SELL",
                            entry_zone_high=ob_high, entry_zone_low=ob_low,
                            target=target, invalidation=invalidation,
                            strength="STRONG" if pips_target / risk_pips > 2.5 else "MEDIUM",
                            description=f"Bearish OB at {ob_low:.5f}-{ob_high:.5f}. "
                                        f"Price returning to institutional supply zone.",
                            pips_to_target=round(pips_target, 1),
                            risk_pips=round(risk_pips, 1),
                        ))

    setups.sort(key=lambda s: abs((s.entry_zone_high + s.entry_zone_low) / 2 - current_price))
    return setups[:2]


def _detect_fvg(df: pd.DataFrame, pair: str,
                current_price: float) -> List[ScalpSetup]:
    setups = []
    pip    = _get_pip(pair)

    if df is None or len(df) < 10:
        return setups

    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    highs  = df['High'].values
    lows   = df['Low'].values
    closes = df['Close'].values

    lookback = min(30, len(df) - 3)

    for i in range(1, lookback):
        bull_fvg_top    = lows[i+1]
        bull_fvg_bottom = highs[i-1]

        if bull_fvg_top > bull_fvg_bottom:
            gap_size = (bull_fvg_top - bull_fvg_bottom) / pip
            if (bull_fvg_bottom <= current_price <= bull_fvg_top and gap_size >= 3):
                target      = float(max(closes[i+1], closes[i+2]) if i+2 < len(df) else closes[i+1])
                invalidation= bull_fvg_bottom - 3 * pip
                pips_target = (target - current_price) / pip
                risk_pips   = (current_price - invalidation) / pip

                if pips_target > 0 and risk_pips > 0:
                    setups.append(ScalpSetup(
                        pair=pair, setup_type="FVG", direction="BUY",
                        entry_zone_high=bull_fvg_top, entry_zone_low=bull_fvg_bottom,
                        target=target, invalidation=invalidation,
                        strength="STRONG" if gap_size > 8 else "MEDIUM",
                        description=f"Bullish FVG {bull_fvg_bottom:.5f}-{bull_fvg_top:.5f} "
                                    f"({gap_size:.1f} pips). Price filling imbalance.",
                        pips_to_target=round(pips_target, 1),
                        risk_pips=round(risk_pips, 1),
                    ))

        bear_fvg_bottom = highs[i+1]
        bear_fvg_top    = lows[i-1]

        if bear_fvg_top > bear_fvg_bottom:
            gap_size = (bear_fvg_top - bear_fvg_bottom) / pip
            if (bear_fvg_bottom <= current_price <= bear_fvg_top and gap_size >= 3):
                target      = float(min(closes[i+1], closes[i+2]) if i+2 < len(df) else closes[i+1])
                invalidation= bear_fvg_top + 3 * pip
                pips_target = (current_price - target) / pip
                risk_pips   = (invalidation - current_price) / pip

                if pips_target > 0 and risk_pips > 0:
                    setups.append(ScalpSetup(
                        pair=pair, setup_type="FVG", direction="SELL",
                        entry_zone_high=bear_fvg_top, entry_zone_low=bear_fvg_bottom,
                        target=target, invalidation=invalidation,
                        strength="STRONG" if gap_size > 8 else "MEDIUM",
                        description=f"Bearish FVG {bear_fvg_bottom:.5f}-{bear_fvg_top:.5f} "
                                    f"({gap_size:.1f} pips). Price filling imbalance.",
                        pips_to_target=round(pips_target, 1),
                        risk_pips=round(risk_pips, 1),
                    ))

    setups.sort(key=lambda s: abs((s.entry_zone_high + s.entry_zone_low) / 2 - current_price))
    return setups[:2]


def _detect_liquidity_sweep(df: pd.DataFrame, pair: str,
                             current_price: float) -> List[ScalpSetup]:
    setups = []
    pip    = _get_pip(pair)

    if df is None or len(df) < 15:
        return setups

    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]

    recent  = df.tail(20)
    highs   = recent['High'].values
    lows    = recent['Low'].values
    closes  = recent['Close'].values

    swing_high = float(np.max(highs[:-3]))
    swing_low  = float(np.min(lows[:-3]))
    last_high  = float(highs[-1])
    last_low   = float(lows[-1])
    last_close = float(closes[-1])

    if last_low < swing_low and last_close > swing_low:
        sweep_size  = (swing_low - last_low) / pip
        if sweep_size >= 2:
            target      = swing_high
            invalidation= last_low - 2 * pip
            pips_target = (target - current_price) / pip
            risk_pips   = (current_price - invalidation) / pip

            if pips_target > 0 and risk_pips > 0 and pips_target / risk_pips > 1.5:
                setups.append(ScalpSetup(
                    pair=pair, setup_type="LIQ_SWEEP", direction="BUY",
                    entry_zone_high=swing_low + 2 * pip,
                    entry_zone_low=swing_low - pip,
                    target=target, invalidation=invalidation,
                    strength="STRONG" if sweep_size > 5 else "MEDIUM",
                    description=f"Bullish liquidity sweep below {swing_low:.5f}. "
                                f"Stop hunt ({sweep_size:.1f} pips) — reversal expected.",
                    pips_to_target=round(pips_target, 1),
                    risk_pips=round(risk_pips, 1),
                ))

    if last_high > swing_high and last_close < swing_high:
        sweep_size  = (last_high - swing_high) / pip
        if sweep_size >= 2:
            target      = swing_low
            invalidation= last_high + 2 * pip
            pips_target = (current_price - target) / pip
            risk_pips   = (invalidation - current_price) / pip

            if pips_target > 0 and risk_pips > 0 and pips_target / risk_pips > 1.5:
                setups.append(ScalpSetup(
                    pair=pair, setup_type="LIQ_SWEEP", direction="SELL",
                    entry_zone_high=swing_high + pip,
                    entry_zone_low=swing_high - 2 * pip,
                    target=target, invalidation=invalidation,
                    strength="STRONG" if sweep_size > 5 else "MEDIUM",
                    description=f"Bearish liquidity sweep above {swing_high:.5f}. "
                                f"Stop hunt ({sweep_size:.1f} pips) — reversal expected.",
                    pips_to_target=round(pips_target, 1),
                    risk_pips=round(risk_pips, 1),
                ))

    return setups


def _detect_asian_range(df_1h: pd.DataFrame, current_price: float) -> Optional[dict]:
    if df_1h is None or df_1h.empty:
        return None
    try:
        df = df_1h.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.capitalize() for c in df.columns]

        now   = pd.Timestamp.now(tz='UTC')
        start = now.replace(hour=22, minute=0, second=0) - pd.Timedelta(days=1)
        end   = now.replace(hour=7,  minute=0, second=0)
        asian = df[(df.index >= start) & (df.index <= end)]

        if len(asian) < 3:
            return None

        asian_high = float(asian['High'].max())
        asian_low  = float(asian['Low'].min())
        asian_mid  = (asian_high + asian_low) / 2
        range_size = asian_high - asian_low

        breakout_up   = current_price > asian_high
        breakout_down = current_price < asian_low
        inside_range  = asian_low <= current_price <= asian_high

        return {
            "high":          asian_high,
            "low":           asian_low,
            "mid":           asian_mid,
            "range_pips":    range_size / 0.0001,
            "breakout_up":   breakout_up,
            "breakout_down": breakout_down,
            "inside":        inside_range,
            "status": (
                "🔼 BREAKING UP — Potential London long" if breakout_up else
                "🔽 BREAKING DOWN — Potential London short" if breakout_down else
                "📦 INSIDE RANGE — Wait for breakout"
            )
        }
    except Exception as e:
        print(f"[scalping] Asian range error: {e}")
        return None


# ── ENHANCED NAS100 LIQUIDITY SWEEP DETECTION ─────────────────────────────────

def _detect_nas100_liquidity_sweeps(df_5m: pd.DataFrame,
                                     df_1d: pd.DataFrame,
                                     current_price: float,
                                     ratio: float = 40.0) -> List[LiquiditySweepDetail]:
    """
    Enhanced NAS100 liquidity sweep detection covering:
    A. Equal highs / equal lows sweeps with RVOL confirmation
    B. Overnight session high/low sweeps
    C. Previous day high/low sweeps

    Returns list of detected sweeps ordered by recency/strength.
    """
    sweeps = []
    if df_5m is None or len(df_5m) < 20:
        return sweeps

    try:
        df = df_5m.copy()
        df.columns = [c.capitalize() for c in df.columns]
        df.index   = pd.to_datetime(df.index, utc=True)

        highs  = df['High'].values * ratio
        lows   = df['Low'].values  * ratio
        closes = df['Close'].values * ratio
        vols   = df['Volume'].values

        # RVOL: compare last 3 candles vs 20-bar average
        vol_avg  = float(np.mean(vols[-20:-3])) if len(vols) > 23 else float(np.mean(vols))
        vol_last = float(np.mean(vols[-3:])) if len(vols) >= 3 else vol_avg
        rvol     = vol_last / vol_avg if vol_avg > 0 else 1.0

        # ── A. Equal Highs / Equal Lows sweep ────────────────────────────────
        lookback = min(30, len(df) - 5)
        recent_highs = highs[-lookback:-3]
        recent_lows  = lows[-lookback:-3]

        if len(recent_highs) > 0:
            # Equal highs: two or more highs within 0.05% of each other
            max_recent_high = float(np.max(recent_highs))
            last_high       = float(highs[-1])
            last_close_h    = float(closes[-1])

            if last_high > max_recent_high * 1.0005:  # swept above
                if last_close_h < max_recent_high:    # rejected back below
                    sweep_pts = last_high - max_recent_high
                    sweeps.append(LiquiditySweepDetail(
                        sweep_type="EQUAL_HIGHS",
                        swept_level=round(max_recent_high, 0),
                        sweep_size_pts=round(sweep_pts, 0),
                        rvol_spike=rvol > 1.5,
                        rvol_ratio=round(rvol, 2),
                        rejection_confirmed=True,
                        fade_direction="SELL",
                        signal_text=(
                            f"🔴 Equal highs swept at {max_recent_high:,.0f}. "
                            f"Wick {sweep_pts:.0f} pts above — stop hunt confirmed. "
                            f"{'RVOL spike confirms institutional activity. ' if rvol > 1.5 else ''}"
                            f"Liquidity sweep detected → fade SELL setup."
                        ),
                    ))

            # Equal lows sweep
            min_recent_low  = float(np.min(recent_lows))
            last_low        = float(lows[-1])
            last_close_l    = float(closes[-1])

            if last_low < min_recent_low * 0.9995:
                if last_close_l > min_recent_low:
                    sweep_pts = min_recent_low - last_low
                    sweeps.append(LiquiditySweepDetail(
                        sweep_type="EQUAL_LOWS",
                        swept_level=round(min_recent_low, 0),
                        sweep_size_pts=round(sweep_pts, 0),
                        rvol_spike=rvol > 1.5,
                        rvol_ratio=round(rvol, 2),
                        rejection_confirmed=True,
                        fade_direction="BUY",
                        signal_text=(
                            f"🟢 Equal lows swept at {min_recent_low:,.0f}. "
                            f"Wick {sweep_pts:.0f} pts below — stop hunt confirmed. "
                            f"{'RVOL spike confirms institutional activity. ' if rvol > 1.5 else ''}"
                            f"Liquidity sweep detected → fade BUY setup."
                        ),
                    ))

        # ── B. Overnight high/low (Asian session levels) ──────────────────────
        today_utc = pd.Timestamp.now(tz='UTC').date()
        overnight = df[df.index.date < today_utc]

        if len(overnight) >= 5:
            overnight_high = float((overnight['High'] * ratio / ratio).max()) * ratio  # already scaled
            overnight_low  = float((overnight['Low']  * ratio / ratio).min()) * ratio

            # Re-extract from raw (already scaled above in highs/lows arrays)
            ov_mask = df.index.date < today_utc
            if ov_mask.any():
                ov_h = float(np.max(df.loc[ov_mask, 'High'].values * ratio))
                ov_l = float(np.min(df.loc[ov_mask, 'Low'].values  * ratio))

                if current_price < ov_l * 0.9998 and float(closes[-1]) > ov_l:
                    sweeps.append(LiquiditySweepDetail(
                        sweep_type="OVERNIGHT_LOW",
                        swept_level=round(ov_l, 0),
                        sweep_size_pts=round(ov_l - float(lows[-1]), 0),
                        rvol_spike=rvol > 1.3,
                        rvol_ratio=round(rvol, 2),
                        rejection_confirmed=True,
                        fade_direction="BUY",
                        signal_text=(
                            f"🟢 Overnight low ({ov_l:,.0f}) swept and reclaimed. "
                            f"Institutions hunted stops below overnight range. "
                            f"Liquidity sweep detected → fade BUY setup."
                        ),
                    ))

                if current_price > ov_h * 1.0002 and float(closes[-1]) < ov_h:
                    sweeps.append(LiquiditySweepDetail(
                        sweep_type="OVERNIGHT_HIGH",
                        swept_level=round(ov_h, 0),
                        sweep_size_pts=round(float(highs[-1]) - ov_h, 0),
                        rvol_spike=rvol > 1.3,
                        rvol_ratio=round(rvol, 2),
                        rejection_confirmed=True,
                        fade_direction="SELL",
                        signal_text=(
                            f"🔴 Overnight high ({ov_h:,.0f}) swept and rejected. "
                            f"Institutions hunted stops above overnight range. "
                            f"Liquidity sweep detected → fade SELL setup."
                        ),
                    ))

        # ── C. Previous day high/low sweep ───────────────────────────────────
        if df_1d is not None and len(df_1d) >= 2:
            d1 = df_1d.copy()
            d1.columns = [c.capitalize() for c in d1.columns]

            prev_day_high = float(d1['High'].iloc[-2])  * ratio
            prev_day_low  = float(d1['Low'].iloc[-2])   * ratio

            if float(highs[-1]) > prev_day_high and float(closes[-1]) < prev_day_high:
                sweeps.append(LiquiditySweepDetail(
                    sweep_type="PREV_DAY_HIGH",
                    swept_level=round(prev_day_high, 0),
                    sweep_size_pts=round(float(highs[-1]) - prev_day_high, 0),
                    rvol_spike=rvol > 1.3,
                    rvol_ratio=round(rvol, 2),
                    rejection_confirmed=True,
                    fade_direction="SELL",
                    signal_text=(
                        f"🔴 Previous day high ({prev_day_high:,.0f}) swept. "
                        f"Classic stop-hunt above key daily level. "
                        f"Liquidity sweep detected → fade SELL setup."
                    ),
                ))

            if float(lows[-1]) < prev_day_low and float(closes[-1]) > prev_day_low:
                sweeps.append(LiquiditySweepDetail(
                    sweep_type="PREV_DAY_LOW",
                    swept_level=round(prev_day_low, 0),
                    sweep_size_pts=round(prev_day_low - float(lows[-1]), 0),
                    rvol_spike=rvol > 1.3,
                    rvol_ratio=round(rvol, 2),
                    rejection_confirmed=True,
                    fade_direction="BUY",
                    signal_text=(
                        f"🟢 Previous day low ({prev_day_low:,.0f}) swept. "
                        f"Classic stop-hunt below key daily level. "
                        f"Liquidity sweep detected → fade BUY setup."
                    ),
                ))

        return sweeps

    except Exception as e:
        print(f"[scalping] NAS100 liquidity sweep error: {e}")
        return []


# ── NAS100 SCALPING DETECTORS (unchanged from original) ─────────────────────

def _calc_vwap(df_5m: pd.DataFrame) -> Optional[float]:
    if df_5m is None or df_5m.empty:
        return None
    try:
        df = df_5m.copy()
        df.columns = [c.capitalize() for c in df.columns]
        df.index   = pd.to_datetime(df.index, utc=True)
        today = pd.Timestamp.now(tz='UTC').date()
        df    = df[df.index.date == today]
        if len(df) < 5:
            return None
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return float(vwap.iloc[-1])
    except Exception:
        return None


def _detect_vwap_setup_scaled(df_5m: pd.DataFrame, vwap_nas: float,
                               current_price: float,
                               ratio: float = 40.0) -> Optional[ScalpSetup]:
    if df_5m is None or vwap_nas is None:
        return None
    try:
        df = df_5m.copy()
        df.columns = [c.capitalize() for c in df.columns]
        df.index   = pd.to_datetime(df.index, utc=True)
        today = pd.Timestamp.now(tz='UTC').date()
        df    = df[df.index.date == today]
        if len(df) < 10:
            return None

        typical_qqq = (df['High'] + df['Low'] + df['Close']) / 3
        typical_nas = typical_qqq * ratio
        std_nas = float((typical_nas - vwap_nas).std())
        std_nas = max(std_nas, 50.0)

        upper_2  = vwap_nas + 2 * std_nas
        lower_2  = vwap_nas - 2 * std_nas
        dev_pct  = (current_price - vwap_nas) / vwap_nas * 100

        if current_price > upper_2 and dev_pct > 0.4:
            return ScalpSetup(
                pair="NAS100", setup_type="VWAP_DEV", direction="SELL",
                entry_zone_high=current_price + 20,
                entry_zone_low=current_price - 20,
                target=vwap_nas,
                invalidation=upper_2 + std_nas * 0.5,
                strength="STRONG" if dev_pct > 0.7 else "MEDIUM",
                description=(
                    f"NAS100 {dev_pct:+.2f}% above VWAP ({vwap_nas:,.0f}). "
                    f"Extended above 2σ ({upper_2:,.0f}). "
                    f"Mean-reversion SELL — target VWAP {vwap_nas:,.0f}."
                ),
                pips_to_target=round(current_price - vwap_nas, 0),
                risk_pips=round(std_nas * 0.5, 0),
            )
        elif current_price < lower_2 and dev_pct < -0.4:
            return ScalpSetup(
                pair="NAS100", setup_type="VWAP_DEV", direction="BUY",
                entry_zone_high=current_price + 20,
                entry_zone_low=current_price - 20,
                target=vwap_nas,
                invalidation=lower_2 - std_nas * 0.5,
                strength="STRONG" if abs(dev_pct) > 0.7 else "MEDIUM",
                description=(
                    f"NAS100 {dev_pct:+.2f}% below VWAP ({vwap_nas:,.0f}). "
                    f"Extended below 2σ ({lower_2:,.0f}). "
                    f"Mean-reversion BUY — target VWAP {vwap_nas:,.0f}."
                ),
                pips_to_target=round(vwap_nas - current_price, 0),
                risk_pips=round(std_nas * 0.5, 0),
            )

        one_std_up   = vwap_nas + std_nas
        one_std_down = vwap_nas - std_nas

        if vwap_nas < current_price <= upper_2:
            return ScalpSetup(
                pair="NAS100", setup_type="VWAP_DEV", direction="BUY",
                entry_zone_high=current_price + 15,
                entry_zone_low=max(vwap_nas, current_price - 30),
                target=current_price + std_nas,
                invalidation=vwap_nas - 20,
                strength="MEDIUM",
                description=(
                    f"NAS100 {dev_pct:+.2f}% above VWAP ({vwap_nas:,.0f}). "
                    f"Bullish VWAP momentum — price holding above institutional fair value. "
                    f"Target: {current_price + std_nas:,.0f}. Invalidation: below VWAP."
                ),
                pips_to_target=round(std_nas, 0),
                risk_pips=round(current_price - vwap_nas + 20, 0),
            )
        elif lower_2 <= current_price < vwap_nas:
            return ScalpSetup(
                pair="NAS100", setup_type="VWAP_DEV", direction="SELL",
                entry_zone_high=min(vwap_nas, current_price + 30),
                entry_zone_low=current_price - 15,
                target=current_price - std_nas,
                invalidation=vwap_nas + 20,
                strength="MEDIUM",
                description=(
                    f"NAS100 {dev_pct:+.2f}% below VWAP ({vwap_nas:,.0f}). "
                    f"Bearish VWAP momentum — price failing to reclaim institutional fair value. "
                    f"Target: {current_price - std_nas:,.0f}. Invalidation: above VWAP."
                ),
                pips_to_target=round(std_nas, 0),
                risk_pips=round(vwap_nas - current_price + 20, 0),
            )
        return None
    except Exception as e:
        print(f"[scalping] VWAP scaled error: {e}")
        return None


def _detect_gap_fill(df_1d: pd.DataFrame, df_5m: pd.DataFrame,
                      current_price: float,
                      ratio: float = 40.0) -> Optional[ScalpSetup]:
    if df_1d is None or df_5m is None or len(df_1d) < 2:
        return None
    try:
        df1 = df_1d.copy()
        df1.columns = [c.capitalize() for c in df1.columns]
        df5 = df_5m.copy()
        df5.columns = [c.capitalize() for c in df5.columns]

        prev_close  = float(df1['Close'].iloc[-2]) * ratio
        today_open  = float(df5['Open'].iloc[0])   * ratio
        gap_size    = today_open - prev_close
        gap_pct     = abs(gap_size) / prev_close * 100

        if gap_pct < 0.1:
            return None

        target = prev_close

        if gap_size > 0:
            if current_price > prev_close:
                return ScalpSetup(
                    pair="NAS100", setup_type="GAP_FILL", direction="SELL",
                    entry_zone_high=today_open + 10,
                    entry_zone_low=today_open - 10,
                    target=prev_close,
                    invalidation=today_open + gap_size * 0.5,
                    strength="STRONG" if gap_pct > 0.3 else "MEDIUM",
                    description=(f"Gap UP {gap_size:+.0f} pts ({gap_pct:.2f}%). "
                                 f"Open: {today_open:,.0f} vs Prev Close: {prev_close:,.0f}. "
                                 f"Gap fill target: {prev_close:,.0f}."),
                    pips_to_target=round(current_price - prev_close, 0),
                    risk_pips=round(gap_size * 0.5, 0),
                )
        else:
            if current_price < prev_close:
                return ScalpSetup(
                    pair="NAS100", setup_type="GAP_FILL", direction="BUY",
                    entry_zone_high=today_open + 10,
                    entry_zone_low=today_open - 10,
                    target=prev_close,
                    invalidation=today_open + gap_size * 0.5,
                    strength="STRONG" if gap_pct > 0.3 else "MEDIUM",
                    description=(f"Gap DOWN {gap_size:+.0f} pts ({gap_pct:.2f}%). "
                                 f"Open: {today_open:,.0f} vs Prev Close: {prev_close:,.0f}. "
                                 f"Gap fill target: {prev_close:,.0f}."),
                    pips_to_target=round(prev_close - current_price, 0),
                    risk_pips=round(abs(gap_size) * 0.5, 0),
                )
        return None
    except Exception as e:
        print(f"[scalping] Gap fill error: {e}")
        return None


def _detect_key_levels(df_1d: pd.DataFrame, df_5m: pd.DataFrame,
                        current_price: float,
                        ratio: float = 40.0) -> tuple:
    levels = []
    if df_1d is not None and len(df_1d) >= 2:
        df1 = df_1d.copy()
        df1.columns = [c.capitalize() for c in df1.columns]
        levels.extend([
            float(df1['High'].iloc[-1])   * ratio,
            float(df1['Low'].iloc[-1])    * ratio,
            float(df1['Close'].iloc[-2])  * ratio,
            float(df1['High'].iloc[-2])   * ratio,
            float(df1['Low'].iloc[-2])    * ratio,
        ])

    base   = round(current_price / 500) * 500
    levels.extend([base - 500, base, base + 500, base + 1000])

    if not levels:
        return [], None

    levels = sorted(set([round(l, 0) for l in levels if l > 0]))
    nearest = min(levels, key=lambda l: abs(l - current_price))
    dist    = abs(nearest - current_price)

    if dist / current_price > 0.0015:
        return levels, None

    direction = "BUY" if current_price <= nearest else "SELL"
    return levels, ScalpSetup(
        pair="NAS100", setup_type="KEY_LEVEL", direction=direction,
        entry_zone_high=nearest + 10,
        entry_zone_low=nearest - 10,
        target=nearest + 100 if direction == "BUY" else nearest - 100,
        invalidation=nearest - 30 if direction == "BUY" else nearest + 30,
        strength="STRONG" if dist < current_price * 0.0005 else "MEDIUM",
        description=(f"Price at key level {nearest:,.0f}. "
                     f"{'Round number / structural support' if nearest % 500 == 0 else 'Daily high/low / prev close'}. "
                     f"Distance: {dist:.0f} pts."),
        pips_to_target=100.0,
        risk_pips=30.0,
    )


def _detect_open_drive(df_5m: pd.DataFrame) -> str:
    if df_5m is None or len(df_5m) < 5:
        return "UNKNOWN"
    try:
        df = df_5m.copy()
        df.columns = [c.capitalize() for c in df.columns]
        df.index   = pd.to_datetime(df.index, utc=True)

        now = pd.Timestamp.now(tz='UTC')
        ny_open = now.replace(hour=14, minute=30, second=0, microsecond=0)
        if now < ny_open:
            return "PRE-OPEN"

        first_15 = df[(df.index >= ny_open) &
                      (df.index <= ny_open + pd.Timedelta(minutes=15))]

        if len(first_15) < 2:
            return "AWAITING"

        open_price  = float(first_15['Open'].iloc[0])
        close_price = float(first_15['Close'].iloc[-1])
        move_pct    = (close_price - open_price) / open_price * 100

        if move_pct > 0.15:   return "BULLISH"
        if move_pct < -0.15:  return "BEARISH"
        return "CHOPPY"
    except Exception:
        return "UNKNOWN"


# ── MAIN ANALYSIS FUNCTIONS ───────────────────────────────────────────────────

def analyse_forex_scalp(pair: str, df_1h: pd.DataFrame,
                         current_price: float) -> ScalpReport:
    session = _get_session()
    report  = ScalpReport(ticker=pair, current_price=current_price, session=session)

    report.order_blocks = _detect_order_blocks(df_1h, pair, current_price)
    report.fvgs         = _detect_fvg(df_1h, pair, current_price)
    report.liq_sweeps   = _detect_liquidity_sweep(df_1h, pair, current_price)
    report.asian_range  = _detect_asian_range(df_1h, current_price)

    all_setups = report.order_blocks + report.fvgs + report.liq_sweeps
    strong     = [s for s in all_setups if s.strength == "STRONG"]
    if strong:
        report.best_setup = max(strong, key=lambda s: s.pips_to_target / max(s.risk_pips, 1))
    elif all_setups:
        report.best_setup = max(all_setups, key=lambda s: s.pips_to_target / max(s.risk_pips, 1))

    return report


def analyse_nas100_scalp(df_5m: pd.DataFrame,
                          df_1d: pd.DataFrame,
                          current_price: float,
                          qqq_to_nas100_ratio: float = 40.0) -> ScalpReport:
    """
    Run all NAS100 scalping detectors including enhanced liquidity sweep detection.
    current_price is NAS100 index points (~19000).
    """
    session = _get_session()
    report  = ScalpReport(ticker="NAS100", current_price=current_price, session=session)

    _raw_vwap = _calc_vwap(df_5m)
    report.vwap = round(_raw_vwap * qqq_to_nas100_ratio, 0) if _raw_vwap else None
    if report.vwap:
        report.vwap_deviation_pct = round(
            (current_price - report.vwap) / report.vwap * 100, 3
        )
        report.vwap_setup = _detect_vwap_setup_scaled(
            df_5m, report.vwap, current_price, qqq_to_nas100_ratio
        )

    report.gap_fill_setup = _detect_gap_fill(df_1d, df_5m, current_price, qqq_to_nas100_ratio)
    report.key_levels, report.key_level_setup = _detect_key_levels(
        df_1d, df_5m, current_price, qqq_to_nas100_ratio
    )
    report.open_drive = _detect_open_drive(df_5m)

    # ── ENHANCED: Liquidity sweeps with RVOL confirmation ────────────────────
    report.liquidity_sweeps = _detect_nas100_liquidity_sweeps(
        df_5m, df_1d, current_price, qqq_to_nas100_ratio
    )
    # Most recent / strongest sweep becomes the active fade setup
    strong_sweeps = [s for s in report.liquidity_sweeps if s.rvol_spike]
    if strong_sweeps:
        report.active_fade_setup = strong_sweeps[0]
    elif report.liquidity_sweeps:
        report.active_fade_setup = report.liquidity_sweeps[0]

    # Best overall setup
    candidates = [s for s in [report.vwap_setup, report.gap_fill_setup,
                               report.key_level_setup] if s is not None]
    strong = [s for s in candidates if s.strength == "STRONG"]
    if strong:
        report.best_setup = strong[0]
    elif candidates:
        report.best_setup = candidates[0]

    return report
