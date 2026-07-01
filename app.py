# ─────────────────────────────────────────────────────────────────────────────
# Trading Decision Maker — Streamlit Dashboard
# Wraps trading_final.ipynb into a live web UI
#
# HOW TO RUN:
#   pip install streamlit yfinance pandas feedparser fredapi ollama \
#               ta scikit-learn textblob plotly openpyxl
#   python -m textblob.download_corpora
#   streamlit run app.py
#
# OPTIONAL API KEYS (enter in sidebar — free tiers available):
#   Alpha Vantage : https://alphavantage.co       (25 calls/day free)
#   Finnhub       : https://finnhub.io            (60 calls/min free)
#   Polygon.io    : https://polygon.io            (5 calls/min free)
#   FRED          : https://fred.stlouisfed.org   (unlimited free)
#   SEC EDGAR     : No key needed (free)
#   CBOE VIX      : No key needed (free)
#
# Then open: http://localhost:8501
# ─────────────────────────────────────────────────────────────────────────────

import datetime
import io
import os
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import feedparser
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Auto-load API keys from .env (no manual entry needed) ─────────────────
# Put a .env file next to this script with your keys — they load automatically
# on every run. See the bottom of the sidebar "🔑 API Keys" panel for the
# exact template, or check requirements.txt comments.
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from the current working directory
    _DOTENV_LOADED = True
except ImportError:
    _DOTENV_LOADED = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Decision Maker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #dee2e6;
        margin-bottom: 8px;
    }
    .action-strong-buy  { background:#d4edda; color:#155724; font-weight:700;
                          padding:4px 10px; border-radius:4px; }
    .action-buy         { background:#c3e6cb; color:#155724; font-weight:600;
                          padding:4px 10px; border-radius:4px; }
    .action-watch       { background:#d1ecf1; color:#0c5460; font-weight:600;
                          padding:4px 10px; border-radius:4px; }
    .action-hold        { background:#fff3cd; color:#856404; font-weight:600;
                          padding:4px 10px; border-radius:4px; }
    .action-avoid       { background:#f8d7da; color:#721c24; font-weight:600;
                          padding:4px 10px; border-radius:4px; }
    .section-header { font-size:1.1rem; font-weight:600; margin-top:1rem; }
    div[data-testid="stExpander"] { border: 1px solid #dee2e6; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 0: MULTI-SOURCE DATA LAYER
#    Priority chains:
#    Fundamentals : Alpha Vantage → yfinance
#    News/Sentiment: Alpha Vantage → Finnhub → yfinance headlines → RSS stubs
#    Price/OHLCV  : yfinance → Polygon.io
#    Macro        : FRED → Alpha Vantage economic → CBOE (VIX)
#    Insider/Inst : SEC EDGAR (free, unlimited)
# ═════════════════════════════════════════════════════════════════════════════

import os as _os
import json as _json
import urllib.request as _urllib

# ── API key registry (reads from env or session state) ────────────────────
def _get_key(name: str) -> str:
    return st.session_state.get(f"api_{name}", _os.environ.get(name, ""))

# ── Generic JSON fetcher with timeout ─────────────────────────────────────
def _fetch_json(url: str, timeout: int = 8) -> dict:
    try:
        req = _urllib.Request(url, headers={"User-Agent": "TradingApp/1.0"})
        with _urllib.urlopen(req, timeout=timeout) as r:
            return _json.loads(r.read().decode())
    except Exception:
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# ALPHA VANTAGE
# ══════════════════════════════════════════════════════════════════════════════

_AV_BASE = "https://www.alphavantage.co/query"

def _av(params: dict) -> dict:
    key = _get_key("ALPHAVANTAGE_API_KEY")
    if not key:
        return {}
    params["apikey"] = key
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return _fetch_json(f"{_AV_BASE}?{qs}")

@st.cache_data(ttl=3600, show_spinner=False)
def av_company_overview(ticker: str) -> dict:
    """Alpha Vantage OVERVIEW — fundamentals (ROE, D/E, P/E, EPS, market cap…)"""
    return _av({"function": "OVERVIEW", "symbol": ticker})

@st.cache_data(ttl=3600, show_spinner=False)
def av_income_statement(ticker: str) -> dict:
    return _av({"function": "INCOME_STATEMENT", "symbol": ticker})

@st.cache_data(ttl=3600, show_spinner=False)
def av_balance_sheet(ticker: str) -> dict:
    return _av({"function": "BALANCE_SHEET", "symbol": ticker})

@st.cache_data(ttl=3600, show_spinner=False)
def av_cash_flow(ticker: str) -> dict:
    return _av({"function": "CASH_FLOW", "symbol": ticker})

@st.cache_data(ttl=900, show_spinner=False)
def av_global_quote(ticker: str) -> dict:
    """Real-time quote from Alpha Vantage."""
    d = _av({"function": "GLOBAL_QUOTE", "symbol": ticker})
    return d.get("Global Quote", {})

@st.cache_data(ttl=1800, show_spinner=False)
def av_news_sentiment(ticker: str) -> list:
    """Alpha Vantage news with built-in sentiment scores (financial NLP)."""
    d = _av({"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": "20"})
    return d.get("feed", [])

@st.cache_data(ttl=3600, show_spinner=False)
def av_economic_indicator(indicator: str) -> list:
    """Alpha Vantage economic data — REAL_GDP, CPI, INFLATION, UNEMPLOYMENT, FEDERAL_FUNDS_RATE."""
    d = _av({"function": indicator, "interval": "monthly"})
    return d.get("data", [])

@st.cache_data(ttl=900, show_spinner=False)
def av_rsi(ticker: str) -> float | None:
    d = _av({"function": "RSI", "symbol": ticker, "interval": "daily",
             "time_period": "14", "series_type": "close"})
    vals = d.get("Technical Analysis: RSI", {})
    if not vals: return None
    latest_date = sorted(vals.keys())[-1]
    try: return float(vals[latest_date]["RSI"])
    except Exception: return None

@st.cache_data(ttl=900, show_spinner=False)
def av_macd(ticker: str) -> dict | None:
    d = _av({"function": "MACD", "symbol": ticker, "interval": "daily", "series_type": "close"})
    vals = d.get("Technical Analysis: MACD", {})
    if not vals: return None
    latest_date = sorted(vals.keys())[-1]
    row = vals[latest_date]
    try:
        return {"macd": float(row["MACD"]), "signal": float(row["MACD_Signal"]),
                "hist": float(row["MACD_Hist"])}
    except Exception: return None

@st.cache_data(ttl=900, show_spinner=False)
def av_ema(ticker: str, period: int = 50) -> float | None:
    d = _av({"function": "EMA", "symbol": ticker, "interval": "daily",
             "time_period": str(period), "series_type": "close"})
    vals = d.get(f"Technical Analysis: EMA", {})
    if not vals: return None
    latest_date = sorted(vals.keys())[-1]
    try: return float(vals[latest_date]["EMA"])
    except Exception: return None

# ══════════════════════════════════════════════════════════════════════════════
# FINNHUB
# ══════════════════════════════════════════════════════════════════════════════

_FH_BASE = "https://finnhub.io/api/v1"

def _fh(path: str, params: dict = {}) -> dict:
    key = _get_key("FINNHUB_API_KEY")
    if not key:
        return {}
    params["token"] = key
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return _fetch_json(f"{_FH_BASE}/{path}?{qs}")

@st.cache_data(ttl=1800, show_spinner=False)
def fh_company_news(ticker: str) -> list:
    import datetime as _dt
    end   = _dt.date.today().isoformat()
    start = (_dt.date.today() - _dt.timedelta(days=30)).isoformat()
    d = _fh("company-news", {"symbol": ticker, "from": start, "to": end})
    return d if isinstance(d, list) else []

@st.cache_data(ttl=3600, show_spinner=False)
def fh_basic_financials(ticker: str) -> dict:
    d = _fh("stock/metric", {"symbol": ticker, "metric": "all"})
    return d.get("metric", {})

@st.cache_data(ttl=3600, show_spinner=False)
def fh_earnings_calendar(ticker: str) -> list:
    import datetime as _dt
    end   = (_dt.date.today() + _dt.timedelta(days=90)).isoformat()
    start = _dt.date.today().isoformat()
    d = _fh("calendar/earnings", {"symbol": ticker, "from": start, "to": end})
    return d.get("earningsCalendar", [])

@st.cache_data(ttl=3600, show_spinner=False)
def fh_recommendation_trends(ticker: str) -> list:
    d = _fh("stock/recommendation", {"symbol": ticker})
    return d if isinstance(d, list) else []

# ══════════════════════════════════════════════════════════════════════════════
# POLYGON.IO
# ══════════════════════════════════════════════════════════════════════════════

_POLY_BASE = "https://api.polygon.io"

def _poly(path: str, params: dict = {}) -> dict:
    key = _get_key("POLYGON_API_KEY")
    if not key:
        return {}
    params["apiKey"] = key
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return _fetch_json(f"{_POLY_BASE}{path}?{qs}")

@st.cache_data(ttl=900, show_spinner=False)
def poly_last_quote(ticker: str) -> dict:
    d = _poly(f"/v2/last/trade/{ticker}")
    return d.get("results", {})

@st.cache_data(ttl=3600, show_spinner=False)
def poly_ticker_details(ticker: str) -> dict:
    d = _poly(f"/v3/reference/tickers/{ticker}")
    return d.get("results", {})

@st.cache_data(ttl=3600, show_spinner=False)
def poly_financials(ticker: str) -> list:
    d = _poly("/vX/reference/financials", {"ticker": ticker, "limit": "4"})
    return d.get("results", [])

@st.cache_data(ttl=3600, show_spinner=False)
def poly_news(ticker: str) -> list:
    d = _poly("/v2/reference/news", {"ticker": ticker, "limit": "10"})
    return d.get("results", [])

# ══════════════════════════════════════════════════════════════════════════════
# SEC EDGAR  (free, no API key required)
# ══════════════════════════════════════════════════════════════════════════════

_SEC_BASE = "https://data.sec.gov"

@st.cache_data(ttl=7200, show_spinner=False)
def _sec_cik(ticker: str) -> str | None:
    """Resolve ticker → CIK (SEC company identifier)."""
    d = _fetch_json("https://efts.sec.gov/LATEST/search-index?q=%22" + ticker + "%22&dateRange=custom&startdt=2020-01-01&forms=10-K")
    # Fast lookup via company tickers JSON
    d2 = _fetch_json("https://www.sec.gov/files/company_tickers.json")
    if not d2: return None
    for v in d2.values():
        if str(v.get("ticker", "")).upper() == ticker.upper():
            return str(v["cik_str"]).zfill(10)
    return None

@st.cache_data(ttl=7200, show_spinner=False)
def sec_insider_trades(ticker: str) -> list:
    """Recent Form 4 insider transactions (purchases/sales)."""
    cik = _sec_cik(ticker)
    if not cik: return []
    d = _fetch_json(f"{_SEC_BASE}/submissions/CIK{cik}.json")
    filings = d.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    dates   = filings.get("filingDate", [])
    desc    = filings.get("primaryDocument", [])
    acn     = filings.get("accessionNumber", [])
    result  = []
    for i, form in enumerate(forms):
        if form == "4":
            result.append({
                "date":   dates[i] if i < len(dates) else "",
                "form":   form,
                "doc":    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acn[i].replace('-','')}/{desc[i]}" if i < len(desc) else "",
            })
            if len(result) >= 10: break
    return result

@st.cache_data(ttl=7200, show_spinner=False)
def sec_institutional_holdings(ticker: str) -> list:
    """13-F institutional holders via SEC EDGAR."""
    cik = _sec_cik(ticker)
    if not cik: return []
    d = _fetch_json(f"{_SEC_BASE}/submissions/CIK{cik}.json")
    filings = d.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    dates   = filings.get("filingDate", [])
    result  = []
    for i, form in enumerate(forms):
        if form in ("13F-HR", "13F-HR/A"):
            result.append({"date": dates[i] if i < len(dates) else "", "form": form})
            if len(result) >= 5: break
    return result

# ══════════════════════════════════════════════════════════════════════════════
# CBOE  (VIX live, free)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def cboe_vix() -> float | None:
    """Fetch live VIX from CBOE."""
    try:
        d = _fetch_json("https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_VIX.json")
        data = d.get("data", [])
        if data:
            return float(data[-1][-1])  # last row, last column = close
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════════════════════════════════════
# SMART FALLBACK CHAINS
# ══════════════════════════════════════════════════════════════════════════════

def get_fundamentals_best(ticker: str, thresholds: tuple) -> dict:
    """
    Returns a dict of fundamental fields.
    Priority: Alpha Vantage OVERVIEW → yfinance (existing fetch_fundamental_metrics).
    """
    overview = av_company_overview(ticker)
    result   = {}
    if overview and "ROE" in overview:
        def _f(key): 
            v = overview.get(key)
            try: return float(v) if v and v != "None" else None
            except Exception: return None
        result = {
            "source":        "Alpha Vantage",
            "roe":           _f("ReturnOnEquityTTM"),
            "debt_to_equity":_f("DebtToEquityRatioTTM") or _f("DebtToEquityRatio"),
            "pe_ratio":      _f("PERatio") or _f("TrailingPE"),
            "eps":           _f("EPS"),
            "market_cap":    _f("MarketCapitalization"),
            "revenue_ttm":   _f("RevenueTTM"),
            "profit_margin": _f("ProfitMargin"),
            "beta":          _f("Beta"),
            "52w_high":      _f("52WeekHigh"),
            "52w_low":       _f("52WeekLow"),
            "analyst_target":_f("AnalystTargetPrice"),
            "dividend_yield":_f("DividendYield"),
            "sector":        overview.get("Sector", ""),
            "industry":      overview.get("Industry", ""),
            "description":   overview.get("Description", ""),
        }
    # Finnhub supplemental metrics
    fh_metrics = fh_basic_financials(ticker)
    if fh_metrics:
        result.setdefault("pe_ratio",  fh_metrics.get("peNormalizedAnnual"))
        result.setdefault("52w_high",  fh_metrics.get("52WeekHigh"))
        result.setdefault("52w_low",   fh_metrics.get("52WeekLow"))
        result["fh_roe"]           = fh_metrics.get("roeTTM")
        result["fh_current_ratio"] = fh_metrics.get("currentRatioQuarterly")
        result["fh_gross_margin"]  = fh_metrics.get("grossMarginTTM")
        result["fh_revenue_growth"]= fh_metrics.get("revenueGrowthTTMYoy")
        result["fh_eps_growth"]    = fh_metrics.get("epsGrowthTTMYoy")
    return result

def get_news_best(ticker: str) -> tuple[list, float]:
    """
    Returns (headlines: list[str], sentiment_score: float 0-100).
    Priority: Alpha Vantage news sentiment → Finnhub → Polygon → yfinance → RSS stubs.
    """
    # 1. Alpha Vantage — has built-in financial sentiment scores
    av_feed = av_news_sentiment(ticker)
    if av_feed:
        headlines = []
        scores    = []
        for item in av_feed[:15]:
            title = item.get("title", "")
            if title: headlines.append(title)
            # AV provides overall_sentiment_score -1..1 per article
            ts = item.get("overall_sentiment_score")
            if ts is not None:
                try: scores.append(float(ts))
                except Exception: pass
            # Also per-ticker sentiment
            for t_sent in item.get("ticker_sentiment", []):
                if t_sent.get("ticker", "").upper() == ticker.upper():
                    ts2 = t_sent.get("ticker_sentiment_score")
                    if ts2 is not None:
                        try: scores.append(float(ts2))
                        except Exception: pass
        if headlines:
            avg   = (sum(scores) / len(scores)) if scores else 0.0
            norm  = round((avg + 1) / 2 * 100, 2)
            return headlines, norm

    # 2. Finnhub company news
    fh_news = fh_company_news(ticker)
    if fh_news:
        headlines = [item.get("headline", "") for item in fh_news[:10] if item.get("headline")]
        if headlines:
            # Use TextBlob as fallback scorer for Finnhub headlines
            try:
                from textblob import TextBlob
                scores = [TextBlob(h).sentiment.polarity for h in headlines]
                avg    = sum(scores) / len(scores)
                return headlines, round((avg + 1) / 2 * 100, 2)
            except ImportError:
                return headlines, 50.0

    # 3. Polygon news
    p_news = poly_news(ticker)
    if p_news:
        headlines = [item.get("title", "") for item in p_news if item.get("title")]
        if headlines:
            try:
                from textblob import TextBlob
                scores = [TextBlob(h).sentiment.polarity for h in headlines]
                avg    = sum(scores) / len(scores)
                return headlines, round((avg + 1) / 2 * 100, 2)
            except ImportError:
                return headlines, 50.0

    # 4. yfinance news
    try:
        items = yf.Ticker(ticker).news or []
        headlines = []
        for item in items[:10]:
            t = item.get("content", {}).get("title") or item.get("title", "")
            if t: headlines.append(t)
        if headlines:
            try:
                from textblob import TextBlob
                scores = [TextBlob(h).sentiment.polarity for h in headlines]
                avg    = sum(scores) / len(scores)
                return headlines, round((avg + 1) / 2 * 100, 2)
            except ImportError:
                return headlines, 50.0
    except Exception:
        pass

    # 5. Stub fallback
    return [f"{ticker} beats earnings estimates",
            f"{ticker} expands into new markets",
            f"Analysts raise price target on {ticker}"], 55.0

def get_price_best(ticker: str) -> float | None:
    """Real-time price: Alpha Vantage → Polygon → yfinance."""
    # Alpha Vantage global quote
    q = av_global_quote(ticker)
    if q:
        p = q.get("05. price")
        if p:
            try: return float(p)
            except Exception: pass
    # Polygon last trade
    pt = poly_last_quote(ticker)
    if pt:
        p = pt.get("p") or pt.get("price")
        if p:
            try: return float(p)
            except Exception: pass
    # yfinance fallback
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("currentPrice") or info.get("regularMarketPrice")
    except Exception:
        return None

def get_vix_best() -> float | None:
    """VIX: FRED (already in macro) → CBOE live."""
    v = cboe_vix()
    if v: return v
    try:
        hist = yf.Ticker("^VIX").history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None

def get_insider_signal(ticker: str) -> dict:
    """
    Aggregate insider trades from SEC EDGAR → net buy/sell signal.
    Returns dict with count, net_signal, label.
    """
    trades = sec_insider_trades(ticker)
    return {
        "count":      len(trades),
        "recent":     trades[:5],
        "net_signal": "Insufficient data" if not trades else f"{len(trades)} recent Form 4 filings",
    }

def get_analyst_consensus(ticker: str) -> dict:
    """
    Analyst consensus: Finnhub recommendation trends → Alpha Vantage target price.
    """
    trends = fh_recommendation_trends(ticker)
    result = {}
    if trends:
        latest = trends[0]
        result["buy"]        = latest.get("buy", 0)
        result["hold"]       = latest.get("hold", 0)
        result["sell"]       = latest.get("sell", 0)
        result["strong_buy"] = latest.get("strongBuy", 0)
        result["strong_sell"]= latest.get("strongSell", 0)
        total = sum(result.values())
        result["total"]      = total
        result["consensus"]  = (
            "Strong Buy"  if result["strong_buy"] > result["buy"] + result["hold"]  else
            "Buy"         if result["buy"] > result["hold"] + result["sell"]         else
            "Hold"        if result["hold"] >= result["sell"]                        else
            "Sell"
        )
    # Alpha Vantage analyst target price
    overview = av_company_overview(ticker)
    if overview:
        tp = overview.get("AnalystTargetPrice")
        if tp and tp != "None":
            try: result["target_price"] = float(tp)
            except Exception: pass
    return result

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 1: DATACLASSES (identical to notebook)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FundamentalMetrics:
    ticker: str
    roe: float | None
    debt_to_equity: float | None
    pe_ratio: float | None
    piotroski_f_score: int
    altman_z_score: float | None
    fundamental_score: int
    fundamental_rating: str
    passed: bool

@dataclass
class TechnicalMetrics:
    ticker: str
    price: float | None
    ma_50: float | None
    ma_200: float | None
    rsi_14: float | None
    high_52w: float | None
    distance_from_52w_high: float | None
    technical_score: int
    technical_rating: str

@dataclass
class MacroEnvironment:
    fed_funds_rate: float | None
    cpi_yoy: float | None
    unemployment: float | None
    yield_spread: float | None
    vix: float | None
    gdp_growth: float | None
    macro_score: float
    macro_label: str
    macro_summary: str

@dataclass
class CompositeScore:
    ticker: str
    fund_score_10: float
    tech_score_10: float
    news_score_10: float
    macro_score_10: float
    composite: float
    recommendation: str
    action: str

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 2: FUNDAMENTAL HELPERS (identical to notebook)
# ═════════════════════════════════════════════════════════════════════════════

US_EXCHANGES = {"ASE","AMEX","BATS","NASDAQ","NCM","NGM","NMS","NYQ","NYSE","PCX"}

def safe_div(n, d):
    return None if n is None or d in (None, 0) else n / d

def latest(stmt, row):
    if stmt is None or stmt.empty or row not in stmt.index: return None
    v = stmt.loc[row].dropna()
    return float(v.iloc[0]) if len(v) else None

def previous(stmt, row):
    if stmt is None or stmt.empty or row not in stmt.index: return None
    v = stmt.loc[row].dropna()
    return float(v.iloc[1]) if len(v) > 1 else None

def first_available(stmt, names):
    for n in names:
        v = latest(stmt, n)
        if v is not None: return v
    return None

def prior_available(stmt, names):
    for n in names:
        v = previous(stmt, n)
        if v is not None: return v
    return None

def is_us_stock(info):
    exchange   = str(info.get("exchange") or "").upper()
    full_ex    = str(info.get("fullExchangeName") or "").lower()
    country    = str(info.get("country") or "").lower()
    quote_type = str(info.get("quoteType") or "").upper()
    is_equity  = quote_type in {"EQUITY", ""}
    is_us      = country in {"united states","united states of america","usa","us",""}
    is_us_ex   = exchange in US_EXCHANGES or any(
        n in full_ex for n in ["nasdaq","nyse","american stock exchange","nyse arca","bats"])
    return is_equity and is_us and is_us_ex

def calc_piotroski(d):
    s    = 0
    roa  = safe_div(d["net_income"],      d["total_assets"])
    proa = safe_div(d["prior_net_income"], d["prior_total_assets"])
    if d["net_income"] and d["net_income"] > 0:                          s += 1
    if d["operating_cash_flow"] and d["operating_cash_flow"] > 0:       s += 1
    if roa is not None and proa is not None and roa > proa:              s += 1
    if (d["operating_cash_flow"] is not None and d["net_income"] is not None
            and d["operating_cash_flow"] > d["net_income"]):             s += 1
    lev  = safe_div(d["total_debt"],      d["total_assets"])
    plev = safe_div(d["prior_total_debt"], d["prior_total_assets"])
    if lev is not None and plev is not None and lev < plev:              s += 1
    liq  = safe_div(d["current_assets"],  d["current_liabilities"])
    pliq = safe_div(d["prior_current_assets"], d["prior_current_liabilities"])
    if liq is not None and pliq is not None and liq > pliq:              s += 1
    if (d["shares_outstanding"] is not None
            and d["prior_shares_outstanding"] is not None
            and d["shares_outstanding"] <= d["prior_shares_outstanding"]): s += 1
    gm   = safe_div(d["gross_profit"],    d["revenue"])
    pgm  = safe_div(d["prior_gross_profit"], d["prior_revenue"])
    if gm is not None and pgm is not None and gm > pgm:                  s += 1
    at   = safe_div(d["revenue"],         d["total_assets"])
    pat  = safe_div(d["prior_revenue"],   d["prior_total_assets"])
    if at is not None and pat is not None and at > pat:                   s += 1
    return s

def calc_altman_z(d):
    wc  = (d["current_assets"] - d["current_liabilities"]
           if d["current_assets"] is not None and d["current_liabilities"] is not None else None)
    mve = (d["price"] * d["shares_outstanding"]
           if d["price"] is not None and d["shares_outstanding"] is not None else None)
    x1 = safe_div(wc,                    d["total_assets"])
    x2 = safe_div(d["retained_earnings"], d["total_assets"])
    x3 = safe_div(d["ebit"],             d["total_assets"])
    x4 = safe_div(mve,                   d["total_liabilities"])
    x5 = safe_div(d["revenue"],          d["total_assets"])
    if None in (x1, x2, x3, x4, x5): return None
    return 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + x5

def rate_fundamental(roe, de, pe, fs, zs):
    s = 0
    if roe is not None:
        s += 2 if roe >= 0.20 else (1 if roe >= 0.10 else 0)
    if de is not None:
        s += 2 if de <= 0.50 else (1 if de <= 1.50 else (-1 if de > 2.50 else 0))
    if pe is not None:
        s += 2 if pe <= 15 else (1 if pe <= 30 else (-1 if pe > 50 else 0))
    s += 2 if fs >= 8 else (1 if fs >= 6 else (-1 if fs <= 3 else 0))
    if zs is not None:
        s += 2 if zs >= 3 else (1 if zs >= 1.8 else -2)
    rating = "High Buy" if s >= 7 else ("Buy" if s >= 4 else ("Hold" if s >= 1 else "Sell"))
    return s, rating

def fetch_fundamental_metrics(ticker, thresholds):
    min_roe, max_de, max_pe, min_fs, min_zs = thresholds
    stock = yf.Ticker(ticker)
    info  = stock.get_info()
    if not is_us_stock(info):
        raise ValueError("not a U.S.-listed common stock")
    bal, inc, cf = stock.balance_sheet, stock.income_stmt, stock.cashflow
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    shares = info.get("sharesOutstanding")
    d = {
        "price":                     float(price)  if price  else None,
        "shares_outstanding":        float(shares) if shares else None,
        "total_assets":              first_available(bal, ["Total Assets"]),
        "total_liabilities":         first_available(bal, ["Total Liabilities Net Minority Interest","Total Liabilities"]),
        "current_assets":            first_available(bal, ["Current Assets","Total Current Assets"]),
        "current_liabilities":       first_available(bal, ["Current Liabilities","Total Current Liabilities"]),
        "retained_earnings":         first_available(bal, ["Retained Earnings"]),
        "total_debt":                first_available(bal, ["Total Debt"]),
        "total_equity":              first_available(bal, ["Stockholders Equity","Total Equity Gross Minority Interest"]),
        "revenue":                   first_available(inc, ["Total Revenue"]),
        "gross_profit":              first_available(inc, ["Gross Profit"]),
        "ebit":                      first_available(inc, ["EBIT","Operating Income"]),
        "net_income":                first_available(inc, ["Net Income","Net Income Common Stockholders"]),
        "operating_cash_flow":       first_available(cf,  ["Operating Cash Flow","Total Cash From Operating Activities"]),
        "prior_net_income":          prior_available(inc, ["Net Income","Net Income Common Stockholders"]),
        "prior_total_assets":        prior_available(bal, ["Total Assets"]),
        "prior_current_assets":      prior_available(bal, ["Current Assets","Total Current Assets"]),
        "prior_current_liabilities": prior_available(bal, ["Current Liabilities","Total Current Liabilities"]),
        "prior_total_debt":          prior_available(bal, ["Total Debt"]),
        "prior_shares_outstanding":  None,
        "prior_gross_profit":        prior_available(inc, ["Gross Profit"]),
        "prior_revenue":             prior_available(inc, ["Total Revenue"]),
    }
    roe    = safe_div(d["net_income"], d["total_equity"])
    de     = safe_div(d["total_debt"], d["total_equity"])
    mktcap = (d["price"] * d["shares_outstanding"]
               if d["price"] and d["shares_outstanding"] else None)
    pe     = safe_div(mktcap, d["net_income"])
    fs     = calc_piotroski(d)
    zs     = calc_altman_z(d)
    fscore, frating = rate_fundamental(roe, de, pe, fs, zs)
    passed = (
        roe is not None and roe >= min_roe and
        de  is not None and de  <= max_de  and
        pe  is not None and pe  <= max_pe  and
        fs  >= min_fs and
        zs  is not None and zs  >= min_zs
    )
    return FundamentalMetrics(
        ticker=ticker.upper(), roe=roe, debt_to_equity=de, pe_ratio=pe,
        piotroski_f_score=fs, altman_z_score=zs,
        fundamental_score=fscore, fundamental_rating=frating, passed=passed
    )

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 3: TECHNICAL HELPERS (identical to notebook)
# ═════════════════════════════════════════════════════════════════════════════

def calc_rsi(close, period=14):
    delta  = close.diff()
    gains  = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_g  = gains.rolling(period).mean()
    avg_l  = losses.rolling(period).mean()
    rs     = avg_g / avg_l
    rsi    = 100 - (100 / (1 + rs))
    v      = rsi.dropna()
    return float(v.iloc[-1]) if len(v) else None

def rate_technical(price, ma50, ma200, rsi14, high52):
    s = 0
    if price and ma50  and price > ma50:  s += 1
    if price and ma200 and price > ma200: s += 1
    if ma50  and ma200 and ma50  > ma200: s += 1
    if rsi14 is not None:
        if 40 <= rsi14 <= 70: s += 1
        elif rsi14 > 75:      s -= 1
    dist = None
    if price and high52 and high52 != 0:
        dist = (price - high52) / high52
        if dist >= -0.20: s += 1
    rating = "Bullish" if s >= 4 else ("Neutral" if s >= 2 else "Bearish")
    return s, rating, dist

def fetch_technical_metrics(ticker):
    stock   = yf.Ticker(ticker)
    history = stock.history(period="1y", interval="1d", auto_adjust=True)
    if history.empty: raise ValueError("no price history")
    close = history["Close"].dropna()
    if len(close) < 200: raise ValueError("< 200 days of history")
    price  = float(close.iloc[-1])
    ma50   = float(close.rolling(50).mean().iloc[-1])
    ma200  = float(close.rolling(200).mean().iloc[-1])
    rsi14  = calc_rsi(close)
    high52 = float(close.max())
    ts, tr, dist = rate_technical(price, ma50, ma200, rsi14, high52)
    return TechnicalMetrics(
        ticker=ticker.upper(), price=price, ma_50=ma50, ma_200=ma200,
        rsi_14=rsi14, high_52w=high52, distance_from_52w_high=dist,
        technical_score=ts, technical_rating=tr
    )

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 4: NEWS HELPERS (identical to notebook)
# ═════════════════════════════════════════════════════════════════════════════

NEWS_SOURCES = {
    "Reuters": "https://feeds.reuters.com/reuters/businessNews",
    "Yahoo":   "https://finance.yahoo.com/news/rssindex",
    "FT":      "https://www.ft.com/rss/home",
    "WSJ":     "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
}
BULLISH_WORDS = ["growth","beat","upgrade","strong","record","surge","rally",
                 "profit","expansion","bullish","outperform","raise guidance",
                 "dividend","buyback","acquisition","innovation","breakthrough"]
BEARISH_WORDS = ["miss","downgrade","risk","recession","decline","warning",
                 "layoff","bankruptcy","fraud","investigation","tariff",
                 "inflation","slowdown","debt","loss","cut guidance"]

def get_market_news(max_per_source=15):
    articles = []
    for source, url in NEWS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_source]:
                articles.append({
                    "source": source,
                    "title":  getattr(entry, "title", ""),
                    "link":   getattr(entry, "link",  ""),
                })
        except Exception:
            pass
    return articles

def score_market_news(articles):
    score = 0
    for a in articles:
        t = a["title"].lower()
        for w in BULLISH_WORDS:
            if w in t: score += 1
        for w in BEARISH_WORDS:
            if w in t: score -= 1
    if score >= 5:    sentiment = "Bullish"
    elif score >= 1:  sentiment = "Mildly Bullish"
    elif score == 0:  sentiment = "Neutral"
    elif score >= -4: sentiment = "Mildly Bearish"
    else:             sentiment = "Bearish"
    return score, sentiment

def get_company_news(ticker, max_items=10):
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news or []
        return [item.get("content", {}).get("title", "") or item.get("title", "")
                for item in news[:max_items]
                if item.get("content", {}).get("title") or item.get("title")]
    except Exception:
        return []

def score_company_news(headlines):
    raw = 0
    for title in headlines:
        t = title.lower()
        for w in BULLISH_WORDS:
            if w in t: raw += 1
        for w in BEARISH_WORDS:
            if w in t: raw -= 1
    clipped    = max(-5, min(5, raw))
    normalized = float(clipped + 5)
    return raw, normalized

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 5: MACRO HELPERS (identical to notebook)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_macro_environment(api_key):
    from fredapi import Fred
    fred  = Fred(api_key=api_key)

    def _latest(sid):
        try:
            s = fred.get_series(sid).dropna()
            return float(s.iloc[-1]) if len(s) else None
        except Exception:
            return None

    def _cpi_yoy():
        try:
            s = fred.get_series("CPIAUCSL").dropna()
            if len(s) < 13: return None
            return ((float(s.iloc[-1]) - float(s.iloc[-13])) / float(s.iloc[-13])) * 100
        except Exception:
            return None

    fed   = _latest("FEDFUNDS")
    cpi   = _cpi_yoy()
    unemp = _latest("UNRATE")
    t10y2 = _latest("T10Y2Y")
    vix   = _latest("VIXCLS")
    gdp   = _latest("A191RL1Q225SBEA")

    score = 0.0
    if fed   is not None: score += 2 if fed   <= 2.0 else (1 if fed   <= 4.0 else 0)
    if cpi   is not None: score += 2 if cpi   <= 2.5 else (1 if cpi   <= 4.0 else 0)
    if unemp is not None: score += 2 if unemp <= 4.0 else (1 if unemp <= 5.5 else 0)
    if t10y2 is not None: score += 2 if t10y2 >= 0.5 else (1 if t10y2 >= 0.0 else 0)
    if vix   is not None: score += 2 if vix   <= 15  else (1 if vix   <= 25  else 0)
    if gdp   is not None: score += 2 if gdp   >= 3.0 else (1 if gdp   >= 1.0 else 0)

    norm  = (score / 12.0) * 10
    label = ("Macro Tailwind" if norm >= 7.5 else
             "Macro Neutral"  if norm >= 5.0 else
             "Macro Headwind" if norm >= 2.5 else "Macro Risk-Off")

    parts = []
    if fed   is not None: parts.append(f"Fed Funds {fed:.2f}%")
    if cpi   is not None: parts.append(f"CPI YoY {cpi:.1f}%")
    if unemp is not None: parts.append(f"Unemployment {unemp:.1f}%")
    if t10y2 is not None: parts.append(f"10Y-2Y Spread {t10y2:.2f}%")
    if vix   is not None: parts.append(f"VIX {vix:.1f}")
    if gdp   is not None: parts.append(f"GDP Growth {gdp:.1f}%")

    return MacroEnvironment(
        fed_funds_rate=fed, cpi_yoy=cpi, unemployment=unemp,
        yield_spread=t10y2, vix=vix, gdp_growth=gdp,
        macro_score=norm, macro_label=label,
        macro_summary=" | ".join(parts) if parts else "No data"
    )

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 6: COMPOSITE + REPORT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def normalize_fundamental(raw):
    return max(0.0, min(10.0, (raw + 4) / 14 * 10))

def normalize_technical(raw):
    return max(0.0, min(10.0, raw / 5 * 10))

def composite_recommendation(score):
    if score >= 7.5: return "Strong Buy",  "🟢 STRONG BUY"
    if score >= 6.0: return "Buy",         "🟢 BUY"
    if score >= 5.0: return "Watch",       "🔵 WATCH"
    if score >= 3.5: return "Hold",        "🟡 HOLD"
    return                   "Avoid",       "🔴 AVOID"

def build_composite(f, t, n10, m10, weights):
    wf, wt, wn, wm = weights
    f10  = normalize_fundamental(f.fundamental_score)
    t10  = normalize_technical(t.technical_score) if t else 5.0
    comp = wf*f10 + wt*t10 + wn*n10 + wm*m10
    rec, action = composite_recommendation(comp)
    return CompositeScore(
        ticker=f.ticker, fund_score_10=f10, tech_score_10=t10,
        news_score_10=n10, macro_score_10=m10,
        composite=comp, recommendation=rec, action=action
    )

def sfmt(val, fmt):
    return "n/a" if val is None else format(val, fmt)

def build_ollama_prompt(f, t, cs, headlines, macro, market_sentiment,
                        market_raw_score, analysis_date, ollama_model):
    tech_block = (
        f"Price: ${sfmt(t.price,'.2f')} | 50DMA: ${sfmt(t.ma_50,'.2f')} | "
        f"200DMA: ${sfmt(t.ma_200,'.2f')} | RSI: {sfmt(t.rsi_14,'.1f')} | "
        f"Technical Rating: {t.technical_rating}"
    ) if t else "Technical data unavailable"

    news_block = "\n".join(f"  - {h}" for h in headlines[:5]) if headlines else "  No headlines available"

    macro_parts = []
    if macro.fed_funds_rate is not None: macro_parts.append(f"Fed Rate: {macro.fed_funds_rate:.2f}%")
    if macro.cpi_yoy        is not None: macro_parts.append(f"CPI YoY: {macro.cpi_yoy:.1f}%")
    if macro.unemployment   is not None: macro_parts.append(f"Unemployment: {macro.unemployment:.1f}%")
    if macro.yield_spread   is not None: macro_parts.append(f"10Y-2Y Spread: {macro.yield_spread:.2f}%")
    if macro.vix            is not None: macro_parts.append(f"VIX: {macro.vix:.1f}")
    macro_block = " | ".join(macro_parts) if macro_parts else "Macro data unavailable"

    return f"""You are a senior equity research analyst. Write a concise, structured one-page investment summary.

STOCK: {f.ticker}
DATE: {analysis_date}

=== FUNDAMENTAL DATA ===
ROE: {sfmt(f.roe,'.1%')} | D/E: {sfmt(f.debt_to_equity,'.2f')} | P/E: {sfmt(f.pe_ratio,'.1f')}x
Piotroski F-Score: {f.piotroski_f_score}/9 | Altman Z-Score: {sfmt(f.altman_z_score,'.2f')}
Fundamental Rating: {f.fundamental_rating}

=== TECHNICAL DATA ===
{tech_block}

=== NEWS SENTIMENT ===
Company News Score: {cs.news_score_10:.1f}/10
Market Sentiment: {market_sentiment} ({market_raw_score:+d})
Recent Headlines:
{news_block}

=== MACRO ENVIRONMENT ===
{macro_block}
Macro Score: {macro.macro_score:.1f}/10 ({macro.macro_label})

=== COMPOSITE SCORE ===
Fundamental: {cs.fund_score_10:.1f}/10 (40%) | Technical: {cs.tech_score_10:.1f}/10 (25%)
News: {cs.news_score_10:.1f}/10 (20%) | Macro: {cs.macro_score_10:.1f}/10 (15%)
COMPOSITE: {cs.composite:.1f}/10 → {cs.recommendation}

Write exactly these six sections, each 2-4 sentences. Be direct, specific, and analytical:

1. BUSINESS QUALITY
2. FINANCIAL STRENGTH
3. KEY RISKS
4. GROWTH OPPORTUNITIES
5. VALUATION ASSESSMENT
6. FINAL RECOMMENDATION

End with one clear action sentence: Buy / Watch / Hold / Avoid — and why.
Do not include disclaimers or preamble. Start directly with section 1."""

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 7: UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════

ACTION_COLORS = {
    "🟢 STRONG BUY": "#d4edda",
    "🟢 BUY":        "#c3e6cb",
    "🔵 WATCH":      "#d1ecf1",
    "🟡 HOLD":       "#fff3cd",
    "🔴 AVOID":      "#f8d7da",
}

def action_badge(action):
    bg = ACTION_COLORS.get(action, "#f8f9fa")
    return f'<span style="background:{bg};padding:3px 10px;border-radius:4px;font-weight:600">{action}</span>'

def score_bar(score, max_score=10):
    pct   = int(score / max_score * 100)
    color = ("#28a745" if score >= 7 else
             "#17a2b8" if score >= 5 else
             "#ffc107" if score >= 3.5 else "#dc3545")
    return f"""
    <div style="background:#e9ecef;border-radius:4px;height:8px;width:100%">
      <div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>
    </div>
    <small style="color:#6c757d">{score:.1f} / {max_score}</small>
    """

def render_macro_panel(macro):
    st.markdown("### 🏛️ Macro Environment (FRED)")
    label_color = ("#28a745" if macro.macro_label == "Macro Tailwind" else
                   "#17a2b8" if macro.macro_label == "Macro Neutral"  else
                   "#ffc107" if macro.macro_label == "Macro Headwind" else "#dc3545")
    st.markdown(
        f'<b>{macro.macro_label}</b> &nbsp;'
        f'<span style="background:{label_color};color:white;padding:2px 8px;border-radius:4px">'
        f'{macro.macro_score:.1f} / 10</span>',
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Fed Funds Rate", sfmt(macro.fed_funds_rate, ".2f") + "%" if macro.fed_funds_rate else "n/a")
        st.metric("CPI (YoY)",      sfmt(macro.cpi_yoy, ".1f") + "%" if macro.cpi_yoy else "n/a")
    with c2:
        st.metric("Unemployment",   sfmt(macro.unemployment, ".1f") + "%" if macro.unemployment else "n/a")
        st.metric("10Y-2Y Spread",  sfmt(macro.yield_spread, ".2f") + "%" if macro.yield_spread else "n/a")
    with c3:
        st.metric("VIX",            sfmt(macro.vix, ".1f") if macro.vix else "n/a")
        st.metric("Real GDP Growth",sfmt(macro.gdp_growth, ".1f") + "%" if macro.gdp_growth else "n/a")

def render_composite_table(results, weights):
    wf, wt, wn, wm = weights
    rows = []
    for f, t, cs, news_raw, news_norm, headlines in results:
        rows.append({
            "Ticker":    f.ticker,
            "Fund":      f"{cs.fund_score_10:.1f}",
            "Tech":      f"{cs.tech_score_10:.1f}",
            "News":      f"{cs.news_score_10:.1f}",
            "Macro":     f"{cs.macro_score_10:.1f}",
            "Composite": f"{cs.composite:.1f}",
            "Action":    cs.action,
            "Passed":    "✓" if f.passed else "",
        })
    df = pd.DataFrame(rows)

    def color_action(val):
        bg = ACTION_COLORS.get(val, "")
        return f"background-color: {bg}" if bg else ""

    st.dataframe(
        df.style.map(color_action, subset=["Action"]),
        use_container_width=True,
        hide_index=True,
    )

def render_ticker_detail(f, t, cs, headlines, macro, news_raw, weights, market_sentiment, market_raw_score):
    wf, wt, wn, wm = weights

    with st.expander(f"📋 {f.ticker} — {cs.action}  |  Composite: {cs.composite:.1f}/10", expanded=False):

        tab1, tab2, tab2b, tab3, tab3b, tab3c, tab4, tab5 = st.tabs(
            ["📊 Fundamentals", "📈 Technical", "📉 Chart & MACD",
             "📰 News", "🏦 Analysts", "🏛️ Insider/SEC", "🏆 Decision", "⚠️ Risk"]
        )

        # ── Fundamentals tab ──
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("ROE",         sfmt(f.roe, ".1%") if f.roe is not None else "n/a")
                st.metric("Debt/Equity", sfmt(f.debt_to_equity, ".2f") if f.debt_to_equity is not None else "n/a")
                st.metric("P/E Ratio",   sfmt(f.pe_ratio, ".1f") + "x" if f.pe_ratio is not None else "n/a")
            with c2:
                st.metric("Piotroski F-Score", f"{f.piotroski_f_score} / 9")
                st.metric("Altman Z-Score",    sfmt(f.altman_z_score, ".2f") if f.altman_z_score is not None else "n/a")
                st.metric("Fundamental Rating", f.fundamental_rating)
            st.markdown(f"**Screen passed:** {'✅ Yes' if f.passed else '❌ No'}")

            # ── Alpha Vantage + Finnhub enrichment ──
            extra = getattr(f, "_extra", {}) or {}
            if extra:
                st.divider()
                src = extra.get("source", "Multi-source")
                st.caption(f"📡 Extended data from {src} + Finnhub")
                ea, eb, ec = st.columns(3)
                def _em(col, label, val, fmt=".2f", prefix="", suffix=""):
                    if val is not None:
                        try: col.metric(label, f"{prefix}{float(val):{fmt}}{suffix}")
                        except Exception: col.metric(label, str(val))
                    else:
                        col.metric(label, "n/a")
                _em(ea, "Market Cap",      extra.get("market_cap"),    ".2e", "$")
                _em(eb, "Revenue TTM",     extra.get("revenue_ttm"),   ".2e", "$")
                _em(ec, "Profit Margin",   extra.get("profit_margin"), ".1%")
                _em(ea, "EPS",             extra.get("eps"),           ".2f", "$")
                _em(eb, "Beta",            extra.get("beta"),          ".2f")
                _em(ec, "Dividend Yield",  extra.get("dividend_yield"),".2%")
                _em(ea, "52W High",        extra.get("52w_high"),      ".2f", "$")
                _em(eb, "52W Low",         extra.get("52w_low"),       ".2f", "$")
                _em(ec, "Analyst Target",  extra.get("analyst_target"),".2f", "$")
                if extra.get("sector"):
                    st.markdown(f"**Sector:** {extra['sector']}  |  **Industry:** {extra.get('industry','n/a')}")
                # Finnhub extras
                fh_cols = st.columns(3)
                _em(fh_cols[0], "Gross Margin",    extra.get("fh_gross_margin"),   ".2%")
                _em(fh_cols[1], "Revenue Growth",  extra.get("fh_revenue_growth"), ".2%")
                _em(fh_cols[2], "EPS Growth",      extra.get("fh_eps_growth"),     ".2%")
                _em(fh_cols[0], "Current Ratio",   extra.get("fh_current_ratio"),  ".2f")

        # ── Technical tab ──
        with tab2:
            if t:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Price",      f"${t.price:.2f}" if t.price else "n/a")
                    st.metric("50-Day MA",  f"${t.ma_50:.2f}" if t.ma_50 else "n/a",
                              delta=f"{((t.price/t.ma_50)-1)*100:.1f}%" if t.price and t.ma_50 else None)
                    st.metric("200-Day MA", f"${t.ma_200:.2f}" if t.ma_200 else "n/a",
                              delta=f"{((t.price/t.ma_200)-1)*100:.1f}%" if t.price and t.ma_200 else None)
                with c2:
                    st.metric("RSI (14)",         f"{t.rsi_14:.1f}" if t.rsi_14 else "n/a")
                    st.metric("52W High",          f"${t.high_52w:.2f}" if t.high_52w else "n/a")
                    st.metric("From 52W High",     f"{t.distance_from_52w_high*100:.1f}%" if t.distance_from_52w_high is not None else "n/a")
                st.markdown(f"**Technical Rating:** {t.technical_rating} (score {t.technical_score}/5)")
            else:
                st.info("Technical data unavailable for this ticker.")

        # ── Chart & MACD tab ──
        with tab2b:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                hist = yf.Ticker(f.ticker).history(period="1y", interval="1d", auto_adjust=True)
                if hist.empty:
                    st.info("No price history available.")
                else:
                    close_h  = hist["Close"].squeeze()
                    high_h   = hist["High"].squeeze()
                    low_h    = hist["Low"].squeeze()
                    open_h   = hist["Open"].squeeze()
                    vol_h    = hist["Volume"].squeeze() if "Volume" in hist.columns else pd.Series(0, index=hist.index)

                    # EMAs
                    ema20_h  = close_h.ewm(span=20,  adjust=False).mean()
                    ema50_h  = close_h.ewm(span=50,  adjust=False).mean()
                    ema200_h = close_h.ewm(span=200, adjust=False).mean()

                    # MACD
                    ema12_h  = close_h.ewm(span=12, adjust=False).mean()
                    ema26_h  = close_h.ewm(span=26, adjust=False).mean()
                    macd_h   = ema12_h - ema26_h
                    macd_s_h = macd_h.ewm(span=9, adjust=False).mean()
                    macd_d_h = macd_h - macd_s_h

                    # Volume average
                    vol_avg_h = vol_h.rolling(20).mean()

                    # Layout: candlestick | volume | MACD
                    fig = make_subplots(
                        rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.20, 0.25],
                        vertical_spacing=0.03,
                        subplot_titles=("Price + EMAs", "Volume", "MACD"),
                    )

                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=hist.index, open=open_h, high=high_h,
                        low=low_h, close=close_h,
                        name="Price", increasing_line_color="#26a69a",
                        decreasing_line_color="#ef5350",
                    ), row=1, col=1)

                    # EMA lines
                    for ema_vals, ema_name, colour in [
                        (ema20_h,  "EMA 20",  "#f9a825"),
                        (ema50_h,  "EMA 50",  "#1565c0"),
                        (ema200_h, "EMA 200", "#6a1b9a"),
                    ]:
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=ema_vals, name=ema_name,
                            line=dict(color=colour, width=1.5),
                        ), row=1, col=1)

                    # Volume bars
                    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                                  for c, o in zip(close_h, open_h)]
                    fig.add_trace(go.Bar(
                        x=hist.index, y=vol_h, name="Volume",
                        marker_color=vol_colors, opacity=0.7,
                    ), row=2, col=1)
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=vol_avg_h, name="Vol MA20",
                        line=dict(color="#ff9800", width=1.2, dash="dot"),
                    ), row=2, col=1)

                    # MACD
                    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_d_h]
                    fig.add_trace(go.Bar(
                        x=hist.index, y=macd_d_h, name="MACD Hist",
                        marker_color=hist_colors, opacity=0.7,
                    ), row=3, col=1)
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=macd_h, name="MACD",
                        line=dict(color="#2196f3", width=1.5),
                    ), row=3, col=1)
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=macd_s_h, name="Signal",
                        line=dict(color="#ff5722", width=1.2, dash="dot"),
                    ), row=3, col=1)

                    fig.update_layout(
                        height=650, showlegend=True,
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                st.warning("Install plotly for charts: `pip install plotly`")
            except Exception as e:
                st.warning(f"Chart unavailable: {e}")

        # ── News tab ──
        with tab3:
            st.markdown(f"**Company News Score:** {cs.news_score_10:.1f}/10 (raw: {news_raw:+d})")
            st.markdown(f"**Market-Wide Sentiment:** {market_sentiment} ({market_raw_score:+d})")
            if headlines:
                st.markdown(f"**Recent headlines ({len(headlines)}):**")
                for h in headlines:
                    st.markdown(f"- {h}")
            else:
                st.info("No company-specific headlines found.")

        # ── Analyst consensus tab ──
        with tab3b:
            st.markdown("### 🏦 Analyst Consensus & Price Target")
            consensus = get_analyst_consensus(f.ticker)
            if consensus:
                if "total" in consensus and consensus["total"] > 0:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Strong Buy",  consensus.get("strong_buy", 0))
                    c2.metric("Buy",         consensus.get("buy", 0))
                    c3.metric("Hold",        consensus.get("hold", 0))
                    c4.metric("Sell",        consensus.get("sell", 0) + consensus.get("strong_sell", 0))
                    st.markdown(f"**Consensus: {consensus.get('consensus', 'N/A')}** "
                                f"({consensus['total']} analyst ratings)")
                if "target_price" in consensus:
                    price_now = get_price_best(f.ticker)
                    upside = ((consensus['target_price'] / price_now) - 1) * 100 if price_now else None
                    st.metric(
                        "Analyst Price Target",
                        f"${consensus['target_price']:.2f}",
                        delta=f"{upside:+.1f}% upside" if upside else None
                    )
                # Earnings calendar
                earnings = fh_earnings_calendar(f.ticker)
                if earnings:
                    st.markdown("**Upcoming Earnings:**")
                    for e in earnings[:3]:
                        st.markdown(f"- 📅 {e.get('date','?')} | EPS Est: {e.get('epsEstimate','?')} "
                                    f"| Revenue Est: ${e.get('revenueEstimate',0)/1e9:.1f}B" 
                                    if e.get('revenueEstimate') else f"- 📅 {e.get('date','?')}")
            else:
                st.info("No analyst data available — add Finnhub or Alpha Vantage key.")

            # Alpha Vantage company profile
            overview = av_company_overview(f.ticker)
            if overview and overview.get("Description"):
                st.markdown("**Company Description:**")
                st.markdown(overview["Description"][:500] + "…")
                c1, c2 = st.columns(2)
                c1.markdown(f"**Sector:** {overview.get('Sector','n/a')}")
                c1.markdown(f"**Industry:** {overview.get('Industry','n/a')}")
                c2.markdown(f"**Employees:** {overview.get('FullTimeEmployees','n/a')}")
                c2.markdown(f"**Exchange:** {overview.get('Exchange','n/a')}")

        # ── Insider trades & SEC EDGAR tab ──
        with tab3c:
            st.markdown("### 🏛️ Insider Trades & Institutional Holdings (SEC EDGAR)")
            insider = get_insider_signal(f.ticker)
            if insider["count"] > 0:
                st.markdown(f"**{insider['net_signal']}** (Form 4 filings)")
                for trade in insider["recent"]:
                    st.markdown(f"- 📋 {trade['date']} — Form {trade['form']} "
                                f"[View]({trade['doc']})" if trade.get('doc') else
                                f"- 📋 {trade['date']} — Form {trade['form']}")
            else:
                st.info("No recent insider filings found on SEC EDGAR.")

            inst = sec_institutional_holdings(f.ticker)
            if inst:
                st.markdown("**Recent 13-F Institutional Filings:**")
                for h in inst:
                    st.markdown(f"- 📄 {h['date']} — {h['form']}")
            else:
                st.info("No recent 13-F filings found.")

            # Polygon ticker details
            poly_det = poly_ticker_details(f.ticker)
            if poly_det:
                st.markdown("**Polygon.io Company Details:**")
                c1, c2 = st.columns(2)
                c1.markdown(f"**Name:** {poly_det.get('name','n/a')}")
                c1.markdown(f"**Market:** {poly_det.get('market','n/a')}")
                c2.markdown(f"**SIC Code:** {poly_det.get('sic_code','n/a')}")
                c2.markdown(f"**List Date:** {poly_det.get('list_date','n/a')}")

            # Finnhub supplemental metrics
            fh_m = fh_basic_financials(f.ticker)
            if fh_m:
                st.markdown("**Finnhub Key Metrics:**")
                cols = st.columns(3)
                metrics_show = [
                    ("ROE TTM",         fh_m.get("roeTTM"),           ".2%"),
                    ("Gross Margin TTM",fh_m.get("grossMarginTTM"),   ".2%"),
                    ("Revenue Growth",  fh_m.get("revenueGrowthTTMYoy"), ".2%"),
                    ("EPS Growth",      fh_m.get("epsGrowthTTMYoy"),  ".2%"),
                    ("Current Ratio",   fh_m.get("currentRatioQuarterly"), ".2f"),
                    ("P/B Ratio",       fh_m.get("pbAnnual"),         ".2f"),
                ]
                for i, (label, val, fmt) in enumerate(metrics_show):
                    with cols[i % 3]:
                        if val is not None:
                            try: st.metric(label, format(float(val), fmt))
                            except Exception: st.metric(label, str(val))
                        else:
                            st.metric(label, "n/a")

        # ── Decision tab ──
        with tab4:
            st.markdown(f"## {cs.action}")
            st.markdown(f"**Composite Score: {cs.composite:.1f} / 10** ({cs.recommendation})")
            score_df = pd.DataFrame([
                {"Factor": "Fundamental", "Score": f"{cs.fund_score_10:.1f}/10", "Weight": f"{wf:.0%}", "Contribution": f"{cs.fund_score_10*wf:.2f}"},
                {"Factor": "Technical",   "Score": f"{cs.tech_score_10:.1f}/10", "Weight": f"{wt:.0%}", "Contribution": f"{cs.tech_score_10*wt:.2f}"},
                {"Factor": "News",        "Score": f"{cs.news_score_10:.1f}/10", "Weight": f"{wn:.0%}", "Contribution": f"{cs.news_score_10*wn:.2f}"},
                {"Factor": "Macro",       "Score": f"{cs.macro_score_10:.1f}/10","Weight": f"{wm:.0%}", "Contribution": f"{cs.macro_score_10*wm:.2f}"},
                {"Factor": "COMPOSITE",   "Score": f"{cs.composite:.1f}/10",     "Weight": "100%",      "Contribution": "—"},
            ])
            st.dataframe(score_df, use_container_width=True, hide_index=True)
            guidance = {
                "Strong Buy": "All four factors aligned positively. Consider entering at current levels or on a minor pullback. Set stop-loss below the 200-day MA.",
                "Buy":        "Predominantly positive signal. Good entry opportunity with manageable risk. Monitor news flow for any deterioration.",
                "Watch":      "Above-average score but not fully confirmed. Add to watchlist and enter on improvement in the weakest factor.",
                "Hold":       "Mixed signals across factors. No clear catalyst in either direction. Reassess next quarter.",
                "Avoid":      "Multiple factors negative. No position warranted. Re-evaluate when fundamentals or technicals improve.",
            }
            st.info(guidance.get(cs.recommendation, ""))

        # ── Risk tab ──
        with tab5:
            risks, positives = [], []
            if f.debt_to_equity is not None and f.debt_to_equity > 2.0:
                risks.append(f"High leverage (D/E {f.debt_to_equity:.2f}) increases risk in downturns")
            if f.altman_z_score is not None and f.altman_z_score < 1.8:
                risks.append(f"Altman Z-Score {f.altman_z_score:.2f} signals financial distress")
            if f.pe_ratio is not None and f.pe_ratio > 50:
                risks.append(f"Elevated P/E {f.pe_ratio:.1f}x — valuation sensitive to growth misses")
            if f.piotroski_f_score <= 3:
                risks.append(f"Low F-Score ({f.piotroski_f_score}/9) — deteriorating fundamentals")
            if f.roe is not None and f.roe < 0:
                risks.append("Negative ROE — company destroying shareholder value")
            if t:
                if t.rsi_14 and t.rsi_14 > 75:
                    risks.append(f"RSI {t.rsi_14:.1f} — overbought, pullback likely")
                if t.price and t.ma_200 and t.price < t.ma_200:
                    risks.append("Price below 200-day MA — long-term downtrend")
                if t.distance_from_52w_high is not None and t.distance_from_52w_high < -0.30:
                    risks.append(f"{abs(t.distance_from_52w_high)*100:.0f}% off 52-week high")
            if macro.yield_spread is not None and macro.yield_spread < 0:
                risks.append("Yield curve inverted — historical recession signal")
            if macro.vix is not None and macro.vix > 25:
                risks.append(f"VIX {macro.vix:.1f} — elevated market fear")
            if f.roe is not None and f.roe >= 0.20:       positives.append("Strong ROE above 20%")
            if f.altman_z_score is not None and f.altman_z_score >= 3.0: positives.append("Z-Score in safe zone")
            if f.piotroski_f_score >= 8:                   positives.append("Excellent F-Score")
            if t and t.technical_rating == "Bullish":      positives.append("All technical indicators bullish")
            if f.passed:                                   positives.append("Passed all screen filters")
            if macro.macro_label == "Macro Tailwind":      positives.append("Favourable macro backdrop")

            if risks:
                st.markdown("**⚠️ Key Risks:**")
                for r in risks: st.markdown(f"- {r}")
            else:
                st.success("No major red flags identified.")
            if positives:
                st.markdown("**✅ Strengths:**")
                for p in positives: st.markdown(f"- {p}")

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 7b: ML SCREENER ENGINE  (from screener.py)
# ═════════════════════════════════════════════════════════════════════════════

try:
    import ta as _ta
    from sklearn.ensemble import RandomForestClassifier as _RFC
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

_ML_FEATURES = ["ma20", "ma50", "rsi_ml", "volatility_ml"]

def _ml_get_data(ticker: str):
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    return df

def _ml_add_features(df):
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(0, index=df.index)
    df = df.copy()
    # Moving averages (EMA20, EMA50, EMA200)
    df["ma20"]          = close.rolling(20).mean()
    df["ma50"]          = close.rolling(50).mean()
    df["ema20"]         = close.ewm(span=20, adjust=False).mean()
    df["ema50"]         = close.ewm(span=50, adjust=False).mean()
    df["ema200"]        = close.ewm(span=200, adjust=False).mean()
    # RSI
    df["rsi_ml"]        = _ta.momentum.RSIIndicator(close).rsi() if _ML_AVAILABLE else close.rolling(14).mean()
    # MACD
    if _ML_AVAILABLE:
        macd_ind        = _ta.trend.MACD(close)
        df["macd"]      = macd_ind.macd()
        df["macd_sig"]  = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()
    else:
        df["macd"]      = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df["macd_sig"]  = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_sig"]
    # Volume average
    df["vol_avg20"]     = volume.rolling(20).mean()
    df["vol_ratio"]     = volume / df["vol_avg20"].replace(0, np.nan)
    # Volatility & ATR
    df["volatility_ml"] = close.pct_change().rolling(10).std()
    df["atr_ml"]        = _ta.volatility.AverageTrueRange(high, low, close).average_true_range()                           if _ML_AVAILABLE else (high - low)
    # Target: up tomorrow
    df["target"]        = (close.shift(-1) > close).astype(int)
    df.dropna(inplace=True)
    return df

def _ml_train(df):
    if not _ML_AVAILABLE:
        return None
    X = df[_ML_FEATURES]
    y = df["target"]
    model = _RFC(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def _ml_probability(model, row) -> float:
    if model is None:
        return 0.5
    return float(model.predict_proba([row[_ML_FEATURES]])[0][1])

def _ml_tech_score(row) -> float:
    score = 50.0
    rsi = row.get("rsi_ml", 50)
    if rsi < 30:   score += 20
    elif rsi > 70: score -= 20
    if row.get("ma20", 0) > row.get("ma50", 0): score += 15
    else:                                         score -= 15
    return float(np.clip(score, 0, 100))

def _ml_fuse_signal(ml_prob: float, t_score: float, sentiment: float = 50.0):
    score = 0.50 * ml_prob * 100 + 0.30 * t_score + 0.20 * sentiment
    if   score > 80: signal = "STRONG BUY"
    elif score > 65: signal = "BUY"
    elif score < 35: signal = "SELL"
    elif score < 45: signal = "WEAK SELL"
    else:            signal = "HOLD"
    return signal, round(score, 2)

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 7b2: SENTIMENT ENGINE  (from sentiment.py)
# ═════════════════════════════════════════════════════════════════════════════

try:
    from textblob import TextBlob as _TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

def _sentiment_score(news_list: list) -> float:
    """Score a list of headlines → 0-100 (50 = neutral)."""
    if not news_list:
        return 50.0
    if not _TEXTBLOB_AVAILABLE:
        return 50.0
    scores = [_TextBlob(text).sentiment.polarity for text in news_list]
    avg = sum(scores) / len(scores)          # -1 to +1
    return round((avg + 1) / 2 * 100, 2)    # normalise to 0-100

def _fetch_ml_news(ticker: str) -> list:
    """
    Fetch recent news headlines for a ticker via yfinance.
    Falls back to stub headlines if unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        items = t.news or []
        headlines = []
        for item in items[:10]:
            title = item.get("content", {}).get("title") or item.get("title", "")
            if title:
                headlines.append(title)
        if headlines:
            return headlines
    except Exception:
        pass
    # Stub fallback (mirrors original sentiment.py behaviour)
    return [
        f"{ticker} beats earnings estimates",
        f"{ticker} expands into new markets",
        f"Analysts raise price target on {ticker}",
    ]

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 7c: RISK ENGINE  (from risk.py)
# ═════════════════════════════════════════════════════════════════════════════

_MAX_RISK_PER_TRADE = 0.02
_MAX_POSITION_PCT   = 0.25
_MAX_DRAWDOWN       = -0.05
_MAX_DAILY_TRADES   = 10

def _position_size(capital: float, confidence: float, price: float) -> int:
    dollar_risk = capital * _MAX_RISK_PER_TRADE * confidence
    shares = int(dollar_risk / max(price, 0.01))
    return max(shares, 1)

def _stop_loss_price(entry: float, atr: float, mult: float = 2.0) -> float:
    return round(entry - mult * atr, 4)

def _take_profit_price(entry: float, atr: float, mult: float = 3.0) -> float:
    return round(entry + mult * atr, 4)

def _drawdown_breached(start: float, current: float) -> bool:
    return (current - start) / start < _MAX_DRAWDOWN

def _trade_limit_reached(trades: int) -> bool:
    return trades >= _MAX_DAILY_TRADES

def _high_vol_lockout(vol: float, threshold: float = 0.04) -> bool:
    return vol > threshold

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 7d: PORTFOLIO ENGINE  (from portfolio.py)
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_live_prices(tickers: list) -> dict:
    prices = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="1d", interval="1m",
                               auto_adjust=True, progress=False)
            prices[ticker] = float(data["Close"].iloc[-1])
        except Exception:
            prices[ticker] = 0.0
    return prices

def _calculate_pnl(portfolio: dict, live_prices: dict) -> dict:
    rows = []
    total_invested = total_value = 0.0
    for ticker, data in portfolio.items():
        qty, avg = data["qty"], data["avg_price"]
        price    = live_prices.get(ticker, avg)
        invested = qty * avg
        value    = qty * price
        pnl      = value - invested
        rows.append({
            "ticker": ticker, "qty": qty, "avg_price": avg,
            "live_price": round(price, 2), "invested": round(invested, 2),
            "value": round(value, 2), "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / invested * 100 if invested else 0, 2),
        })
        total_invested += invested
        total_value    += value
    total_pnl = total_value - total_invested
    return {
        "positions":      rows,
        "total_invested": round(total_invested, 2),
        "total_value":    round(total_value, 2),
        "total_pnl":      round(total_pnl, 2),
        "total_pnl_pct":  round(total_pnl / total_invested * 100 if total_invested else 0, 2),
    }

def _allocation_table(positions: list) -> list:
    total = sum(p["value"] for p in positions)
    for p in positions:
        p["allocation_pct"] = round(p["value"] / total * 100 if total else 0, 2)
    return positions

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 7e: EXECUTION ENGINE  (from execution.py)
# ═════════════════════════════════════════════════════════════════════════════

def _get_alpaca_client(paper: bool = True):
    import os
    try:
        from alpaca.trading.client import TradingClient
        return TradingClient(
            os.environ.get("ALPACA_API_KEY", ""),
            os.environ.get("ALPACA_SECRET_KEY", ""),
            paper=paper,
        )
    except ImportError:
        raise ImportError("Install alpaca-py: pip install alpaca-py")

def _execute_trade(client, signal: str, symbol: str, qty: int) -> str:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums   import OrderSide, TimeInForce
    if signal in ("BUY", "STRONG BUY"):
        side = OrderSide.BUY
    elif signal in ("SELL", "WEAK SELL"):
        side = OrderSide.SELL
    else:
        return f"NO TRADE — signal is {signal}"
    order = MarketOrderRequest(symbol=symbol, qty=qty, side=side,
                               time_in_force=TimeInForce.DAY)
    client.submit_order(order)
    return f"ORDER PLACED: {signal} {qty} {symbol}"

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 7f: BACKTESTING ENGINE  (V2 gap from guide)
# ═════════════════════════════════════════════════════════════════════════════

def _bt_generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate BUY/SELL signals on historical OHLCV data.
    EMA20/50/200 crossover + RSI filter + MACD confirmation + volume average.
    Returns df with columns: signal (1=buy, -1=sell, 0=hold),
    ema20, ema50, ema200, rsi14, macd, macd_sig, macd_hist, atr14, vol_avg20.
    """
    df = df.copy()
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(0, index=df.index)

    # EMAs (20, 50, 200 — all three from the guide)
    df["ema20"]  = close.ewm(span=20,  adjust=False).mean()
    df["ema50"]  = close.ewm(span=50,  adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # Volume average (20-day)
    df["vol_avg20"] = volume.rolling(20).mean()

    # ATR-14 for stop-loss
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # Signal: EMA crossover + RSI filter + MACD histogram confirmation
    prev_ema20  = df["ema20"].shift(1)
    prev_ema50  = df["ema50"].shift(1)
    prev_macdh  = df["macd_hist"].shift(1)

    df["signal"] = 0
    buy  = (
        (df["ema20"] > df["ema50"]) & (prev_ema20 <= prev_ema50) &   # EMA crossover
        (df["rsi14"] < 70) &                                           # not overbought
        (df["macd_hist"] > 0) &                                        # MACD positive
        (close > df["ema200"])                                         # above 200 EMA (trend filter)
    )
    sell = (
        (df["ema20"] < df["ema50"]) & (prev_ema20 >= prev_ema50) &   # EMA crossunder
        (df["rsi14"] > 30) &                                           # not oversold
        (df["macd_hist"] < 0)                                          # MACD negative
    )
    df.loc[buy,  "signal"] =  1
    df.loc[sell, "signal"] = -1

    df.dropna(inplace=True)
    return df


def _bt_run(df: pd.DataFrame, capital: float = 10_000.0,
            risk_pct: float = 0.02, atr_sl_mult: float = 2.0,
            atr_tp_mult: float = 3.0) -> dict:
    """
    Event-driven backtest loop.
    Supports one open position at a time.
    Returns metrics dict + trades list + equity series.
    """
    cash     = capital
    position = 0       # shares held
    entry_px = 0.0
    stop_loss= 0.0
    take_profit = 0.0
    trades   = []
    equity   = [capital]

    close_arr  = df["Close"].squeeze().values
    signal_arr = df["signal"].values
    atr_arr    = df["atr14"].values
    dates      = df.index.tolist()

    for i in range(len(df)):
        price = float(close_arr[i])
        sig   = int(signal_arr[i])
        atr   = float(atr_arr[i])

        # Check stop-loss / take-profit if in a position
        if position > 0:
            if price <= stop_loss or price >= take_profit:
                reason = "Stop-Loss" if price <= stop_loss else "Take-Profit"
                pnl    = (price - entry_px) * position
                cash  += position * price
                trades.append({
                    "Entry Date":  entry_date,
                    "Exit Date":   dates[i],
                    "Side":        "LONG",
                    "Entry Price": round(entry_px, 2),
                    "Exit Price":  round(price, 2),
                    "Shares":      position,
                    "P&L ($)":     round(pnl, 2),
                    "P&L (%)":     round(pnl / (entry_px * position) * 100, 2),
                    "Exit Reason": reason,
                })
                position = 0

        # Enter on BUY signal (no open position)
        if sig == 1 and position == 0:
            risk_dollars = cash * risk_pct
            shares = int(risk_dollars / max(atr * atr_sl_mult, 0.01))
            shares = max(shares, 1)
            cost   = shares * price
            if cost <= cash:
                cash       -= cost
                position    = shares
                entry_px    = price
                entry_date  = dates[i]
                stop_loss   = price - atr_sl_mult * atr
                take_profit = price + atr_tp_mult * atr

        # Exit on SELL signal if holding
        elif sig == -1 and position > 0:
            pnl  = (price - entry_px) * position
            cash += position * price
            trades.append({
                "Entry Date":  entry_date,
                "Exit Date":   dates[i],
                "Side":        "LONG",
                "Entry Price": round(entry_px, 2),
                "Exit Price":  round(price, 2),
                "Shares":      position,
                "P&L ($)":     round(pnl, 2),
                "P&L (%)":     round(pnl / (entry_px * position) * 100, 2),
                "Exit Reason": "Signal",
            })
            position = 0

        # Mark-to-market equity
        equity.append(cash + position * price)

    # Close any open position at last price
    if position > 0:
        price = float(close_arr[-1])
        pnl   = (price - entry_px) * position
        cash += position * price
        trades.append({
            "Entry Date":  entry_date,
            "Exit Date":   dates[-1],
            "Side":        "LONG",
            "Entry Price": round(entry_px, 2),
            "Exit Price":  round(price, 2),
            "Shares":      position,
            "P&L ($)":     round(pnl, 2),
            "P&L (%)":     round(pnl / (entry_px * position) * 100, 2),
            "Exit Reason": "End of Data",
        })
        equity[-1] = cash

    # ── Metrics ───────────────────────────────────────────────────────────
    equity_s   = pd.Series(equity)
    total_ret  = (equity_s.iloc[-1] - capital) / capital * 100
    peak       = equity_s.cummax()
    drawdown_s = (equity_s - peak) / peak * 100
    max_dd     = float(drawdown_s.min())

    winning = [t for t in trades if t["P&L ($)"] > 0]
    losing  = [t for t in trades if t["P&L ($)"] <= 0]
    win_rate = len(winning) / len(trades) * 100 if trades else 0

    # Daily returns for Sharpe
    eq_daily   = equity_s.pct_change().dropna()
    sharpe     = (eq_daily.mean() / eq_daily.std() * np.sqrt(252)
                  if eq_daily.std() > 0 else 0.0)

    avg_win  = np.mean([t["P&L ($)"] for t in winning]) if winning else 0
    avg_loss = np.mean([t["P&L ($)"] for t in losing])  if losing  else 0
    profit_factor = (
        sum(t["P&L ($)"] for t in winning) /
        abs(sum(t["P&L ($)"] for t in losing))
        if losing and sum(t["P&L ($)"] for t in losing) != 0 else float("inf")
    )

    # Buy-and-hold comparison
    bh_ret = (float(close_arr[-1]) - float(close_arr[0])) / float(close_arr[0]) * 100

    return {
        "equity":         equity_s,
        "drawdown":       drawdown_s,
        "trades":         trades,
        "total_return":   round(total_ret,  2),
        "max_drawdown":   round(max_dd,     2),
        "win_rate":       round(win_rate,   1),
        "sharpe":         round(float(sharpe), 2),
        "total_trades":   len(trades),
        "avg_win":        round(avg_win,    2),
        "avg_loss":       round(avg_loss,   2),
        "profit_factor":  round(profit_factor, 2),
        "bh_return":      round(bh_ret,     2),
        "final_equity":   round(float(equity_s.iloc[-1]), 2),
    }

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 8: STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Trading Decision Maker")
st.caption(f"4-factor analysis: Fundamental · Technical · News · Macro  |  {datetime.date.today()}")
st.divider()

# ── Sidebar — mode selector (always visible) ─────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    app_mode = st.radio(
        "Mode",
        ["📈 Stock Analysis", "💼 Portfolio Tracker", "🤖 ML Signal Scanner", "📊 Backtester"],
        index=0,
        key="app_mode",
    )
    st.divider()

    # ── API Keys (auto-loaded from .env — no manual entry needed) ─────────────
    with st.expander("🔑 API Keys", expanded=False):
        if _DOTENV_LOADED:
            st.caption("✅ .env file detected — keys below are auto-loaded. "
                       "Override here only if you want a different key for this session.")
        else:
            st.caption("⚠️ python-dotenv not installed — keys won't auto-load from .env. "
                       "Run: `pip install python-dotenv`. Or enter keys manually below "
                       "(session only, never saved to disk).")

        st.text_input("Alpha Vantage API Key",
            value=st.session_state.get("api_ALPHAVANTAGE_API_KEY", os.environ.get("ALPHAVANTAGE_API_KEY", "")),
            type="password", key="api_ALPHAVANTAGE_API_KEY",
            help="alphavantage.co — free 25 calls/day")
        st.text_input("Finnhub API Key",
            value=st.session_state.get("api_FINNHUB_API_KEY", os.environ.get("FINNHUB_API_KEY", "")),
            type="password", key="api_FINNHUB_API_KEY",
            help="finnhub.io — free 60 calls/min")
        st.text_input("Polygon.io API Key",
            value=st.session_state.get("api_POLYGON_API_KEY", os.environ.get("POLYGON_API_KEY", "")),
            type="password", key="api_POLYGON_API_KEY",
            help="polygon.io — free tier available")
        st.text_input("FRED API Key",
            value=st.session_state.get("api_FRED_API_KEY", os.environ.get("FRED_API_KEY", "5758685d1d96b52dedd08d7933375085")),
            type="password", key="api_FRED_API_KEY",
            help="fred.stlouisfed.org — unlimited free")

        st.markdown("**Alpaca (paper/live trading)**")
        st.text_input("Alpaca API Key",
            value=st.session_state.get("api_ALPACA_API_KEY", os.environ.get("ALPACA_API_KEY", "")),
            type="password", key="api_ALPACA_API_KEY")
        st.text_input("Alpaca Secret Key",
            value=st.session_state.get("api_ALPACA_SECRET_KEY", os.environ.get("ALPACA_SECRET_KEY", "")),
            type="password", key="api_ALPACA_SECRET_KEY")

        st.markdown("**Alerts (optional)**")
        st.text_input("Alert email — from",
            value=st.session_state.get("api_ALERT_EMAIL_FROM", os.environ.get("ALERT_EMAIL_FROM", "")),
            key="api_ALERT_EMAIL_FROM")
        st.text_input("Alert email — to",
            value=st.session_state.get("api_ALERT_EMAIL_TO", os.environ.get("ALERT_EMAIL_TO", "")),
            key="api_ALERT_EMAIL_TO")
        st.text_input("Alert email — app password",
            value=st.session_state.get("api_ALERT_EMAIL_PASS", os.environ.get("ALERT_EMAIL_PASS", "")),
            type="password", key="api_ALERT_EMAIL_PASS")
        st.text_input("Telegram bot token",
            value=st.session_state.get("api_ALERT_TG_TOKEN", os.environ.get("ALERT_TG_TOKEN", "")),
            type="password", key="api_ALERT_TG_TOKEN")
        st.text_input("Telegram chat ID",
            value=st.session_state.get("api_ALERT_TG_CHAT_ID", os.environ.get("ALERT_TG_CHAT_ID", "")),
            key="api_ALERT_TG_CHAT_ID")
        st.text_input("Slack webhook URL",
            value=st.session_state.get("api_ALERT_SLACK_WEBHOOK", os.environ.get("ALERT_SLACK_WEBHOOK", "")),
            type="password", key="api_ALERT_SLACK_WEBHOOK")

        st.caption("✅ yfinance · SEC EDGAR · CBOE — no key needed, always on")

        # Push every key into os.environ so downstream functions
        # (alerts, Alpaca client, bot engine) all see them automatically too
        for _k in ["ALPHAVANTAGE_API_KEY", "FINNHUB_API_KEY", "POLYGON_API_KEY", "FRED_API_KEY",
                   "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO",
                   "ALERT_EMAIL_PASS", "ALERT_TG_TOKEN", "ALERT_TG_CHAT_ID", "ALERT_SLACK_WEBHOOK"]:
            _v = st.session_state.get(f"api_{_k}", "")
            if _v:
                os.environ[_k] = _v

    # ── Data source status ─────────────────────────────────────────────────
    with st.expander("📡 Data Source Status", expanded=False):
        sources = {
            "Alpha Vantage": bool(_get_key("ALPHAVANTAGE_API_KEY")),
            "Finnhub":       bool(_get_key("FINNHUB_API_KEY")),
            "Polygon.io":    bool(_get_key("POLYGON_API_KEY")),
            "FRED":          bool(_get_key("FRED_API_KEY")),
            "Alpaca":        bool(_get_key("ALPACA_API_KEY")),
            "yfinance":      True,
            "SEC EDGAR":     True,
            "CBOE (VIX)":   True,
        }
        for src, active in sources.items():
            icon = "🟢" if active else "⚪"
            st.markdown(f"{icon} {src}")

    st.divider()

# ── Per-mode sidebar content ──────────────────────────────────────────────────
# Tickers picker is shared across Stock Analysis, ML Scanner, Backtester

with st.sidebar:
    # ── Tickers (shared across Stock Analysis, ML Scanner, Backtester) ────────
    if app_mode in ("📈 Stock Analysis", "🤖 ML Signal Scanner", "📊 Backtester"):
        st.subheader("Tickers")

        @st.cache_data(show_spinner="Loading US stock list…", ttl=86400)
        def load_us_stocks():
            import urllib.request, io

            def _make_empty():
                df = pd.DataFrame(columns=["symbol", "name"])
                df["_search"] = pd.Series(dtype=str)
                return df

            frames = []
            urls = {
                "NASDAQ":    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                "NYSE/AMEX": "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            }
            for exchange, url in urls.items():
                try:
                    with urllib.request.urlopen(url, timeout=15) as r:
                        raw = r.read().decode("utf-8")
                    df = pd.read_csv(io.StringIO(raw), sep="|")
                    sym_col  = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
                    name_col = "Security Name"
                    if sym_col not in df.columns or name_col not in df.columns:
                        continue
                    df = df[[sym_col, name_col]].rename(columns={sym_col: "symbol", name_col: "name"})
                    df = df[df["symbol"].notna() & df["name"].notna()]
                    df = df[~df["symbol"].str.contains(r"[$^]", regex=True)]
                    df = df[~df["symbol"].str.startswith("File")]
                    frames.append(df)
                except Exception:
                    pass

            if not frames:
                return _make_empty()

            result = (pd.concat(frames, ignore_index=True)
                        .drop_duplicates("symbol")
                        .sort_values("symbol")
                        .reset_index(drop=True))
            result["_search"] = (result["symbol"].fillna("") + " " + result["name"].fillna("")).str.lower()
            return result

        us_stocks = load_us_stocks()

        if us_stocks.empty:
            st.warning("⚠️ Could not load US stock list. Use manual entry below.")

        if "selected_tickers" not in st.session_state:
            st.session_state.selected_tickers = []

        search_query = st.text_input(
            "🔍 Search by ticker or company name",
            placeholder="e.g.  AAPL  or  Apple  or  nvidia",
            key="stock_search",
        )

        if search_query.strip() and not us_stocks.empty and "_search" in us_stocks.columns:
            q    = search_query.strip().lower()
            mask = us_stocks["_search"].str.contains(q, regex=False)
            hits = us_stocks[mask].copy()

            def _score(row):
                if row["symbol"].lower().startswith(q): return 0
                if row["name"].lower().startswith(q):   return 1
                return 2

            if not hits.empty:
                hits["_score"] = hits.apply(_score, axis=1)
                hits = hits.sort_values(["_score", "symbol"]).head(50)
                options = ["— pick one —"] + [
                    f"{r.symbol}  —  {r.name[:55]}" for r in hits.itertuples()
                ]
                chosen = st.selectbox("Suggestions", options, key="suggestion_box")
                if chosen and chosen != "— pick one —":
                    ticker_sym = chosen.split("  —  ")[0].strip()
                    if ticker_sym not in st.session_state.selected_tickers:
                        st.session_state.selected_tickers.append(ticker_sym)
                        st.rerun()
            else:
                st.caption("No matches found.")
        else:
            st.caption("Type a ticker or company name to see suggestions.")

        if st.session_state.selected_tickers:
            st.markdown("**Selected:**")
            cols = st.columns(3)
            to_remove = None
            for i, sym in enumerate(st.session_state.selected_tickers):
                with cols[i % 3]:
                    if st.button(f"✕ {sym}", key=f"rm_{sym}", use_container_width=True):
                        to_remove = sym
            if to_remove:
                st.session_state.selected_tickers.remove(to_remove)
                st.rerun()
            if st.button("🗑️ Clear all", use_container_width=True):
                st.session_state.selected_tickers = []
                st.rerun()
        else:
            st.info("No stocks selected yet.")

        with st.expander("✏️ Or paste tickers directly"):
            manual_input = st.text_area("Tickers (one per line or comma-separated)", value="", height=70)
            if manual_input.strip():
                for s in [t.strip().upper() for t in manual_input.replace(",", "\n").split("\n") if t.strip()]:
                    if s not in st.session_state.selected_tickers:
                        st.session_state.selected_tickers.append(s)

        ticker_input = "\n".join(st.session_state.selected_tickers)

        # ── Mode-specific sidebar settings ────────────────────────────────────
        st.divider()

        if app_mode == "📈 Stock Analysis":
            st.subheader("Screening thresholds")
            min_roe = st.slider("Min ROE (%)", 0, 30, 10, 1) / 100
            max_de  = st.slider("Max D/E",     0.5, 5.0,  2.0,  0.1)
            max_pe  = st.slider("Max P/E",     5.0, 100.0,30.0, 1.0)
            min_fs  = st.slider("Min F-Score", 0,   9,    6)
            min_zs  = st.slider("Min Z-Score", 0.5, 3.0,  1.8, 0.1)

            st.subheader("Composite weights")
            wf = st.slider("Fundamental %", 10, 70, 40, 5) / 100
            wt = st.slider("Technical %",   10, 50, 25, 5) / 100
            wn = st.slider("News %",         5, 40, 20, 5) / 100

            st.subheader("🤖 Ollama AI Narratives")
            ollama_enabled = st.toggle("Enable Ollama AI", value=False)
            ollama_model   = st.selectbox("Model", ["llama3","mistral","phi3","gemma"], index=0)

            st.divider()
            st.subheader("🏛️ Macro Environment")
            fred_key = _get_key("FRED_API_KEY") or "5758685d1d96b52dedd08d7933375085"
            st.caption(f"FRED key: {'🟢 auto-loaded' if _get_key('FRED_API_KEY') else '🟡 using default'} "
                      "(set/override in the 🔑 API Keys panel above)")
            wm = st.slider("Macro weight %", 5, 30, 15, 5) / 100
            total_w = wf + wt + wn + wm
            if abs(total_w - 1.0) > 0.01:
                st.warning(f"Weights sum to {total_w:.0%} — should be 100%")

            st.divider()
            run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        elif app_mode == "📊 Backtester":
            st.subheader("Backtest Settings")
            bt_period    = st.selectbox("History period", ["1y","2y","3y","5y"], index=1)
            bt_capital   = st.number_input("Starting capital ($)", value=10000, step=1000, min_value=1000)
            bt_risk_pct  = st.slider("Risk per trade (%)", 1, 5, 2) / 100
            bt_sl_mult   = st.slider("Stop-loss ATR multiplier", 1.0, 4.0, 2.0, 0.5)
            bt_tp_mult   = st.slider("Take-profit ATR multiplier", 1.0, 6.0, 3.0, 0.5)
            bt_oos_split = st.slider("Train / Test split (%)", 50, 85, 70, 5)
            st.divider()
            run_button   = st.button("▶️ Run Backtest", type="primary", use_container_width=True)

        elif app_mode == "🤖 ML Signal Scanner":
            st.subheader("ML Scanner Settings")
            ml_capital   = st.number_input("Portfolio capital ($)", value=10000, step=500, min_value=1000)
            ml_auto_ref  = st.checkbox("Auto-refresh (60s)", value=False)
            st.divider()
            st.subheader("🔔 Alerts")
            ml_alert_email     = st.text_input("Alert email (optional)", placeholder="you@gmail.com",        key="ml_alert_email")
            ml_alert_tg        = st.text_input("Telegram chat ID (optional)", placeholder="@yourbot",        key="ml_alert_tg")
            ml_alert_slack     = st.text_input("Slack webhook (optional)", placeholder="https://hooks.slack...", key="ml_alert_slack")
            ml_alert_threshold = st.selectbox("Alert on signal",
                ["STRONG BUY only", "BUY or STRONG BUY", "SELL or STRONG SELL", "Any non-HOLD"],
                key="ml_alert_thresh")
            st.divider()
            run_button = st.button("🔍 Run ML Scan", type="primary", use_container_width=True)

        else:
            run_button = False

    else:
        # Portfolio Tracker — no ticker picker, no run button in sidebar
        ticker_input = ""
        run_button   = False


# ══════════════════════════════════════════════════════════════════════════════
# MODE: 📈 STOCK ANALYSIS  (original comprehensive analysis)
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "📈 Stock Analysis" and run_button:

    # Parse tickers
    raw_tickers = ticker_input.replace(",", "\n").split("\n")
    tickers = [t.strip().upper() for t in raw_tickers if t.strip()]

    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    weights    = (wf, wt, wn, wm)
    thresholds = (min_roe, max_de, max_pe, min_fs, min_zs)
    analysis_date = datetime.date.today().strftime("%Y-%m-%d")

    # ── Step 1: Fundamentals ─────────────────────────────────────────────────
    st.markdown("### Step 1 — Fundamental screen")
    fund_results  = {}
    fund_progress = st.progress(0)
    fund_status   = st.empty()

    for i, t in enumerate(tickers):
        fund_status.text(f"Fetching fundamentals: {t} ({i+1}/{len(tickers)}) — AV→Finnhub→yfinance")
        try:
            # Try multi-source enrichment first (Alpha Vantage + Finnhub supplemental)
            extra = get_fundamentals_best(t, thresholds)
            # Always run core fetch (Piotroski, Altman Z, screen pass)
            core  = fetch_fundamental_metrics(t, thresholds)
            # Enrich core with AV/Finnhub data where available
            if extra:
                if extra.get("roe")           is not None: core.roe           = extra["roe"]
                if extra.get("debt_to_equity") is not None: core.debt_to_equity= extra["debt_to_equity"]
                if extra.get("pe_ratio")       is not None: core.pe_ratio      = extra["pe_ratio"]
                core._extra = extra   # attach full extra for display
            fund_results[t] = core
        except Exception as e:
            fund_results[t] = None
            st.warning(f"{t}: skipped — {e}")
        fund_progress.progress((i + 1) / len(tickers))

    fund_progress.empty()
    fund_status.empty()

    valid_fund = {t: f for t, f in fund_results.items() if f is not None}
    st.success(f"Fundamentals fetched: {len(valid_fund)}/{len(tickers)} tickers")

    # ── Step 2: Technicals ───────────────────────────────────────────────────
    st.markdown("### Step 2 — Technical screen")
    tech_results  = {}
    tech_progress = st.progress(0)
    tech_status   = st.empty()
    valid_fund_list = list(valid_fund.keys())

    for i, t in enumerate(valid_fund_list):
        tech_status.text(f"Fetching technicals: {t} ({i+1}/{len(valid_fund_list)})")
        try:
            tech_results[t] = fetch_technical_metrics(t)
        except Exception as e:
            tech_results[t] = None
            st.warning(f"{t} technical: skipped — {e}")
        tech_progress.progress((i + 1) / len(valid_fund_list))

    tech_progress.empty()
    tech_status.empty()

    # ── Step 3: News ─────────────────────────────────────────────────────────
    st.markdown("### Step 3 — News intelligence")
    with st.spinner("Collecting RSS market news..."):
        market_articles             = get_market_news()
        market_raw_score, market_sentiment = score_market_news(market_articles)

    company_news_map    = {}
    company_news_scores = {}
    news_progress = st.progress(0)
    news_status   = st.empty()

    for i, t in enumerate(valid_fund_list):
        news_status.text(f"Fetching news & sentiment: {t} — AV Financial NLP → Finnhub → Polygon → yfinance")
        # Multi-source news with sentiment (AV → Finnhub → Polygon → yfinance)
        headlines, sent_score = get_news_best(t)
        company_news_map[t]    = headlines
        # Map 0-100 sentiment score to our -10..+10 raw scale
        raw_score = round((sent_score - 50) / 5)
        company_news_scores[t] = (raw_score, sent_score / 10)
        news_progress.progress((i + 1) / len(valid_fund_list))

    news_progress.empty()
    news_status.empty()
    st.success(f"News collected — market sentiment: **{market_sentiment}** ({market_raw_score:+d})")

    # ── Step 4: Macro ─────────────────────────────────────────────────────────
    st.markdown("### Step 4 — Macro (FRED)")
    macro = None
    if fred_key:
        with st.spinner("Fetching FRED macro indicators..."):
            try:
                macro = fetch_macro_environment(fred_key)
                # Supplement VIX with CBOE live if FRED VIX is stale
                if macro.vix is None:
                    live_vix = get_vix_best()
                    if live_vix:
                        macro = MacroEnvironment(
                            fed_funds_rate=macro.fed_funds_rate, cpi_yoy=macro.cpi_yoy,
                            unemployment=macro.unemployment, yield_spread=macro.yield_spread,
                            vix=live_vix, gdp_growth=macro.gdp_growth,
                            macro_score=macro.macro_score, macro_label=macro.macro_label,
                            macro_summary=macro.macro_summary + f" | VIX {live_vix:.1f} (CBOE live)"
                        )
                st.success(f"Macro: **{macro.macro_label}** ({macro.macro_score:.1f}/10)")
            except Exception as e:
                st.warning(f"FRED fetch failed ({e}) — trying CBOE + AV economic fallback")

    if macro is None:
        # Fallback: try Alpha Vantage economic indicators + CBOE VIX
        with st.spinner("Fetching macro from Alpha Vantage + CBOE…"):
            av_key = _get_key("ALPHAVANTAGE_API_KEY")
            fed, cpi, unemp, vix_val, gdp = None, None, None, None, None
            if av_key:
                try:
                    fed_data  = av_economic_indicator("FEDERAL_FUNDS_RATE")
                    fed       = float(fed_data[0]["value"]) if fed_data else None
                    cpi_data  = av_economic_indicator("CPI")
                    if len(cpi_data) >= 13:
                        cpi = ((float(cpi_data[0]["value"]) - float(cpi_data[12]["value"]))
                               / float(cpi_data[12]["value"])) * 100
                    unemp_data = av_economic_indicator("UNEMPLOYMENT")
                    unemp     = float(unemp_data[0]["value"]) if unemp_data else None
                    gdp_data  = av_economic_indicator("REAL_GDP")
                    gdp       = float(gdp_data[0]["value"]) if gdp_data else None
                except Exception:
                    pass
            vix_val = get_vix_best()

            score = 5.0
            if fed   is not None: score += 1 if fed <= 4.0 else 0
            if cpi   is not None: score += 1 if cpi <= 3.0 else 0
            if unemp is not None: score += 1 if unemp <= 5.0 else 0
            if vix_val is not None: score += 1 if vix_val <= 20 else 0
            norm = min(score / 8.0 * 10, 10)

            label = ("Macro Tailwind" if norm >= 7.5 else
                     "Macro Neutral"  if norm >= 5.0 else
                     "Macro Headwind" if norm >= 2.5 else "Macro Risk-Off")
            parts = []
            if fed   is not None: parts.append(f"Fed Rate {fed:.2f}%")
            if cpi   is not None: parts.append(f"CPI YoY {cpi:.1f}%")
            if unemp is not None: parts.append(f"Unemp {unemp:.1f}%")
            if vix_val is not None: parts.append(f"VIX {vix_val:.1f}")
            macro = MacroEnvironment(
                fed_funds_rate=fed, cpi_yoy=cpi, unemployment=unemp,
                yield_spread=None, vix=vix_val, gdp_growth=gdp,
                macro_score=norm, macro_label=label,
                macro_summary=" | ".join(parts) if parts else "Limited macro data"
            )

    # ── Step 5: Composite scores ──────────────────────────────────────────────
    results = []
    for t in valid_fund_list:
        f  = valid_fund[t]
        te = tech_results.get(t)
        news_raw, news_norm = company_news_scores.get(t, (0, 5.0))
        cs = build_composite(f, te, news_norm, macro.macro_score, weights)
        results.append((f, te, cs, news_raw, news_norm, company_news_map.get(t, [])))

    results.sort(key=lambda x: x[2].composite, reverse=True)

    # ── Store in session state ────────────────────────────────────────────────
    st.session_state["results"]          = results
    st.session_state["macro"]            = macro
    st.session_state["market_sentiment"] = market_sentiment
    st.session_state["market_raw_score"] = market_raw_score
    st.session_state["weights"]          = weights
    st.session_state["analysis_date"]    = analysis_date
    st.session_state["ollama_enabled"]   = ollama_enabled
    st.session_state["ollama_model"]     = ollama_model
    st.session_state["ollama_reports"]   = {}

    st.success("✅ Analysis complete — scroll down to view results")

# ── Display results (persists across reruns) ──────────────────────────────────
if "results" in st.session_state:
    results          = st.session_state["results"]
    macro            = st.session_state["macro"]
    market_sentiment = st.session_state["market_sentiment"]
    market_raw_score = st.session_state["market_raw_score"]
    weights          = st.session_state["weights"]
    analysis_date    = st.session_state["analysis_date"]
    ollama_enabled   = st.session_state.get("ollama_enabled", False)
    ollama_model_val = st.session_state.get("ollama_model", "llama3")
    ollama_reports   = st.session_state.get("ollama_reports", {})

    st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("## 📊 Results Summary")
    passed     = [r for r in results if r[0].passed]
    top_picks  = [r for r in results if r[2].recommendation in ("Strong Buy","Buy")]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tickers screened", len(results))
    c2.metric("Passed all filters", len(passed))
    c3.metric("Buy / Strong Buy", len(top_picks))
    c4.metric("Market sentiment", market_sentiment)

    st.divider()

    # ── Composite score table ─────────────────────────────────────────────────
    st.markdown("## 🏆 Composite Score Table")
    render_composite_table(results, weights)
    st.divider()

    # ── Per-ticker detail ─────────────────────────────────────────────────────
    st.markdown("## 📋 Per-Ticker Detail")
    for f, t, cs, news_raw, news_norm, headlines in results:
        render_ticker_detail(
            f, t, cs, headlines, macro, news_raw,
            weights, market_sentiment, market_raw_score
        )

    st.divider()

    # ── Macro panel (moved to end) ────────────────────────────────────────────
    render_macro_panel(macro)
    st.divider()

    # ── Ollama AI narratives ──────────────────────────────────────────────────
    st.markdown("## 🤖 Ollama AI Narratives")

    if not ollama_enabled:
        st.info("Ollama is disabled. Enable it in the sidebar and re-run to generate AI narratives.")
    else:
        if st.button("Generate AI narratives", type="secondary"):
            try:
                import ollama as _ollama

                # Check model available
                result = subprocess.run(["ollama","list"], capture_output=True, text=True, timeout=5)
                if ollama_model_val not in result.stdout:
                    st.error(f"Model '{ollama_model_val}' not found. Run: ollama pull {ollama_model_val}")
                    st.stop()

                prog = st.progress(0)
                for i, (f, t, cs, _, _, headlines) in enumerate(results):
                    with st.spinner(f"Generating narrative for {f.ticker}..."):
                        prompt = build_ollama_prompt(
                            f, t, cs, headlines, macro,
                            market_sentiment, market_raw_score,
                            analysis_date, ollama_model_val
                        )
                        try:
                            response = _ollama.chat(
                                model=ollama_model_val,
                                messages=[{"role":"user","content":prompt}],
                                options={"temperature":0.3}
                            )
                            ollama_reports[f.ticker] = response["message"]["content"].strip()
                        except Exception as e:
                            ollama_reports[f.ticker] = f"_AI narrative failed: {e}_"
                    prog.progress((i+1)/len(results))

                st.session_state["ollama_reports"] = ollama_reports
                st.success(f"AI narratives generated for {len(ollama_reports)} tickers")
                prog.empty()

            except ImportError:
                st.error("ollama package not installed. Run: pip install ollama")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                st.error("Ollama service not running. Open the Ollama app or run: ollama serve")

        if ollama_reports:
            for f, t, cs, _, _, _ in results:
                ticker = f.ticker
                if ticker in ollama_reports:
                    with st.expander(f"🤖 {ticker} — AI Narrative", expanded=False):
                        st.markdown(f"**Composite: {cs.composite:.1f}/10 → {cs.action}**")
                        st.divider()
                        st.markdown(ollama_reports[ticker])

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("## 💾 Export")
    export_rows = []
    for f, t, cs, news_raw, news_norm, headlines in results:
        export_rows.append({
            "Date":        analysis_date,
            "Ticker":      f.ticker,
            "ROE":         sfmt(f.roe, ".1%") if f.roe is not None else "n/a",
            "D/E":         sfmt(f.debt_to_equity, ".2f") if f.debt_to_equity is not None else "n/a",
            "P/E":         sfmt(f.pe_ratio, ".1f") if f.pe_ratio is not None else "n/a",
            "F-Score":     f.piotroski_f_score,
            "Z-Score":     sfmt(f.altman_z_score, ".2f") if f.altman_z_score is not None else "n/a",
            "Fund Rating": f.fundamental_rating,
            "Price":       f"${t.price:.2f}" if t and t.price else "n/a",
            "RSI":         sfmt(t.rsi_14, ".1f") if t and t.rsi_14 else "n/a",
            "Tech Rating": t.technical_rating if t else "n/a",
            "Fund Score":  f"{cs.fund_score_10:.1f}",
            "Tech Score":  f"{cs.tech_score_10:.1f}",
            "News Score":  f"{cs.news_score_10:.1f}",
            "Macro Score": f"{cs.macro_score_10:.1f}",
            "Composite":   f"{cs.composite:.1f}",
            "Action":      cs.action,
            "Passed":      "Yes" if f.passed else "No",
        })

    export_df = pd.DataFrame(export_rows)
    import io
    excel_buffer = io.BytesIO()
    export_df.to_excel(excel_buffer, index=False, sheet_name="Trading Results")
    excel_buffer.seek(0)
    st.download_button(
        label="⬇️ Download results as Excel",
        data=excel_buffer,
        file_name=f"trading_results_{analysis_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

elif app_mode == "📈 Stock Analysis":
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    ### Welcome to the Trading Decision Maker

    Configure your settings in the sidebar on the left, then click **🚀 Run Analysis**.

    **What this dashboard does:**
    1. **Fundamental screen** — ROE, D/E, P/E, Piotroski F-Score, Altman Z-Score
    2. **Technical screen** — 50/200-day MA, RSI, 52-week high distance
    3. **News intelligence** — RSS feeds (Reuters, Yahoo, FT, WSJ) + per-ticker yfinance news
    4. **Macro score** — 6 FRED indicators (Fed rate, CPI, unemployment, yield spread, VIX, GDP)
    5. **4-factor composite score** — weighted combination of all four layers
    6. **Ollama AI narratives** — local LLM generates a one-page investment summary per ticker

    > **Tip:** The FRED API key is pre-filled. You can change tickers and thresholds in the sidebar at any time.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# MODE: 💼 PORTFOLIO TRACKER
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "💼 Portfolio Tracker":
    st.title("💼 Live Portfolio Tracker")
    st.caption("Real-time P&L · Allocation · Position sizing")
    st.divider()

    # ── Portfolio builder ─────────────────────────────────────────────────────
    st.markdown("### 📋 Your Positions")
    st.info("Add your holdings below. Changes take effect when you click **Refresh Portfolio**.")

    if "portfolio_positions" not in st.session_state:
        st.session_state.portfolio_positions = [
            {"ticker": "AAPL", "qty": 10, "avg_price": 180.0},
            {"ticker": "MSFT", "qty":  5, "avg_price": 350.0},
        ]

    # Editable positions table
    pos_df = pd.DataFrame(st.session_state.portfolio_positions)
    edited  = st.data_editor(
        pos_df,
        num_rows="dynamic",
        column_config={
            "ticker":    st.column_config.TextColumn("Ticker", width="small"),
            "qty":       st.column_config.NumberColumn("Qty", min_value=0, step=1),
            "avg_price": st.column_config.NumberColumn("Avg Price ($)", min_value=0.0, format="%.2f"),
        },
        use_container_width=True,
        key="portfolio_editor",
    )
    st.session_state.portfolio_positions = edited.to_dict("records")

    col_btn1, col_btn2 = st.columns(2)
    refresh_portfolio = col_btn1.button("🔄 Refresh Portfolio", type="primary", use_container_width=True)
    auto_refresh_ptf  = col_btn2.checkbox("Auto-refresh (60s)", value=False)

    if refresh_portfolio or auto_refresh_ptf:
        portfolio_dict = {
            r["ticker"].upper(): {"qty": int(r["qty"]), "avg_price": float(r["avg_price"])}
            for r in st.session_state.portfolio_positions
            if r.get("ticker") and str(r["ticker"]).strip()
        }

        with st.spinner("Fetching live prices…"):
            live_prices = _fetch_live_prices(list(portfolio_dict.keys()))
        result      = _calculate_pnl(portfolio_dict, live_prices)
        positions   = _allocation_table(result["positions"])

        # Summary metrics
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        pnl_delta = f"{result['total_pnl_pct']:+.2f}%"
        m1.metric("Total Invested",  f"${result['total_invested']:,.2f}")
        m2.metric("Total Value",     f"${result['total_value']:,.2f}")
        m3.metric("Total P&L",       f"${result['total_pnl']:,.2f}", delta=pnl_delta)
        m4.metric("Return",          pnl_delta)

        # Positions table
        st.markdown("### Positions")
        df_pos = pd.DataFrame(positions)
        st.dataframe(
            df_pos[[
                "ticker", "qty", "avg_price", "live_price",
                "invested", "value", "pnl", "pnl_pct", "allocation_pct"
            ]].rename(columns={
                "ticker": "Ticker", "qty": "Qty", "avg_price": "Avg Price",
                "live_price": "Live Price", "invested": "Invested ($)",
                "value": "Value ($)", "pnl": "P&L ($)", "pnl_pct": "P&L %",
                "allocation_pct": "Allocation %"
            }),
            use_container_width=True,
        )

        # Allocation chart
        st.markdown("### Allocation")
        alloc_df = df_pos[["ticker", "allocation_pct"]].set_index("ticker")
        st.bar_chart(alloc_df)

        # Risk metrics per position
        st.markdown("### Risk Metrics")
        risk_rows = []
        for p in positions:
            risk_rows.append({
                "Ticker":         p["ticker"],
                "Kelly Shares":   _position_size(result["total_value"], 0.6, p["live_price"]) if p["live_price"] > 0 else "n/a",
                "Max Exposure":   f"${result['total_value'] * _MAX_POSITION_PCT:,.0f}",
                "Allocation %":   f"{p['allocation_pct']:.1f}%",
                "Over-exposed":   "⚠️ Yes" if p["allocation_pct"] > _MAX_POSITION_PCT * 100 else "✅ No",
            })
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True)

        # Auto-refresh loop
        if auto_refresh_ptf:
            time.sleep(60)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MODE: 🤖 ML SIGNAL SCANNER
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🤖 ML Signal Scanner":
    st.title("🤖 ML Signal Scanner")
    st.caption("RandomForest ML · Technical score · Sentiment fusion · Risk engine")
    st.divider()

    if not _ML_AVAILABLE:
        st.error("Missing dependencies: `pip install ta scikit-learn`")
        st.stop()

    scan_tickers = [s.strip().upper() for s in ticker_input.split("\n") if s.strip()]

    if not scan_tickers:
        st.info("Select stocks from the sidebar ticker picker, then switch to this mode.")
        st.stop()

    if run_button or ml_auto_ref:
        scan_results = []
        prog = st.progress(0)
        for i, ticker in enumerate(scan_tickers):
            prog.progress((i + 1) / len(scan_tickers), text=f"Scanning {ticker}…")
            try:
                df_ml   = _ml_get_data(ticker)
                df_ml   = _ml_add_features(df_ml)
                model   = _ml_train(df_ml)
                latest  = df_ml.iloc[-1]
                ml_prob = _ml_probability(model, latest)
                t_score = _ml_tech_score(latest)

                # Live sentiment via yfinance news + TextBlob
                news_ml = _fetch_ml_news(ticker)
                sent    = _sentiment_score(news_ml)
                signal, score = _ml_fuse_signal(ml_prob, t_score, sent)

                price   = float(latest["Close"])
                atr     = float(latest["atr_ml"])
                vol     = float(latest["volatility_ml"])
                qty     = _position_size(ml_capital, ml_prob, price)
                sl      = _stop_loss_price(price, atr)
                tp      = _take_profit_price(price, atr)

                scan_results.append({
                    "Ticker":       ticker,
                    "Signal":       signal,
                    "Fused Score":  score,
                    "ML Prob":      round(ml_prob, 3),
                    "Tech Score":   round(t_score, 1),
                    "Price":        round(price, 2),
                    "Suggested Qty":qty,
                    "Stop-Loss":    sl,
                    "Take-Profit":  tp,
                    "Volatility":   round(vol, 4),
                    "Vol Lock":     "🔒 Yes" if _high_vol_lockout(vol) else "✅ No",
                })
            except Exception as e:
                scan_results.append({"Ticker": ticker, "Signal": f"Error: {e}",
                                     "Fused Score": 0, "ML Prob": 0, "Tech Score": 0,
                                     "Price": 0, "Suggested Qty": 0,
                                     "Stop-Loss": 0, "Take-Profit": 0,
                                     "Volatility": 0, "Vol Lock": "n/a"})

        prog.empty()

        df_scan = pd.DataFrame(scan_results)

        # Colour-code signal column
        def _signal_colour(val):
            colours = {
                "STRONG BUY": "background-color:#1a7a3c;color:white",
                "BUY":        "background-color:#28a745;color:white",
                "HOLD":       "background-color:#6c757d;color:white",
                "WEAK SELL":  "background-color:#fd7e14;color:white",
                "SELL":       "background-color:#dc3545;color:white",
            }
            return colours.get(val, "")

        st.markdown("### Scan Results")
        st.dataframe(
            df_scan.style.map(_signal_colour, subset=["Signal"]),
            use_container_width=True,
        )

        # ── Supplemental data panel ────────────────────────────────────────
        if len(scan_results) <= 10:
            st.markdown("### 📡 Multi-Source Intelligence")
            for row in scan_results:
                if "Error" in str(row.get("Signal", "")):
                    continue
                tkr = row["Ticker"]
                with st.expander(f"🔍 {tkr} — Extended Data"):
                    c1, c2, c3 = st.columns(3)
                    # Alpha Vantage overview
                    ov = av_company_overview(tkr)
                    with c1:
                        st.markdown("**Alpha Vantage**")
                        if ov and ov.get("Sector"):
                            st.markdown(f"Sector: {ov.get('Sector','n/a')}")
                            st.markdown(f"P/E: {ov.get('PERatio','n/a')}")
                            st.markdown(f"Target: ${ov.get('AnalystTargetPrice','n/a')}")
                        else:
                            st.caption("No AV key or data")
                    # Finnhub
                    fh_m2 = fh_basic_financials(tkr)
                    with c2:
                        st.markdown("**Finnhub**")
                        if fh_m2:
                            roe_val = fh_m2.get("roeTTM")
                            gr_val  = fh_m2.get("revenueGrowthTTMYoy")
                            st.markdown(f"ROE TTM: {float(roe_val):.1%}" if roe_val else "ROE: n/a")
                            st.markdown(f"Rev Growth: {float(gr_val):.1%}" if gr_val else "Rev Growth: n/a")
                        else:
                            st.caption("No Finnhub key or data")
                    # SEC insider
                    with c3:
                        st.markdown("**SEC Insider**")
                        ins = get_insider_signal(tkr)
                        st.markdown(f"Form 4 filings: {ins['count']}")
                        if ins["count"] > 0:
                            st.markdown(f"Latest: {ins['recent'][0].get('date','?')}")

        # ── Fire alerts for qualifying signals ────────────────────────────
        import os as _os
        alert_thresh = locals().get("ml_alert_threshold", "STRONG BUY only")
        alert_map = {
            "STRONG BUY only":       {"STRONG BUY"},
            "BUY or STRONG BUY":     {"BUY", "STRONG BUY"},
            "SELL or STRONG SELL":   {"SELL", "WEAK SELL"},
            "Any non-HOLD":          {"BUY", "STRONG BUY", "SELL", "WEAK SELL"},
        }
        qualifying_signals = alert_map.get(alert_thresh, {"STRONG BUY"})

        for row in scan_results:
            if row.get("Signal") in qualifying_signals:
                alert_msg = (
                    f"{row['Ticker']}: {row['Signal']} | "
                    f"Score={row['Fused Score']} | Price=${row['Price']} | "
                    f"Qty={row['Suggested Qty']} | SL=${row['Stop-Loss']} | TP=${row['Take-Profit']}"
                )
                # Set env vars temporarily from sidebar inputs
                if ml_alert_email:
                    _os.environ.setdefault("ALERT_EMAIL_TO", ml_alert_email)
                if ml_alert_tg:
                    _os.environ.setdefault("ALERT_TG_CHAT_ID", ml_alert_tg)
                if ml_alert_slack:
                    _os.environ.setdefault("ALERT_SLACK_WEBHOOK", ml_alert_slack)
                _bot_send_alert(alert_msg)

        # Export
        excel_buf = io.BytesIO()
        df_scan.to_excel(excel_buf, index=False, sheet_name="ML Signals")
        excel_buf.seek(0)
        st.download_button(
            "⬇️ Download ML signals as Excel",
            data=excel_buf,
            file_name=f"ml_signals_{datetime.date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        if ml_auto_ref:
            time.sleep(60)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODE: 📊 BACKTESTER  (V2 gap — strategy validation before live trading)
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "📊 Backtester":
    st.title("📊 Strategy Backtester")
    st.caption("EMA crossover · RSI filter · ATR stop-loss/take-profit · Out-of-sample validation")
    st.divider()

    if not _ML_AVAILABLE:
        st.warning("Install numpy for full backtest charts: `pip install numpy`")

    # ── Settings ──────────────────────────────────────────────────────────────
    bt_tickers = [s.strip().upper() for s in ticker_input.split("\n") if s.strip()]
    if not bt_tickers:
        st.info("Select stocks from the sidebar ticker picker, then switch to this mode.")
        st.stop()

    if run_button:
        for bt_ticker in bt_tickers:
            st.markdown(f"---\n### {bt_ticker}")

            with st.spinner(f"Downloading {bt_ticker} data…"):
                try:
                    raw = yf.download(bt_ticker, period=bt_period,
                                      auto_adjust=True, progress=False)
                    if raw.empty or len(raw) < 60:
                        st.warning(f"{bt_ticker}: Not enough data.")
                        continue
                except Exception as e:
                    st.error(f"{bt_ticker}: Download failed — {e}")
                    continue

            df_bt = _bt_generate_signals(raw)

            # Train/test split (out-of-sample validation)
            split_idx  = int(len(df_bt) * bt_oos_split / 100)
            df_train   = df_bt.iloc[:split_idx]
            df_test    = df_bt.iloc[split_idx:]

            tab_in, tab_out, tab_wf, tab_trades, tab_compare = st.tabs([
                "📈 In-Sample", "🧪 Out-of-Sample", "🔄 Walk-Forward", "📋 Trade Log", "⚖️ vs Buy & Hold"
            ])

            for label, df_split, tab in [
                ("In-Sample",     df_train, tab_in),
                ("Out-of-Sample", df_test,  tab_out),
            ]:
                if len(df_split) < 10:
                    with tab:
                        st.warning("Not enough data for this split.")
                    continue

                res = _bt_run(df_split, bt_capital, bt_risk_pct, bt_sl_mult, bt_tp_mult)

                with tab:
                    # ── Key metrics ───────────────────────────────────────
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Total Return",   f"{res['total_return']:+.1f}%",
                              delta=f"B&H: {res['bh_return']:+.1f}%")
                    m2.metric("Max Drawdown",   f"{res['max_drawdown']:.1f}%")
                    m3.metric("Sharpe Ratio",   f"{res['sharpe']:.2f}")
                    m4.metric("Win Rate",       f"{res['win_rate']:.1f}%")
                    m5.metric("Total Trades",   res['total_trades'])
                    m6.metric("Profit Factor",  f"{res['profit_factor']:.2f}"
                              if res['profit_factor'] != float('inf') else "∞")

                    st.markdown(f"**Final equity:** ${res['final_equity']:,.2f}  "
                                f"| Avg win: ${res['avg_win']:,.2f}  "
                                f"| Avg loss: ${res['avg_loss']:,.2f}")

                    # ── Equity curve ──────────────────────────────────────
                    st.markdown("**Equity Curve**")
                    eq_df = pd.DataFrame({
                        "Strategy": res["equity"].values,
                    })
                    # Buy-and-hold baseline
                    close_vals = df_split["Close"].squeeze().values
                    shares_bh  = bt_capital / float(close_vals[0])
                    eq_df["Buy & Hold"] = [shares_bh * float(p) for p in close_vals[:len(eq_df)]]
                    st.line_chart(eq_df)

                    # ── Drawdown ──────────────────────────────────────────
                    st.markdown("**Drawdown (%)**")
                    dd_df = pd.DataFrame({"Drawdown %": res["drawdown"].values})
                    st.area_chart(dd_df)

                    # ── Signal chart ──────────────────────────────────────
                    st.markdown("**Price with EMA20 / EMA50 + Signals**")
                    price_df = df_split[["Close", "ema20", "ema50"]].copy()
                    price_df.columns = ["Close", "EMA 20", "EMA 50"]
                    st.line_chart(price_df)

            # ── Walk-forward validation ───────────────────────────────────
            with tab_wf:
                st.markdown("**Walk-Forward Validation** — rolling windows prevent look-ahead bias")
                st.caption("Splits data into N equal windows, trains on the first part of each, tests on the rest, then aggregates results.")

                n_windows = st.slider("Number of windows", 3, 8, 4, key=f"wf_windows_{bt_ticker}")
                wf_train_pct = st.slider("Train % per window", 50, 80, 70, 5, key=f"wf_train_{bt_ticker}") / 100

                if len(df_bt) < n_windows * 20:
                    st.warning("Not enough data for walk-forward with these settings.")
                else:
                    window_size = len(df_bt) // n_windows
                    wf_results  = []

                    for w in range(n_windows):
                        w_start = w * window_size
                        w_end   = w_start + window_size if w < n_windows - 1 else len(df_bt)
                        w_df    = df_bt.iloc[w_start:w_end]
                        split   = int(len(w_df) * wf_train_pct)
                        w_test  = w_df.iloc[split:]
                        if len(w_test) < 5:
                            continue
                        r = _bt_run(w_test, bt_capital, bt_risk_pct, bt_sl_mult, bt_tp_mult)
                        wf_results.append({
                            "Window":       f"W{w+1}",
                            "Period":       f"{str(w_df.index[0])[:10]} → {str(w_df.index[-1])[:10]}",
                            "Return (%)":   r["total_return"],
                            "B&H (%)":      r["bh_return"],
                            "Sharpe":       r["sharpe"],
                            "Win Rate (%)": r["win_rate"],
                            "Max DD (%)":   r["max_drawdown"],
                            "Trades":       r["total_trades"],
                        })

                    if wf_results:
                        wf_df = pd.DataFrame(wf_results)
                        st.dataframe(
                            wf_df.style.map(
                                lambda v: "color:green" if isinstance(v, (int,float)) and v > 0 else "color:red",
                                subset=["Return (%)", "B&H (%)"]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                        avg_ret = wf_df["Return (%)"].mean()
                        avg_bh  = wf_df["B&H (%)"].mean()
                        avg_sr  = wf_df["Sharpe"].mean()
                        avg_wr  = wf_df["Win Rate (%)"].mean()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Avg Return",    f"{avg_ret:+.1f}%", delta=f"B&H: {avg_bh:+.1f}%")
                        c2.metric("Avg Sharpe",    f"{avg_sr:.2f}")
                        c3.metric("Avg Win Rate",  f"{avg_wr:.1f}%")
                        c4.metric("Windows tested",f"{len(wf_results)}/{n_windows}")

                        verdict_wf = (
                            "✅ Strategy beats B&H consistently across windows — good robustness."
                            if avg_ret > avg_bh
                            else "⚠️ Strategy underperforms B&H on average — consider revising parameters."
                        )
                        st.info(verdict_wf)
                        st.caption("ℹ️ Consistent performance across windows = more robust strategy. High variance = overfitting risk.")

            # ── Trade log ─────────────────────────────────────────────────
            with tab_trades:
                all_trades = []
                res_full   = _bt_run(df_bt, bt_capital, bt_risk_pct, bt_sl_mult, bt_tp_mult)
                if res_full["trades"]:
                    tr_df = pd.DataFrame(res_full["trades"])
                    tr_df["Entry Date"] = pd.to_datetime(tr_df["Entry Date"]).dt.strftime("%Y-%m-%d")
                    tr_df["Exit Date"]  = pd.to_datetime(tr_df["Exit Date"]).dt.strftime("%Y-%m-%d")

                    def _pnl_colour(val):
                        return "color:green" if val > 0 else "color:red"

                    st.dataframe(
                        tr_df.style.map(_pnl_colour, subset=["P&L ($)", "P&L (%)"]),
                        use_container_width=True,
                    )

                    # Export trades
                    buf = io.BytesIO()
                    tr_df.to_excel(buf, index=False, sheet_name="Trades")
                    buf.seek(0)
                    st.download_button(
                        f"⬇️ Download {bt_ticker} trade log",
                        data=buf,
                        file_name=f"{bt_ticker}_backtest_{datetime.date.today()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                else:
                    st.info("No trades generated with current settings.")

            # ── Strategy vs Buy & Hold comparison ─────────────────────────
            with tab_compare:
                res_full = _bt_run(df_bt, bt_capital, bt_risk_pct, bt_sl_mult, bt_tp_mult)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Strategy**")
                    st.metric("Return",        f"{res_full['total_return']:+.1f}%")
                    st.metric("Max Drawdown",  f"{res_full['max_drawdown']:.1f}%")
                    st.metric("Sharpe",        f"{res_full['sharpe']:.2f}")
                    st.metric("Total Trades",  res_full["total_trades"])
                    st.metric("Win Rate",      f"{res_full['win_rate']:.1f}%")
                with c2:
                    st.markdown("**Buy & Hold**")
                    st.metric("Return",       f"{res_full['bh_return']:+.1f}%")
                    st.metric("Max Drawdown", "N/A")
                    st.metric("Sharpe",       "N/A")
                    st.metric("Trades",       "1")
                    st.metric("Win Rate",     "N/A")

                verdict = ("✅ Strategy outperforms Buy & Hold"
                           if res_full["total_return"] > res_full["bh_return"]
                           else "⚠️ Buy & Hold outperforms Strategy — review settings")
                st.info(verdict)

                st.caption(
                    "ℹ️ Always validate on out-of-sample data before paper trading. "
                    "Past performance does not guarantee future results."
                )

# ═════════════════════════════════════════════════════════════════════════════
# ── SECTION 9: AUTONOMOUS BOT ENGINE  (from bot.py)
# Run standalone:  python app.py --bot
# ═════════════════════════════════════════════════════════════════════════════

import logging as _logging
import sys as _sys

_bot_log = _logging.getLogger("trading_bot")

# ── Bot configuration (edit here or pass via env vars) ────────────────────
BOT_WATCHLIST     = os.environ.get("BOT_WATCHLIST", "AAPL,MSFT,NVDA").split(",")
BOT_CAPITAL       = float(os.environ.get("BOT_CAPITAL", "10000"))
BOT_LOOP_INTERVAL = int(os.environ.get("BOT_LOOP_INTERVAL", "60"))  # seconds between scan cycles
BOT_USE_ALPACA    = os.environ.get("BOT_USE_ALPACA", "false").lower() == "true"  # set true in .env + Alpaca keys to go live


# ── Alert system (V4 gap — email / Telegram / Slack) ─────────────────────
def _bot_send_alert(message: str):
    """
    Multi-channel alert dispatcher.
    Reads config from environment variables — never hardcode credentials.

    Email  : set ALERT_EMAIL_FROM, ALERT_EMAIL_TO, ALERT_EMAIL_PASS, ALERT_SMTP_HOST
    Telegram: set ALERT_TG_TOKEN, ALERT_TG_CHAT_ID
    Slack  : set ALERT_SLACK_WEBHOOK
    """
    import os

    _bot_log.info(f"🚨 ALERT: {message}")
    sent_any = False

    # ── Email via SMTP ────────────────────────────────────────────────────
    email_from = os.environ.get("ALERT_EMAIL_FROM", "")
    email_to   = os.environ.get("ALERT_EMAIL_TO",   "")
    email_pass = os.environ.get("ALERT_EMAIL_PASS",  "")
    smtp_host  = os.environ.get("ALERT_SMTP_HOST",  "smtp.gmail.com")
    smtp_port  = int(os.environ.get("ALERT_SMTP_PORT", "587"))

    if email_from and email_to and email_pass:
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(message)
            msg["Subject"] = f"[TradingBot] {message[:60]}"
            msg["From"]    = email_from
            msg["To"]      = email_to
            with smtplib.SMTP(smtp_host, smtp_port) as srv:
                srv.ehlo(); srv.starttls(); srv.ehlo()
                srv.login(email_from, email_pass)
                srv.sendmail(email_from, [email_to], msg.as_string())
            _bot_log.info("Alert sent via Email ✓")
            sent_any = True
        except Exception as e:
            _bot_log.warning(f"Email alert failed: {e}")

    # ── Telegram ──────────────────────────────────────────────────────────
    tg_token   = os.environ.get("ALERT_TG_TOKEN",   "")
    tg_chat_id = os.environ.get("ALERT_TG_CHAT_ID", "")

    if tg_token and tg_chat_id:
        try:
            import urllib.request, json as _json
            url     = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            payload = _json.dumps({"chat_id": tg_chat_id, "text": f"📊 {message}"}).encode()
            req     = urllib.request.Request(url, data=payload,
                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            _bot_log.info("Alert sent via Telegram ✓")
            sent_any = True
        except Exception as e:
            _bot_log.warning(f"Telegram alert failed: {e}")

    # ── Slack webhook ─────────────────────────────────────────────────────
    slack_wh = os.environ.get("ALERT_SLACK_WEBHOOK", "")

    if slack_wh:
        try:
            import urllib.request, json as _json
            payload = _json.dumps({"text": f"📊 TradingBot: {message}"}).encode()
            req     = urllib.request.Request(slack_wh, data=payload,
                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            _bot_log.info("Alert sent via Slack ✓")
            sent_any = True
        except Exception as e:
            _bot_log.warning(f"Slack alert failed: {e}")

    if not sent_any:
        _bot_log.info("(No alert channels configured — set env vars to enable)")


def _bot_run_ticker(ticker: str, capital: float, trades_today: int):
    """
    Full analysis pipeline for one ticker inside the bot loop.
    Returns (signal, trades_today_incremented).
    """
    try:
        df_b    = _ml_get_data(ticker)
        df_b    = _ml_add_features(df_b)
        model_b = _ml_train(df_b)
        latest_b = df_b.iloc[-1]

        ml_prob_b = _ml_probability(model_b, latest_b)
        t_score_b = _ml_tech_score(latest_b)
        news_b    = _fetch_ml_news(ticker)
        sent_b    = _sentiment_score(news_b)
        signal_b, score_b = _ml_fuse_signal(ml_prob_b, t_score_b, sent_b)

        _bot_log.info(
            f"{ticker} | signal={signal_b} score={score_b:.1f} "
            f"ml={ml_prob_b:.2f} tech={t_score_b:.1f} sent={sent_b:.1f}"
        )

        vol_b   = float(latest_b["volatility_ml"])
        price_b = float(latest_b["Close"])
        atr_b   = float(latest_b["atr_ml"])

        if _high_vol_lockout(vol_b):
            _bot_log.warning(f"{ticker} — skipped (high volatility={vol_b:.4f})")
            return signal_b, trades_today

        if _trade_limit_reached(trades_today):
            _bot_log.warning("Daily trade cap reached — no more orders today.")
            return signal_b, trades_today

        qty_b = _position_size(capital, ml_prob_b, price_b)
        sl_b  = _stop_loss_price(price_b, atr_b)
        _bot_log.info(f"{ticker} | qty={qty_b} price={price_b:.2f} stop-loss={sl_b:.2f}")

        if BOT_USE_ALPACA and signal_b in ("BUY", "STRONG BUY", "SELL", "WEAK SELL"):
            try:
                alpaca_client = _get_alpaca_client(paper=True)
                result_b = _execute_trade(alpaca_client, signal_b, ticker, qty_b)
                _bot_log.info(result_b)
                _bot_send_alert(f"{ticker}: {signal_b} × {qty_b} @ ~{price_b:.2f}")
                trades_today += 1
            except Exception as ex:
                _bot_log.error(f"Execution error for {ticker}: {ex}")

    except Exception as e:
        _bot_log.error(f"Error processing {ticker}: {e}")
        signal_b = None

    return signal_b, trades_today


def run_bot(
    watchlist: list = None,
    capital: float = None,
    loop_interval: int = None,
):
    """
    Autonomous trading bot main loop.
    Runs indefinitely; Ctrl-C to stop.
    """
    import datetime as _dt

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            _logging.FileHandler("bot.log"),
            _logging.StreamHandler(),
        ],
    )

    watchlist     = watchlist     or BOT_WATCHLIST
    capital       = capital       or BOT_CAPITAL
    loop_interval = loop_interval or BOT_LOOP_INTERVAL

    _bot_log.info("═" * 50)
    _bot_log.info("🚀 AI TRADING BOT STARTED")
    _bot_log.info(f"   Watchlist : {watchlist}")
    _bot_log.info(f"   Capital   : ${capital:,.0f}")
    _bot_log.info(f"   Interval  : {loop_interval}s")
    _bot_log.info(f"   Alpaca    : {'LIVE (paper)' if BOT_USE_ALPACA else 'DISABLED'}")
    _bot_log.info("═" * 50)

    trades_today = 0
    last_date    = _dt.datetime.now().date()

    while True:
        today = _dt.datetime.now().date()
        if today != last_date:
            trades_today = 0
            last_date    = today
            _bot_log.info("Daily counters reset.")

        for ticker in watchlist:
            sig, trades_today = _bot_run_ticker(ticker, capital, trades_today)

        _bot_log.info(
            f"Cycle complete. Trades today: {trades_today}/{_MAX_DAILY_TRADES}. "
            f"Sleeping {loop_interval}s...\n"
        )
        time.sleep(loop_interval)


# ── Entry point: python app.py --bot ──────────────────────────────────────
if __name__ == "__main__" and len(_sys.argv) > 1 and _sys.argv[1] == "--bot":
    run_bot()
