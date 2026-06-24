# ─────────────────────────────────────────────────────────────────────────────
# Trading Decision Maker — Streamlit Dashboard
# Wraps trading_final.ipynb into a live web UI
#
# HOW TO RUN:
#   pip install streamlit yfinance pandas feedparser fredapi ollama
#   streamlit run app.py
#
# Then open: http://localhost:8501
# ─────────────────────────────────────────────────────────────────────────────

import datetime
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import feedparser
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

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

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Fundamentals", "📈 Technical", "📰 News", "🏆 Decision", "⚠️ Risk"]
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
# ── SECTION 8: STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Trading Decision Maker")
st.caption(f"4-factor analysis: Fundamental · Technical · News · Macro  |  {datetime.date.today()}")
st.divider()

# ── Sidebar — configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Tickers
    st.subheader("Tickers")
    ticker_input = st.text_area(
        "Enter tickers (one per line or comma-separated)",
        value="AAPL\nMSFT\nNVDA\nAMZN\nGOOGL\nMETA\nJPM\nTSLA",
        height=160,
    )

    # FRED key
    st.subheader("FRED API Key")
    fred_key = st.text_input(
        "FRED API Key",
        value=st.secrets.get("FRED_API_KEY", ""),
        type="password",
    )

    # Screening thresholds
    st.subheader("Screening thresholds")
    min_roe = st.slider("Min ROE (%)", 0, 30, 10, 1) / 100
    max_de  = st.slider("Max D/E",     0.5, 5.0,  2.0,  0.1)
    max_pe  = st.slider("Max P/E",     5.0, 100.0,30.0, 1.0)
    min_fs  = st.slider("Min F-Score", 0,   9,    6)
    min_zs  = st.slider("Min Z-Score", 0.5, 3.0,  1.8, 0.1)

    # Weights
    st.subheader("Composite weights")
    wf = st.slider("Fundamental %", 10, 70, 40, 5) / 100
    wt = st.slider("Technical %",   10, 50, 25, 5) / 100
    wn = st.slider("News %",         5, 40, 20, 5) / 100
    wm = st.slider("Macro %",        5, 30, 15, 5) / 100
    total_w = wf + wt + wn + wm
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_w:.0%} — should be 100%")

    # Ollama
    st.subheader("🤖 Ollama AI Narratives")
    ollama_enabled = st.toggle("Enable Ollama AI", value=False)
    ollama_model   = st.selectbox("Model", ["llama3","mistral","phi3","gemma"], index=0)

    # Run button
    st.divider()
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ── Main area — run analysis ──────────────────────────────────────────────────
if run_button:

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
        fund_status.text(f"Fetching fundamentals: {t} ({i+1}/{len(tickers)})")
        try:
            fund_results[t] = fetch_fundamental_metrics(t, thresholds)
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
        news_status.text(f"Fetching company news: {t}")
        headlines = get_company_news(t)
        company_news_map[t]    = headlines
        company_news_scores[t] = score_company_news(headlines)
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
                st.success(f"Macro: **{macro.macro_label}** ({macro.macro_score:.1f}/10)")
            except Exception as e:
                st.warning(f"FRED fetch failed ({e}) — macro defaulted to 5.0/10")
    if macro is None:
        macro = MacroEnvironment(
            fed_funds_rate=None, cpi_yoy=None, unemployment=None,
            yield_spread=None, vix=None, gdp_growth=None,
            macro_score=5.0, macro_label="Macro Neutral (no key)",
            macro_summary="FRED key not set or fetch failed"
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

    # ── Macro panel ───────────────────────────────────────────────────────────
    render_macro_panel(macro)
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

else:
    # ── Landing state (no analysis run yet) ───────────────────────────────────
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
