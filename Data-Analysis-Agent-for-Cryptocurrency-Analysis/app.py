import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "spoof-x-logo.png"


ASSET_NAMES = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "dogecoin": "DOGE",
    "cardano": "ADA",
    "litecoin": "LTC",
    "polkadot": "DOT",
    "avalanche": "AVAX",
    "chainlink": "LINK",
    "bitcoin cash": "BCH",
    "stellar": "XLM",
    "ripple": "XRP",
    "shiba inu": "SHIB",
    "polygon": "POL",
}

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "LTC": "litecoin",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "BCH": "bitcoin-cash",
    "XLM": "stellar",
    "XRP": "ripple",
    "SHIB": "shiba-inu",
    "POL": "polygon-ecosystem-token",
}

PRICE_DATA_VERSION = 2


class MarketDataUnavailable(Exception):
    """Raised when no configured public market-data provider can be reached."""


st.set_page_config(
    page_title="SPOOF-X | Crypto Analysis",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_secret(name: str) -> str:
    """Read a key from Streamlit secrets first, then the environment."""
    try:
        value = st.secrets.get(name, "")
    except FileNotFoundError:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def get_api_key(name: str, label: str) -> str:
    """Allow session-only key entry when the server has no configured key."""
    configured_key = get_secret(name)
    if configured_key:
        return configured_key
    return st.sidebar.text_input(label, type="password", key=f"input_{name}").strip()


def resolve_symbol(asset: str) -> str:
    """Use a ticker for Alpha Vantage when a common coin name is entered."""
    normalized_name = " ".join(asset.split()).casefold()
    return ASSET_NAMES.get(normalized_name, normalized_name.upper())


def fetch_alpha_vantage_prices(symbol: str, api_key: str) -> pd.DataFrame:
    """Fetch prices from Alpha Vantage when the user has provided a key."""
    response = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": "USD",
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    series = payload.get("Time Series (Digital Currency Daily)")
    if not series:
        message = (payload.get("Error Message") or payload.get("Note")
                   or payload.get("Information") or "No price data was returned.")
        raise ValueError(message)

    prices = pd.DataFrame.from_dict(series, orient="index")
    close_column = "4. close"
    if close_column not in prices:
        raise ValueError("The price response did not include daily closing prices.")
    prices = prices[[close_column]].rename(columns={close_column: "close"})
    prices.index = pd.to_datetime(prices.index)
    prices["close"] = pd.to_numeric(prices["close"])
    return prices.sort_index().tail(60)


def fetch_coingecko_prices(symbol: str) -> pd.DataFrame:
    """Fetch a no-key price series from CoinGecko."""
    coin_id = COINGECKO_IDS.get(symbol)
    if not coin_id:
        lookup = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": symbol},
            headers={"User-Agent": "SignalDesk/1.0"},
            timeout=15,
        )
        lookup.raise_for_status()
        matches = lookup.json().get("coins", [])
        exact_match = next(
            (coin for coin in matches if str(coin.get("symbol", "")).upper() == symbol),
            None,
        )
        if not exact_match:
            raise ValueError(f"No cryptocurrency was found for {symbol}.")
        coin_id = exact_match["id"]

    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": 60, "interval": "daily"},
        headers={"User-Agent": "SignalDesk/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    values = response.json().get("prices", [])
    if not values:
        raise ValueError("The fallback provider returned no price data.")

    prices = pd.DataFrame(values, columns=["timestamp", "close"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    prices["close"] = pd.to_numeric(prices["close"])
    return prices.set_index("timestamp").sort_index().tail(60)


def fetch_cryptocompare_prices(symbol: str) -> pd.DataFrame:
    """Fetch live history from an independent no-key provider."""
    response = requests.get(
        "https://min-api.cryptocompare.com/data/v2/histoday",
        params={"fsym": symbol, "tsym": "USD", "limit": 59},
        headers={"User-Agent": "SignalDesk/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("Response") == "Error":
        raise ValueError(payload.get("Message", "No price data was returned."))
    values = payload.get("Data", {}).get("Data", [])
    if not values:
        raise ValueError("The backup provider returned no price data.")

    prices = pd.DataFrame(values)
    prices["timestamp"] = pd.to_datetime(prices["time"], unit="s", utc=True).dt.tz_localize(None)
    prices["close"] = pd.to_numeric(prices["close"])
    return prices.set_index("timestamp")[["close"]].sort_index().tail(60)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(symbol: str, api_key: str) -> tuple[pd.DataFrame, bool]:
    """Try independent live providers; never substitute synthetic market data."""
    providers = [lambda: fetch_coingecko_prices(symbol), lambda: fetch_cryptocompare_prices(symbol)]
    if api_key:
        providers.append(lambda: fetch_alpha_vantage_prices(symbol, api_key))

    for provider in providers:
        try:
            return provider(), True
        except (requests.RequestException, ValueError, KeyError):
            # Never surface the exception: request URLs can include the user's API key.
            continue

    raise MarketDataUnavailable(
        f"Live market data is temporarily unavailable for {symbol}. Check the symbol and try again."
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol: str, api_key: str) -> list[dict[str, Any]]:
    from exa_py import Exa

    results = Exa(api_key=api_key).search_and_contents(
        f"{symbol} cryptocurrency market news",
        summary=True,
        num_results=6,
    ).results
    return [
        {
            "title": item.title or "Untitled article",
            "url": item.url,
            "summary": getattr(item, "summary", "") or "Summary unavailable.",
            "published_date": getattr(item, "published_date", "") or "",
        }
        for item in results
    ]


def build_market_prompt(symbol: str, prices: pd.DataFrame, news: list[dict[str, Any]]) -> str:
    price_lines = "\n".join(
        f"{date.strftime('%Y-%m-%d')}: ${value:,.2f}"
        for date, value in prices.tail(14)["close"].items()
    )
    news_lines = "\n".join(f"- {item['title']}: {item['summary']}" for item in news)
    return f"""Analyze {symbol} as a cautious cryptocurrency market analyst.

Daily prices:
{price_lines}

Recent news:
{news_lines or '- No news was retrieved.'}

Return three short sections: Market read, Key drivers, and Outlook. Clearly label the outlook as upward, downward, or neutral. Do not present this as financial advice."""


def generate_analysis(symbol: str, prices: pd.DataFrame, news: list[dict[str, Any]], api_key: str) -> str:
    from langchain_groq import ChatGroq

    response = ChatGroq(
        temperature=0.2,
        model_name=get_secret("GROQ_MODEL") or "openai/gpt-oss-120b",
        groq_api_key=api_key,
    ).invoke(build_market_prompt(symbol, prices, news))
    return response.content


def render_chart(prices: pd.DataFrame, symbol: str) -> None:
    chart = go.Figure(
        go.Scatter(
            x=prices.index,
            y=prices["close"],
            mode="lines",
            line={"color": "#35d0ba", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(53, 208, 186, 0.10)",
            hovertemplate="$%{y:,.2f}<extra></extra>",
        )
    )
    chart.update_layout(
        height=330,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#b7c7d9"},
        xaxis={"showgrid": False, "zeroline": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(183, 199, 217, 0.12)", "tickprefix": "$"},
        hovermode="x unified",
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink: #eaf2f8; --muted: #8fa5b9; --line: rgba(183,199,217,.16); --teal: #35d0ba; --orange: #ff9f68; }
        .stApp { background: radial-gradient(circle at 85% 0%, rgba(53,208,186,.12), transparent 32rem), #08131e; color: var(--ink); font-family: 'DM Sans', sans-serif; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
        h1 { font-size: clamp(2.2rem, 5vw, 4.6rem); line-height: .98; margin-bottom: .6rem; }
        .eyebrow { color: var(--teal); font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
        .subtle { color: var(--muted); font-size: 1rem; }
        .panel { border: 1px solid var(--line); border-radius: 8px; padding: 1.2rem; background: rgba(12,29,43,.74); }
        .metric-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
        .metric-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; margin-top: .3rem; }
        .article { border-top: 1px solid var(--line); padding: .9rem 0; }
        .article a { color: var(--ink); font-weight: 600; text-decoration: none; }
        .article p { color: var(--muted); font-size: .9rem; line-height: 1.5; margin: .35rem 0 0; }
        [data-testid='stSidebar'] { background: #0b1b2a; border-right: 1px solid var(--line); }
        [data-testid='stSidebar'] [data-testid='stImage'] { display: flex; justify-content: center; margin: -.7rem auto .2rem; }
        [data-testid='stSidebar'] [data-testid='stImage'] img { filter: drop-shadow(0 10px 22px rgba(0,0,0,.28)); transform: translateX(32px); }
        .brand-caption { color: var(--muted); font-size: .82rem; letter-spacing: .04em; margin: -.2rem 0 1.35rem; text-align: center; }
        [data-testid='stMetricValue'] { color: var(--teal); }
        .stButton > button { background: var(--teal); border: 0; border-radius: 6px; color: #06141c; font-weight: 700; min-height: 2.8rem; }
        .stButton > button:hover { background: #70e3d3; color: #06141c; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    st.sidebar.image(str(LOGO_PATH), width=190)
    st.sidebar.markdown(
        '<div class="brand-caption">AI-POWERED CRYPTO INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    asset = st.sidebar.text_input(
        "Asset name or symbol", value="BTC", max_chars=40,
        help="For example: BTC, Ethereum, or Solana.",
    ).strip()
    symbol = resolve_symbol(asset)
    if asset and symbol != asset.upper():
        st.sidebar.caption(f"Using ticker: {symbol}")
    run_analysis = st.sidebar.button("Run analysis", width="stretch")
    st.sidebar.divider()
    st.sidebar.caption("Enter API keys here if they are not available to the app. They are kept for this session only.")
    alpha_key = get_api_key("ALPHAVANTAGE_API_KEY", "Alpha Vantage API key (optional)")
    exa_key = get_api_key("EXA_API_KEY", "Exa API key (optional)")
    groq_key = get_api_key("GROQ_API_KEY", "Groq API key")
    st.sidebar.divider()
    st.sidebar.caption("Data refreshes every 15 minutes. Market insights are informational, not financial advice.")

    st.markdown('<div class="eyebrow">Multi-agent market intelligence</div>', unsafe_allow_html=True)
    st.title("Read the market before it moves.")
    st.markdown('<p class="subtle">A focused view of price momentum, live reporting, and an AI-generated market read.</p>', unsafe_allow_html=True)

    if not symbol:
        st.info("Enter a cryptocurrency symbol in the sidebar to begin.")
        return

    if (
        run_analysis
        or "prices" not in st.session_state
        or st.session_state.get("price_data_version") != PRICE_DATA_VERSION
    ):
        with st.spinner(f"Gathering the {symbol} market signal..."):
            try:
                prices, has_live_prices = fetch_prices(symbol, alpha_key)
            except MarketDataUnavailable as error:
                # Do not show raw request errors because they can contain API keys.
                st.error(str(error))
                return
            news = []
            news_error = None
            if exa_key:
                try:
                    news = fetch_news(symbol, exa_key)
                except Exception as error:
                    news_error = str(error)
            st.session_state.prices = prices
            st.session_state.news = news
            st.session_state.news_error = news_error
            st.session_state.symbol = symbol
            st.session_state.analysis = None
            st.session_state.has_live_prices = has_live_prices
            st.session_state.price_data_version = PRICE_DATA_VERSION

    prices = st.session_state.prices
    news = st.session_state.get("news", [])
    active_symbol = st.session_state.get("symbol", symbol)
    has_live_prices = st.session_state.get("has_live_prices", True)
    latest = prices["close"].iloc[-1]
    change = (latest / prices["close"].iloc[0] - 1) * 100

    left, right = st.columns([1.35, .65], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"<span class=\"eyebrow\">{active_symbol} / USD · 60 DAY VIEW</span>", unsafe_allow_html=True)
        render_chart(prices, active_symbol)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Market snapshot")
        metric_one, metric_two = st.columns(2)
        metric_one.metric("Last close", f"${latest:,.2f}")
        metric_two.metric("60d change", f"{change:+.2f}%")
        st.markdown('<div class="panel"><span class="metric-label">Signal state</span><div class="metric-value">Awaiting synthesis</div><p class="subtle">Run the analyst crew to combine price action and current reporting.</p></div>', unsafe_allow_html=True)

    st.markdown("### Current reporting")
    if news:
        news_columns = st.columns(2)
        for index, article in enumerate(news):
            with news_columns[index % 2]:
                st.markdown(
                    f'<div class="article"><a href="{article["url"]}" target="_blank">{article["title"]}</a><p>{article["summary"]}</p></div>',
                    unsafe_allow_html=True,
                )
    else:
        if st.session_state.get("news_error"):
            st.warning(f"Prices loaded, but current reporting is unavailable: {st.session_state.news_error}")
        elif exa_key:
            st.info("No current reporting was found for this asset.")
        else:
            st.info("Add EXA_API_KEY to show current reporting.")

    st.markdown("### Analyst synthesis")
    if st.button("Generate market read", type="primary", disabled=not groq_key):
        with st.spinner("The analyst crew is writing the market read..."):
            try:
                st.session_state.analysis = generate_analysis(active_symbol, prices, news, groq_key)
            except Exception as error:
                st.error(f"Could not generate analysis: {error}")
    if not groq_key:
        st.info("Enter a Groq API key in the sidebar to enable the AI market read.")
    elif st.session_state.get("analysis"):
        st.markdown(f'<div class="panel">{st.session_state.analysis}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
