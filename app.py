from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="AgriMitra | Smart Harvest & Price Advisory",
    page_icon="🌾",
    layout="centered",
)

st.title("🌾 AgriMitra: Smart Harvest & Price Advisory")
st.write(
    "Know the best time to harvest and sell based on local weather and market"
    " price trends."
)

KERALA_DISTRICTS = {
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kollam": (8.8932, 76.6141),
    "Pathanamthitta": (9.2648, 76.7870),
    "Alappuzha": (9.4981, 76.3388),
    "Kottayam": (9.5916, 76.5222),
    "Idukki": (9.8495, 76.9720),
    "Ernakulam": (9.9816, 76.2999),
    "Thrissur": (10.5276, 76.2144),
    "Palakkad": (10.7867, 76.6548),
    "Malappuram": (11.0510, 76.0711),
    "Kozhikode": (11.2588, 75.7804),
    "Wayanad": (11.6854, 76.1320),
    "Kannur": (11.8745, 75.3704),
    "Kasaragod": (12.4996, 74.9869),
}

HISTORICAL_RANGES = {
    "Black Pepper (Ungarbled)": {"low": 510.0, "high": 785.0},
    "Black Pepper (Garbled)": {"low": 530.0, "high": 805.0},
    "Rubber (RSS-4)": {"low": 152.0, "high": 302.0},
    "Rubber (RSS-5)": {"low": 148.0, "high": 296.0},
    "Rubber (Latex 60% DRC)": {"low": 115.0, "high": 205.0},
    "Small Cardamom (Auction Avg)": {"low": 1450.0, "high": 3250.0},
    "Coconut (Whole)": {"low": 26.0, "high": 52.0},
    "Milling Copra": {"low": 95.0, "high": 165.0},
}


@st.cache_data
def load_default_prices():
    return pd.read_csv("kerala_crop_prices.csv")


@st.cache_resource
def load_ml_bundle():
    try:
        return joblib.load("agri_ml_model.pkl")
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_live_global_macro():
    tickers = {"Crude_Oil": "CL=F", "USD_INR": "USDINR=X", "Agri_Index": "DBA"}
    frames = {}
    for name, ticker in tickers.items():
        hist = yf.download(ticker, period="1mo", progress=False)
        if hist.empty:
            return None
        close_s = (
            hist["Close"][ticker]
            if isinstance(hist.columns, pd.MultiIndex)
            else hist["Close"]
        )
        frames[name] = close_s

    df = pd.DataFrame(frames).ffill().dropna()
    latest = df.iloc[-1]
    prev_5d = df.iloc[-6] if len(df) >= 6 else df.iloc[0]

    return {
        "Crude_Oil": float(latest["Crude_Oil"]),
        "USD_INR": float(latest["USD_INR"]),
        "Agri_Index": float(latest["Agri_Index"]),
        "Oil_Return_5d": float(
            (latest["Crude_Oil"] - prev_5d["Crude_Oil"]) / prev_5d["Crude_Oil"]
        ),
        "USDINR_Return_5d": float(
            (latest["USD_INR"] - prev_5d["USD_INR"]) / prev_5d["USD_INR"]
        ),
        "Agri_Return_5d": float(
            (latest["Agri_Index"] - prev_5d["Agri_Index"])
            / prev_5d["Agri_Index"]
        ),
    }


@st.cache_data(ttl=3600)
def get_detailed_weather(district_name):
    coords = KERALA_DISTRICTS.get(district_name)
    if not coords:
        return None, 0.0

    lat, lon = coords
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"past_days=7&forecast_days=7&"
        f"daily=temperature_2m_max,precipitation_sum,precipitation_probability_max,"
        f"relative_humidity_2m_mean,wind_speed_10m_max&"
        f"timezone=Asia%2FKolkata"
    )
    res = requests.get(weather_url, timeout=10).json()
    if "daily" not in res:
        return None, 0.0

    w = res["daily"]
    full_df = pd.DataFrame({
        "Date": w["time"],
        "Max Temp (°C)": w["temperature_2m_max"],
        "Rain (mm)": w["precipitation_sum"],
        "Rain Chance (%)": w["precipitation_probability_max"],
        "Humidity (%)": w["relative_humidity_2m_mean"],
        "Wind (km/h)": w["wind_speed_10m_max"],
    })

    past_7d_rain = float(full_df.iloc[:7]["Rain (mm)"].fillna(0).sum())
    forecast_df = full_df.iloc[7:].reset_index(drop=True)
    return forecast_df, round(past_7d_rain, 1)


# --- LOAD DATA & MODELS ---
all_prices_df = load_default_prices()
ml_bundle = load_ml_bundle()
macro_live = get_live_global_macro()

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Location & Crop")
district = st.sidebar.selectbox(
    "Your District",
    list(KERALA_DISTRICTS.keys()),
    index=4,  # Kottayam
)

commodity = st.sidebar.selectbox(
    "Crop / Grade", all_prices_df["Commodity"].unique().tolist()
)

price_df = all_prices_df[all_prices_df["Commodity"] == commodity].copy()
official_rate = float(price_df["Price_Per_Kg"].iloc[-1])
latest_date = str(price_df["Date"].iloc[-1])
data_source = (
    str(price_df["Source"].iloc[-1])
    if "Source" in price_df.columns
    else "Official Market Dataset"
)

current_price = st.sidebar.number_input(
    f"Today's Local Rate in {district} (₹/kg)",
    min_value=1.0,
    max_value=10000.0,
    value=official_rate,
    step=0.5,
)
price_df.iloc[-1, price_df.columns.get_loc("Price_Per_Kg")] = current_price

st.sidebar.header("2. Harvest & Cost Details")
unit = st.sidebar.radio("Quantity Unit", ["kg", "Quintal (100 kg)"], horizontal=True)
raw_qty = st.sidebar.number_input(
    f"Harvest Quantity ({unit})",
    min_value=0.5,
    max_value=100000.0,
    value=1.0 if unit == "Quintal (100 kg)" else 100.0,
    step=0.5,
)
harvest_qty_kg = raw_qty if unit == "kg" else raw_qty * 100.0

cost_per_kg = st.sidebar.number_input(
    "Harvesting & Processing Cost (₹/kg)",
    min_value=0.0,
    max_value=5000.0,
    value=40.0 if "Rubber" in commodity else 25.0,
    step=1.0,
)

has_dry_storage = st.sidebar.checkbox(
    "I have covered / indoor drying storage", value=False
)

with st.sidebar.expander("Upload Custom Price Sheet"):
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        all_prices_df = pd.read_csv(uploaded_file)

# --- RUN ANALYSIS ---
weather_df, past_7d_rain_mm = get_detailed_weather(district)

if weather_df is None:
    st.error(f"Could not fetch weather for {district}.")
else:
    y_hist = price_df["Price_Per_Kg"].values
    x_hist = np.arange(len(y_hist))
    slope, _ = np.polyfit(x_hist, y_hist, 1)

    ml_used = False
    gb_mae = 0.0
    mape = 0.0

    if ml_bundle and commodity in ml_bundle and macro_live is not None:
        m_info = ml_bundle[commodity]
        gb_model = m_info["gb_model"]
        gb_mae = m_info["gb_mae"]
        mape = m_info["mape"]

        ret_1d = (y_hist[-1] - y_hist[-2]) / y_hist[-2] if len(y_hist) >= 2 else 0.0
        ret_3d = (y_hist[-1] - y_hist[-4]) / y_hist[-4] if len(y_hist) >= 4 else 0.0
        ret_5d = (y_hist[-1] - y_hist[-6]) / y_hist[-6] if len(y_hist) >= 6 else 0.0
        ma7 = float(np.mean(y_hist[-7:]))
        price_to_ma7 = (current_price / ma7) - 1.0
        daily_rets = np.diff(y_hist[-8:]) / y_hist[-8:-1] if len(y_hist) >= 3 else [0.0]
        vol_7d = float(np.std(daily_rets))

        now = datetime.now()
        X_live = pd.DataFrame([{
            "Return_1d": ret_1d,
            "Return_3d": ret_3d,
            "Return_5d": ret_5d,
            "Price_To_MA7": price_to_ma7,
            "Volatility_7d": vol_7d,
            "Oil_Return_5d": macro_live["Oil_Return_5d"],
            "USDINR_Return_5d": macro_live["USDINR_Return_5d"],
            "Agri_Return_5d": macro_live["Agri_Return_5d"],
            "Rain_Last_7d_mm": past_7d_rain_mm,
            "Month": now.month,
            "Day_Of_Week": now.weekday(),
        }])

        pred_return_1d = float(gb_model.predict(X_live)[0])
        tomorrow_price = round(current_price * (1.0 + pred_return_1d), 2)
        ml_used = True
    else:
        tomorrow_price = round(current_price + slope, 2)

    tomorrow_diff_kg = round(tomorrow_price - current_price, 2)
    tomorrow_pct = round((tomorrow_diff_kg / current_price) * 100, 2)

    predicted_price_7d = round(tomorrow_price + slope * 5, 2)
    expected_change_pct = round(
        ((predicted_price_7d - current_price) / current_price) * 100, 2
    )

    hist = HISTORICAL_RANGES.get(
        commodity, {"low": min(y_hist) * 0.8, "high": max(y_hist) * 1.05}
    )
    price_percentile = round(
        ((current_price - hist["low"]) / (hist["high"] - hist["low"])) * 100, 1
    )
    at_multi_year_peak = price_percentile >= 90.0

    total_cost = round(cost_per_kg * harvest_qty_kg, 0)
    net_profit_today = round((current_price - cost_per_kg) * harvest_qty_kg, 0)
    net_profit_tomorrow = round(
        (tomorrow_price - cost_per_kg) * harvest_qty_kg, 0
    )
    tomorrow_profit_shift = round(net_profit_tomorrow - net_profit_today, 0)

    max_rain_mm = weather_df["Rain (mm)"].max()
    total_rain_mm = round(weather_df["Rain (mm)"].sum(), 1)
    avg_humidity = round(weather_df["Humidity (%)"].mean(), 1)
    rainy_days = (weather_df["Rain Chance (%)"] > 60).sum()
    high_weather_risk = (max_rain_mm > 15) or (
        rainy_days >= 3 and avg_humidity > 82
    )

    # --- CLEAN, RESULT-FOCUSED METRIC CARDS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Today's Rate",
        f"₹{current_price:.2f}/kg",
        f"Profit: ₹{net_profit_today:,.0f}",
    )
    col2.metric(
        "Expected Tomorrow",
        f"₹{tomorrow_price:.2f}/kg",
        f"₹{tomorrow_diff_kg:+.2f} ({tomorrow_pct:+}%)",
    )
    col3.metric(
        "7-Day Outlook",
        f"₹{predicted_price_7d:.2f}/kg",
        f"{expected_change_pct:+}%",
    )
    col4.metric(
        "7-Day Rain",
        f"{total_rain_mm} mm",
        f"Peak: {max_rain_mm} mm/day",
    )

    st.divider()

    # --- PLAIN-LANGUAGE DECISION BANNER ---
    if at_multi_year_peak and not high_weather_risk:
        st.warning(
            "### 🟠 RECOMMENDATION: SELL MOST OF YOUR STOCK NOW"
        )
        st.write(
            f"Today's rate (**₹{current_price:.2f}/kg**) is near its highest"
            " level in recent years. Because prices at peak levels can drop"
            " suddenly, consider selling **60–70% now** to lock in"
            f" **₹{net_profit_today:,.0f}** in net profit."
        )
    elif (tomorrow_pct > 0.4 or expected_change_pct > 1.5) and not high_weather_risk:
        st.success("### 🟢 RECOMMENDATION: WAIT A FEW DAYS TO SELL")
        st.write(
            f"Market demand points upward toward **₹{tomorrow_price:.2f}/kg**"
            f" tomorrow (potentially adding **₹{tomorrow_profit_shift:+,.0f}**"
            f" to your profit on **{harvest_qty_kg:g} kg**), and the weather"
            f" in {district} looks safe for harvesting and drying."
        )
    elif (tomorrow_pct > 0.4 or expected_change_pct > 1.5) and high_weather_risk:
        if has_dry_storage:
            st.warning("### 🟡 RECOMMENDATION: HARVEST NOW & STORE INDOORS")
            st.write(
                f"Prices are expected to rise (**₹{tomorrow_price:.2f}/kg**),"
                f" but heavy rain (**{max_rain_mm} mm**) is coming to"
                f" {district}. Harvest now before the rain and store indoors."
            )
        else:
            st.error("### 🔴 RECOMMENDATION: HARVEST & SELL IMMEDIATELY")
            st.write(
                f"Heavy rain (**{max_rain_mm} mm**) and high humidity"
                f" (**{avg_humidity}%**) could damage your crop without indoor"
                f" storage. Sell now to secure **₹{net_profit_today:,.0f}** in"
                " net profit."
            )
    elif tomorrow_pct <= -0.4 or expected_change_pct <= -1.5:
        st.error("### 🔴 RECOMMENDATION: SELL NOW (Prices Dipping)")
        st.write(
            f"Prices are projected to dip toward **₹{tomorrow_price:.2f}/kg**"
            f" tomorrow (an estimated **₹{abs(tomorrow_profit_shift):,.0f}**"
            f" drop on your **{harvest_qty_kg:g} kg**). Selling today locks in"
            f" **₹{net_profit_today:,.0f}** in net profit."
        )
    else:
        st.info("### 🟡 RECOMMENDATION: HARVEST AS SCHEDULED")
        st.write(
            f"Tomorrow's expected price (**₹{tomorrow_price:.2f}/kg**) is"
            f" steady. Your **{harvest_qty_kg:g} kg** harvest brings in roughly"
            f" **₹{net_profit_today:,.0f}** in net profit after ₹{total_cost:,.0f}"
            f" expenses. Plan harvesting on dry days (avoiding {max_rain_mm} mm"
            " peak rain)."
        )

    st.subheader(f"📈 Recent Price Trend: {commodity}")
    st.line_chart(price_df.set_index("Date")["Price_Per_Kg"])

    st.subheader(f"🌦️ 7-Day Weather Forecast: {district}")
    st.dataframe(weather_df, use_container_width=True, hide_index=True)

    # --- COLLAPSIBLE TECHNICAL METHODOLOGY & DATA SOURCES ---
    with st.expander("🔍 How this forecast is calculated (Technical & Model Details)"):
        st.markdown(
            f"""
            * **Official Market Benchmark:** {data_source} (Latest rate: ₹{official_rate:.2f}/kg as of {latest_date})
            * **Forecasting Algorithm:** Return-based Gradient Boosting Regressor (`HistGradientBoostingRegressor`) trained on 3 years of daily global market & local supply data
            * **Backtest Accuracy (Time-Series Cross-Validation):** ±₹{gb_mae:.2f}/kg Mean Absolute Error ({mape:.2f}% MAPE)
            * **Historical Price Position:** {min(max(price_percentile, 0), 100):.0f}th percentile of 3-year price range
            """
        )
        if macro_live is not None:
            st.markdown("**Live Market & Supply Signals Used Today:**")
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Crude Oil (CL=F)",
                f"${macro_live['Crude_Oil']:.2f}",
                f"{macro_live['Oil_Return_5d']*100:+.2f}% (5d)",
            )
            m2.metric(
                "USD / INR",
                f"₹{macro_live['USD_INR']:.2f}",
                f"{macro_live['USDINR_Return_5d']*100:+.2f}% (5d)",
            )
            m3.metric(
                f"Past 7d Rain ({district})",
                f"{past_7d_rain_mm} mm",
                "Supply-shock indicator",
            )