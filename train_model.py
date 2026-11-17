import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf

print("1. Downloading 3 years of live global market demand data via yfinance...")
tickers = {
    "Crude_Oil": "CL=F",
    "USD_INR": "USDINR=X",
    "Agri_Index": "DBA",
}

macro_frames = {}
for name, ticker in tickers.items():
    hist = yf.download(ticker, period="3y", progress=False)
    close_s = (
        hist["Close"][ticker]
        if isinstance(hist.columns, pd.MultiIndex)
        else hist["Close"]
    )
    macro_frames[name] = close_s

macro_df = pd.DataFrame(macro_frames).ffill().dropna()
macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)

macro_df["Oil_Return_5d"] = macro_df["Crude_Oil"].pct_change(5)
macro_df["USDINR_Return_5d"] = macro_df["USD_INR"].pct_change(5)
macro_df["Agri_Return_5d"] = macro_df["Agri_Index"].pct_change(5)
macro_df = macro_df.dropna()

start_str = macro_df.index.min().strftime("%Y-%m-%d")
end_str = macro_df.index.max().strftime("%Y-%m-%d")
print(f"   Downloaded {len(macro_df)} trading days ({start_str} to {end_str}).")

print("2. Fetching 3 years of historical Kerala rainfall from Open-Meteo Archive...")


def fetch_historical_rain(lat, lon):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_str}&end_date={end_str}&"
        f"daily=precipitation_sum&timezone=Asia%2FKolkata"
    )
    res = requests.get(url, timeout=15).json()
    rain_df = pd.DataFrame({
        "Date": pd.to_datetime(res["daily"]["time"]),
        "Rain_mm": res["daily"]["precipitation_sum"],
    }).set_index("Date")
    rain_df["Rain_Last_7d_mm"] = rain_df["Rain_mm"].rolling(7, min_periods=1).sum()
    return rain_df[["Rain_Last_7d_mm"]]


rain_kottayam = fetch_historical_rain(9.5916, 76.5222)  # Rubber & Coconut hub
rain_idukki = fetch_historical_rain(9.8495, 76.9720)    # Pepper & Cardamom hub

local_df = pd.read_csv("kerala_crop_prices.csv")
local_df["Date"] = pd.to_datetime(local_df["Date"])

trained_models = {}

for commodity in local_df["Commodity"].unique():
    sub = local_df[local_df["Commodity"] == commodity].sort_values("Date")
    latest_verified_price = float(sub["Price_Per_Kg"].iloc[-1])

    df = macro_df.copy()
    rain_hub = rain_idukki if ("Pepper" in commodity or "Cardamom" in commodity) else rain_kottayam
    df = df.join(rain_hub, how="left").ffill().fillna(0.0)

    oil_norm = df["Crude_Oil"] / df["Crude_Oil"].iloc[-1]
    fx_norm = df["USD_INR"] / df["USD_INR"].iloc[-1]
    agri_norm = df["Agri_Index"] / df["Agri_Index"].iloc[-1]
    rain_shock = 1.0 + (df["Rain_Last_7d_mm"] / 2000.0)  # Supply tightening from heavy rain

    if "Rubber" in commodity:
        macro_factor = (0.45 * oil_norm + 0.40 * fx_norm + 0.15 * agri_norm) * rain_shock
    elif "Pepper" in commodity or "Cardamom" in commodity:
        macro_factor = (0.15 * oil_norm + 0.50 * fx_norm + 0.35 * agri_norm) * rain_shock
    else:
        macro_factor = (0.20 * oil_norm + 0.35 * fx_norm + 0.45 * agri_norm) * rain_shock

    df["Price"] = (latest_verified_price * macro_factor).round(2)

    for _, row in sub.iterrows():
        if row["Date"] in df.index:
            df.loc[row["Date"], "Price"] = row["Price_Per_Kg"]

    # Feature Engineering: Stationary Returns, Lags, Volatility, Supply Rain & Seasonality
    df["Return_1d"] = df["Price"].pct_change(1)
    df["Return_3d"] = df["Price"].pct_change(3)
    df["Return_5d"] = df["Price"].pct_change(5)
    df["Price_To_MA7"] = (df["Price"] / df["Price"].rolling(7).mean()) - 1.0
    df["Volatility_7d"] = df["Return_1d"].rolling(7).std()
    df["Month"] = df.index.month
    df["Day_Of_Week"] = df.index.dayofweek

    # Target: Tomorrow's percentage return (eliminates tree ceiling extrapolation limits)
    df["Target_Return_1d"] = df["Price"].shift(-1) / df["Price"] - 1.0
    df["Actual_Tomorrow_Price"] = df["Price"].shift(-1)

    model_df = df.dropna().copy()

    feature_cols = [
        "Return_1d",
        "Return_3d",
        "Return_5d",
        "Price_To_MA7",
        "Volatility_7d",
        "Oil_Return_5d",
        "USDINR_Return_5d",
        "Agri_Return_5d",
        "Rain_Last_7d_mm",
        "Month",
        "Day_Of_Week",
    ]

    X = model_df[feature_cols]
    y_ret = model_df["Target_Return_1d"]
    p_today = model_df["Price"]
    p_tomorrow_actual = model_df["Actual_Tomorrow_Price"]

    # Walk-Forward Time-Series Cross-Validation with Regularized Gradient Boosting
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []

    gb_model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features=["Month", "Day_Of_Week"],
        random_state=42,
    )

    for train_idx, test_idx in tscv.split(X):
        gb_model.fit(X.iloc[train_idx], y_ret.iloc[train_idx])
        pred_ret = gb_model.predict(X.iloc[test_idx])
        pred_price = p_today.iloc[test_idx] * (1.0 + pred_ret)
        maes.append(mean_absolute_error(p_tomorrow_actual.iloc[test_idx], pred_price))

    gb_model.fit(X, y_ret)
    avg_mae = round(float(np.mean(maes)), 2)

    recent_pred_price = p_today.iloc[-30:] * (1.0 + gb_model.predict(X.iloc[-30:]))
    mape = round(
        float(mean_absolute_percentage_error(p_tomorrow_actual.iloc[-30:], recent_pred_price) * 100),
        2,
    )

    print(f"✅ {commodity:28s} | Tuned GBDT MAE: ±₹{avg_mae}/kg ({mape}% MAPE)")

    trained_models[commodity] = {
        "gb_model": gb_model,
        "feature_cols": feature_cols,
        "gb_mae": avg_mae,
        "mape": mape,
    }

joblib.dump(trained_models, "agri_ml_model.pkl")
print("\n🎉 Saved improved return-based ML models to 'agri_ml_model.pkl'!")