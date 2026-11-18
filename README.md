# 🌾 AgriMitra: Smart Harvest & Price Advisory

🚀 **Live App:** [agrimitra-kerala.streamlit.app](https://agrimitra-kerala.streamlit.app)

An end-to-end decision-support web application and Power BI market intelligence suite for Kerala cash-crop farmers across all **14 districts**.

## 🚀 Key Features
- **Live Agricultural Weather Risk (Open-Meteo API):** Fetches 7-day rainfall, humidity, temperature, and past 7-day supply-shock precipitation for any Kerala district.
- **ML Price Forecasting (`<1% MAPE`):** Return-based Gradient Boosting model (`HistGradientBoostingRegressor`) trained on 3 years of global macro demand signals (`Crude Oil CL=F`, `USD/INR`, `DBA` via `yfinance`) and historical Kerala rainfall under `TimeSeriesSplit` cross-validation.
- **Net Profit & Multi-Year Peak Protection:** Calculates take-home profit after harvesting/processing expenses and triggers partial-sell alerts when prices cross the 90th historical percentile.
- **Executive Power BI Dashboard:** Interactive BI report (`AgriMitra_Market_BI.pbix` & `AgriMitra_BI_Dashboard.pdf`) built with Power Query and DAX to track volatility and multi-grade trends across 8 Kerala cash-crop grades.

## 📊 Backtest Validation Accuracy (Time-Series CV)
| Commodity / Grade | Benchmark Source | MAE (₹/kg) | MAPE |
| :--- | :--- | :---: | :---: |
| Rubber (RSS-4) | Rubber Board India | ±₹3.38 | 0.79% |
| Rubber (Latex 60% DRC) | Rubber Board India | ±₹2.22 | 0.80% |
| Black Pepper (Garbled) | Spices Board India | ±₹6.09 | 0.56% |
| Small Cardamom (Auction Avg) | Spices Board India | ±₹23.91 | 0.81% |
| Coconut (Whole) | Coconut Development Board | ±₹0.48 | 0.76% |

## 🛠️ Tech Stack
- **Application & ML:** Python, Streamlit, Scikit-learn, Pandas, NumPy, `yfinance`, Open-Meteo REST APIs
- **Business Intelligence:** Power BI, Power Query, DAX

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
