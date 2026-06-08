# ✈️ Flight Delay Prediction System

**Project Theme:** Aerospace  
**Technology Theme:** Prediction (Machine Learning & Data Science)  
**Dataset:** [Kaggle – Flight Delay Dataset 2018–2024](https://kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024)

---

## 📋 Problem Statement

Flight delays cause inconvenience to passengers, increase airline costs, and disrupt airport operations. This system predicts whether a flight will be delayed (departure ≥ 15 minutes) using historical flight data, real-time weather conditions, and an interactive route map.

---

## 🏗️ Project Structure

```
flight_delay_app/
├── app.py                     # Main Streamlit application
├── requirements.txt
├── README.md                  # This file
├── readme.html                # Full HTML documentation
├── models/                    # ✨ Pre-trained ML model binaries
│   ├── encoders.pkl
│   ├── feature_importances.pkl
│   ├── metrics.pkl
│   └── models.pkl
└── utils/
    ├── data_generator.py      # Synthetic dataset generator (mimics Kaggle dataset)
    ├── ml_models.py           # ML model training & prediction
    └── weather_fetcher.py     # ✨ NEW: Live weather integration (Open-Meteo)
```

---

## 🚀 Setup & Run

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install map dependencies (optional but recommended)
pip install folium streamlit-folium

# 3. Run the app
streamlit run app.py
```

---

## 🤖 ML Models

| Model | Description |
|---|---|
| Random Forest | Ensemble of 100 decision trees, balanced class weights |
| Gradient Boosting | Sequential boosting, learning rate 0.1 |
| Logistic Regression | Linear baseline with StandardScaler pipeline |

**Features used:**
- Month, Day of Week, Day of Month
- Scheduled departure time
- Distance, Scheduled elapsed time
- Airline (encoded), Origin (encoded), Destination (encoded)
- Weather condition (encoded)
- ✨ **NEW:** 25 live weather features (see [Weather Integration](#-weather-integration) below)

**Target:** `IS_DELAYED` — 1 if departure delay ≥ 15 minutes

---

## 📊 App Pages

| # | Page | Status | Description |
|---|---|---|---|
| 1 | 🏠 Overview | — | Project summary, KPIs, dataset stats, workflow diagram |
| 2 | 📊 EDA & Insights | — | Delay by month, hour, airline, weather, route map |
| 3 | 🤖 Model Training | — | Train RF, GBM, LR with one click; class distribution |
| 4 | 🎯 Predict a Flight | ✨ **Updated** | Live weather fetch + delay probability gauge + interactive route map |
| 5 | 📈 Model Performance | — | ROC curves, confusion matrices, feature importances |

---

## 🌦️ Weather Integration

> **NEW feature** — Real-time weather is fetched at prediction time using the [Open-Meteo API](https://open-meteo.com/) — completely free, no API key or signup required.

### How it works

1. When the user clicks **Predict**, the app calls `get_route_weather(origin, dest)` from `utils/weather_fetcher.py`.
2. Live conditions are fetched for both airports simultaneously via `api.open-meteo.com/v1/forecast`.
3. Raw weather data is converted into **25 model-ready features** via `build_weather_features()` and merged into the prediction input.
4. Each airport card displays a **LIVE** badge (green) when data is fresh, or an **OFFLINE** badge (red) with safe fallback defaults when the API is unreachable.

### Supported airports (30 major US hubs)

`ATL` `LAX` `ORD` `DFW` `DEN` `JFK` `SFO` `SEA` `LAS` `MCO` `EWR` `MIA` `PHX` `IAH` `BOS` `MSP` `DTW` `FLL` `PHL` `LGA` `BWI` `SLC` `DCA` `SAN` `MDW` `TPA` `PDX` `HNL` `DAL` `STL`

### Weather data fields returned

| Field | Type | Description |
|---|---|---|
| `airport` | str | IATA airport code |
| `live` | bool | `True` if freshly fetched, `False` if fallback defaults |
| `temperature_c` | float | Air temperature at 2 m height (°C) |
| `feels_like_c` | float | Apparent temperature (°C) |
| `humidity_pct` | int | Relative humidity (%) |
| `pressure_hpa` | float | Surface pressure (hPa) |
| `wind_speed_ms` | float | Wind speed at 10 m (m/s) |
| `wind_deg` | int | Wind direction (degrees) |
| `visibility_m` | float | Visibility (metres) |
| `clouds_pct` | int | Cloud cover (%) |
| `precipitation` | float | Precipitation in current hour (mm) |
| `weather_code` | int | WMO weather interpretation code |
| `weather_main` | str | Category: `Clear / Clouds / Rain / Snow / Fog / Thunderstorm` |
| `weather_desc` | str | Human-readable description (e.g. `"Moderate rain"`) |
| `fetched_at` | str | UTC fetch timestamp (`HH:MM UTC`) |

### WMO weather codes supported

| Code(s) | Icon | Label |
|---|---|---|
| 0 | ☀️ | Clear sky |
| 1 | 🌤️ | Mainly clear |
| 2 | ⛅ | Partly cloudy |
| 3 | ☁️ | Overcast |
| 45, 48 | 🌫️ | Fog / Icy fog |
| 51, 53, 55 | 🌦️ | Drizzle (light / moderate / dense) |
| 61, 63, 65 | 🌧️ | Rain (slight / moderate / heavy) |
| 71, 73, 75, 77 | ❄️ | Snow / Snow grains |
| 80, 81, 82 | 🌧️ | Showers |
| 85, 86 | ❄️ | Snow showers |
| 95, 96, 99 | ⛈️ | Thunderstorm (± hail) |

### Weather risk score (0–10)

A composite route risk score is computed from conditions at both airports:

| Condition | Points |
|---|---|
| Wind > 10 m/s | +1.5 per airport |
| Wind > 17 m/s (gale force) | +2.0 per airport |
| Visibility < 5 km | +0.5 per airport |
| Visibility < 1 km | +1.5 per airport |
| Visibility < 200 m | +2.5 per airport |
| Any precipitation | +0.5 per airport |
| Precipitation > 2.5 mm | +1.0 per airport |
| Precipitation > 7.5 mm | +1.5 per airport |
| Thunderstorm | +3.0 per airport |
| Snow or Fog | +2.0 per airport |
| Rain or Drizzle | +1.0 per airport |

Score is capped at **10.0** and colour-coded: 🟢 Low (< 3) · 🟡 Moderate (3–6) · 🔴 High (> 6)

### Key functions in `utils/weather_fetcher.py`

| Function | Returns | Description |
|---|---|---|
| `fetch_weather(iata)` | `dict` | Fetches live conditions for one airport; falls back gracefully on failure |
| `get_route_weather(origin, dest)` | `tuple[dict, dict]` | Fetches both airports; returns `(origin_wx, dest_wx)` |
| `compute_risk_score(origin_wx, dest_wx)` | `float` | Composite 0–10 weather risk score for the full route |
| `build_weather_features(origin_wx, dest_wx)` | `dict` (25 keys) | Converts raw weather into model-ready numeric features |

### Fallback behaviour

If the API call fails for any reason (timeout, unknown airport, HTTP error), `_default_weather()` returns safe neutral values — 20 °C, Clear sky, 10 km visibility, 5 m/s wind — so prediction always completes without crashing the app.

---

## 🗺️ Interactive Route Map

> **NEW feature** — The **Predict a Flight** page now renders a live [Folium](https://python-visualization.github.io/folium/) map mounted into Streamlit via `streamlit-folium`.

### Map components

| Component | Details |
|---|---|
| Base tile | CartoDB Dark Matter — matches the app's dark theme |
| Origin marker | Blue circle; popup shows live weather summary |
| Destination marker | Red circle; popup shows live weather summary |
| Flight path | Polyline coloured by weather risk score (🟢 / 🟡 / 🔴), weight 4 |
| Risk caption | Below map: route name, colour meaning, and risk score / 10 |
| Dimensions | 700 × 420 px via `st_folium()` |

### Risk-coloured flight path

- 🟢 **Green** — weather risk score < 3 (low risk)
- 🟡 **Orange** — weather risk score 3–6 (moderate risk)
- 🔴 **Red** — weather risk score > 6 (high risk)

### Optional dependency

The map requires `folium` and `streamlit-folium`. The app checks availability at startup (`FOLIUM_AVAILABLE` flag). If the packages are absent, the map section is hidden and a pip install hint is displayed — the rest of the app works normally.

```bash
pip install folium streamlit-folium
```

---

## 📦 Using the Real Kaggle Dataset

Download `flight_delay_dataset.csv` from Kaggle and load it:

```python
import pandas as pd
df = pd.read_csv("flight_delay_dataset.csv")
# Add IS_DELAYED column if missing
df["IS_DELAYED"] = (df["DEP_DELAY"] >= 15).astype(int)
```

Then in `app.py`, replace the `generate_flight_data()` call with your CSV load.

---

## 📐 Record Layout

Fields from the TranStats On-Time database in order of appearance:

| Field | Description |
|---|---|
| Year | Year |
| Quarter | Quarter (1–4) |
| Month | Month |
| DayofMonth | Day of Month |
| DayOfWeek | Day of Week |
| FlightDate | Flight Date (yyyymmdd) |
| Marketing_Airline_Network | Unique Marketing Carrier Code — use for cross-year analysis |
| Operated_or_Branded_Code_Share_Partners | Reporting Carrier Operated or Branded Code Share Partners |
| DOT_ID_Marketing_Airline | US DOT unique airline identifier |
| IATA_Code_Marketing_Airline | IATA carrier code (not always unique over time) |
| Flight_Number_Marketing_Airline | Flight Number |
| Originally_Scheduled_Code_Share_Airline | Unique Scheduled Operating Carrier Code |
| DOT_ID_Originally_Scheduled_Code_Share_Airline | DOT ID for scheduled operating carrier |
| IATA_Code_Originally_Scheduled_Code_Share_Airline | IATA code for scheduled operating carrier |
| Flight_Num_Originally_Scheduled_Code_Share_Airline | Flight Number (scheduled operating carrier) |
| Operating_Airline | Unique Carrier Code — use for cross-year analysis |
| DOT_ID_Operating_Airline | DOT ID for operating carrier |
| IATA_Code_Operating_Airline | IATA code for operating carrier |
| Tail_Number | Aircraft Tail Number |
| Flight_Number_Operating_Airline | Flight Number (operating carrier) |
| OriginAirportID | Origin airport DOT ID — stable across airport code changes |
| OriginAirportSeqID | Origin airport sequence ID — point-in-time identifier |
| OriginCityMarketID | City Market ID — consolidates multiple airports in same city |
| Origin | Origin Airport Code |
| OriginCityName | Origin City Name |
| OriginState | Origin State Code |
| OriginStateFips | Origin State FIPS code |
| OriginStateName | Origin State Name |
| OriginWac | Origin World Area Code |
| DestAirportID | Destination airport DOT ID |
| DestAirportSeqID | Destination airport sequence ID |
| DestCityMarketID | Destination City Market ID |
| Dest | Destination Airport Code |
| DestCityName | Destination City Name |
| DestState | Destination State Code |
| DestStateFips | Destination State FIPS code |
| DestStateName | Destination State Name |
| DestWac | Destination World Area Code |
| CRSDepTime | Scheduled Departure Time (local hhmm) |
| DepTime | Actual Departure Time (local hhmm) |
| DepDelay | Departure delay in minutes (negative = early) |
| DepDelayMinutes | Departure delay in minutes — early departures set to 0 |
| DepDel15 | Departure Delay ≥ 15 min indicator (1=Yes) |
| DepartureDelayGroups | Delay interval every 15 min from <-15 to >180 |
| DepTimeBlk | CRS Departure Time Block (hourly intervals) |
| TaxiOut | Taxi-out time (minutes) |
| WheelsOff | Wheels-off time (local hhmm) |
| WheelsOn | Wheels-on time (local hhmm) |
| TaxiIn | Taxi-in time (minutes) |
| CRSArrTime | Scheduled Arrival Time (local hhmm) |
| ArrTime | Actual Arrival Time (local hhmm) |
| ArrDelay | Arrival delay in minutes (negative = early) |
| ArrDelayMinutes | Arrival delay in minutes — early arrivals set to 0 |
| ArrDel15 | Arrival Delay ≥ 15 min indicator (1=Yes) |
| ArrivalDelayGroups | Arrival delay interval every 15 min |
| ArrTimeBlk | CRS Arrival Time Block (hourly intervals) |
| Cancelled | Cancelled Flight Indicator (1=Yes) |
| CancellationCode | Reason for cancellation |
| Diverted | Diverted Flight Indicator (1=Yes) |
| CRSElapsedTime | Scheduled elapsed flight time (minutes) |
| ActualElapsedTime | Actual elapsed flight time (minutes) |
| AirTime | Airborne time (minutes) |
| Flights | Number of Flights |
| Distance | Distance between airports (miles) |
| DistanceGroup | Distance interval (every 250 miles) |
| CarrierDelay | Carrier Delay (minutes) |
| WeatherDelay | Weather Delay (minutes) |
| NASDelay | National Air System Delay (minutes) |
| SecurityDelay | Security Delay (minutes) |
| LateAircraftDelay | Late Aircraft Delay (minutes) |
| FirstDepTime | First gate departure time at origin |
| TotalAddGTime | Total ground time away from gate (gate return / cancelled) |
| LongestAddGTime | Longest time away from gate (gate return / cancelled) |
| DivAirportLandings | Number of diverted airport landings |
| DivReachedDest | Diverted flight reached scheduled destination (1=Yes) |
| DivActualElapsedTime | Elapsed time for diverted flight reaching scheduled destination (minutes) |
| DivArrDelay | Arrival delay for diverted flight reaching scheduled destination (minutes) |
| DivDistance | Distance from scheduled destination to final diverted airport (miles; 0 if reached) |
| Div1Airport … Div5Airport | Diverted airport codes 1–5 |
| Div1AirportID … Div5AirportID | DOT Airport IDs for diverted airports 1–5 |
| Div1AirportSeqID … Div5AirportSeqID | Sequence IDs for diverted airports 1–5 |
| Div1WheelsOn … Div5WheelsOn | Wheels-on time at diverted airports 1–5 (local hhmm) |
| Div1TotalGTime … Div5TotalGTime | Total ground time at diverted airports 1–5 |
| Div1LongestGTime … Div5LongestGTime | Longest ground time at diverted airports 1–5 |
| Div1WheelsOff … Div5WheelsOff | Wheels-off time at diverted airports 1–5 (local hhmm) |
| Div1TailNum … Div5TailNum | Aircraft tail number at diverted airports 1–5 |
| Duplicate | Y if flight is swapped based on Form-3A data |

---

*Weather data provided by [Open-Meteo](https://open-meteo.com/) · Maps powered by [Folium](https://python-visualization.github.io/folium/)*
