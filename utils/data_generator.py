"""
Synthetic data generator mimicking the Kaggle Flight Delay Dataset 2018-2024.
Dataset source: kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

AIRLINES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
}

AIRPORTS = {
    "ATL": ("Atlanta Hartsfield-Jackson", 33.6407, -84.4277),
    "LAX": ("Los Angeles International", 33.9425, -118.4081),
    "ORD": ("Chicago O'Hare International", 41.9742, -87.9073),
    "DFW": ("Dallas/Fort Worth International", 32.8998, -97.0403),
    "DEN": ("Denver International", 39.8561, -104.6737),
    "JFK": ("John F. Kennedy International", 40.6413, -73.7781),
    "SFO": ("San Francisco International", 37.6213, -122.379),
    "SEA": ("Seattle-Tacoma International", 47.4502, -122.3088),
    "LAS": ("Harry Reid International", 36.0840, -115.1537),
    "MCO": ("Orlando International", 28.4312, -81.3081),
    "MIA": ("Miami International", 25.7959, -80.2870),
    "CLT": ("Charlotte Douglas International", 35.2140, -80.9431),
    "EWR": ("Newark Liberty International", 40.6895, -74.1745),
    "PHX": ("Phoenix Sky Harbor International", 33.4373, -112.0078),
    "BOS": ("Boston Logan International", 42.3656, -71.0096),
}

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy", "Thunderstorm", "Windy"]
CANCELLATION_CODES = {
    "A": "Carrier",
    "B": "Weather",
    "C": "National Air System",
    "D": "Security"
}


def generate_flight_data(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic flight delay data mimicking the Kaggle dataset."""
    np.random.seed(seed)
    random.seed(seed)

    airport_codes = list(AIRPORTS.keys())
    airline_codes = list(AIRLINES.keys())

    records = []
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    for _ in range(n_samples):
        flight_date = start_date + timedelta(days=random.randint(0, date_range))
        month = flight_date.month
        day_of_week = flight_date.weekday() + 1  # 1=Monday
        
        origin = random.choice(airport_codes)
        dest_options = [a for a in airport_codes if a != origin]
        dest = random.choice(dest_options)
        
        airline = random.choice(airline_codes)
        flight_num = random.randint(100, 9999)

        # Scheduled departure (0000-2359)
        sched_dep_hour = np.random.choice(range(5, 24), p=_hour_distribution())
        sched_dep_min = random.choice([0, 15, 30, 45])
        sched_dep_time = sched_dep_hour * 100 + sched_dep_min

        # Distance based on airport pairs
        dist = _estimate_distance(origin, dest)

        # Weather
        season_weather_weight = _weather_weights(month)
        weather = np.random.choice(WEATHER_CONDITIONS, p=season_weather_weight)

        # Delay factors
        carrier_delay_prob = 0.15 + (airline in ["NK", "F9", "G4"]) * 0.10
        weather_delay_prob = _weather_delay_prob(weather)
        nas_delay_prob = 0.08 if origin in ["ATL", "ORD", "JFK", "LAX"] else 0.04
        late_aircraft_prob = 0.12 if sched_dep_hour >= 18 else 0.06

        dep_delay = 0
        carrier_delay = 0
        weather_delay_min = 0
        nas_delay = 0
        security_delay = 0
        late_aircraft_delay = 0

        if random.random() < carrier_delay_prob:
            carrier_delay = int(np.random.exponential(25))
        if random.random() < weather_delay_prob:
            weather_delay_min = int(np.random.exponential(35))
        if random.random() < nas_delay_prob:
            nas_delay = int(np.random.exponential(20))
        if random.random() < 0.01:
            security_delay = int(np.random.exponential(15))
        if random.random() < late_aircraft_prob:
            late_aircraft_delay = int(np.random.exponential(30))

        dep_delay = carrier_delay + weather_delay_min + nas_delay + security_delay + late_aircraft_delay
        dep_delay = max(dep_delay - random.randint(0, 5), -15)  # Some early departures

        # Air time
        air_time = max(30, int(dist / 7 + np.random.normal(0, 15)))
        arr_delay = dep_delay + int(np.random.normal(0, 5))

        # Cancellation
        cancelled = 0
        cancellation_code = ""
        if weather == "Thunderstorm" and random.random() < 0.04:
            cancelled = 1
            cancellation_code = "B"
        elif weather == "Snowy" and random.random() < 0.03:
            cancelled = 1
            cancellation_code = "B"
        elif random.random() < 0.005:
            cancelled = 1
            cancellation_code = random.choice(["A", "C", "D"])

        # Diverted
        diverted = 1 if not cancelled and random.random() < 0.002 else 0

        # Is delayed (target variable): dep_delay >= 15 min
        is_delayed = 1 if dep_delay >= 15 else 0

        records.append({
            "FL_DATE": flight_date.strftime("%Y-%m-%d"),
            "YEAR": flight_date.year,
            "MONTH": month,
            "DAY_OF_MONTH": flight_date.day,
            "DAY_OF_WEEK": day_of_week,
            "OP_CARRIER": airline,
            "OP_CARRIER_FL_NUM": flight_num,
            "ORIGIN": origin,
            "DEST": dest,
            "CRS_DEP_TIME": sched_dep_time,
            "DEP_DELAY": dep_delay if not cancelled else np.nan,
            "ARR_DELAY": arr_delay if not cancelled else np.nan,
            "CANCELLED": cancelled,
            "CANCELLATION_CODE": cancellation_code,
            "DIVERTED": diverted,
            "CRS_ELAPSED_TIME": air_time + 15,
            "ACTUAL_ELAPSED_TIME": air_time + dep_delay if not cancelled else np.nan,
            "AIR_TIME": air_time if not cancelled else np.nan,
            "DISTANCE": dist,
            "CARRIER_DELAY": carrier_delay if dep_delay > 0 else 0,
            "WEATHER_DELAY": weather_delay_min if dep_delay > 0 else 0,
            "NAS_DELAY": nas_delay if dep_delay > 0 else 0,
            "SECURITY_DELAY": security_delay if dep_delay > 0 else 0,
            "LATE_AIRCRAFT_DELAY": late_aircraft_delay if dep_delay > 0 else 0,
            "WEATHER_CONDITION": weather,
            "IS_DELAYED": is_delayed,
        })

    return pd.DataFrame(records)


def _hour_distribution():
    """Realistic flight departure hour distribution."""
    weights = [0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.08, 0.07, 0.07, 0.07,
               0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02]
    total = sum(weights)
    return [w / total for w in weights]


def _weather_weights(month: int):
    """Season-adjusted weather probabilities."""
    if month in [12, 1, 2]:  # Winter
        return [0.25, 0.20, 0.15, 0.20, 0.08, 0.04, 0.08]
    elif month in [3, 4, 5]:  # Spring
        return [0.30, 0.25, 0.18, 0.05, 0.07, 0.08, 0.07]
    elif month in [6, 7, 8]:  # Summer
        return [0.35, 0.20, 0.15, 0.00, 0.05, 0.15, 0.10]
    else:  # Fall
        return [0.35, 0.28, 0.18, 0.05, 0.07, 0.03, 0.04]


def _weather_delay_prob(weather: str) -> float:
    probs = {
        "Clear": 0.02,
        "Cloudy": 0.05,
        "Rainy": 0.15,
        "Snowy": 0.25,
        "Foggy": 0.18,
        "Thunderstorm": 0.35,
        "Windy": 0.10,
    }
    return probs.get(weather, 0.05)


def _estimate_distance(origin: str, dest: str) -> int:
    """Estimate distance between airports using lat/lon."""
    lat1, lon1 = AIRPORTS[origin][1], AIRPORTS[origin][2]
    lat2, lon2 = AIRPORTS[dest][1], AIRPORTS[dest][2]
    # Haversine approximation
    dlat = abs(lat2 - lat1)
    dlon = abs(lon2 - lon1)
    dist = int(np.sqrt(dlat**2 + dlon**2) * 69)  # degrees to miles approx
    return max(100, dist)

