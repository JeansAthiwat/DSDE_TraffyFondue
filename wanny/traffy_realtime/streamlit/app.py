from streamlit_autorefresh import st_autorefresh
import streamlit as st
import pandas as pd
import redis
import pydeck as pdk
import plotly.express as px
import requests
import time
import re
import threading
import json
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

st.set_page_config(layout="wide")
st.title("Realtime Traffy Reports (from Redis) + Prediction")

# ---> AUTO REFRESH every 3 seconds
count = st_autorefresh(interval=3 * 1000, key="datarefresh")
st.write(f"Refreshed {count} times")

# ---------------------- CONNECT REDIS ------------------------
rds = redis.Redis(host="localhost", port=6379, decode_responses=True)


# --------------------- SCRAPING FUNCTIONS --------------------
def fetch_holiday(r):
    url = "https://www.myhora.com/calendar/ical/holiday.aspx?latest.txt"
    thai_months = {
        "ม.ค.": "01",
        "ก.พ.": "02",
        "มี.ค.": "03",
        "เม.ย.": "04",
        "พ.ค.": "05",
        "มิ.ย.": "06",
        "ก.ค.": "07",
        "ส.ค.": "08",
        "ก.ย.": "09",
        "ต.ค.": "10",
        "พ.ย.": "11",
        "ธ.ค.": "12",
    }

    def convert(date_str):
        match = re.match(r"(\d{1,2})\s(\S+)\s(\d{4})", date_str.strip())
        if not match:
            return None
        d, m, y = match.groups()
        return f"{int(d):02d}/{thai_months.get(m)}/{int(y) - 543}"

    try:
        resp = requests.get(url)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print("⚠️ Error fetching holiday page:", e)
        return

    count = 0
    for div in soup.find_all("div", class_="mb-5"):
        cols = div.find_all("div")
        if len(cols) >= 2:
            raw = cols[0].text.strip()
            name = cols[1].text.strip()
            formatted = convert(raw)
            if formatted:
                iso = datetime.strptime(formatted, "%d/%m/%Y").strftime("%Y-%m-%d")
                r.set(f"holiday:{iso}", name)
                r.sadd(f"holidays:{iso[:4]}", iso)
                count += 1
    print(f"✅ fetched {count} holidays")


def fetch_air_quality(r, driver):
    url = "https://airquality.airbkk.com/PublicWebClient/#/Modules/Aqs/HomePage"
    driver.get(url)
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("div", class_="table-responsive")
    if not table:
        print("❌ Could not find air quality table")
        return

    rows = table.find("tbody", class_="table-bordered").find_all("tr")
    count = 0

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 6:
            b_tag = cols[1].find("b")
            pm25 = cols[5].text.strip()
            if b_tag and pm25:
                location = b_tag.text.strip()
                r.set(f"aqi:{location}", pm25)
                count += 1
    print(f"✅ fetched PM2.5 for {count} locations")


def fetch_weather_forecast(r, driver):
    provinces = [
        "Bangkok",
        "Nakhon Pathom",
        "Pathum Thani",
        "Nonthaburi",
        "Samut Prakan",
        "Samut Sakhon",
    ]

    total = 0

    for province in provinces:
        url = f"https://www.tmd.go.th/en/weatherForecast7Days?province={province}&culture=en-US"
        driver.get(url)
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        container = soup.select_one("div.d-flex.gap-3.h-100")
        if not container:
            print(f"❌ Cannot find forecast container for {province}")
            continue

        cards = container.select(".card")
        for card in cards:
            try:
                date_text = card.select_one(".today-header .text-dark2").text.strip()
                date_obj = datetime.strptime(
                    date_text + f" {datetime.now().year}", "%d %b %Y"
                )
                date_iso = date_obj.strftime("%Y-%m-%d")

                weather = card.select(".font-tiny.text-center")[0].text.strip()
                rain = card.select(".font-tiny.text-center")[1].text.strip()
                temps = card.select(".sub-heading div")

                max_temp = temps[0].text.strip() if len(temps) > 0 else ""
                min_temp = temps[2].text.strip() if len(temps) > 2 else ""
                wind = card.select_one("span.ps-1").text.strip()

                redis_key = f"weather:{date_iso}:{province}"
                data = {
                    "province": province,
                    "date": date_iso,
                    "weather": weather,
                    "rain": rain,
                    "max_temp": max_temp.replace("°", ""),
                    "min_temp": min_temp.replace("°", ""),
                    "wind_speed": wind.replace(" km./hr.", ""),
                }

                r.set(redis_key, json.dumps(data))
                total += 1

            except Exception as e:
                print(f"⚠️ Error parsing weather card for {province}: {e}")

    print(f"✅ fetched {total} weather forecast entries")


# ----------------- BACKGROUND SCRAPER THREAD -----------------
def scraper_thread_func(redis_conn):
    """Background thread for periodic data scraping"""
    try:
        # Set up Selenium headless browser for scraping
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=options)

        while True:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n⏰ Running scraper at {timestamp}")
                redis_conn.set("scraper:last_run", timestamp)

                # Run the scraping functions
                fetch_weather_forecast(redis_conn, driver)
                fetch_air_quality(redis_conn, driver)
                fetch_holiday(redis_conn)

                # Sleep for 30 seconds as requested
                time.sleep(30)
            except Exception as e:
                error_msg = f"⚠️ Error in scraper thread: {e}"
                print(error_msg)
                redis_conn.set("scraper:last_error", error_msg)
                time.sleep(10)  # Short sleep before retry after error

    except Exception as e:
        print(f"⚠️ Fatal error in scraper thread: {e}")
    finally:
        if "driver" in locals():
            driver.quit()


# Start the background scraper thread (runs only once when app first loads)
if "scraper_thread_started" not in st.session_state:
    scraper_thread = threading.Thread(
        target=scraper_thread_func,
        args=(rds,),
        daemon=True,  # Will exit when main thread exits
    )
    scraper_thread.start()
    st.session_state["scraper_thread_started"] = True


# @st.cache_data
def load_data_from_redis():
    keys = rds.keys("feat:*")
    rows = []

    for key in keys:
        feats = rds.hgetall(key)
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df


data = load_data_from_redis()

# Show scraper status
last_run = rds.get("scraper:last_run")
if last_run:
    st.sidebar.success(f"Data scraper last run: {last_run}")
else:
    st.sidebar.info("Data scraper starting up...")

# Display scraped data in sidebar
with st.sidebar.expander("View Scraped Data"):
    # Show holidays
    st.subheader("Holidays")
    holiday_keys = rds.keys("holiday:*")
    if holiday_keys:
        holiday_data = {k: rds.get(k) for k in holiday_keys[:10]}  # Limit to 10
        st.dataframe(
            pd.DataFrame(
                {
                    "Date": [k.split(":")[-1] for k in holiday_data.keys()],
                    "Holiday": list(holiday_data.values()),
                }
            )
        )
    else:
        st.write("No holiday data available")

    # Show weather forecast
    st.subheader("Weather")
    weather_keys = rds.keys("weather:*")
    if weather_keys:
        weather_data = []
        for k in weather_keys[:5]:
            try:
                w_data = json.loads(rds.get(k))
                weather_data.append(
                    {
                        "Date": w_data.get("date"),
                        "Province": w_data.get("province"),
                        "Weather": w_data.get("weather"),
                        "Rain": w_data.get("rain"),
                    }
                )
            except:
                pass
        st.dataframe(pd.DataFrame(weather_data))
    else:
        st.write("No weather data available")

    # Show air quality
    st.subheader("Air Quality (PM 2.5)")
    aqi_keys = rds.keys("aqi:*")
    if aqi_keys:
        aqi_data = {k.split(":")[-1]: rds.get(k) for k in aqi_keys[:10]}
        st.dataframe(
            pd.DataFrame(
                {"Location": list(aqi_data.keys()), "PM2.5": list(aqi_data.values())}
            )
        )
    else:
        st.write("No air quality data available")

# Original code continues...
st.header("All Processed Reports (Realtime from Redis)")
st.write(f"Total Reports: {len(data)}")

if not data.empty:
    st.dataframe(data)

# ---------------------- Predict API Section ------------------------

st.sidebar.header("Predict Ticket Resolution Time")

ticket_id_input = st.sidebar.text_input("Ticket ID")

if ticket_id_input:
    try:
        response = requests.post(
            "http://localhost:8000/predict", json={"ticket_id": ticket_id_input}
        )
        if response.status_code == 200:
            result = response.json()
            st.sidebar.success(f"Prediction for Ticket ID {ticket_id_input}")
            st.sidebar.write(f"Start In Minutes: {result['start_in_minutes']}")
            st.sidebar.write(f"Resolve Minutes: {result['resolve_minutes']}")
        else:
            st.sidebar.error(f"Ticket ID not found or error occurred.")
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")


CELL_SIZE = 0.01  # Example cell size (you MUST set this to your real CELL_SIZE used when you created grid_x, grid_y)

# # Convert grid_x and grid_y back to approximate lon/lat
data["grid_x"] = data["grid_x"].astype(int)
data["grid_y"] = data["grid_y"].astype(int)

data["lon"] = data["grid_x"] * CELL_SIZE
data["lat"] = data["grid_y"] * CELL_SIZE

heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=data,
    get_position="[lon, lat]",
    radiusPixels=50,
    aggregation=pdk.types.String("SUM"),
)
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position="[lon, lat]",
    get_color="[255, 0, 0, 160]",  # Red color
    get_radius=30,
    pickable=True,
)

# Set view state
view_state = pdk.ViewState(
    latitude=data["lat"].mean() if not data.empty else 13.7563,
    longitude=data["lon"].mean() if not data.empty else 100.5018,
    zoom=13,
    pitch=40,
)

# Render the map
st.pydeck_chart(
    pdk.Deck(
        layers=[heatmap_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"text": "Ticket ID: {ticket_id}"},
    )
)
