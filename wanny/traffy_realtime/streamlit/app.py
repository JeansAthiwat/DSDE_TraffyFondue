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

import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your Google API key (you should ideally put this in an .env file)
os.environ["GOOGLE_API_KEY"] = (
    "AIzaSyCjfzIgu2KAYY2zK-MJu4bAiAhMXarAulE"  # Replace with your actual key
)
CELL_SIZE = 0.01  # Example cell size (you MUST set this to your real CELL_SIZE used when you created grid_x, grid_y)


# Add this cached function near your other functions
@st.cache_data(ttl=3600)  # Cache results for 1 hour
def get_cached_analysis(ticket_id, prediction_result, context_data, ticket_data):
    """
    Cached wrapper for the LLM analysis to avoid repeated API calls
    """
    # Log that we're generating a new analysis (will only show when cache misses)
    print(f"🤖 Generating new analysis for ticket {ticket_id}...")
    return analyze_with_llm(prediction_result, context_data, ticket_data)


### jeans added
# Add this function to get context data from Redis
def get_context_data_for_ticket(redis_client, ticket_data):
    """
    Fetch relevant contextual data from Redis based on ticket information
    """
    context_data = {"weather": None, "air_quality": None, "holiday": None}

    # Get current date for context
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Find the nearest province based on coordinates (simplified approach)
    # In a real implementation, you'd calculate distance to find the closest province
    bangkok_coords = [100.5018, 13.7563]  # Default to Bangkok if no coords available
    province = "Bangkok"  # Default province

    if "grid_x" in ticket_data and "grid_y" in ticket_data:
        # Convert grid coordinates to approximate lat/lon
        lon = float(ticket_data.get("grid_x", 0)) * CELL_SIZE
        lat = float(ticket_data.get("grid_y", 0)) * CELL_SIZE

        # Here you could use a more sophisticated approach to find nearest province
        # For simplicity, we're just using Bangkok

    # Get weather data for the province and date
    weather_key = f"weather:{current_date}:{province}"
    weather_data = redis_client.get(weather_key)
    if weather_data:
        context_data["weather"] = json.loads(weather_data)

    # Get air quality data for the province
    # Find a matching AQI entry for the location
    aqi_keys = redis_client.keys(f"aqi:*{province}*")
    if aqi_keys and len(aqi_keys) > 0:
        aqi_key = aqi_keys[0]
        context_data["air_quality"] = {
            "location": aqi_key.split(":")[-1],
            "pm25": redis_client.get(aqi_key),
        }

    # Check if today is a holiday
    holiday = redis_client.get(f"holiday:{current_date}")
    if holiday:
        context_data["holiday"] = holiday

    return context_data


# Add a function to analyze with LLM
def analyze_with_llm(prediction_result, context_data, ticket_data):
    """
    Use LangChain and Gemini to analyze the prediction and context
    """
    try:
        # Initialize the LLM
        llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

        # Create prompt template
        template = """
        You are a smart assistant that analyzes urban issue tickets and their resolution predictions.
        We already have a prediction for the ticket resolution time, and we want you to analyze it based on the context data.
        whether the prediction is reasonable or not.
        
        gives professional and practical analysis.
        Please answer the following questions based on the provided data:
        1. is the data todo it reasonable considering the weather, air quality, and holiday included?
        2. What factors might delay or speed up this resolution?
        3. Provide a brief 2-3 sentence explanation of your analysis.
        Please keep your answer brief and practical.
        
        TICKET INFORMATION:
        - Ticket ID: {ticket_id}
        - Type: {ticket_type}
        - Text: {ticket_text}
        - Location: Grid coordinates x={grid_x}, y={grid_y}
        
        PREDICTION:
        - Start handling in: {start_minutes} minutes
        - Expected resolution time: {resolve_minutes} minutes
        
        CONTEXT DATA:
        Weather: {weather_info}
        Air Quality: {air_quality_info}
        Holiday: {holiday_info}
        
        Based on the above information, please analyze:
        1. Is the predicted resolution time reasonable given the context?
        2. What factors might delay or speed up this resolution?
        3. Provide a brief 2-3 sentence explanation of your analysis.
        
        Keep your answer brief and practical.
        """

        prompt = PromptTemplate(
            input_variables=[
                "ticket_id",
                "ticket_type",
                "ticket_text",
                "grid_x",
                "grid_y",
                "start_minutes",
                "resolve_minutes",
                "weather_info",
                "air_quality_info",
                "holiday_info",
            ],
            template=template,
        )

        # Create the chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain
        result = chain.run(
            ticket_id=ticket_data.get("ticket_id", "Unknown"),
            ticket_type=ticket_data.get("type", "Unknown"),
            ticket_text=ticket_data.get("text", "No description"),
            grid_x=ticket_data.get("grid_x", "Unknown"),
            grid_y=ticket_data.get("grid_y", "Unknown"),
            start_minutes=prediction_result.get("start_in_minutes", "Unknown"),
            resolve_minutes=prediction_result.get("resolve_minutes", "Unknown"),
            weather_info=(
                json.dumps(context_data.get("weather", {}), ensure_ascii=False)
                if context_data.get("weather")
                else "No weather data available"
            ),
            air_quality_info=(
                json.dumps(context_data.get("air_quality", {}), ensure_ascii=False)
                if context_data.get("air_quality")
                else "No air quality data available"
            ),
            holiday_info=context_data.get("holiday", "Not a holiday"),
        )

        return result

    except Exception as e:
        return f"Error generating analysis: {str(e)}"


### end jeans added


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

# Then replace the prediction section with this code
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

            # Get ticket data from Redis
            ticket_key = f"feat:{ticket_id_input}"
            ticket_data = rds.hgetall(ticket_key)

            if ticket_data:
                # Get contextual data
                context_data = get_context_data_for_ticket(rds, ticket_data)

                # Use the cached analysis function instead of direct call with spinner
                analysis = get_cached_analysis(
                    ticket_id_input, result, context_data, ticket_data
                )

                # Display the analysis
                st.sidebar.subheader("AI Analysis")
                st.sidebar.info(analysis)

                # Display the context data used
                with st.sidebar.expander("Context Data Used"):
                    if context_data["weather"]:
                        st.write(
                            "**Weather:**",
                            context_data["weather"]["weather"],
                            f"({context_data['weather']['rain']})",
                        )
                    if context_data["air_quality"]:
                        st.write(
                            "**Air Quality (PM2.5):**",
                            context_data["air_quality"]["pm25"],
                        )
                    if context_data["holiday"]:
                        st.write("**Holiday:**", context_data["holiday"])
            else:
                st.sidebar.warning("Could not find ticket data for analysis")
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
