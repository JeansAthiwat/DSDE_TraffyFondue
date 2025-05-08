from streamlit_autorefresh import st_autorefresh
import streamlit as st
import pandas as pd
import redis
import pydeck as pdk
import plotly.express as px
import requests

st.set_page_config(layout="wide")
st.title('Realtime Traffy Reports (from Redis) + Prediction')

# ---> AUTO REFRESH every 10 seconds
count = st_autorefresh(interval=3 * 1000, key="datarefresh")
st.write(f"Refreshed {count} times")


# ---------------------- CONNECT REDIS ------------------------
rds = redis.Redis(host="localhost", port=6379, decode_responses=True)

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

st.header("All Processed Reports (Realtime from Redis)")
st.write(f"Total Reports: {len(data)}")

if not data.empty:
    st.dataframe(data)

# ---------------------- Predict API Section ------------------------

st.sidebar.header("Predict Ticket Resolution Time")

ticket_id_input = st.sidebar.text_input("Ticket ID")

if ticket_id_input:
    try:
        response = requests.post("http://localhost:8000/predict", json={"ticket_id": ticket_id_input})
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
    latitude=data["lat"].mean(),
    longitude=data["lon"].mean(),
    zoom=13,
    pitch=40,
)

# Render the map
st.pydeck_chart(pdk.Deck(
    layers=[heatmap_layer, scatter_layer],
    initial_view_state=view_state,
    tooltip={"text": "Ticket ID: {ticket_id}"}
))