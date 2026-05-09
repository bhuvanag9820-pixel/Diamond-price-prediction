import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Load Models
# --- Load Models with Caching ---
@st.cache_resource
def load_all_models():
    # Loading the feature list and the trained models
    feature_columns = joblib.load("model_features.pkl")
    price_model = joblib.load("best_price_prediction_model.pkl")
    cluster_model = joblib.load("diamond_clustering_model.pkl")
    return feature_columns, price_model, cluster_model

# Calling the function to initialize the models
feature_columns, price_model, cluster_model = load_all_models()

#Title
st.title("💎 Diamond Price Prediction & Market Segmentation")

#🔹 4. User Inputs
st.header("Enter Diamond Details")

carat = st.number_input("Carat", value=0.5)
depth = st.number_input("Depth", value=60.0)
table = st.number_input("Table", value=55.0)

x = st.number_input("Length (x)", value=5.0)
y = st.number_input("Width (y)", value=5.0)
z = st.number_input("Depth (z)", value=3.0)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

#🔹 5. Feature Engineering
volume = x * y * z
dimension_ratio = (x + y) / (2 * z)

if carat < 0.5:
    carat_category = "Light"
elif carat <= 1.5:
    carat_category = "Medium"
else:
    carat_category = "Heavy"

depth_percentage = (z / ((x + y)/2)) * 100

#MODULE 1 — PRICE PREDICTION
#🔹 Prepare Input (One-Hot like training)
# Base numeric features
input_dict = {
    "carat": carat,
    "depth": depth,
    "table": table,
    "x": x,
    "y": y,
    "z": z,
    "volume": volume,
    "dimension_ratio": dimension_ratio,
    "depth_percentage": depth_percentage
}

input_df = pd.DataFrame([input_dict])

#Apply One-Hot Encoding
# Add categorical columns
input_df["cut"] = cut
input_df["color"] = color
input_df["clarity"] = clarity
input_df["carat_category"] = carat_category

# One-hot encode
input_df = pd.get_dummies(input_df)

#Align Columns
# Match training columns
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

#💥 THIS LINE SOLVES EVERYTHING
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

#Final Prediction Code

pred_log = price_model.predict(input_df)[0]
price_usd = np.expm1(pred_log)
price_inr = price_usd * 83


#MODULE 2 — CLUSTERING INPUT (SEPARATE)

cluster_input = {
    "carat": carat,
    "cut": cut,
    "color": color,
    "clarity": clarity,
    "depth": depth,
    "table": table,
    "x": x,
    "y": y,
    "z": z,
    "volume": volume,
    "dimension_ratio": dimension_ratio,
    "carat_category": carat_category,
    "depth_percentage": depth_percentage
}

cluster_input_df = pd.DataFrame([cluster_input])

#🔹 Encoding

cut_map = {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}
color_map = {"D":0,"E":1,"F":2,"G":3,"H":4,"I":5,"J":6}
clarity_map = {"I1":0,"SI2":1,"SI1":2,"VS2":3,"VS1":4,"VVS2":5,"VVS1":6,"IF":7}
carat_map = {"Light":0,"Medium":1,"Heavy":2}

cluster_input_df["cut"] = cluster_input_df["cut"].map(cut_map)
cluster_input_df["color"] = cluster_input_df["color"].map(color_map)
cluster_input_df["clarity"] = cluster_input_df["clarity"].map(clarity_map)
cluster_input_df["carat_category"] = cluster_input_df["carat_category"].map(carat_map)

#🔹 Buttons Layout

col1, col2 = st.columns(2)

#🎯 PRICE BUTTON

with col1:
    if st.button("Predict Price"):

        pred_log = price_model.predict(input_df)[0]
        price_usd = np.expm1(pred_log)
        price_inr = price_usd * 83

        st.success(f"💰 Predicted Price: ₹ {price_inr:,.2f}")

        #🎯 CLUSTER BUTTON

        cluster_names = {
    0: "Affordable Small Diamonds",
    1: "Mid-range Balanced Diamonds",
    2: "Premium Heavy Diamonds"
}

with col2:
    if st.button("Predict Cluster"):

        cluster_pred = cluster_model.predict(cluster_input_df)[0]

        cluster_names={
            0:"Budget segment",
            1:"Premium segment",
            2:"Luxury segment"
        }

        cluster_name = cluster_names.get(cluster_pred, "Unknown")

        st.info(f"📊 Cluster: {cluster_pred} — {cluster_name}")