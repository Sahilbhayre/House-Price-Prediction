import streamlit as st  #type:ignore
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIG

st.set_page_config(
    page_title="House Price Predictor - Sahil Bhayre",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Ames House Price Prediction")
st.write("Enter the house details below to estimate the sale price.")


# LOAD MODEL + FEATURE SCHEMA

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


@st.cache_resource
def load_features():
    return joblib.load("feature_columns.pkl")


model = load_model()
feature_columns = load_features()


# SIDEBAR - MODEL INFORMATION
with st.sidebar:

    st.title("🏠 Model Information")

    st.markdown("### 👨‍💻 Developed by")
    st.markdown("**Sahil Bhayre**")

    st.markdown("---")

    st.subheader("🤖 Model")

    st.write("**Algorithm:** Gradient Boosting Regressor")
    st.write("**Task:** Regression")
    st.write("**Target:** Sale Price")

    st.markdown("---")

    st.subheader("📊 Model Performance")

    st.metric("Test R² Score", "0.93")
    st.metric("Cross-Validation R²", "0.92")
    st.metric("MAE", "$13,344.71")
    st.metric("RMSE", "$22,462.86")

    st.markdown("---")

    st.subheader("📁 Dataset")

    st.write("**Dataset:** Ames Housing")
    st.write("**Total Houses:** 2,930")
    st.write("**Features Used:** 85")
    st.write("**Train/Test Split:** 80% / 20%")

    st.markdown("---")

    st.subheader("⚙️ Model Settings")

    st.write("**Estimators:** 500")
    st.write("**Learning Rate:** 0.05")
    st.write("**Max Depth:** 3")
    st.write("**Subsample:** 0.9")

    st.markdown("---")

    st.caption(
        "🏠 Ames House Price Prediction"
    )

    st.caption(
        "Developed by Sahil Bhayre"
    )
    
    
# USER INPUTS

st.header("🏡 House Information")

col1, col2, col3 = st.columns(3)

with col1:
    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5
    )

    overall_cond = st.slider(
        "Overall Condition",
        min_value=1,
        max_value=10,
        value=5
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=5000,
        value=1500
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

with col2:
    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=800
    )

    first_flr_sf = st.number_input(
        "1st Floor Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000
    )

    second_flr_sf = st.number_input(
        "2nd Floor Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=500
    )

    garage_cars = st.number_input(
        "Garage Capacity",
        min_value=0,
        max_value=5,
        value=2
    )

with col3:
    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=2000,
        value=500
    )

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2
    )

    half_bath = st.number_input(
        "Half Bathrooms",
        min_value=0,
        max_value=4,
        value=1
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=3
    )


# ============================================================
# HOUSE QUALITY & LOCATION
# ============================================================

st.header("🏘️ House Quality & Location")

col1, col2, col3 = st.columns(3)


# ============================================================
# QUALITY MAPPING
# ============================================================

quality_options = {
    "Excellent": "Ex",
    "Good": "Gd",
    "Typical / Average": "TA",
    "Fair": "Fa",
    "Poor": "Po"
}


# ============================================================
# BASEMENT QUALITY
# ============================================================

basement_quality_options = {
    "Excellent": "Ex",
    "Good": "Gd",
    "Typical / Average": "TA",
    "Fair": "Fa",
    "Poor": "Po",
    "No Basement": "None"
}


# ============================================================
# GARAGE FINISH
# ============================================================

garage_finish_options = {
    "Finished": "Fin",
    "Roughly Finished": "RFn",
    "Unfinished": "Unf",
    "No Garage": "None"
}


# ============================================================
# FOUNDATION
# ============================================================

foundation_options = {
    "Poured Concrete": "PConc",
    "Concrete Block": "CBlock",
    "Brick & Tile": "BrkTil",
    "Slab": "Slab",
    "Stone": "Stone",
    "Wood": "Wood"
}


# ============================================================
# CENTRAL AIR
# ============================================================

central_air_options = {
    "Yes": "Y",
    "No": "N"
}


# ============================================================
# HOUSE STYLE
# ============================================================

house_style_options = {
    "1 Story": "1Story",
    "2 Story": "2Story",
    "1.5 Story - Finished": "1.5Fin",
    "1.5 Story - Unfinished": "1.5Unf",
    "Split Foyer": "SFoyer",
    "Split Level": "SLvl",
    "2.5 Story - Unfinished": "2.5Unf",
    "2.5 Story - Finished": "2.5Fin"
}


# ============================================================
# NEIGHBORHOOD
# ============================================================

neighborhood_options = {
    "North Ames": "NAmes",
    "College Creek": "CollgCr",
    "Old Town": "OldTown",
    "Edwards": "Edwards",
    "Somerset": "Somerst",
    "Northridge Heights": "NridgHt",
    "Gilbert": "Gilbert",
    "Sawyer": "Sawyer",
    "Northwest Ames": "NWAmes",
    "Sawyer West": "SawyerW",
    "Northridge": "NoRidge",
    "Meadow Village": "MeadowV",
    "Brookside": "BrkSide",
    "Clear Creek": "ClearCr",
    "Crawford": "Crawfor",
    "Timberland": "Timber",
    "Iowa DOT": "IDOTRR",
    "Mitchell": "Mitchel",
    "Northpark Villa": "NPkVill",
    "Briardale": "BrDale",
    "Bluestem": "Blueste",
    "Veenker": "Veenker",
    "Stone Brook": "StoneBr",
    "Bloomington Heights": "Blmngtn",
    "Landmark": "Landmrk",
    "Green Hills": "Greens",
    "Gilbert North": "Gilbert",
    "South & West of Iowa State": "SWISU"
}


# COLUMN 1
with col1:

    neighborhood_display = st.selectbox(
        "Neighborhood",
        list(neighborhood_options.keys()),
        help="General location of the property."
    )

    neighborhood = neighborhood_options[neighborhood_display]


    kitchen_quality_display = st.selectbox(
        "Kitchen Quality",
        list(quality_options.keys()),
        help="Overall quality of the kitchen."
    )

    kitchen_qual = quality_options[kitchen_quality_display]


    exterior_quality_display = st.selectbox(
        "Exterior Quality",
        list(quality_options.keys()),
        help="Overall quality of the exterior materials."
    )

    exter_qual = quality_options[exterior_quality_display]

# COLUMN 2
with col2:

    basement_quality_display = st.selectbox(
        "Basement Quality",
        list(basement_quality_options.keys()),
        help="Quality of the basement. Select 'No Basement' if there is no basement."
    )

    bsmt_qual = basement_quality_options[basement_quality_display]


    garage_finish_display = st.selectbox(
        "Garage Finish",
        list(garage_finish_options.keys()),
        help="Finish level of the garage."
    )

    garage_finish = garage_finish_options[garage_finish_display]


    heating_quality_display = st.selectbox(
        "Heating Quality",
        list(quality_options.keys()),
        help="Quality and condition of the heating system."
    )

    heating_qc = quality_options[heating_quality_display]

# COLUMN 3
with col3:

    central_air_display = st.selectbox(
        "Central Air Conditioning",
        list(central_air_options.keys()),
        help="Does the house have central air conditioning?"
    )

    central_air = central_air_options[central_air_display]


    house_style_display = st.selectbox(
        "House Style",
        list(house_style_options.keys()),
        help="General architectural style of the house."
    )

    house_style = house_style_options[house_style_display]


    foundation_display = st.selectbox(
        "Foundation Type",
        list(foundation_options.keys()),
        help="Type of foundation used for the house."
    )

    foundation = foundation_options[foundation_display]


# ADDITIONAL INPUTS

st.header("🚗 Additional Information")

col1, col2, col3 = st.columns(3)

with col1:

    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=500,
        max_value=100000,
        value=8000
    )

    fireplaces = st.number_input(
        "Fireplaces",
        min_value=0,
        max_value=5,
        value=1
    )

with col2:

    year_remod = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        max_value=20,
        value=7
    )

with col3:

    parking = st.number_input(
        "Garage Area",
        min_value=0,
        max_value=2000,
        value=500
    )

    sale_year = st.number_input(
        "Sale Year",
        min_value=2006,
        max_value=2026,
        value=2007
    )

# CREATE DEFAULT ROW
def create_input_dataframe():

    # Start with all required columns
    input_data = {}

    for column in feature_columns:

        # Numerical columns → 0
        input_data[column] = 0

    df_input = pd.DataFrame([input_data])


    # Main numerical inputs

    values = {

        "Overall Qual": overall_qual,
        "Overall Cond": overall_cond,
        "Gr Liv Area": gr_liv_area,

        "Year Built": year_built,
        "Year Remod/Add": year_remod,

        "Total Bsmt SF": total_bsmt_sf,
        "1st Flr SF": first_flr_sf,
        "2nd Flr SF": second_flr_sf,

        "Garage Cars": garage_cars,
        "Garage Area": garage_area,

        "Full Bath": full_bath,
        "Half Bath": half_bath,

        "Bedroom AbvGr": bedrooms,

        "Lot Area": lot_area,

        "Fireplaces": fireplaces,

        "TotRms AbvGrd": total_rooms,

        "Yr Sold": sale_year,

        "Central Air": central_air,

        "Neighborhood": neighborhood,

        "Kitchen Qual": kitchen_qual,

        "Exter Qual": exter_qual,

        "Bsmt Qual": bsmt_qual,

        "Garage Finish": garage_finish,

        "Heating QC": heating_qc,

        "House Style": house_style,

        "Foundation": foundation
    }


    # Put values only if the column exists

    for column, value in values.items():

        if column in df_input.columns:
            df_input[column] = value


    # FEATURE ENGINEERING

    if "TotalSF" in df_input.columns:
        df_input["TotalSF"] = (
            df_input["Total Bsmt SF"]
            + df_input["1st Flr SF"]
            + df_input["2nd Flr SF"]
        )

    if "TotalBathrooms" in df_input.columns:
        df_input["TotalBathrooms"] = (
            df_input["Full Bath"]
            + 0.5 * df_input["Half Bath"]
        )

    if "TotalPorchSF" in df_input.columns:

        porch_columns = [
            "Wood Deck SF",
            "Open Porch SF",
            "Enclosed Porch",
            "3Ssn Porch",
            "Screen Porch"
        ]

        df_input["TotalPorchSF"] = sum(
            df_input[column]
            for column in porch_columns
            if column in df_input.columns
        )

    if "HouseAge" in df_input.columns:

        df_input["HouseAge"] = (
            df_input["Yr Sold"]
            - df_input["Year Built"]
        )

        df_input["HouseAge"] = df_input["HouseAge"].clip(lower=0)

    if "RemodAge" in df_input.columns:

        df_input["RemodAge"] = (
            df_input["Yr Sold"]
            - df_input["Year Remod/Add"]
        )

        df_input["RemodAge"] = df_input["RemodAge"].clip(lower=0)

    if "TotalBsmtFinishedSF" in df_input.columns:

        df_input["TotalBsmtFinishedSF"] = (
            df_input["BsmtFin SF 1"]
            + df_input["BsmtFin SF 2"]
        )

    if "TotalHouseArea" in df_input.columns:

        df_input["TotalHouseArea"] = (
            df_input["Gr Liv Area"]
            + df_input["Total Bsmt SF"]
        )

    if "TotalOutdoorSF" in df_input.columns:

        df_input["TotalOutdoorSF"] = (
            df_input["TotalPorchSF"]
            + df_input["Garage Area"]
        )

    if "TotalRooms" in df_input.columns:

        df_input["TotalRooms"] = (
            df_input["TotRms AbvGrd"]
            + df_input["Bedroom AbvGr"]
        )

    if "Qual_GrLivArea" in df_input.columns:

        df_input["Qual_GrLivArea"] = (
            df_input["Overall Qual"]
            * df_input["Gr Liv Area"]
        )


    # ENSURE EXACT COLUMN ORDER

    df_input = df_input[feature_columns]

    return df_input


# PREDICTION

st.divider()

if st.button("🏠 Predict House Price", type="primary"):

    try:

        input_df = create_input_dataframe()

        prediction = model.predict(input_df)[0]

        st.success(
            f"🏠 Estimated House Price: ₹{prediction:,.0f}"
        )

        st.info(
            "This prediction is generated by the trained Gradient Boosting model."
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)