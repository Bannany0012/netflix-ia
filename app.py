import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Loading & Cleaning
@st.cache_data # باش ما يبقاش يعاود التحميل كل مرة
def load_data():
    df = pd.read_csv('netflix.csv')
    df['country'] = df['country'].fillna('United States')
    df['rating'] = df['rating'].fillna('TV-MA')
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip())
    # تحويل المدة لأرقام (Minutes)
    df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float).fillna(0)
    return df

df = load_data()

# 2. Machine Learning (Training on the fly)
le_rating = LabelEncoder()
le_country = LabelEncoder()
le_type = LabelEncoder()

df['rating_enc'] = le_rating.fit_transform(df['rating'])
df['country_enc'] = le_country.fit_transform(df['country'])
df['type_enc'] = le_type.fit_transform(df['type'])

X = df[['release_year', 'rating_enc', 'country_enc', 'duration_num']]
y = df['type_enc']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 3. Streamlit Interface
st.set_page_config(page_title="Netflix AI", layout="wide")
st.title("🎬 Netflix Smart Analytics & Predictor")

# --- Tabs لترتيب المشروع ---
tab1, tab2, tab3 = st.tabs(["📊 Data Review", "🤖 ML Prediction", "🆕 New Content"])

with tab1:
    st.header("Analyse des données (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df, names='type', title="Films vs Séries")
        st.plotly_chart(fig1)
    with col2:
        top_countries = df['country'].value_counts().head(10)
        st.bar_chart(top_countries)

with tab2:
    st.header("Prédire le Type de Contenu")
    year = st.slider("Année", 1950, 2025, 2021)
    rating_in = st.selectbox("Rating", le_rating.classes_)
    country_in = st.selectbox("Country", le_country.classes_)
    duration_in = st.number_input("Durée (min ou saisons)", value=90)
    
    if st.button("Lancer l'IA"):
        r_enc = le_rating.transform([rating_in])[0]
        c_enc = le_country.transform([country_in])[0]
        pred = model.predict([[year, r_enc, c_enc, duration_in]])
        prob = model.predict_proba([[year, r_enc, c_enc, duration_in]])
        
        st.success(f"Résultat : **{le_type.inverse_transform(pred)[0]}**")
        st.info(f"Confiance : {np.max(prob)*100:.2f}%")

with tab3:
    st.header("Derniers ajouts sur Netflix")
    nouveautes = df.sort_values(by='date_added', ascending=False).head(15)
    st.dataframe(nouveautes[['title', 'type', 'release_year', 'date_added']])