import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import requests
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder
from haversine import haversine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. LOADING DATA SET ---
print("--- Section 1: Loaded Dataset ---")
# Path to the dataset file in the current project folder
# Make sure this matches the actual file name on disk
DATASET_FILE = "fakeuserdetection_ecommerce_data (1).csv"

# Read dataset
df = pd.read_csv(DATASET_FILE)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Optional: Convert timestamp column if available
if 'LoginTimestamp' in df.columns:
    df['LoginTimestamp'] = pd.to_datetime(df['LoginTimestamp'], errors='coerce')

print("--- Data Head ---")
print(df.head())




# --- 2. FEATURE ENGINEERING ---
print("--- Section 2: Feature Engineering ---")

# IP Clustering
ip_counts = df.groupby('IPAddress')['UserID'].nunique().reset_index()
ip_counts.columns = ['IPAddress', 'IPUserCount']
df = pd.merge(df, ip_counts, on='IPAddress', how='left')

# Geo-inconsistency
df = df.sort_values(by=['UserID', 'LoginTimestamp'])
df['TimeDiff'] = df.groupby('UserID')['LoginTimestamp'].diff().dt.total_seconds().div(3600)
df['PrevLat'] = df.groupby('UserID')['Latitude'].shift()
df['PrevLon'] = df.groupby('UserID')['Longitude'].shift()

def calculate_haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1):
        return 0
        return haversine((lat1, lon1), (lat2, lon2))

df['Distance'] = df.apply(lambda row: calculate_haversine(row['PrevLat'], row['PrevLon'], row['Latitude'], row['Longitude']), axis=1)
df['Speed'] = df['Distance'].div(df['TimeDiff']).fillna(0)

# Behavioral Profiling
df['PurchaseToBrowseRatio'] = df['Purchases'].div(df['BrowsingEvents']).fillna(0)

# Device-switch anomalies
device_counts = df.groupby('UserID')['DeviceType'].nunique().reset_index()
device_counts.columns = ['UserID', 'DeviceCount']
df = pd.merge(df, device_counts, on='UserID', how='left')

# Label Encoding
for col in ['DeviceType', 'Browser']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("Feature engineering complete. New features added: IPUserCount, Speed, PurchaseToBrowseRatio, DeviceCount")
print(df[['UserID', 'IPUserCount', 'Speed', 'PurchaseToBrowseRatio', 'DeviceCount']].head())

# --- 3. MACHINE LEARNING PIPELINE ---
print("--- Section 3: Machine Learning Pipeline ---")

features = [
    'IPUserCount', 'SessionDuration', 'BrowsingEvents', 'Purchases', 
    'AverageOrderValue', 'CartAbandonmentRate', 'Speed', 
    'PurchaseToBrowseRatio', 'DeviceCount', 'DeviceType', 'Browser'
]
target = 'IsFake'

X = df[features].fillna(0)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print(f'--- {name} ---')
    a=accuracy_score(y_test, y_pred)
    print(a)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred):.4f}')
    print(f'Recall: {recall_score(y_test, y_pred):.4f}')
    print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')
    print(f'ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fake'], yticklabels=['Legit', 'Fake'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_filename = f"confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(cm_filename)
    print(f"Saved {cm_filename}")
    plt.close()

# Anomaly Detection
iso_forest = IsolationForest(contamination='auto', random_state=42)
df['AnomalyScore_ISO'] = iso_forest.fit_predict(X.fillna(0))
# -1 is anomaly, 1 is normal. We map it to 1 for anomaly, 0 for normal.
df['IsAnomaly_ISO'] = df['AnomalyScore_ISO'].apply(lambda x: 1 if x == -1 else 0)

print('--- Isolation Forest Anomaly Detection ---')
print(f"Detected {df['IsAnomaly_ISO'].sum()} anomalies out of {len(df)} records.")

# --- 4. FAKE USER DETECTION DASHBOARD (Script Version) ---
print("--- Section 4: Fake User Detection Dashboard ---")

# Rich Table for Suspicious Users
suspicious_users = df[df['IsAnomaly_ISO'] == 1].copy()
suspicious_users['Reason'] = ''
suspicious_users.loc[suspicious_users['IPUserCount'] > 2, 'Reason'] += 'IP Cluster; '
suspicious_users.loc[suspicious_users['Speed'] > 1000, 'Reason'] += 'Impossible Travel; '
suspicious_users.loc[suspicious_users['Purchases'] > 10, 'Reason'] += 'Transaction Spike; '
suspicious_users.loc[suspicious_users['DeviceCount'] > 2, 'Reason'] += 'Device Switch; '

table = Table(title="Top 10 Suspicious User Activities (Detected by Isolation Forest)")
table.add_column("Username", style="cyan")
table.add_column("IP Address", style="magenta")
table.add_column("Detected Reason(s)", style="green")

for _, row in suspicious_users.head(10).iterrows():
    table.add_row(row['Username'], row['IPAddress'], row['Reason'])

console = Console()
console.print(table)

# Generate Interactive Charts as HTML files
print("Generating interactive charts as HTML files...")

# Geo-map of suspicious accounts
color_sequence = px.colors.qualitative.Bold
fig_map = px.scatter_geo(
    suspicious_users, 
    lat='Latitude', lon='Longitude', 
    color='Reason',
    hover_name='Username', size='AverageOrderValue',
    title='Suspicious Accounts Geo-Map (Anomalies)',
    projection="natural earth",
    color_discrete_sequence=color_sequence
)
fig_map.update_layout(
    paper_bgcolor="#f5f9ff",
    plot_bgcolor="#eef3fb",
    title_font_color="#0f172a",
    legend_title_font_color="#0f172a",
    legend_font_color="#0f172a"
)
map_filename = "dashboard_geo_map.html"
fig_map.write_html(map_filename)
print(f"Saved {map_filename}")

# Anomaly scores distribution
fig_hist = px.histogram(
    df, 
    x='IsAnomaly_ISO', 
    color='IsFake', 
    barmode='group', 
    title='Anomaly Detection vs. True Labels',
    color_discrete_sequence=["#2563eb", "#0ea5e9"]
)
fig_hist.update_layout(
    paper_bgcolor="#f5f9ff",
    plot_bgcolor="#eef3fb",
    title_font_color="#0f172a",
    legend_title_font_color="#0f172a",
    legend_font_color="#0f172a",
    bargap=0.2
)
hist_filename = "dashboard_histogram.html"
fig_hist.write_html(hist_filename)
print(f"Saved {hist_filename}")

# Timeline of user activity
fig_timeline = px.scatter(
    df.sample(n=min(2000, len(df))), # Sample to keep timeline readable
    x='LoginTimestamp', y='Username', color='IsFake', 
    title='User Activity Timeline (Sample)',
    labels={'Username': 'Users'},
    color_discrete_sequence=["#0ea5e9", "#2563eb"]
)
fig_timeline.update_traces(marker=dict(size=6, opacity=0.75))
fig_timeline.update_layout(
    paper_bgcolor="#f5f9ff",
    plot_bgcolor="#eef3fb",
    title_font_color="#0f172a",
    legend_title_font_color="#0f172a",
    legend_font_color="#0f172a",
    yaxis_title="Users",
    xaxis_title="Login Timestamp"
)
timeline_filename = "dashboard_timeline.html"
fig_timeline.write_html(timeline_filename)
print(f"Saved {timeline_filename}")

print("--- Project Execution Complete ---")
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fake User Detection Lab", layout="wide", page_icon="🛡️")

# Custom CSS for an "Attractive Interface"
st.markdown("""
    <style>
    /* Page */
    .main { 
        background: linear-gradient(180deg, #f5f9ff 0%, #eef3fb 40%, #fdfdff 100%); 
        color: #0f172a;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .sidebar-content {
        padding: 1rem 0.75rem;
    }
    /* Headings */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Metrics / cards */
    .stMetric { 
        background: #ffffff; 
        padding: 15px; 
        border-radius: 12px; 
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08); 
        border: 1px solid #e2e8f0;
    }
    /* Tabs */
    button[data-baseweb="tab"] { 
        background: #e2e8f0; 
        border-radius: 12px 12px 0 0; 
        color: #0f172a;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        margin-right: 0.25rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #0ea5e9;
        color: #ffffff;
    }
    /* Tables */
    table {
        border-radius: 8px;
        overflow: hidden;
    }
    thead tr th {
        background: #0ea5e9;
        color: #ffffff;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(37, 99, 235, 0.32);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Fake User Analysis & Detection Dashboard")
st.write("Comprehensive overview of synthetic account detection and model performance.")

# --- SIDEBAR: TOP 10 FAKE USERS ---
st.sidebar.header("🚨 Top 10 High-Risk Users")
try:
    # Attempting to load the dataset
    df = pd.read_csv(DATASET_FILE)
    
    # Filter for fake users (assuming column 'IsFake' exists)
    # If using Isolation Forest, change condition to df['IsAnomaly_ISO'] == -1
    fake_users = df[df['IsFake'] == 1].head(10)
    
    # Display simplified table in sidebar
    st.sidebar.dataframe(fake_users[['UserID', 'Username', 'IPAddress', 'Country']], hide_index=True)
    st.sidebar.info("These users showed the highest statistical similarity to known bot patterns.")
except:
    st.sidebar.warning(f"Dataset not found. Please ensure '{DATASET_FILE}' is in this folder.")

# --- MAIN INTERFACE TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Interactive Dashboards", "🧠 ML Performance", "📋 Dataset Explorer"])

with tab1:
    st.subheader("Visual Anomaly Detection")
    
    # Displaying HTML Dashboards using Components
    dashboard_files = {
        "Global Activity Map": "dashboard_geo_map.html",
        "Login Timeline": "dashboard_timeline.html",
        "Feature Distribution": "dashboard_histogram.html"
    }
    
    selected_db = st.selectbox("Choose a Dashboard to View:", list(dashboard_files.keys()))
    
    file_path = dashboard_files[selected_db]
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
    else:
        st.error(f"File {file_path} not found.")

with tab2:
    st.subheader("Model Confusion Matrices")
    st.write("Comparing the accuracy of Logistic Regression, Random Forest, and XGBoost.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("confusion_matrix_Logistic_Regression.png", caption="Logistic Regression")
    with col2:
        st.image("confusion_matrix_Random_Forest.png", caption="Random Forest")
    with col3:
        st.image("confusion_matrix_XGBoost.png", caption="XGBoost")

with tab3:
    st.subheader("Raw Data Inspection")
    if 'df' in locals():
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Upload the CSV file to view data.")

    
    

    
