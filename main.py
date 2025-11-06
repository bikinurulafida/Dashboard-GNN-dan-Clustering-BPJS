# main.py
import streamlit as st
from load_data import load_data_visualization, load_data_gnn
from helper import plot_table, plot_diagnosis_distribution, compute_kmeans, visualize_graph
import numpy as np
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="Dashboard Analisis GNN & Clustering Peserta Tuberculosis BPJS Kesehatan Jakarta",
    page_icon="🩺",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown("""
    <h1 style='text-align:center; color:#145A32'>
        Dashboard Analisis GNN & Clustering Peserta Tuberculosis BPJS Kesehatan Jakarta
    </h1>
""", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #145A32;'>", unsafe_allow_html=True)

# ================== LOAD DATA ==================
try:
    df_vis = load_data_visualization()
    node_peserta, node_faskes, edges = load_data_gnn()
except FileNotFoundError as e:
    st.error("❌ Gagal memuat data. Pastikan file CSV sudah ada di folder `Data/` di repository kamu.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# Pastikan tipe data sinkron
for df in [df_vis, node_peserta, edges, node_faskes]:
    for col in ['peserta_id', 'faskes_id']:
        if col in df.columns:
            df[col] = df[col].astype(str)

# ================== K-MEANS ==================
embedding_cols = [col for col in ['Usia', 'Jarak_km', 'Biaya_Klaim'] if col in node_peserta.columns]
embeddings = node_peserta[embedding_cols].fillna(0).values if embedding_cols else np.random.rand(len(node_peserta), 2)

node_peserta_clustered, edges_clustered, mode_per_faskes = compute_kmeans(embeddings, node_peserta, edges)
if 'k-means_cluster' in node_peserta_clustered.columns:
    node_peserta_clustered['k-means_cluster'] = node_peserta_clustered['k-means_cluster'].astype(int)

# ================== DETEKSI ANOMALI ==================
def detect_anomalies(node_peserta, embedding_cols, threshold=3.0):
    if 'k-means_cluster' not in node_peserta.columns:
        raise ValueError("Kolom 'k-means_cluster' tidak ditemukan")
    centroids = node_peserta.groupby('k-means_cluster')[embedding_cols].mean()
    distances = [
        np.linalg.norm(row[embedding_cols].values - centroids.loc[row['k-means_cluster']].values)
        for _, row in node_peserta.iterrows()
    ]
    node_peserta['distance_to_centroid'] = distances
    z_scores = (node_peserta['distance_to_centroid'] - np.mean(distances)) / np.std(distances)
    node_peserta['anomali'] = z_scores > threshold
    return node_peserta

if embedding_cols:
    node_peserta_clustered = detect_anomalies(node_peserta_clustered, embedding_cols)

# ================== FILTER DROPDOWN ==================
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    cluster_options = ["All"] + sorted(node_peserta_clustered['k-means_cluster'].unique().tolist())
    selected_cluster = st.selectbox("Cluster", cluster_options)
with col2:
    gender_options = ["All"] + sorted(df_vis['jenis_kelamin'].dropna().unique().tolist()) if 'jenis_kelamin' in df_vis.columns else ["All"]
    selected_gender = st.selectbox("Jenis Kelamin", gender_options)
with col3:
    segment_options = ["All"] + sorted(df_vis['Segmentasi'].dropna().unique().tolist()) if 'Segmentasi' in df_vis.columns else ["All"]
    selected_segment = st.selectbox("Segmentasi", segment_options)
with col4:
    provinsi_options = ["All"] + sorted(node_faskes['Provinsi'].dropna().unique().tolist()) if 'Provinsi' in node_faskes.columns else ["All"]
    selected_provinsi = st.selectbox("Provinsi", provinsi_options)
with col5:
    anomali_options = ["All", "Normal", "Anomali"]
    selected_anomali = st.selectbox("Status Anomali", anomali_options)

# ================== APPLY FILTER ==================
df_filtered = df_vis.copy()
node_peserta_filtered = node_peserta_clustered.copy()
edges_filtered = edges_clustered.copy()

if selected_gender != "All" and 'jenis_kelamin' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['jenis_kelamin'] == selected_gender]

if selected_segment != "All" and 'Segmentasi' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Segmentasi'] == selected_segment]

if selected_cluster != "All":
    node_peserta_filtered = node_peserta_filtered[node_peserta_filtered['k-means_cluster'] == int(selected_cluster)]
    df_filtered = df_filtered[df_filtered['peserta_id'].isin(node_peserta_filtered['peserta_id'])]

if selected_provinsi != "All" and 'Provinsi' in node_faskes.columns:
    faskes_in_prov = node_faskes[node_faskes['Provinsi'] == selected_provinsi]['faskes_id'].tolist()
    edges_filtered = edges_filtered[edges_filtered['faskes_id'].isin(faskes_in_prov)]
    peserta_in_edges = edges_filtered['peserta_id'].unique().tolist()
    node_peserta_filtered = node_peserta_filtered[node_peserta_filtered['peserta_id'].isin(peserta_in_edges)]
    df_filtered = df_filtered[df_filtered['peserta_id'].isin(peserta_in_edges)]

if selected_anomali != "All":
    node_peserta_filtered = node_peserta_filtered[node_peserta_filtered['anomali'] == (selected_anomali == "Anomali")]
    df_filtered = df_filtered[df_filtered['peserta_id'].isin(node_peserta_filtered['peserta_id'])]
    edges_filtered = edges_filtered[edges_filtered['peserta_id'].isin(node_peserta_filtered['peserta_id'])]

# ================== SUMMARY CARDS ==================
total_peserta = len(df_filtered)
usia_mean = round(df_filtered['Usia'].astype(float).mean(), 1) if 'Usia' in df_filtered.columns and total_peserta > 0 else 0
merged = edges_filtered.merge(node_peserta_filtered[['peserta_id']], on='peserta_id', how='inner')
merged['Biaya_Klaim'] = pd.to_numeric(merged.get('Biaya_Klaim', 0), errors='coerce').fillna(0)
rata_klaim = round(merged['Biaya_Klaim'].sum() / total_peserta, 2) if total_peserta > 0 else 0
jumlah_cluster = len(node_peserta_filtered['k-means_cluster'].unique()) if total_peserta > 0 else 0

c1, c2, c3, c4 = st.columns(4)
card_style = """
background-color:#D5F5E3; 
padding:18px; 
border-radius:12px; 
text-align:center; 
border:1px solid #82E0AA
"""
with c1:
    st.markdown(f"<div style='{card_style}'><h4 style='color:#145A32'>Total Peserta</h4><h2>{total_peserta}</h2></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='{card_style}'><h4 style='color:#145A32'>Rata-rata Usia</h4><h2>{usia_mean}</h2></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div style='{card_style}'><h4 style='color:#145A32'>Rata-rata Klaim (FKRTL)</h4><h2>{rata_klaim}</h2></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div style='{card_style}'><h4 style='color:#145A32'>Jumlah Cluster Terfilter</h4><h2>{jumlah_cluster}</h2></div>", unsafe_allow_html=True)

# ================== VISUALISASI ==================
st.subheader("📋 Tabel Peserta")
st.dataframe(df_filtered, use_container_width=True)

# ---------- Karakteristik Peserta ----------
col1, col2 = st.columns(2)
with col1:
    if 'jenis_kelamin' in df_filtered.columns and total_peserta > 0:
        gender_count = df_filtered['jenis_kelamin'].value_counts().reset_index()
        gender_count.columns = ['Jenis Kelamin', 'Jumlah']
        fig_gender = px.pie(gender_count, names='Jenis Kelamin', values='Jumlah', hole=0.4,
                            title="Distribusi Jenis Kelamin Peserta",
                            color_discrete_sequence=px.colors.sequential.Teal)
        fig_gender.update_traces(textinfo='percent+label', textfont_size=16)
        fig_gender.update_layout(title_font_color='#145A32')
        st.plotly_chart(fig_gender, use_container_width=True)
with col2:
    if 'Segmentasi' in df_filtered.columns and total_peserta > 0:
        seg_count = df_filtered['Segmentasi'].value_counts().reset_index()
        seg_count.columns = ['Segmentasi', 'Jumlah']
        fig_seg = px.pie(seg_count, names='Segmentasi', values='Jumlah', hole=0.4,
                         title="Distribusi Segmentasi Peserta",
                         color_discrete_sequence=px.colors.sequential.Teal)
        fig_seg.update_traces(textinfo='percent+label', textfont_size=16)
        fig_seg.update_layout(title_font_color='#145A32')
        st.plotly_chart(fig_seg, use_container_width=True)

# ---------- Distribusi Usia ----------
st.subheader("📊 Distribusi Usia Peserta")
if 'Usia' in df_filtered.columns and total_peserta > 0:
    fig_age = px.histogram(df_filtered, x='Usia', nbins=20, color_discrete_sequence=['#1ABC9C'],
                           title="Distribusi Usia Peserta")
    fig_age.update_layout(title_font_color='#145A32')
    st.plotly_chart(fig_age, use_container_width=True)

# ---------- Distribusi Diagnosis Primer ----------
st.subheader("🩺 Distribusi Diagnosis Primer")
if total_peserta > 0:
    plot_diagnosis_distribution(df_filtered)

# ---------- Distribusi Biaya Klaim ----------
st.subheader("💰 Distribusi Biaya Klaim")
if total_peserta > 0 and any(c in df_filtered.columns for c in ['Biaya_Klaim', 'rata-rata_klaim_biaya']):
    col_cost = 'rata-rata_klaim_biaya' if 'rata-rata_klaim_biaya' in df_filtered.columns else 'Biaya_Klaim'
    fig_cost = px.histogram(df_filtered, x=col_cost, nbins=50, color_discrete_sequence=['#1ABC9C'],
                            title="Distribusi Biaya Klaim")
    fig_cost.update_layout(title_font_color='#145A32')
    st.plotly_chart(fig_cost, use_container_width=True)

# ---------- Distribusi Usia per Cluster ----------
st.subheader("📦 Distribusi Usia per Cluster")
if total_peserta > 0 and {'Usia', 'k-means_cluster'}.issubset(node_peserta_filtered.columns):
    fig_box = px.box(node_peserta_filtered, x='k-means_cluster', y='Usia', color='k-means_cluster',
                     color_discrete_sequence=px.colors.qualitative.Vivid,
                     labels={'k-means_cluster': 'Cluster', 'Usia': 'Usia'})
    st.plotly_chart(fig_box, use_container_width=True)

# ---------- Proporsi Cluster ----------
st.subheader("📈 Proporsi Peserta per Cluster")
if total_peserta > 0:
    cluster_count = node_peserta_filtered['k-means_cluster'].value_counts().reset_index()
    cluster_count.columns = ['Cluster', 'Jumlah']
    cluster_count = cluster_count.sort_values('Cluster')
    fig_cluster = px.bar(cluster_count, x='Cluster', y='Jumlah', text='Jumlah', color='Cluster',
                         color_discrete_sequence=px.colors.qualitative.Vivid,
                         title="Proporsi Peserta per Cluster")
    fig_cluster.update_layout(title_font_color='#145A32')
    st.plotly_chart(fig_cluster, use_container_width=True)

# ---------- Visualisasi Peserta-Faskes ----------
st.subheader("🌐 Visualisasi Peserta-Faskes (Graph)")
if total_peserta > 0:
    visualize_graph(node_peserta_filtered, edges_filtered)

# ---------- Tabel Indikasi Anomali ----------
st.subheader("🚨 Peserta Indikasi Anomali")
anomali_df = node_peserta_filtered[node_peserta_filtered['anomali'] == True]
if len(anomali_df) > 0:
    st.dataframe(anomali_df, use_container_width=True)
else:
    st.info("Tidak ada peserta yang terindikasi anomali.")
