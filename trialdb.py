# app.py
# Streamlit Dashboard (single-page) - Interaktif (Plotly + Matplotlib)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
st.set_page_config(page_title="Dashboard Analisis GNN & Cluster Peserta TB BPJS Jakarta",
                   page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

# ---------- STYLE ----------
st.markdown("""
    <style>
    .block-container { padding: 1rem 2rem; }
    .title {font-size:1.6rem; font-weight:800; color:#0b66c3; text-align:center; margin-bottom:0.2rem;}
    .subtitle {color:#6b7280; text-align:center; margin-top:0; margin-bottom:1.0rem;}
    .card {background:#f0fdfa; padding:0.8rem; border-radius:10px; box-shadow:0 1px 6px rgba(11,102,195,0.06); text-align:center;}
    .section {margin-top:1rem; margin-bottom:0.6rem; font-weight:700; color:#117a65;}
    .small-muted {font-size:0.85rem; color:#6b7280;}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# (Gunakan fungsi load_and_process_data yang sudah kamu pakai)
# -------------------------
@st.cache_data(show_spinner=False)
def load_and_process_data(base_path=r"D:\Intern at BPJS\Projek Dashboard\Data"):
    try:
        # --- (kode load & cleaning sama seperti yang kamu kirim) ---
        tb_fkrtl = pd.read_stata(os.path.join(base_path, "TB2023_fkrtl.dta"))
        tb_peserta = pd.read_stata(os.path.join(base_path, "TB2023_kepesertaan.dta"))
        tb_diagnosis = pd.read_stata(os.path.join(base_path, "TB2023_fkrtldxsekunder.dta"))

        tb_fkrtl_clean = tb_fkrtl[
            (tb_fkrtl['FKL09'] != 'Missing') &
            (tb_fkrtl['FKL11'] != "Missing") &
            (tb_fkrtl['FKL14'] != "Tidak Tahu") &
            (tb_fkrtl['FKL21'] != "Missing") &
            (tb_fkrtl['FKL27'] != "MISSING") &
            (tb_fkrtl['FKL29'] != "MISSING")
        ].copy()
        tb_fkrtl_clean['FKL48'] = pd.to_numeric(tb_fkrtl_clean['FKL48'], errors='coerce')

        total_klaim = tb_fkrtl_clean.groupby('PSTV01')['FKL48'].sum().reset_index()
        jumlah_klaim = tb_fkrtl_clean.groupby('PSTV01')['FKL48'].count().reset_index()
        jumlah_klaim.rename(columns={'FKL48': 'jumlah_klaim_fkrtl'}, inplace=True)
        total_klaim.rename(columns={'FKL48': 'total_klaim_fkrtl'}, inplace=True)

        tb_peserta = tb_peserta.merge(jumlah_klaim, on='PSTV01', how='left')
        tb_peserta = tb_peserta.merge(total_klaim, on='PSTV01', how='left')
        tb_peserta['jumlah_klaim_fkrtl'].fillna(0, inplace=True)
        tb_peserta['total_klaim_fkrtl'].fillna(0, inplace=True)

        tb_peserta_clean = tb_peserta[
            (tb_peserta['PSTV06'] != "Tidak terdefinisi") &
            (tb_peserta['PSTV07'] != "MISSING") &
            (tb_peserta['PSTV08'] != "MISSING") &
            (tb_peserta['PSTV09'] != "Tidak terdefinisi") &
            (tb_peserta['PSTV09_NEW'] != "Tidak terdefinisi") &
            (tb_peserta['PSTV11'] != "Missing") &
            (tb_peserta['PSTV12'] != "Missing") &
            (tb_peserta['PSTV13'] != "Tidak terdefinisi") &
            (tb_peserta['PSTV17'] != "Missing")
        ].copy()

        tb_fkrtl_plus = pd.merge(tb_fkrtl_clean, tb_diagnosis, on='FKL02', how='inner')
        tb_fkrtl_plus_clean = tb_fkrtl_plus.drop_duplicates(subset=['FKL02'])

        id1 = set(tb_peserta_clean['PSTV01'])
        id2 = set(tb_fkrtl_plus_clean['PSTV01'])

        tb_fkrtl_baru = tb_fkrtl_plus_clean[tb_fkrtl_plus_clean['PSTV01'].isin(id1)].copy()
        tb_kepesertaan_baru = tb_peserta_clean[tb_peserta_clean['PSTV01'].isin(id2)].copy()

        tb_dki_peserta = tb_kepesertaan_baru[
            (tb_kepesertaan_baru['PSTV13'] == 'DKI JAKARTA') |
            (tb_kepesertaan_baru['PSTV09_NEW'] == 'DKI JAKARTA')
        ].copy()
        tb_top_peserta = tb_dki_peserta.nlargest(1000, 'jumlah_klaim_fkrtl')
        id3 = set(tb_top_peserta['PSTV01'])
        tb_top_fkrtl = tb_fkrtl_baru[tb_fkrtl_baru['PSTV01'].isin(id3)].copy()
        kep = tb_top_peserta.drop(columns=['PSTV15'], errors='ignore').drop_duplicates().copy()

        kep['PSTV03'] = pd.to_datetime(kep['PSTV03'], errors='coerce')
        cutoff = pd.to_datetime("2023-12-31")
        kep['Usia'] = cutoff.year - kep['PSTV03'].dt.year
        mask = ((cutoff.month < kep['PSTV03'].dt.month) |
                ((cutoff.month == kep['PSTV03'].dt.month) & (cutoff.day < kep['PSTV03'].dt.day)))
        kep.loc[mask, 'Usia'] = kep.loc[mask, 'Usia'] - 1

        node_peserta = kep.rename(columns={
            'PSTV01': 'peserta_id',
            'PSTV02': 'keluarga_id',
            'PSTV03': 'tanggal_lahir',
            'PSTV04': 'hub_keluarga',
            'PSTV05': 'jenis_kelamin',
            'PSTV06': 'status_perkawinan',
            'PSTV07': 'Kelas_Rawat',
            'PSTV08': 'Segmentasi',
        })[['peserta_id', 'Usia', 'hub_keluarga', 'jenis_kelamin', 'Segmentasi', 'Kelas_Rawat', 'jumlah_klaim_fkrtl', 'total_klaim_fkrtl']].copy()

        faskes = tb_top_fkrtl[['FKL07', 'FKL08', 'FKL05', 'FKL06', 'FKL48', 'FKL09']].copy()
        faskes['faskes_id'] = (faskes['FKL05'].astype(str) + "_" +
                               faskes['FKL06'].astype(str) + "_" +
                               faskes['FKL07'].astype(str) + "_" +
                               faskes['FKL08'].astype(str) + "_" +
                               faskes['FKL09'].astype(str))
        node_faskes = faskes.groupby('faskes_id').agg(
            Kepemilikan_Faskes=('FKL07', 'first'),
            Jenis_Faskes=('FKL08', 'first'),
            Tipe_Faskes=('FKL09', 'first'),
            Provinsi=('FKL05', 'first'),
            Kabupaten=('FKL06', 'first'),
            Biaya_Klaim_Rata2=('FKL48', 'mean')
        ).reset_index()

        fk = tb_top_fkrtl[['PSTV01', 'FKL02', 'FKL05', 'FKL06', 'FKL07', 'FKL08', 'FKL09',
                           'FKL03', 'FKL04', 'FKL47', 'FKL24A', 'FKL17A', 'FKL48', 'FKL10']].copy()
        fk['faskes_id'] = (fk['FKL05'].astype(str) + "_" +
                           fk['FKL06'].astype(str) + "_" +
                           fk['FKL07'].astype(str) + "_" +
                           fk['FKL08'].astype(str) + "_" +
                           fk['FKL09'].astype(str))
        fk['FKL03'] = pd.to_datetime(fk['FKL03'], errors='coerce')
        fk['FKL04'] = pd.to_datetime(fk['FKL04'], errors='coerce')
        fk['Durasi_Rawat'] = (fk['FKL04'] - fk['FKL03']).dt.days

        df_fkrtl_sorted = fk.sort_values(by=['PSTV01', 'FKL03']).reset_index(drop=True)
        df_fkrtl_sorted['tanggal_kunjungan_sebelumnya'] = df_fkrtl_sorted.groupby('PSTV01')['FKL03'].shift()
        df_fkrtl_sorted['Hari_Setelah_Kunjungan_Terakhir'] = (df_fkrtl_sorted['FKL03'] - df_fkrtl_sorted['tanggal_kunjungan_sebelumnya']).dt.days
        fk = fk.merge(df_fkrtl_sorted[['PSTV01', 'FKL02', 'Hari_Setelah_Kunjungan_Terakhir']], on=['PSTV01', 'FKL02'], how='left')

        df_fkrtl_sorted['faskes_kunjungan_sebelumnya'] = (
            df_fkrtl_sorted['FKL05'].astype(str) + "_" +
            df_fkrtl_sorted['FKL06'].astype(str) + "_" +
            df_fkrtl_sorted['FKL07'].astype(str) + "_" +
            df_fkrtl_sorted['FKL08'].astype(str) + "_" +
            df_fkrtl_sorted['FKL09'].astype(str)
        )
        prev_faskes = df_fkrtl_sorted.groupby('PSTV01')['faskes_kunjungan_sebelumnya'].shift()
        df_fkrtl_sorted['faskes_kunjungan_sebelumnya'] = np.where(
            df_fkrtl_sorted['faskes_kunjungan_sebelumnya'] != prev_faskes, 'berbeda', 'tidak berbeda'
        )
        df_fkrtl_sorted.loc[prev_faskes.isna(), 'faskes_kunjungan_sebelumnya'] = 'tidak berbeda'
        fk = fk.merge(df_fkrtl_sorted[['PSTV01', 'FKL02', 'faskes_kunjungan_sebelumnya']], on=['PSTV01', 'FKL02'], how='left')

        # koordinat optional
        kordinat_path = os.path.join(base_path, "koordinat_kota.json")
        if os.path.exists(kordinat_path):
            kordinat = pd.read_json(kordinat_path)
            df_fkrtl_sorted = df_fkrtl_sorted.merge(kordinat[['city', 'latitude', 'longitude']], left_on='FKL06', right_on='city', how='left')
            df_fkrtl_sorted.rename(columns={'latitude': 'kotkab_latitude', 'longitude': 'kotkab_longitude'}, inplace=True)
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371.0
                lat1_rad = np.radians(lat1); lon1_rad = np.radians(lon1)
                lat2_rad = np.radians(lat2); lon2_rad = np.radians(lon2)
                dlon = lon2_rad - lon1_rad; dlat = lat2_rad - lat1_rad
                a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a)); return R * c
            jarak_hitung = haversine(
                df_fkrtl_sorted['kotkab_latitude'].shift(), df_fkrtl_sorted['kotkab_longitude'].shift(),
                df_fkrtl_sorted['kotkab_latitude'], df_fkrtl_sorted['kotkab_longitude']
            )
            df_fkrtl_sorted['Jarak_km'] = np.where(df_fkrtl_sorted['faskes_kunjungan_sebelumnya']=='berbeda', jarak_hitung, 0)
        else:
            df_fkrtl_sorted['Jarak_km'] = 0

        fk = fk.merge(df_fkrtl_sorted[['PSTV01', 'FKL02', 'Jarak_km']], on=['PSTV01', 'FKL02'], how='left')
        fk['Selisih_Biaya'] = fk['FKL47'] - fk['FKL48']
        fk['Biaya_Klaim'] = fk['FKL48']

        edges = fk.rename(columns={
            'PSTV01': 'peserta_id', 'FKL10': 'Jenis_Pelayanan',
            'FKL17A': 'Diagnosis_Primer', 'FKL24A': 'Diagnosis_Sekunder'
        })[['peserta_id', 'faskes_id', 'faskes_kunjungan_sebelumnya', 'Jarak_km', 'Hari_Setelah_Kunjungan_Terakhir',
             'Diagnosis_Primer', 'Diagnosis_Sekunder', 'Jenis_Pelayanan', 'Biaya_Klaim']].copy()

        visual = node_peserta.copy()
        visual['rata-rata_jarak_km'] = edges.groupby('peserta_id')['Jarak_km'].mean().reindex(visual['peserta_id']).values
        visual['rata-rata_hari_setelah_kunjungan_terakhir'] = edges.groupby('peserta_id')['Hari_Setelah_Kunjungan_Terakhir'].mean().reindex(visual['peserta_id']).values
        visual['rata-rata_klaim_biaya'] = edges.groupby('peserta_id')['Biaya_Klaim'].mean().reindex(visual['peserta_id']).values
        visual['proporsi_faskes_kunjungan_sebelumnya'] = edges.groupby('peserta_id')['faskes_kunjungan_sebelumnya'].apply(lambda x: x.value_counts(normalize=True).max()).reindex(visual['peserta_id']).values
        visual['modus_diagnosis_primer'] = edges.groupby('peserta_id')['Diagnosis_Primer'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reindex(visual['peserta_id']).values
        visual['modus_diagnosis_sekunder'] = edges.groupby('peserta_id')['Diagnosis_Sekunder'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reindex(visual['peserta_id']).values

        features = node_peserta[['Usia', 'jumlah_klaim_fkrtl']].fillna(0)
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        node_peserta['k-means_cluster'] = clusters
        edges['k-means_cluster'] = edges['peserta_id'].map(node_peserta.set_index('peserta_id')['k-means_cluster'])

        def mode_or_nan(s):
            s_clean = s.dropna()
            if s_clean.empty: return np.nan
            return s_clean.value_counts().idxmax()
        mode_per_faskes = edges.groupby('faskes_id')['k-means_cluster'].agg(mode_or_nan).reset_index().rename(columns={'k-means_cluster': 'k-means_cluster_mode'})
        node_faskes = node_faskes.merge(mode_per_faskes, on='faskes_id', how='left')

        visual['k-means_cluster'] = node_peserta['k-means_cluster']

        return node_peserta, node_faskes, edges, visual

    except Exception as e:
        return None, None, None, f"Error: {e}"

# ---------- HELPER: graph (sama seperti yang kamu pakai) ----------
def create_interactive_graph(edges, node_peserta, top_faskes_count=50):
    try:
        if edges is None or edges.empty:
            fig = go.Figure(); fig.update_layout(title="No edge data to show"); return fig

        faskes_counts = edges['faskes_id'].value_counts()
        top_faskes = faskes_counts.head(top_faskes_count).index.tolist()
        sub_edges = edges[edges['faskes_id'].isin(top_faskes)].copy()

        G = nx.Graph()
        peserta_cluster_kmeans = {}
        if (node_peserta is not None) and ('peserta_id' in node_peserta.columns) and ('k-means_cluster' in node_peserta.columns):
            peserta_cluster_kmeans = node_peserta.set_index('peserta_id')['k-means_cluster'].to_dict()

        for _, row in sub_edges.iterrows():
            peserta_id_raw = row['peserta_id']
            peserta_node = f"peserta_{peserta_id_raw}"
            faskes_node = f"faskes_{row['faskes_id']}"
            cluster_id = peserta_cluster_kmeans.get(peserta_id_raw, None)
            G.add_node(peserta_node, type="peserta", cluster=cluster_id)
            G.add_node(faskes_node, type="faskes")
            G.add_edge(peserta_node, faskes_node)

        if G.number_of_nodes() == 0:
            fig = go.Figure(); fig.update_layout(title="No subgraph to show"); return fig

        pos = nx.spring_layout(G, k=0.25, iterations=50, seed=42)

        edge_x, edge_y = [], []
        for e in G.edges():
            x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

        base_colors = {0:'#3b82f6',1:'#10b981',2:'#8b5cf6',3:'#f97316',4:'#ef4444',5:'#06b6d4',6:'#a78bfa',7:'#f59e0b',8:'#84cc16',9:'#ec4899'}
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for n,d in G.nodes(data=True):
            x,y = pos[n]; node_x.append(x); node_y.append(y)
            node_type = d.get('type','')
            if node_type == 'peserta':
                node_text.append(f"ID: {n}<br>Tipe: peserta<br>Cluster: {d.get('cluster','N/A')}")
                node_color.append(base_colors.get(d.get('cluster'), 'lightgray')); node_size.append(8)
            else:
                node_text.append(f"ID: {n}<br>Tipe: faskes"); node_color.append('#ffb86b'); node_size.append(18)

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                                marker=dict(color=node_color, size=node_size, line_width=1.0))

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(title="Subgraph Peserta-Faskes (Warna Berdasarkan Cluster)", showlegend=False,
                          hovermode='closest', margin=dict(b=10,l=10,r=10,t=50),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
    except Exception:
        fig = go.Figure(); fig.update_layout(title="Error building graph"); return fig

# ---------- VISUALIZE CLUSTER (Plotly version + small matplots where ok) ----------
def visualize_visual_clusters_streamlit(visual, cluster_col='k-means_cluster'):
    """Tampilkan semua visual cluster yang diminta (menggunakan Plotly dan Matplotlib)."""
    # SAFETY: jika kosong
    if visual is None or visual.empty:
        st.info("Tidak ada data visual untuk divisualisasikan.")
        return

    # Ensure cluster col exists
    if cluster_col not in visual.columns:
        st.warning(f"Kolom cluster '{cluster_col}' tidak ditemukan di data. Lewati visual cluster.")
        return

    # --- Top summary visuals (Distribusi usia, jenis kelamin, segmentasi)
    st.markdown("### Visualisasi Peserta")
    c1, c2, c3 = st.columns(3)
    with c1:
        if 'Usia' in visual.columns:
            fig = px.histogram(visual, x='Usia', nbins=20, title='Distribusi Usia', marginal='box')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'jenis_kelamin' in visual.columns:
            fig = px.pie(visual, names='jenis_kelamin', title='Komposisi Jenis Kelamin', hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        if 'Segmentasi' in visual.columns:
            seg_counts = visual['Segmentasi'].value_counts()
            if seg_counts.shape[0] <= 8:
                fig = px.pie(visual, names='Segmentasi', title='Segmentasi', hole=0.35)
            else:
                top_seg = seg_counts.head(10)
                fig = px.bar(x=top_seg.index, y=top_seg.values, title='Top Segmentasi')
            st.plotly_chart(fig, use_container_width=True)

    # --- Distribusi cluster
    st.markdown("### Distribusi Cluster & Usia per Cluster")
    col_a, col_b = st.columns([1,2])
    with col_a:
        st.plotly_chart(px.pie(visual, names=cluster_col, title='Distribusi Cluster', hole=0.3), use_container_width=True)
    with col_b:
        if 'Usia' in visual.columns:
            fig = px.box(visual, x=cluster_col, y='Usia', title='Distribusi Usia per Cluster', points='outliers')
            st.plotly_chart(fig, use_container_width=True)

    # --- Per-cluster categorical distributions (stacked bar)
    def stacked_bar_per_cluster(df, column, top_n=8, title=None):
        if column not in df.columns:
            return None
        temp = df.copy()
        temp[column] = temp[column].fillna('Unknown')
        cross = pd.crosstab(temp[cluster_col], temp[column], normalize='index') * 100
        if cross.empty:
            return None
        # limit categories
        if cross.shape[1] > top_n:
            top = cross.sum().nlargest(top_n).index
            cross = cross[top].copy()
            cross['Others'] = 100 - cross.sum(axis=1)
        cross = cross.reset_index().melt(id_vars=cluster_col, var_name=column, value_name='pct')
        fig = px.bar(cross, x=cluster_col, y='pct', color=column, title=title, labels={'pct':'%','k-means_cluster':'Cluster'})
        fig.update_layout(barmode='stack', legend=dict(orientation='v'))
        return fig

    st.markdown("### Distribusi Kategorikal per Cluster")
    cats = [
        ('hub_keluarga', 'Distribusi Hubungan Keluarga per Cluster'),
        ('jenis_kelamin', 'Distribusi Jenis Kelamin per Cluster'),
        ('Segmentasi', 'Distribusi Segmentasi per Cluster'),
        ('Kelas_Rawat', 'Distribusi Kelas Rawat per Cluster'),
        ('modus_diagnosis_primer', 'Distribusi Modus Diagnosis Primer per Cluster'),
        ('modus_diagnosis_sekunder', 'Distribusi Modus Diagnosis Sekunder per Cluster'),
    ]
    # tampilkan 2 kolom grid
    for i in range(0, len(cats), 2):
        row = cats[i:i+2]
        cols = st.columns(len(row))
        for (col_name, title), place in zip(row, cols):
            fig = stacked_bar_per_cluster(visual, col_name, top_n=8, title=title)
            if fig is not None:
                place.plotly_chart(fig, use_container_width=True)
            else:
                place.info(f"{col_name} tidak tersedia")

    # --- Numeric per cluster (boxplots multiple metrics)
    st.markdown("### Variabel Numerik per Cluster (Boxplots)")
    numeric_cols = [
        'Usia', 'jumlah_klaim_fkrtl', 'total_klaim_fkrtl',
        'rata-rata_jarak_km', 'rata-rata_hari_setelah_kunjungan_terakhir',
        'rata-rata_klaim_biaya', 'proporsi_faskes_kunjungan_sebelumnya'
    ]
    present_nums = [c for c in numeric_cols if c in visual.columns]
    # tampilkan sampai 3 per row
    for i in range(0, len(present_nums), 3):
        row = present_nums[i:i+3]
        cols = st.columns(len(row))
        for col_name, place in zip(row, cols):
            fig = px.box(visual, x=cluster_col, y=col_name, title=f"{col_name} per Cluster", color=cluster_col, points='outliers')
            place.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- View peserta by cluster (table + download)
    st.markdown("### View Peserta per Cluster")
    cluster_options = ['All'] + sorted(visual[cluster_col].dropna().unique().tolist())
    chosen = st.selectbox("Pilih Cluster untuk melihat peserta", options=cluster_options, index=0)
    if chosen == 'All':
        df_view = visual.copy()
    else:
        df_view = visual[visual[cluster_col] == chosen].copy()

    st.markdown(f"Menampilkan peserta: {len(df_view):,}")
    st.dataframe(df_view.reset_index(drop=True), use_container_width=True)
    csv = df_view.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Peserta (CSV)", csv, file_name=f'peserta_cluster_{chosen}.csv', mime='text/csv')

# ---------- MAIN ----------
def main():
    st.markdown("<div class='title'>Dashboard Analisis GNN & Cluster Peserta Tuberculosis BPJS Kesehatan Provinsi Jakarta</div>", unsafe_allow_html=True)

    # load data
    with st.spinner("Memuat data..."):
        node_peserta, node_faskes, edges, visual = load_and_process_data()
    if node_peserta is None:
        st.error(f"Data gagal dimuat: {visual}")
        return

    # -------- FILTERS
    st.markdown("<div class='section'>Filter</div>", unsafe_allow_html=True)
    f1, f2, f3, f4, f5, f6 = st.columns([1.4,1,1,1,1,0.8])

    clusters = ['All'] + sorted(node_peserta['k-means_cluster'].dropna().unique().tolist()) if 'k-means_cluster' in node_peserta.columns else ['All']
    sel_cluster = f1.selectbox("Cluster", options=clusters, index=0)

    genders = ['All'] + sorted(node_peserta['jenis_kelamin'].dropna().unique().tolist()) if 'jenis_kelamin' in node_peserta.columns else ['All']
    sel_gender = f2.selectbox("Jenis Kelamin", options=genders, index=0)

    kelas_opts = ['All'] + sorted(node_peserta['Kelas_Rawat'].dropna().unique().tolist()) if 'Kelas_Rawat' in node_peserta.columns else ['All']
    sel_kelas = f3.selectbox("Kelas Rawat", options=kelas_opts, index=0)

    seg_opts = ['All'] + sorted(node_peserta['Segmentasi'].dropna().unique().tolist()) if 'Segmentasi' in node_peserta.columns else ['All']
    sel_seg = f4.selectbox("Segmentasi", options=seg_opts, index=0)

    prov_opts = ['All'] + sorted(node_faskes['Provinsi'].dropna().unique().tolist()) if 'Provinsi' in node_faskes.columns else ['All']
    sel_prov = f5.selectbox("Provinsi Faskes", options=prov_opts, index=0)

    top_faskes_count = f6.slider("Top Faskes (graph)", min_value=10, max_value=200, value=50)

    s1, s2 = st.columns([1,0.7])
    show_diagnosis = s1.checkbox("Tampilkan Diagnosis Charts", value=True)

    # -------- APPLY FILTERS
    df_vis = visual.copy()
    if sel_cluster != 'All':
        df_vis = df_vis[df_vis['k-means_cluster'] == sel_cluster]
    if sel_gender != 'All' and 'jenis_kelamin' in df_vis.columns:
        df_vis = df_vis[df_vis['jenis_kelamin'] == sel_gender]
    if sel_kelas != 'All' and 'Kelas_Rawat' in df_vis.columns:
        df_vis = df_vis[df_vis['Kelas_Rawat'] == sel_kelas]
    if sel_seg != 'All' and 'Segmentasi' in df_vis.columns:
        df_vis = df_vis[df_vis['Segmentasi'] == sel_seg]

    peserta_ke = set(df_vis['peserta_id'])
    df_edges = edges[edges['peserta_id'].isin(peserta_ke)].copy()

    df_node_faskes = node_faskes.copy()
    if sel_prov != 'All':
        df_node_faskes = df_node_faskes[df_node_faskes['Provinsi'] == sel_prov]
    faskes_ke = set(df_node_faskes['faskes_id'])
    df_edges = df_edges[df_edges['faskes_id'].isin(faskes_ke)]

    peserta_ke = set(df_edges['peserta_id'])
    df_vis = df_vis[df_vis['peserta_id'].isin(peserta_ke)]
    df_node_peserta = node_peserta[node_peserta['peserta_id'].isin(set(df_vis['peserta_id']))].copy()
    df_node_faskes = df_node_faskes[df_node_faskes['faskes_id'].isin(set(df_edges['faskes_id']))]

    # ---------- TOP METRICS ----------
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='card'><div class='small-muted'>Peserta</div><div style='font-weight:700;font-size:1.1rem'>{len(df_node_peserta):,}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><div class='small-muted'>Faskes</div><div style='font-weight:700;font-size:1.1rem'>{len(df_node_faskes):,}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><div class='small-muted'>Edges</div><div style='font-weight:700;font-size:1.1rem'>{len(df_edges):,}</div></div>", unsafe_allow_html=True)
    cluster_count = node_peserta['k-means_cluster'].nunique() if 'k-means_cluster' in node_peserta.columns else 0
    col4.markdown(f"<div class='card'><div class='small-muted'>Jumlah Cluster</div><div style='font-weight:700;font-size:1.1rem'>{cluster_count}</div></div>", unsafe_allow_html=True)

    # ---------- tambahan: visualisasi cluster lebih lengkap (fungsi terpisah) ----------
    st.markdown("<div class='section'>Visualisasi Karakteristik Cluster (Detail)</div>", unsafe_allow_html=True)
    visualize_visual_clusters_streamlit(df_vis, cluster_col='k-means_cluster')

    # ---------- CLUSTER & INSIGHT ----------
    st.markdown("<div class='section'>Cluster & Insight</div>", unsafe_allow_html=True)
    cl1, cl2 = st.columns([1,2])
    with cl1:
        st.plotly_chart(px.pie(df_vis, names='k-means_cluster', title='Distribusi Cluster', hole=0.25) if ('k-means_cluster' in df_vis.columns and not df_vis.empty) else go.Figure().update_layout(title="No cluster"), use_container_width=True)
        if show_diagnosis and 'modus_diagnosis_primer' in df_vis.columns:
            diag_counts = df_vis['modus_diagnosis_primer'].value_counts().head(10)
            fig_diag = px.bar(x=diag_counts.index, y=diag_counts.values, title='Top 10 Diagnosis Primer')
            st.plotly_chart(fig_diag, use_container_width=True)
    with cl2:
        source_for_cluster = df_vis if ('rata-rata_klaim_biaya' in df_vis.columns) and (not df_vis.empty) else node_peserta
        if 'k-means_cluster' in source_for_cluster.columns and 'rata-rata_klaim_biaya' in source_for_cluster.columns:
            cluster_claim = source_for_cluster.groupby('k-means_cluster')['rata-rata_klaim_biaya'].mean().reset_index()
            fig_cluster = px.bar(cluster_claim, x='k-means_cluster', y='rata-rata_klaim_biaya', title='Rata-rata Klaim Biaya per Cluster', color='k-means_cluster')
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            box_figs = []
            for col in ['Usia','jumlah_klaim_fkrtl','total_klaim_fkrtl']:
                if col in df_vis.columns:
                    box_figs.append(px.box(df_vis, x='k-means_cluster', y=col, title=f"{col} per Cluster"))
            for fig in box_figs:
                st.plotly_chart(fig, use_container_width=True)

    try:
        clust_summary = df_vis.groupby('k-means_cluster')[['Usia','jumlah_klaim_fkrtl','total_klaim_fkrtl']].agg(['mean','count']).round(2)
        st.markdown("#### Statistik per Cluster")
        st.dataframe(clust_summary)
    except Exception:
        pass

    # ---------- GRAPH ----------
    st.markdown("<div class='section'>Graph: Subgraph Peserta - Faskes</div>", unsafe_allow_html=True)
    fig_graph = create_interactive_graph(df_edges, df_node_peserta, top_faskes_count=top_faskes_count)
    st.plotly_chart(fig_graph, use_container_width=True, height=600)

    # ---------- ANOMALY DETECTION BERDASARKAN JARAK SAJA ----------
    st.markdown("<div class='section'>Deteksi Anomali: Jarak Kunjungan (Hanya Jarak)</div>", unsafe_allow_html=True)

    df_edges_anom = df_edges.merge(
        df_node_peserta[['peserta_id', 'k-means_cluster', 'Usia', 'jenis_kelamin', 'Segmentasi', 'Kelas_Rawat', 'jumlah_klaim_fkrtl', 'total_klaim_fkrtl']],
        on='peserta_id', how='left'
    ).merge(
        df_node_faskes[['faskes_id', 'k-means_cluster_mode']],
        on='faskes_id', how='left'
    )

    if 'Jarak_km' not in df_edges_anom.columns:
        df_edges_anom['Jarak_km'] = 0

    jarak_threshold = 50  # sesuai syntax kamu
    df_edges_anom['anomali_jarak'] = df_edges_anom['Jarak_km'].fillna(0) > jarak_threshold

    total_edges = len(df_edges_anom)
    total_anomali_jarak = int(df_edges_anom['anomali_jarak'].sum())
    total_normal_by_distance = total_edges - total_anomali_jarak

    st.markdown("#### Proporsi Edges Berdasarkan Anomali Jarak")
    fig_pie_distance = px.pie(
        names=['Anomali Jarak', f'Normal (Jarak <= {jarak_threshold} km)'],
        values=[total_anomali_jarak, total_normal_by_distance],
        title=f'Proporsi Edges: Anomali Jarak (> {jarak_threshold} km) vs Normal',
        hole=0.3
    )
    st.plotly_chart(fig_pie_distance, use_container_width=True)

    # Table peserta anomali (unik per peserta) - semua karakteristik
    peserta_anom_ids = df_edges_anom[df_edges_anom['anomali_jarak'] == True]['peserta_id'].unique()
    df_peserta_anom = df_vis[df_vis['peserta_id'].isin(peserta_anom_ids)].copy()  # gunakan df_vis agar filter tersinkron

    if not df_peserta_anom.empty:
        # tambahkan flag & jarak maks
        anom_flags = df_edges_anom[df_edges_anom['peserta_id'].isin(peserta_anom_ids)].groupby('peserta_id').agg({
            'anomali_jarak': 'any',
            'Jarak_km': 'max'
        }).reset_index().rename(columns={'Jarak_km': 'Jarak_km_maks'})
        df_peserta_anom = df_peserta_anom.merge(anom_flags, on='peserta_id', how='left')

        st.markdown("#### Data Peserta yang Terindikasi Anomali (berdasarkan jarak saja)")
        # tampilkan semua kolom karakteristik yang ada
        st.dataframe(df_peserta_anom.reset_index(drop=True), use_container_width=True)
        csv = df_peserta_anom.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Peserta Anomali (CSV)",
            data=csv,
            file_name='peserta_anomali_jarak_only.csv',
            mime='text/csv',
        )
    else:
        st.info(f"Tidak ada peserta yang terdeteksi anomali berdasarkan jarak (threshold={jarak_threshold} km).")

if __name__ == "__main__":
    main()
