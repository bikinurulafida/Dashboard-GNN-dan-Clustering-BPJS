# helper.py
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

# ================== 📊 BASIC PLOTS ==================
def plot_table(df, n=10):
    st.dataframe(df.head(n), use_container_width=True)

def plot_diagnosis_distribution(df):
    col = 'modus_diagnosis_primer' if 'modus_diagnosis_primer' in df.columns else 'Diagnosis_Primer'
    if col in df.columns:
        diag_count = df[col].dropna().value_counts().reset_index()
        diag_count.columns = [col,'Jumlah']
        fig = px.bar(
            diag_count, x=col, y='Jumlah', text='Jumlah',
            title="Distribusi Diagnosis Primer",
            color=col,
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Kolom 'Diagnosis_Primer' atau 'modus_diagnosis_primer' tidak ditemukan.")

def plot_age_distribution(df):
    """
    Histogram Distribusi Usia Peserta dengan warna flat.
    """
    if 'Usia' in df.columns:
        fig = px.histogram(
            df, x='Usia', nbins=20, title="Distribusi Usia Peserta",
            color_discrete_sequence=['#1abc9c']  # warna flat Teal
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Kolom 'Usia' tidak ditemukan.")

def plot_service_distribution(df):
    col = 'Jenis_Pelayanan' if 'Jenis_Pelayanan' in df.columns else None
    if col and col in df.columns:
        service_count = df[col].dropna().value_counts().reset_index()
        service_count.columns = [col,'Jumlah']
        fig = px.bar(
            service_count, x=col, y='Jumlah', text='Jumlah', 
            title="Distribusi Jenis Pelayanan",
            color=col,
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Kolom 'Jenis_Pelayanan' tidak ditemukan, gunakan 'Segmentasi' atau 'Kelas_Rawat' jika perlu.")

def plot_cost_distribution(df):
    """
    Histogram Distribusi Biaya Klaim dengan warna flat.
    """
    col = 'rata-rata_klaim_biaya' if 'rata-rata_klaim_biaya' in df.columns else 'Biaya_Klaim'
    if col in df.columns:
        fig = px.histogram(
            df, x=col, nbins=50, title="Distribusi Biaya Klaim",
            color_discrete_sequence=['#1abc9c']  # warna flat Teal
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Kolom 'Biaya_Klaim' atau 'rata-rata_klaim_biaya' tidak ditemukan.")

def modus_diagnosis_per_faskes(df):
    col_diag = 'modus_diagnosis_primer' if 'modus_diagnosis_primer' in df.columns else 'Diagnosis_Primer'
    col_faskes = 'faskes_id' if 'faskes_id' in df.columns else None
    if col_diag in df.columns and col_faskes in df.columns:
        mode_diag = df.groupby(col_faskes)[col_diag] \
                      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
        mode_diag.columns = [col_faskes,'Modus_Diagnosis_Primer']
        st.subheader("Modus Diagnosis Primer per Faskes")
        st.dataframe(mode_diag.head(10))
    else:
        st.warning("Kolom diagnosis atau faskes tidak ditemukan untuk modus diagnosis per faskes.")

# ================== 📈 K-MEANS ==================
def compute_kmeans(embeddings: np.ndarray, node_peserta: pd.DataFrame, edges: pd.DataFrame, n_clusters:int=10):
    # Jika sudah ada hasil cluster di data, skip perhitungan ulang
    if 'k-means_cluster' in node_peserta.columns and node_peserta['k-means_cluster'].notna().any():
        print("🔹 Kolom 'k-means_cluster' sudah ada di node_peserta — skip perhitungan ulang.")
    else:
        print("⚙️ Menjalankan KMeans baru...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        node_peserta['k-means_cluster'] = clusters

    # Map cluster ke edges
    edges['k-means_cluster'] = edges['peserta_id'].map(node_peserta.set_index('peserta_id')['k-means_cluster'])
    print(edges[['peserta_id','faskes_id','k-means_cluster']].head())

    # Fungsi aman untuk menghitung modus
    def mode_or_nan(s):
        s_clean = s.dropna()
        if s_clean.empty:
            return np.nan
        vc = s_clean.value_counts()
        return vc.idxmax()

    # Hitung modus cluster per faskes
    mode_per_faskes = edges.groupby('faskes_id')['k-means_cluster'].agg(mode_or_nan).reset_index().rename(columns={'k-means_cluster':'k-means_cluster_mode'})

    # Pastikan node_faskes terbentuk
    if 'faskes_id' in node_peserta.columns:
        node_faskes = node_peserta[['faskes_id']].drop_duplicates()
    else:
        node_faskes = pd.DataFrame({'faskes_id': edges['faskes_id'].unique()})

    # Merge hasil modus
    node_faskes = node_faskes.merge(mode_per_faskes, on='faskes_id', how='left')

    # Normalisasi kolom hasil merge agar hanya ada satu kolom k-means_cluster_mode
    if 'k-means_cluster_mode' not in node_faskes.columns:
        for col in ['k-means_cluster_mode_y', 'k-means_cluster_mode_x']:
            if col in node_faskes.columns:
                node_faskes['k-means_cluster_mode'] = node_faskes[col]
                break

    # Jika belum ada, turunkan dari kolom cluster lain atau isi NaN
    if 'k-means_cluster_mode' not in node_faskes.columns:
        if 'k-means_cluster' in node_faskes.columns:
            node_faskes['k-means_cluster_mode'] = node_faskes['k-means_cluster']
        else:
            node_faskes['k-means_cluster_mode'] = np.nan

    # Casting aman ke Int64
    try:
        node_faskes['k-means_cluster_mode'] = node_faskes['k-means_cluster_mode'].astype('Int64')
    except Exception:
        pass

    # Bersihkan kolom duplikat hasil merge
    for col in ['k-means_cluster_mode_x', 'k-means_cluster_mode_y']:
        if col in node_faskes.columns:
            node_faskes.drop(columns=[col], inplace=True)

    print(node_faskes[['faskes_id','k-means_cluster_mode']].drop_duplicates().head())

    return node_peserta, edges, node_faskes

# ================== 📊 NETWORK GRAPH ==================
def visualize_graph(node_peserta: pd.DataFrame, edges: pd.DataFrame, top_n_faskes:int=175):
    if 'k-means_cluster' not in node_peserta.columns:
        st.warning("Cluster belum dihitung, jalankan compute_kmeans dulu.")
        return

    peserta_cluster_kmeans = node_peserta.set_index('peserta_id')['k-means_cluster'].to_dict()
    top_faskes = edges['faskes_id'].value_counts().head(top_n_faskes).index.tolist()
    sub_edges = edges[edges['faskes_id'].isin(top_faskes)]
    
    G = nx.Graph()
    for _, row in sub_edges.iterrows():
        peserta_node = f"peserta_{row['peserta_id']}"
        faskes_node = f"faskes_{row['faskes_id']}"
        cluster_id = peserta_cluster_kmeans.get(row['peserta_id'], -1)
        G.add_node(peserta_node, type='peserta', cluster=cluster_id)
        G.add_node(faskes_node, type='faskes')
        G.add_edge(peserta_node, faskes_node)
    
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0,x1,None])
        edge_y.extend([y0,y1,None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5,color='#888'),
                            hoverinfo='none', mode='lines')
    
    node_x, node_y, node_text, node_color, node_size = [],[],[],[],[]
    color_map = {i:c for i,c in enumerate([
        '#ff0000','#008000','#0000ff','#800080','#ffff00','#00ffff',
        '#ff00ff','#ffa500','#a52a2a','#00ff00'])}
    color_map[-1] = '#000000'
    
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type = data.get('type')
        node_text.append(f"ID: {node}<br>Tipe: {node_type}<br>Cluster: {data.get('cluster','N/A')}")
        if node_type=='peserta':
            cluster_id = data.get('cluster', -1)
            node_color.append(color_map.get(cluster_id,'#1abc9c'))  # default Teal
            node_size.append(10)
        else:
            node_color.append('orange')
            node_size.append(25)
    
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                            marker=dict(color=node_color, size=node_size, line_width=1.5))
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=dict(text="Subgraph Peserta-Faskes (Cluster)", font=dict(size=16)),
                                     showlegend=False, hovermode='closest',
                                     margin=dict(b=0,l=0,r=0,t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig, use_container_width=True)
