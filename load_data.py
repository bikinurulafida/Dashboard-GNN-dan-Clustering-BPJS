import pandas as pd
import os

# ================= LOAD DATA UNTUK VISUALISASI =================
def load_data_visualization():
    """
    Load dataset peserta enriched untuk visualisasi.
    File harus berada di: data/node_peserta_tb_dki_enriched.csv
    """
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, "data", "node_peserta_tb_dki_enriched.csv")
    
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    
    # Convert kolom numerik yang relevan
    numeric_cols = ['Jarak_km', 'Hari_Setelah_Kunjungan_Terakhir', 'Biaya_Klaim', 'Usia']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


# ================= LOAD DATA UNTUK GNN & CLUSTERING =================
def load_data_gnn():
    """
    Load dataset peserta, faskes, dan edges untuk GNN + clustering.
    Semua file harus berada di dalam folder: data/
    """
    base_path = os.path.dirname(__file__)
    try:
        node_peserta = pd.read_csv(os.path.join(base_path, "data", "node_peserta_tb_dki.csv"), dtype=str)
        node_faskes = pd.read_csv(os.path.join(base_path, "data", "node_faskes_tb_dki.csv"), dtype=str)
        edges = pd.read_csv(os.path.join(base_path, "data", "edges_tb_dki.csv"), dtype=str)

        # Bersihkan spasi di nama kolom
        for df in [node_peserta, node_faskes, edges]:
            df.columns = df.columns.str.strip()

        # Convert kolom numerik jika bisa
        for df in [node_peserta, node_faskes, edges]:
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        return node_peserta, node_faskes, edges
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File CSV tidak ditemukan: {e}")
