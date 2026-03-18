# app.py
# =========================================================
# STREAMLIT DASHBOARD: Transaksi → RFM → K-Means → Cluster → Strategi/Rekomendasi → Dashboard
# =========================================================
# Catatan:
# - Data di-import (upload) lewat fitur file uploader (CSV/XLSX).
# - Kolom yang dipakai: nopo, marketing_id, tglpo, kode_perusahaan_id, total_bayar, nama_produk
# - Periode data dibatasi tahun 2022–2025 (berdasarkan tglpo).
# - Output utama: tabel RFM, hasil cluster per customer, ringkasan cluster, dan tabel strategi/rekomendasi.

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------
# [LANGKAH 0] Konfigurasi halaman
# -----------------------------
st.set_page_config(page_title="RFM + K-Means Recommender Dashboard", layout="wide")
st.title("Dashboard Segmentasi & Rekomendasi (RFM + K-Means)")

st.caption(
    "Alur: Transaksi → RFM → K-Means → Cluster → Strategi/Rekomendasi → Dashboard"
)

# -----------------------------
# [LANGKAH 1] Import data (upload)
# -----------------------------
st.sidebar.header("1) Import Data")
uploaded = st.sidebar.file_uploader(
    "Upload file transaksi (CSV / XLSX)", type=["csv", "xlsx", "xls"]
)

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

if uploaded is None:
    st.info("Silakan upload data transaksi terlebih dahulu.")
    st.stop()

df_raw = load_data(uploaded)

st.subheader("Preview Data (Raw)")
st.dataframe(df_raw.head(20), use_container_width=True)

# -----------------------------
# [LANGKAH 2] Pilih kolom yang digunakan + standarisasi nama kolom
# -----------------------------
st.sidebar.header("2) Seleksi Kolom")
required_cols = ["nopo", "marketing_id", "tglpo", "kode_perusahaan_id", "total_bayar", "nama_produk"]

# Bantu jika 'nama_produk' belum ada, tapi 'nama_pelatihan' ada (opsional)
# (Anda bisa hapus blok ini jika tidak diperlukan.)
if "nama_produk" not in df_raw.columns and "nama_pelatihan" in df_raw.columns:
    df_raw = df_raw.rename(columns={"nama_pelatihan": "nama_produk"})

missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(
        f"Kolom berikut tidak ditemukan: {missing}\n\n"
        f"Kolom yang tersedia: {list(df_raw.columns)}"
    )
    st.stop()

df = df_raw[required_cols].copy()

# Pastikan tipe data tanggal & numerik benar
df["tglpo"] = pd.to_datetime(df["tglpo"], errors="coerce")
df["total_bayar"] = pd.to_numeric(df["total_bayar"], errors="coerce")

# Bersihkan data invalid
df = df.dropna(subset=["tglpo", "kode_perusahaan_id", "nopo", "total_bayar"])
df = df[df["total_bayar"] >= 0]

# -----------------------------
# [LANGKAH 3] Filter periode 2022–2025
# -----------------------------
st.sidebar.header("3) Filter Periode")
start_year, end_year = 2022, 2025
df = df[(df["tglpo"].dt.year >= start_year) & (df["tglpo"].dt.year <= end_year)].copy()

st.subheader("Data Setelah Seleksi Kolom & Filter Tahun (2022–2025)")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah baris", f"{len(df):,}")
c2.metric("Jumlah customer (kode_perusahaan_id)", f"{df['kode_perusahaan_id'].nunique():,}")
c3.metric("Rentang tanggal", f"{df['tglpo'].min().date()} s/d {df['tglpo'].max().date()}" if len(df) else "-")

st.dataframe(df.head(20), use_container_width=True)

if len(df) == 0:
    st.warning("Tidak ada data transaksi pada periode 2022–2025 setelah filter.")
    st.stop()

# -----------------------------
# [LANGKAH 4] Hitung RFM per customer
# -----------------------------
st.sidebar.header("4) RFM Settings")
st.sidebar.caption("Recency dihitung dari tanggal transaksi terakhir dibanding tanggal referensi.")
use_today_as_ref = st.sidebar.checkbox("Gunakan hari ini sebagai tanggal referensi", value=False)

# Tanggal referensi default: 1 hari setelah tanggal transaksi maksimum pada data
max_tx_date = df["tglpo"].max()
ref_date = pd.Timestamp.today().normalize() if use_today_as_ref else (max_tx_date + pd.Timedelta(days=1))

# RFM:
# Recency  = selisih hari antara ref_date dan transaksi terakhir customer
# Frequency = jumlah transaksi (count nopo) per customer
# Monetary  = total nilai transaksi (sum total_bayar) per customer
rfm = (
    df.groupby("kode_perusahaan_id")
      .agg(
          last_date=("tglpo", "max"),
          Frequency=("nopo", "nunique"),
          Monetary=("total_bayar", "sum"),
          marketing_id=("marketing_id", "first")  # opsional (untuk konteks)
      )
      .reset_index()
)

rfm["Recency"] = (ref_date - rfm["last_date"]).dt.days
rfm = rfm.drop(columns=["last_date"])

st.subheader("Tabel RFM (per Customer)")
st.dataframe(rfm.head(20), use_container_width=True)

# -----------------------------
# [LANGKAH 5] Persiapan fitur untuk K-Means (scaling)
# -----------------------------
X = rfm[["Recency", "Frequency", "Monetary"]].copy()

# Hindari issue jika Monetary sangat besar: scaling tetap menangani
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# [LANGKAH 6] K-Means clustering + evaluasi (Silhouette & DBI)
# -----------------------------
st.sidebar.header("5) K-Means")
k = st.sidebar.slider("Pilih jumlah cluster (k)", min_value=2, max_value=10, value=3, step=1)
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=10_000, value=42, step=1)
n_init = st.sidebar.selectbox("n_init", options=[10, 20, 50], index=1)

kmeans = KMeans(n_clusters=k, random_state=int(random_state), n_init=int(n_init))
clusters = kmeans.fit_predict(X_scaled)

rfm["Cluster"] = clusters

# Evaluasi
sil = silhouette_score(X_scaled, clusters) if k > 1 else np.nan
dbi = davies_bouldin_score(X_scaled, clusters) if k > 1 else np.nan

st.subheader("Evaluasi Kualitas Cluster")
e1, e2, e3 = st.columns(3)
e1.metric("k", k)
e2.metric("Silhouette Coefficient", f"{sil:.4f}")
e3.metric("Davies–Bouldin Index (DBI)", f"{dbi:.4f}")

# -----------------------------
# [LANGKAH 7] Ringkasan cluster (profil RFM per cluster)
# -----------------------------
cluster_profile = (
    rfm.groupby("Cluster")
       .agg(
           Customers=("kode_perusahaan_id", "nunique"),
           Avg_Recency=("Recency", "mean"),
           Avg_Frequency=("Frequency", "mean"),
           Avg_Monetary=("Monetary", "mean"),
           Median_Recency=("Recency", "median"),
           Median_Frequency=("Frequency", "median"),
           Median_Monetary=("Monetary", "median"),
       )
       .reset_index()
       .sort_values("Cluster")
)

st.subheader("Profil Cluster (Rata-rata & Median RFM)")
st.dataframe(cluster_profile, use_container_width=True)

# -----------------------------
# [LANGKAH 7B] Visualisasi (Gambar) Cluster
# -----------------------------
st.subheader("Gambar Cluster")

tab1, tab2 = st.tabs(["PCA (2D)", "Recency vs Monetary"])

# --- Tab 1: PCA 2D (gambar cluster paling representatif)
with tab1:
    pca = PCA(n_components=2, random_state=int(random_state))
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=20)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title("Visualisasi Cluster (PCA 2D dari fitur RFM yang sudah diskalakan)")
    st.pyplot(fig, clear_figure=True)

    st.caption(
        f"Explained variance PCA: PC1={pca.explained_variance_ratio_[0]:.2%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.2%}"
    )

# --- Tab 2: Plot bisnis (Recency vs Monetary)
with tab2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"], s=20)
    ax2.set_xlabel("Recency (hari sejak transaksi terakhir)")
    ax2.set_ylabel("Monetary (total nilai transaksi)")
    ax2.set_title("Cluster pada ruang bisnis: Recency vs Monetary")
    st.pyplot(fig2, clear_figure=True)

    st.caption("Warna menunjukkan cluster hasil K-Means.")

# -----------------------------
# [LANGKAH 8] Interpretasi cluster → kategori bisnis + strategi/rekomendasi
#   (Label tidak ditentukan di awal; ditetapkan setelah cluster terbentuk)
# -----------------------------
st.sidebar.header("6) Strategi/Rekomendasi")
st.sidebar.caption("Aturan label berbasis posisi relatif nilai RFM antar cluster (heuristic).")

# Heuristik pemetaan cluster ke label:
# - Loyal    : Recency rendah, Frequency tinggi, Monetary tinggi
# - Pasif    : Recency tinggi, Frequency rendah
# - Prospek  : sisanya (atau yang tidak ekstrem)
#
# Kita gunakan ranking antar cluster (bukan threshold absolut) agar adaptif terhadap data.
tmp = cluster_profile.copy()

# Rank: Recency kecil lebih baik → rank ascending
tmp["r_recency"] = tmp["Avg_Recency"].rank(ascending=True, method="dense")
# Frequency besar lebih baik → rank descending
tmp["r_frequency"] = tmp["Avg_Frequency"].rank(ascending=False, method="dense")
# Monetary besar lebih baik → rank descending
tmp["r_monetary"] = tmp["Avg_Monetary"].rank(ascending=False, method="dense")

# Skor loyalitas sederhana: gabungan rank (semakin kecil semakin "loyal")
tmp["loyal_score"] = tmp["r_recency"] + tmp["r_frequency"] + tmp["r_monetary"]

# Tentukan kandidat Loyal = loyal_score minimum
loyal_cluster = int(tmp.sort_values("loyal_score").iloc[0]["Cluster"])

# Tentukan kandidat Pasif = Avg_Recency paling tinggi + Avg_Frequency paling rendah (skor gabungan)
tmp["pasif_score"] = tmp["Avg_Recency"].rank(ascending=False, method="dense") + tmp["Avg_Frequency"].rank(ascending=True, method="dense")
pasif_cluster = int(tmp.sort_values("pasif_score").iloc[0]["Cluster"])

# Sisa cluster jadi Prospek (atau jika loyal==pasif karena data ekstrem, fallback)
def label_cluster(c: int) -> str:
    if c == loyal_cluster and c != pasif_cluster:
        return "Customer Loyal"
    if c == pasif_cluster and c != loyal_cluster:
        return "Customer Pasif"
    if c == loyal_cluster and c == pasif_cluster:
        # fallback: pilih loyal, sisanya prospek
        return "Customer Loyal"
    return "Customer Prospek"

label_map = {int(c): label_cluster(int(c)) for c in tmp["Cluster"].tolist()}

# Strategi default per label (bisa Anda ubah sesuai kebutuhan perusahaan)
strategy_map = {
    "Customer Loyal": [
        "Program retensi (prioritas layanan, early access materi, atau benefit khusus).",
        "Penawaran eksklusif/bundling pelatihan lanjutan yang relevan.",
        "Upsell layanan konsultasi atau paket premium."
    ],
    "Customer Prospek": [
        "Kampanye nurturing (newsletter kebijakan terbaru, webinar pengantar).",
        "Promo peningkatan frekuensi (diskon pembelian kedua/ketiga, bundling).",
        "Rekomendasi pelatihan lanjutan berdasarkan riwayat produk yang dibeli."
    ],
    "Customer Pasif": [
        "Strategi reaktivasi (voucher/discount time-limited, follow-up personal).",
        "Penawaran materi refresh/update regulasi yang ringkas.",
        "Survey kebutuhan untuk memicu repeat order."
    ],
}

rfm["Kategori_Bisnis"] = rfm["Cluster"].map(label_map)
rfm["Strategi_Rekomendasi"] = rfm["Kategori_Bisnis"].apply(lambda x: " | ".join(strategy_map.get(x, [])))

# [TAMBAHAN] Simpan hasil untuk dipakai landing page
st.session_state["rfm_result"] = rfm.copy()
st.session_state["trx_filtered"] = df.copy()  # transaksi setelah filter kolom dan tahun 2022–2025

st.subheader("Hasil Akhir: Cluster + Kategori Bisnis + Strategi/Rekomendasi (per Customer)")
st.dataframe(
    rfm[["kode_perusahaan_id", "marketing_id", "Recency", "Frequency", "Monetary", "Cluster", "Kategori_Bisnis", "Strategi_Rekomendasi"]]
    .sort_values(["Kategori_Bisnis", "Cluster"]),
    use_container_width=True
)

# -----------------------------
# [LANGKAH 9] Dashboard strategi/rekomendasi (ringkasan per kategori)
# -----------------------------
st.subheader("Ringkasan Strategi/Rekomendasi per Kategori")
summary = (
    rfm.groupby(["Kategori_Bisnis", "Cluster"])
       .agg(
           Customers=("kode_perusahaan_id", "nunique"),
           Avg_Recency=("Recency", "mean"),
           Avg_Frequency=("Frequency", "mean"),
           Avg_Monetary=("Monetary", "mean"),
       )
       .reset_index()
       .sort_values(["Kategori_Bisnis", "Cluster"])
)
st.dataframe(summary, use_container_width=True)

# -----------------------------
# [LANGKAH 10] (Opsional) Drill-down: transaksi per cluster/kategori
# -----------------------------
st.subheader("Detail Transaksi (Opsional) — Filter berdasarkan Kategori/Cluster")
colA, colB = st.columns(2)
selected_cat = colA.selectbox("Pilih kategori bisnis", options=sorted(rfm["Kategori_Bisnis"].unique()))
selected_clusters = sorted(rfm.loc[rfm["Kategori_Bisnis"] == selected_cat, "Cluster"].unique().tolist())
selected_cluster = colB.selectbox("Pilih cluster", options=selected_clusters)

selected_customers = rfm.loc[rfm["Cluster"] == selected_cluster, "kode_perusahaan_id"].unique()
df_detail = df[df["kode_perusahaan_id"].isin(selected_customers)].copy()

st.dataframe(df_detail.sort_values("tglpo", ascending=False), use_container_width=True)

# -----------------------------
# [LANGKAH 11] Export hasil (Opsional)
# -----------------------------
st.subheader("Unduh Hasil")
out = rfm.copy()
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download hasil (CSV)",
    data=csv_bytes,
    file_name="hasil_rfm_kmeans_cluster_strategi.csv",
    mime="text/csv"
)
