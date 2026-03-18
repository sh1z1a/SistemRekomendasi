import streamlit as st
import pandas as pd

st.set_page_config(page_title="Pencarian Rekomendasi Data Pelanggan", layout="wide")

# -----------------------------
# [LANDING PAGE] Validasi data tersedia
# -----------------------------
if "rfm_result" not in st.session_state or "trx_filtered" not in st.session_state:
    st.error("Hasil cluster belum tersedia. Jalankan proses di halaman utama (app.py) terlebih dahulu.")
    st.stop()

rfm = st.session_state["rfm_result"].copy()
trx = st.session_state["trx_filtered"].copy()

# Pastikan kolom yang dipakai tersedia
need_rfm_cols = ["kode_perusahaan_id", "marketing_id", "Kategori_Bisnis", "Strategi_Rekomendasi", "Cluster"]
need_trx_cols = ["kode_perusahaan_id", "nama_produk"]

missing_rfm = [c for c in need_rfm_cols if c not in rfm.columns]
missing_trx = [c for c in need_trx_cols if c not in trx.columns]
if missing_rfm or missing_trx:
    st.error(f"Kolom kurang. Missing rfm: {missing_rfm}, missing trx: {missing_trx}")
    st.stop()

# -----------------------------
# [STYLE] CSS sederhana agar mirip layout contoh
# -----------------------------
st.markdown(
    """
    <style>
      .topbar {
        background:#2E6EA6;
        padding:18px 22px;
        border-radius:10px;
        color:white;
        font-size:28px;
        font-weight:700;
        margin-bottom:14px;
        text-align:center;
      }
      .subtitle {
        text-align:center;
        font-size:22px;
        font-weight:600;
        margin-top:6px;
        margin-bottom:16px;
      }
      .section-title {
        font-size:28px;
        font-weight:700;
        margin: 18px 0 10px 0;
      }
      div[data-testid="stDataFrame"] {
        border-radius:10px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# [HEADER]
# -----------------------------
st.markdown('<div class="topbar">Pencarian Data Pelanggan</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistem informasi pencarian rekomendasi data pelanggan</div>', unsafe_allow_html=True)

# -----------------------------
# [TABEL INFO KELAS] (seperti contoh)
# -----------------------------
info_df = pd.DataFrame([
    {
        "no": 1,
        "Kelas": "Customer Loyal",
        "Informasi": "Pelanggan yang sering bertransaksi, nilai transaksi tinggi, dan masih aktif. Fokus pada retensi dan penawaran eksklusif."
    },
    {
        "no": 2,
        "Kelas": "Customer Pasif",
        "Informasi": "Pelanggan yang sudah lama tidak bertransaksi dan frekuensi rendah. Fokus pada reaktivasi dan penawaran pemicu transaksi ulang."
    },
    {
        "no": 3,
        "Kelas": "Customer Prospek",
        "Informasi": "Pelanggan dengan potensi berkembang. Fokus pada nurturing, bundling, dan promosi untuk meningkatkan frekuensi atau nilai transaksi."
    },
])

st.dataframe(info_df, use_container_width=True, hide_index=True)

# -----------------------------
# [FILTER DATA]
# - filter hanya: kategori bisnis (loyal/pasif/prospek) dan nama produk
# -----------------------------
st.markdown('<div class="section-title">Filter Data</div>', unsafe_allow_html=True)

# Standarisasi pilihan kategori bisnis agar rapi
# (sesuaikan jika label Anda tepatnya "Customer Loyal" dll)
kategori_opsi = sorted(rfm["Kategori_Bisnis"].dropna().unique().tolist())
if not kategori_opsi:
    kategori_opsi = ["Customer Loyal", "Customer Prospek", "Customer Pasif"]

produk_opsi = sorted(trx["nama_produk"].dropna().unique().tolist())

c1, c2, c3 = st.columns([3, 4, 2])

with c1:
    pilih_kategori = st.multiselect(
        "Jenis Pelanggan",
        options=kategori_opsi,
        default=kategori_opsi[:1] if len(kategori_opsi) else None
    )

with c2:
    pilih_produk = st.multiselect(
        "Nama Produk",
        options=produk_opsi,
        default=produk_opsi[:1] if len(produk_opsi) else None
    )

with c3:
    st.write("")  # spacer
    st.write("")
    tombol = st.button("Search", use_container_width=True)

# -----------------------------
# [HASIL]
# -----------------------------
st.markdown('<div class="section-title">Hasil</div>', unsafe_allow_html=True)

# Gabungkan transaksi dengan hasil cluster agar bisa filter produk + tampil strategi/rekomendasi
# Kita ambil produk per customer dari transaksi (unik) agar tidak terlalu duplikat
cust_produk = (
    trx[["kode_perusahaan_id", "nama_produk"]]
    .dropna()
    .drop_duplicates()
)

# Join: customer + cluster + strategi + produk
result = rfm.merge(cust_produk, on="kode_perusahaan_id", how="left")

# Terapkan filter ketika tombol ditekan (atau langsung realtime — di sini saya realtime juga)
filtered = result.copy()

if pilih_kategori:
    filtered = filtered[filtered["Kategori_Bisnis"].isin(pilih_kategori)]

if pilih_produk:
    filtered = filtered[filtered["nama_produk"].isin(pilih_produk)]

# Jika ingin hasil muncul hanya saat klik Search, uncomment:
# if not tombol:
#     st.info("Pilih filter lalu klik Search.")
#     st.stop()

# Rapikan kolom tampilan (sesuai contoh tabel)
# Catatan: kolom "nama" tidak ada di data Anda (yang Anda sebutkan), jadi saya tampilkan kode_perusahaan_id.
# Jika Anda punya kolom nama perusahaan, tinggal join dan tampilkan.
view_cols = [
    "kode_perusahaan_id",
    "marketing_id",
    "nama_produk",
    "Kategori_Bisnis",
    "Cluster",
    "Strategi_Rekomendasi"
]
view_cols = [c for c in view_cols if c in filtered.columns]

st.caption(f"Menampilkan {len(filtered):,} data (hasil filter).")

st.dataframe(
    filtered[view_cols].sort_values(["Kategori_Bisnis", "Cluster"], ascending=[True, True]),
    use_container_width=True,
    hide_index=True
)