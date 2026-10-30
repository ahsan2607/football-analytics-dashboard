"""Anggota 4 — ML Scouting & Similarity.

TODO: Cosine Similarity / K-Means (Scikit-Learn), fitur "cari pemain mirip
[Player X]" lintas liga. Bungkus model dengan @st.cache_resource agar scaler
dan matriks similarity tidak dihitung ulang tiap klik. Normalisasi fitur
per-90 menit dan per posisi.
Kontrak: satu fungsi render(df) yang mengisi seluruh isi tab ini. Gunakan
PLOTLY_TEMPLATE, COLORS, COLOR_SEQ dari utils/theme.py untuk semua grafik, dan
tutup dengan kotak Key Insight berisi kesimpulan tertulis.
"""

import streamlit as st


def render(df):
    st.subheader("ML Scouting & Similarity")
    st.info("Modul ini belum diisi — dikerjakan oleh Anggota 4.")
