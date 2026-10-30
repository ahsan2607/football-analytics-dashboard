"""Anggota 4 — ML Scouting & Similarity.

Fitur "cari pemain mirip [Player X]" lintas liga:
- Fitur dinormalisasi per-90 menit lalu di-scale (StandardScaler).
- Kemiripan dihitung dengan Cosine Similarity antar pemain pada posisi yang
  sama (FW/MF/DF/GK dibandingkan dalam kelompoknya masing-masing karena
  profil statistiknya tidak sebanding lintas posisi).
- K-Means dipakai untuk mengelompokkan pemain dalam posisi yang sama menjadi
  beberapa "tipe peran" dan memberi konteks tambahan di luar daftar mirip.
- Model & similarity matrix dibungkus @st.cache_resource agar tidak dihitung
  ulang setiap klik.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from utils.theme import COLORS, COLOR_SEQ, PLOTLY_TEMPLATE

# fitur per kelompok posisi — semua dihitung per-90 menit sebelum di-scale
POSITION_FEATURES = {
    "FW": ["goals", "xg", "npxg", "shots", "shots_on_target", "sca", "gca", "prog_carries", "key_passes"],
    "MF": ["xag", "key_passes", "prog_passes", "prog_carries", "sca", "gca", "tackles", "interceptions"],
    "DF": ["tackles", "interceptions", "clearances", "recoveries", "prog_passes", "prog_carries", "tackles_interceptions"],
    "GK": ["saves", "save_pct", "cs_pct", "ga90", "psxg"],
}

MIN_NINETIES_FOR_SIMILARITY = 5  # ~450 menit, ambang lebih longgar dari filter global


def _per90(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    nineties = df["nineties"].replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            out[col] = 0.0
            continue
        if col.endswith("_pct") or col.startswith("ga") or col == "psxg":
            out[col] = df[col]  # sudah berupa rasio/rata-rata, tidak perlu dibagi menit
        else:
            out[col] = (df[col] / nineties).fillna(0.0)
    return out.fillna(0.0)


@st.cache_resource
def _build_similarity(df: pd.DataFrame, pos: str):
    """Scaler + matriks cosine similarity untuk satu kelompok posisi."""
    features = POSITION_FEATURES[pos]
    subset = df[df["pos"] == pos].copy()
    subset = subset[subset["nineties"] >= MIN_NINETIES_FOR_SIMILARITY]
    if len(subset) < 2:
        return subset, None, None, features

    feat_df = _per90(subset, features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_df.values)
    sim_matrix = cosine_similarity(scaled)

    n_clusters = min(5, len(subset))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    subset = subset.reset_index(drop=True)
    subset["cluster"] = kmeans.fit_predict(scaled)

    return subset, sim_matrix, scaled, features


def _radar_chart(subset, feat_df, idx_a, idx_b, label_a, label_b):
    cats = feat_df.columns.tolist()
    row_a = feat_df.iloc[idx_a]
    row_b = feat_df.iloc[idx_b]
    max_vals = feat_df.max().replace(0, 1)

    norm_a = (row_a / max_vals * 100).tolist()
    norm_b = (row_b / max_vals * 100).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_a + [norm_a[0]], theta=cats + [cats[0]],
        fill="toself", name=label_a, line_color=COLOR_SEQ[0],
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_b + [norm_b[0]], theta=cats + [cats[0]],
        fill="toself", name=label_b, line_color=COLOR_SEQ[3],
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=True,
        height=420,
        title="Perbandingan profil statistik (relatif terhadap pemain sekelompok posisi, per-90 menit)",
    )
    return fig


def render(df):
    st.subheader("ML Scouting & Similarity")
    st.caption(
        "Cari pemain dengan profil statistik paling mirip — dihitung per posisi, "
        "dinormalisasi per-90 menit, lintas liga."
    )

    available_pos = [p for p in ["FW", "MF", "DF", "GK"] if (df["pos"] == p).sum() >= 2]
    if not available_pos:
        st.warning("Data pemain pada filter saat ini terlalu sedikit untuk analisis kemiripan.")
        return

    col_pos, col_player = st.columns([1, 2])
    with col_pos:
        pos_choice = st.selectbox("Posisi", available_pos)

    subset, sim_matrix, scaled, features = _build_similarity(df, pos_choice)

    if sim_matrix is None:
        st.warning(f"Tidak cukup pemain posisi {pos_choice} (menit main ≥ {MIN_NINETIES_FOR_SIMILARITY * 90}) pada filter saat ini.")
        return

    player_options = subset["player"] + " (" + subset["team"] + ", " + subset["league"] + ")"
    with col_player:
        selected_label = st.selectbox("Pemain acuan", player_options.tolist())
    selected_idx = player_options.tolist().index(selected_label)

    top_n = st.slider("Jumlah rekomendasi", 3, 15, 5)

    sims = sim_matrix[selected_idx]
    order = np.argsort(-sims)
    order = [i for i in order if i != selected_idx][:top_n]

    result = subset.iloc[order][["player", "team", "league", "nineties"]].copy()
    result["similarity"] = [round(float(sims[i]), 3) for i in order]
    result = result.rename(columns={
        "player": "Pemain", "team": "Tim", "league": "Liga",
        "nineties": "90 Menit Dimainkan", "similarity": "Skor Kemiripan",
    })
    st.dataframe(result, use_container_width=True, hide_index=True)

    if order:
        feat_df = _per90(subset, features)
        compare_idx = order[0]
        fig = _radar_chart(
            subset, feat_df, selected_idx, compare_idx,
            subset.iloc[selected_idx]["player"], subset.iloc[compare_idx]["player"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tipe peran (K-Means cluster) pada posisi ini"):
        cluster_counts = subset["cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

    top_match = subset.iloc[order[0]]["player"] if order else None
    st.markdown("**Key Insight**")
    if top_match:
        st.info(
            f"Pemain dengan profil statistik paling mirip **{selected_label}** pada posisi {pos_choice} "
            f"adalah **{top_match}** (skor kemiripan {result.iloc[0]['Skor Kemiripan']}). "
            "Gunakan fitur ini untuk mencari pengganti pemain lintas liga dengan gaya main sebanding."
        )
