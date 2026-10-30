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

# label yang dibaca manusia untuk tiap nama kolom fitur mentah, dipakai di
# sumbu radar chart dan ringkasan cluster (menghindari singkatan FBref mentah)
FEATURE_LABELS = {
    "goals": "Goals",
    "xg": "Expected Goals (xG)",
    "npxg": "Non-Penalty xG",
    "shots": "Shots",
    "shots_on_target": "Shots on Target",
    "sca": "Shot-Creating Actions",
    "gca": "Goal-Creating Actions",
    "prog_carries": "Progressive Carries",
    "prog_passes": "Progressive Passes",
    "key_passes": "Key Passes",
    "xag": "Expected Assisted Goals (xAG)",
    "tackles": "Tackles",
    "interceptions": "Interceptions",
    "tackles_interceptions": "Tackles + Interceptions",
    "clearances": "Clearances",
    "recoveries": "Ball Recoveries",
    "saves": "Saves",
    "save_pct": "Save %",
    "cs_pct": "Clean Sheet %",
    "ga90": "Goals Against / 90",
    "psxg": "Post-Shot xG (PSxG)",
}


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
    cats = [FEATURE_LABELS.get(c, c) for c in feat_df.columns.tolist()]
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
        title="Stat profile comparison (relative to players in the same position group, per 90 minutes)",
    )
    return fig


def _describe_clusters(subset, scaled, features, top_k=2):
    """Untuk tiap cluster, cari fitur dengan rata-rata z-score tertinggi agar
    pengguna tahu gaya main apa yang menonjol — bukan cuma label angka 0-4."""
    scaled_df = pd.DataFrame(scaled, columns=features)
    scaled_df["cluster"] = subset["cluster"].values
    cluster_means = scaled_df.groupby("cluster")[features].mean()

    rows = []
    counts = subset["cluster"].value_counts()
    for cluster_id, row in cluster_means.iterrows():
        top_features = row.sort_values(ascending=False).head(top_k).index.tolist()
        traits = ", ".join(FEATURE_LABELS.get(f, f) for f in top_features)
        rows.append({
            "Cluster": int(cluster_id),
            "Players": int(counts.get(cluster_id, 0)),
            "Stands out in": traits,
        })
    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)


def render(df):
    st.subheader("ML Scouting & Similarity")
    st.caption(
        "Find players with the closest statistical profile — computed per position, "
        "normalized per 90 minutes, across leagues."
    )

    available_pos = [p for p in ["FW", "MF", "DF", "GK"] if (df["pos"] == p).sum() >= 2]
    if not available_pos:
        st.warning("Not enough players under the current filters for similarity analysis.")
        return

    col_pos, col_player = st.columns([1, 2])
    with col_pos:
        pos_choice = st.selectbox("Position", available_pos)

    subset, sim_matrix, scaled, features = _build_similarity(df, pos_choice)

    if sim_matrix is None:
        st.warning(f"Not enough {pos_choice} players (minutes ≥ {MIN_NINETIES_FOR_SIMILARITY * 90}) under the current filters.")
        return

    player_options = subset["player"] + " (" + subset["team"] + ", " + subset["league"] + ")"
    with col_player:
        selected_label = st.selectbox("Reference player", player_options.tolist())
    selected_idx = player_options.tolist().index(selected_label)

    top_n = st.slider("Number of recommendations", 3, 15, 5)

    sims = sim_matrix[selected_idx]
    order = np.argsort(-sims)
    order = [i for i in order if i != selected_idx][:top_n]

    result = subset.iloc[order][["player", "team", "league", "nineties"]].copy()
    result["similarity"] = [round(float(sims[i]), 3) for i in order]
    result = result.rename(columns={
        "player": "Player", "team": "Team", "league": "League",
        "nineties": "Full Matches Played (90-min equiv.)", "similarity": "Similarity Score",
    })
    st.dataframe(result, use_container_width=True, hide_index=True)

    if order:
        feat_df = _per90(subset, features)
        compare_labels = [player_options.tolist()[i] for i in order]
        compare_label = st.selectbox("Compare reference player against", compare_labels)
        compare_idx = order[compare_labels.index(compare_label)]
        fig = _radar_chart(
            subset, feat_df, selected_idx, compare_idx,
            subset.iloc[selected_idx]["player"], subset.iloc[compare_idx]["player"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Playing styles (K-Means clusters) for this position"):
        st.caption(
            "K-Means grouped players at this position into clusters purely from their "
            "statistical profile (no inherent names — cluster IDs are arbitrary). "
            "Each row below shows how many players fall into the cluster and which "
            "stats stand out the most for that group."
        )
        cluster_summary = _describe_clusters(subset, scaled, features)
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
        st.bar_chart(subset["cluster"].value_counts().sort_index())

    top_match = subset.iloc[order[0]]["player"] if order else None
    st.markdown("**Key Insight**")
    if top_match:
        st.info(
            f"The player with the closest statistical profile to **{selected_label}** at {pos_choice} "
            f"is **{top_match}** (similarity score {result.iloc[0]['Similarity Score']}). "
            "Use this to find a cross-league replacement with a comparable playing style."
        )
