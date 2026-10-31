
"""Anggota 1 — Attacking Analytics & Finishing.

TODO: Gol vs xG (efisiensi finishing), pemain paling produktif, grafik Plotly.
Kontrak: satu fungsi render(df) yang mengisi seluruh isi tab ini. Gunakan
PLOTLY_TEMPLATE, COLORS, COLOR_SEQ dari utils/theme.py untuk semua grafik, dan
tutup dengan kotak Key Insight berisi kesimpulan tertulis.
""

Analisis:
- Goals vs Expected Goals (xG)
- Finishing efficiency: Goals - xG
- Shot-Creating Actions (SCA) per 90 minutes

render(df) menerima DataFrame yang sudah difilter oleh app.py.
Modul ini tidak membaca CSV sendiri dan tidak mengubah DataFrame asli.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.theme import COLORS, COLOR_SEQ, PLOTLY_TEMPLATE


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Ambil kolom numerik; jika kolom tidak ada, gunakan nilai default."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Pembagian aman untuk menghindari pembagian dengan nol."""
    return numerator.div(denominator.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Siapkan metrik attacking tanpa memodifikasi DataFrame dari app.py."""
    data = df.copy().reset_index(drop=True)

    for col in ["goals", "xg", "npxg", "shots", "shots_on_target", "sca", "minutes"]:
        data[col] = _numeric(data, col)

    if "nineties" in data.columns:
        data["nineties"] = _numeric(data, "nineties")
    else:
        data["nineties"] = data["minutes"] / 90.0

    defaults = {
        "player": "Unknown player",
        "team": "—",
        "league": "—",
        "pos": "Other"
    }

    for col, fallback in defaults.items():
        if col not in data.columns:
            data[col] = fallback
        else:
            data[col] = data[col].fillna(fallback).astype(str)

    data["finishing_delta"] = data["goals"] - data["xg"]
    data["goals_per90"] = _safe_divide(data["goals"], data["nineties"])
    data["xg_per90"] = _safe_divide(data["xg"], data["nineties"])
    data["shots_per90"] = _safe_divide(data["shots"], data["nineties"])

    data["shot_accuracy_pct"] = _safe_divide(
        data["shots_on_target"] * 100, data["shots"]
    )

    calculated_sca_per90 = _safe_divide(
        data["sca"], data["nineties"]
    )

    if "sca_per90" in data.columns:
        supplied = pd.to_numeric(data["sca_per90"], errors="coerce")
        data["sca_per90"] = supplied.fillna(calculated_sca_per90)
    else:
        data["sca_per90"] = calculated_sca_per90

    return data


def _player_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "player", "team", "pos", "goals", "xg",
        "finishing_delta", "shots",
        "shot_accuracy_pct", "sca_per90",
    ]

    table = frame[columns].copy().rename(columns={
        "player": "Player",
        "team": "Team",
        "pos": "Position",
        "goals": "Goals",
        "xg": "xG",
        "finishing_delta": "Goals − xG",
        "shots": "Shots",
        "shot_accuracy_pct": "Shot accuracy (%)",
        "sca_per90": "SCA / 90",
    })

    return table.round(2).reset_index(drop=True)


def _goals_vs_xg_chart(data: pd.DataFrame) -> None:
    """Grafik perbandingan gol aktual dengan expected goals."""

    chart_data = data.dropna(subset=["goals", "xg"]).copy()

    if chart_data.empty:
        st.info("Data Goals dan xG belum tersedia untuk filter ini.")
        return

    position_colors = {
        pos: COLORS.get(pos, COLOR_SEQ[i % len(COLOR_SEQ)])
        for i, pos in enumerate(chart_data["pos"].unique())
    }

    fig = px.scatter(
        chart_data,
        x="xg",
        y="goals",
        color="pos",
        color_discrete_map=position_colors,
        size="shots",
        size_max=19,
        hover_name="player",
        hover_data={
            "team": True,
            "league": True,
            "goals": ":.0f",
            "xg": ":.2f",
            "finishing_delta": ":+.2f",
            "shots": ":.0f",
            "shot_accuracy_pct": ":.1f",
            "sca_per90": ":.2f",
        },
        labels={
            "xg": "Expected Goals (xG)",
            "goals": "Goals scored",
            "pos": "Position"
        },
        title="Actual goals vs expected goals",
    )

    axis_max = max(
        float(chart_data["xg"].max()),
        float(chart_data["goals"].max()),
        1.0
    ) * 1.05

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=axis_max,
        y1=axis_max,
        line={
            "color": COLORS["muted"],
            "width": 1.5,
            "dash": "dash"
        },
    )

    fig.add_annotation(
        x=axis_max * 0.72,
        y=axis_max * 0.76,
        text="Goals = xG",
        showarrow=False,
        font={"color": COLORS["muted"]},
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=470,
        xaxis_range=[0, axis_max],
        yaxis_range=[0, axis_max],
        legend_title_text="Position",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


def _finishing_chart(data: pd.DataFrame) -> None:
    """Grafik efisiensi finishing berdasarkan Goals - xG."""

    if data.empty:
        st.info("Tidak ada pemain yang memenuhi minimum xG.")
        return

    top = data.nlargest(
        10, "finishing_delta"
    ).sort_values("finishing_delta")

    colors = [
        COLORS["positive"] if value >= 0 else COLORS["negative"]
        for value in top["finishing_delta"]
    ]

    fig = go.Figure(go.Bar(
        x=top["finishing_delta"],
        y=top["player"],
        orientation="h",
        marker_color=colors,
        customdata=np.column_stack([
            top["goals"],
            top["xg"]
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Goals − xG: %{x:+.2f}"
            "<br>Goals: %{customdata[0]:.0f}"
            "<br>xG: %{customdata[1]:.2f}"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(
        x=0,
        line_color=COLORS["muted"],
        line_width=1
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Finishing over/underperformance (Goals − xG)",
        xaxis_title="Goals − xG",
        yaxis_title="",
        height=430,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


def _sca_chart(data: pd.DataFrame) -> None:
    """Grafik ranking Shot-Creating Actions per 90 menit."""

    chart_data = data.dropna(
        subset=["sca_per90"]
    ).copy()

    if chart_data.empty or chart_data["sca_per90"].max() <= 0:
        st.info(
            "Data Shot-Creating Actions belum tersedia untuk filter ini."
        )
        return

    top = chart_data.nlargest(
        10, "sca_per90"
    ).sort_values("sca_per90")

    colors = [
        COLORS.get(pos, COLOR_SEQ[0])
        for pos in top["pos"]
    ]

    fig = go.Figure(go.Bar(
        x=top["sca_per90"],
        y=top["player"],
        orientation="h",
        marker_color=colors,
        customdata=np.column_stack([
            top["sca"],
            top["nineties"]
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SCA / 90: %{x:.2f}"
            "<br>Total SCA: %{customdata[0]:.0f}"
            "<br>90s played: %{customdata[1]:.1f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Top Shot-Creating Actions per 90 minutes",
        xaxis_title="SCA per 90 minutes",
        yaxis_title="",
        height=430,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


def render(df: pd.DataFrame) -> None:
    """Render seluruh analisis attacking pada tab dashboard Streamlit."""

    st.subheader("Attacking Analytics & Finishing")

    st.caption(
        "Analisis gol, kualitas peluang (xG), efisiensi finishing, "
        "volume tembakan, dan kontribusi penciptaan peluang. "
        "Hasil mengikuti filter global."
    )

    if df is None or df.empty:
        st.warning(
            "Tidak ada pemain yang cocok dengan filter saat ini. "
            "Coba turunkan minimum minutes atau pilih liga/posisi lain."
        )
        return

    if "goals" not in df.columns or "xg" not in df.columns:
        st.error(
            "Kolom wajib 'goals' dan/atau 'xg' tidak ditemukan. "
            "Periksa utils/data_loader.py."
        )
        return

    data = _prepare_data(df)

    total_goals = data["goals"].sum()
    total_xg = data["xg"].sum()
    total_shots = data["shots"].sum()
    total_sot = data["shots_on_target"].sum()
    total_nineties = data["nineties"].sum()
    total_sca = data["sca"].sum()

    shot_accuracy = (
        total_sot / total_shots * 100
        if total_shots > 0 else np.nan
    )

    overall_sca_per90 = (
        total_sca / total_nineties
        if total_nineties > 0 else np.nan
    )

    # KPI utama
    metrics = st.columns(4)

    metrics[0].metric(
        "Goals scored",
        f"{total_goals:,.0f}"
    )

    metrics[1].metric(
        "Expected Goals (xG)",
        f"{total_xg:,.1f}"
    )

    metrics[2].metric(
        "Goals − xG",
        f"{total_goals - total_xg:+.1f}",
        help=(
            "Positif berarti gol aktual lebih banyak dari "
            "ekspektasi xG; bukan bukti tunggal kemampuan finishing."
        ),
    )

    metrics[3].metric(
        "Shot accuracy",
        f"{shot_accuracy:.1f}%"
        if pd.notna(shot_accuracy) else "—",
        help="Total shots on target dibagi total shots.",
    )

    # 1. GOALS VS EXPECTED GOALS
    st.divider()

    st.markdown("#### 1. Actual Goals vs Expected Goals")

    st.caption(
        "Setiap titik mewakili satu pemain. Titik di atas garis "
        "putus-putus mencetak gol lebih banyak daripada xG; "
        "titik di bawah garis mencetak lebih sedikit. "
        "Ukuran titik menunjukkan jumlah tembakan."
    )

    _goals_vs_xg_chart(data)

    # 2. FINISHING EFFICIENCY
    st.divider()

    st.markdown("#### 2. Finishing Efficiency")

    max_xg = float(data["xg"].max())

    slider_max = max(
        5.0,
        min(20.0, float(np.ceil(max_xg * 2) / 2))
    )

    min_xg = st.slider(
        "Minimum xG untuk ranking finishing",
        min_value=0.0,
        max_value=slider_max,
        value=min(3.0, slider_max),
        step=0.5,
        help=(
            "Ambang xG mengurangi ranking yang terlalu "
            "dipengaruhi sampel kecil."
        ),
    )

    eligible = data[data["xg"] >= min_xg].copy()

    st.caption(
        f"{len(eligible)} pemain memenuhi ambang xG ≥ {min_xg:.1f}. "
        "Nilai positif berarti overperformance terhadap xG."
    )

    finish_col, table_col = st.columns([1.15, 1])

    with finish_col:
        _finishing_chart(eligible)

    with table_col:
        st.markdown("**Pemain dengan Goals − xG tertinggi**")

        if eligible.empty:
            st.info(
                "Turunkan ambang minimum xG untuk melihat ranking."
            )
        else:
            st.dataframe(
                _player_table(
                    eligible.nlargest(10, "finishing_delta")
                ),
                use_container_width=True,
                hide_index=True,
            )

    # 3. SHOT-CREATING ACTIONS
    st.divider()

    st.markdown("#### 3. Shot-Creating Actions")

    st.caption(
        "SCA menghitung aksi ofensif yang berujung pada tembakan. "
        "Nilai per 90 menit membantu membandingkan pemain "
        "dengan waktu bermain berbeda."
    )

    sca_col, note_col = st.columns([1.15, 1])

    with sca_col:
        _sca_chart(data)

    with note_col:
        creators = data.dropna(
            subset=["sca_per90"]
        )

        if not creators.empty and creators["sca_per90"].max() > 0:
            creator = creators.loc[
                creators["sca_per90"].idxmax()
            ]

            st.metric(
                "Highest SCA / 90",
                f"{creator['sca_per90']:.2f}"
            )

            st.write(
                f"**{creator['player']}** "
                f"({creator['team']}, {creator['pos']}) "
                f"mencatat {creator['sca']:.0f} "
                "Shot-Creating Actions dalam "
                f"{creator['nineties']:.1f} "
                "90-menit pertandingan."
            )
        else:
            st.info(
                "Belum ada catatan SCA pada pemain yang dipilih."
            )

    # KEY INSIGHT DINAMIS
    insights = []

    if not eligible.empty:
        best = eligible.loc[
            eligible["finishing_delta"].idxmax()
        ]

        worst = eligible.loc[
            eligible["finishing_delta"].idxmin()
        ]

        insights.append(
            f"Pemain dengan Goals − xG tertinggi adalah "
            f"**{best['player']}** ({best['team']}): "
            f"{best['goals']:.0f} gol dari {best['xg']:.1f} xG "
            f"(**{best['finishing_delta']:+.1f}** terhadap ekspektasi)."
        )

        insights.append(
            f"Selisih terendah pada sampel ini dimiliki "
            f"**{worst['player']}** ({worst['team']}) "
            f"dengan Goals − xG "
            f"**{worst['finishing_delta']:+.1f}**."
        )

    top_scorer = data.loc[
        data["goals"].idxmax()
    ]

    insights.append(
        f"Top scorer pada filter ini adalah "
        f"**{top_scorer['player']}** ({top_scorer['team']}) "
        f"dengan {top_scorer['goals']:.0f} gol "
        f"dari {top_scorer['xg']:.1f} xG."
    )

    if pd.notna(overall_sca_per90):
        insights.append(
            f"Rata-rata tertimbang SCA pada sampel adalah "
            f"**{overall_sca_per90:.2f} per 90 menit**."
        )

    st.divider()

    st.info(
        "**Key Insight**\n\n" + " ".join(insights)
    )

    st.caption(
        "Catatan: Goals − xG dapat dipengaruhi variasi, penalti, "
        "kualitas model xG, dan ukuran sampel. Gunakan bersama "
        "menit bermain, jumlah tembakan, serta konteks pertandingan."
    )
