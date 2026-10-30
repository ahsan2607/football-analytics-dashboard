"""Tema visual bersama untuk seluruh grafik Plotly di EuroScout AI.

Semua modul (attacking, playmaking, defensive, recommender) WAJIB memakai
PLOTLY_TEMPLATE, COLORS, dan COLOR_SEQ dari file ini agar tampilan dashboard
konsisten.
"""

import plotly.graph_objects as go
import plotly.io as pio

COLORS = {
    "background": "#0E1117",
    "surface": "#161B22",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "grid": "#30363D",
    "primary": "#2DD4BF",
    "accent": "#F59E0B",
    "positive": "#4ADE80",
    "negative": "#F87171",
    # warna tetap per posisi, dipakai di seluruh modul
    "FW": "#F87171",
    "MF": "#2DD4BF",
    "DF": "#60A5FA",
    "GK": "#FBBF24",
}

# urutan warna kategorikal untuk grafik multi-seri (liga, tim, pemain, dll)
COLOR_SEQ = [
    "#2DD4BF", "#F59E0B", "#60A5FA", "#F87171",
    "#A78BFA", "#4ADE80", "#FB923C", "#38BDF8",
]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], family="Inter, Segoe UI, sans-serif", size=13),
        colorway=COLOR_SEQ,
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=COLORS["surface"], font_size=13),
    )
)

pio.templates["euroscout"] = PLOTLY_TEMPLATE
