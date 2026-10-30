"""Pemuat & pembersih data pemain untuk EuroScout AI.

Sumber: data/players_data-2024_2025_cleansing.csv (FBref, Big 5 liga Eropa
2024-2025). Kolom asli adalah gabungan beberapa tabel statistik FBref
(standard, shooting, passing, passing_types, gca, defense, possession,
playing_time, misc, keeper, keeper_adv), sehingga banyak nama kolom berulang
yang sudah dibedakan lewat suffix (mis. `Tkl` vs `Tkl_stats_defense`).

load_data() melakukan:
1. Membuang kolom index mentah (`source_row`, `Rk`).
2. Menormalkan SEMUA nama kolom ke snake_case huruf kecil
   (mis. `xAG` -> `xag`, `Tkl+Int` -> `tkl_int`, `SoT%` -> `sot_pct`).
3. Mengganti nama kolom inti ke istilah standar lintas modul lewat RENAME_MAP
   (player, team, league, pos, minutes, goals, assists, xg, xag, dst).
4. Menambahkan `pos`: penyederhanaan `Pos` asli (bisa multi-posisi, mis.
   "MF,FW") menjadi satu label utama FW / MF / DF / GK.
5. Menambahkan `league`: pembersihan `Comp` asli (mis. "eng Premier League"
   -> "Premier League").

Modul lain TIDAK BOLEH membaca CSV sendiri — selalu panggil load_data().
"""

import re

import pandas as pd
import streamlit as st

DATA_PATH = "data/players_data-2024_2025_cleansing.csv"

# kolom asli (setelah snake_case) -> nama standar yang dipakai di seluruh modul
RENAME_MAP = {
    "player": "player",
    "nation": "nation",
    "squad": "team",
    "comp": "league_raw",
    "pos": "pos_raw",
    "age": "age",
    "min": "minutes",
    "90s": "nineties",
    "gls": "goals",
    "ast": "assists",
    "g_a": "goals_assists",
    "xg": "xg",
    "npxg": "npxg",
    "xag": "xag",
    "npxg_xag": "npxg_xag",
    "sh": "shots",
    "sot": "shots_on_target",
    "prgc": "prog_carries",
    "prgp": "prog_passes",
    "prgr": "prog_receptions",
    "kp": "key_passes",
    "sca": "sca",
    "sca90": "sca_per90",
    "gca": "gca",
    "gca90": "gca_per90",
    "tkl": "tackles",
    "int": "interceptions",
    "tkl_int": "tackles_interceptions",
    "clr": "clearances",
    "err": "errors",
    "recov": "recoveries",
    "touches": "touches",
}

_POS_MAP = {"FW": "FW", "MF": "MF", "DF": "DF", "GK": "GK"}


def _snake_case(col: str) -> str:
    col = col.strip()
    col = col.replace("%", "_pct").replace("+", "_").replace("/", "_")
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col.lower()


def _simplify_pos(raw) -> str:
    """Ambil posisi utama dari nilai Pos asli (mis. 'MF,FW' -> 'MF')."""
    if not isinstance(raw, str) or not raw:
        return "DF"
    first = raw.split(",")[0].strip()
    return _POS_MAP.get(first, "DF")


def _simplify_league(raw):
    """Buang kode negara di depan nama liga (mis. 'eng Premier League' -> 'Premier League')."""
    if not isinstance(raw, str):
        return raw
    parts = raw.split(" ", 1)
    return parts[1] if len(parts) == 2 else raw


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.drop(columns=[c for c in ("source_row", "Rk") if c in df.columns])

    df.columns = [_snake_case(c) for c in df.columns]
    df = df.rename(columns=RENAME_MAP)

    df["pos"] = df["pos_raw"].apply(_simplify_pos)
    df["league"] = df["league_raw"].apply(_simplify_league)

    numeric_cols = [
        "minutes", "nineties", "goals", "assists", "xg", "npxg", "xag",
        "prog_carries", "prog_passes", "prog_receptions", "tackles",
        "interceptions", "recoveries",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
