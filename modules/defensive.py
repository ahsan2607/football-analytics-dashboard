"""Defensive activity within the globally filtered player–team population."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.theme import COLORS, COLOR_SEQ, PLOTLY_TEMPLATE


METRICS = {
    "tackles_interceptions": "Tackles + interceptions",
    "tackles": "Tackles",
    "interceptions": "Interceptions",
    "recoveries": "Recoveries",
    "blocks_stats_defense": "Defensive blocks",
    "clearances": "Clearances",
}
ZONES = {"def_3rd": "Defensive third", "mid_3rd": "Middle third", "att_3rd": "Attacking third"}
IDENTITY = ["player", "team", "league", "pos", "minutes"]
COUNTS = list(METRICS) + list(ZONES) + [
    "tklw", "errors", "att_stats_defense", "lost", "won", "lost_stats_misc",
]
PERCENTAGES = ["tkl_pct", "won_pct"]
SMALL_SAMPLE_MINUTES = 900  


def _prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate a local copy; keep unknown events unknown and retain club stints."""
    missing = [c for c in IDENTITY if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names prevent safe metric selection.")
    data = df.copy(deep=True).reset_index(drop=True)
    data["record_id"] = np.arange(len(data)) + 1
    notes = []
    for col in IDENTITY[:-1]:
        data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
    unknown = data[IDENTITY[:-1]].isna().any(axis=1)
    if unknown.any():
        notes.append(f"{int(unknown.sum())} records have incomplete identity; labels show Unknown.")
    for col in IDENTITY[:-1]:
        data[col] = data[col].fillna("Unknown")
    data["pos"] = data["pos"].str.split(",").str[0].str.strip().str.upper()
    
    totals = data["team"].str.fullmatch(
        r"(?i)(?:\d+\s+(?:squads?|clubs?|teams?)|(?:all\s+)?(?:squads|clubs|teams)|total)", na=False,
    )
    gk = data["pos"].eq("GK")
    if totals.any():
        notes.append(f"{int(totals.sum())} explicitly marked multi-team totals excluded to avoid double counting.")
    if gk.any():
        notes.append(f"{int(gk.sum())} goalkeeper records excluded; analysis is outfield only.")
    data = data.loc[~(totals | gk)].copy()
    absent = [c for c in METRICS if c not in data.columns]
    if absent:
        notes.append("Absent source metrics: " + ", ".join(METRICS[c] for c in absent) + ".")
    for col in ["minutes"] + COUNTS + PERCENTAGES:
        original = data[col] if col in data else pd.Series(np.nan, index=data.index)
        numeric = pd.to_numeric(original, errors="coerce").astype(float)
        valid = np.isfinite(numeric) & (numeric >= 0)
        if col == "minutes":
            valid &= numeric > 0
        elif col in PERCENTAGES:
            valid &= numeric <= 100
        invalid = original.notna() & ~valid
        if invalid.any():
            notes.append(f"{col}: {int(invalid.sum())} invalid values treated as unavailable.")
        data[col] = numeric.where(valid)
    components = data["tackles"] + data["interceptions"]
    supplied = data["tackles_interceptions"]
    mismatch = supplied.notna() & components.notna() & ~np.isclose(supplied, components)
    if mismatch.any():
        notes.append(f"{int(mismatch.sum())} supplied tackles + interceptions totals disagree with components; supplied values retained.")
    fill = supplied.isna() & components.notna()
    data["tackles_interceptions"] = supplied.fillna(components)
    if fill.any():
        notes.append(f"{int(fill.sum())} tackles + interceptions totals derived from two valid components.")
    zone_sum = data[list(ZONES)].sum(axis=1, min_count=3)
    mismatch = zone_sum.notna() & data["tackles"].notna() & ~np.isclose(zone_sum, data["tackles"])
    if mismatch.any():
        notes.append(f"{int(mismatch.sum())} zone sums differ from tackles; zone shares use the sum of recorded zones.")
    for col in METRICS:
        data[col + "_per90"] = (data[col] * 90 / data["minutes"]).replace([np.inf, -np.inf], np.nan)
    data["label"] = (data["player"] + " · " + data["team"] + " · " + data["league"]
                     + " [#" + data["record_id"].astype(str) + "]")
    return data, notes


def _pooled_per90(data: pd.DataFrame, metric: str) -> tuple[float, int]:
    """Pool events and minutes from exactly the same valid observations."""
    valid = data.dropna(subset=[metric, "minutes"])
    if valid.empty:
        return np.nan, 0
    return float(valid[metric].sum() * 90 / valid["minutes"].sum()), len(valid)


def _rank(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data.dropna(subset=[column]).sort_values(
        [column, "player", "team", "league", "record_id"],
        ascending=[False, True, True, True, True], kind="stable",
    )


def _table(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["record_id"] + IDENTITY
    if "pos_raw" in data:
        columns.append("pos_raw")
    columns += [c for metric in METRICS for c in (metric, metric + "_per90")]
    return data[columns].rename(columns={c: c + "_total" for c in METRICS})


def _show_chart(fig: go.Figure, key: str, height: int = 450) -> None:
    fig.update_layout(template=PLOTLY_TEMPLATE, height=height, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def _comparison(data: pd.DataFrame, selected: list[int]) -> None:
    chosen = data[data["record_id"].isin(selected)]
    if chosen.empty:
        st.info("Choose up to three player–team records to compare.")
        return
    metrics = list(METRICS)[1:]
    long = chosen.melt(
        id_vars=["label", "minutes", "pos"],
        value_vars=[c + "_per90" for c in metrics], var_name="metric", value_name="per90",
    ).dropna(subset=["per90"])
    long["metric"] = long["metric"].map({c + "_per90": METRICS[c] for c in metrics})
    if not long.empty:
        fig = px.bar(
            long, x="metric", y="per90", color="label", barmode="group",
            color_discrete_sequence=COLOR_SEQ, hover_data={"minutes": ":.0f", "pos": True},
            labels={"metric": "", "per90": "Actions per 90 minutes", "label": "Player–team"},
            title="How do the selected activity profiles differ?",
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        _show_chart(fig, "def_comparison", 510)
    else:
        st.info("Per-90 comparison is unavailable for these records.")
    st.dataframe(_table(chosen).round(2), hide_index=True, key="def_selected_table")
    references = []
    for pos in sorted(chosen["pos"].unique()):
        peers = data[data["pos"].eq(pos)]
        for metric in metrics:
            value, n = _pooled_per90(peers, metric)
            references.append({"Primary position": pos, "Activity": METRICS[metric],
                               "Pooled actions_per90": value, "Valid records": n})
    st.caption("Same-primary-position context within the current global filters, including selected records. "
               "Pooled rates weight by valid minutes; small peer groups are descriptive, not benchmarks of quality.")
    st.dataframe(pd.DataFrame(references).round(2), hide_index=True, key="def_peers")


def render(df: pd.DataFrame) -> None:
    """Render using only the globally filtered input; never mutate it."""
    st.subheader("Defensive Activity & Workrate")
    st.caption("Workrate here means recorded defensive activity, not physical exertion, running or pressures. "
               "All views respect the global league, position and minimum-minute filters.")
    if df is None or df.empty:
        st.info("No records match the global filters. Adjust league, position or minimum minutes.")
        return
    try:
        data, notes = _prepare_data(df)
    except ValueError as exc:
        st.error(str(exc))
        return
    if data.empty:
        st.info("No outfield player–team records remain. Select an outfield position in the global filters.")
        if notes:
            st.caption(" ".join(notes))
        return
    minutes = data["minutes"].dropna()
    minute_range = f"{minutes.min():,.0f}–{minutes.max():,.0f} valid minutes" if not minutes.empty else "minutes unavailable"
    st.caption(f"{len(data):,} player–team records · {data['player'].nunique():,} distinct player names "
               f"(not verified unique people) · {minute_range}. "
               f"Leagues: {', '.join(sorted(data['league'].unique()))}. "
               f"Primary positions: {', '.join(sorted(data['pos'].unique()))}. "
               "Club stints remain separate; record IDs refer to rows in this filtered input.")
    small = int(data["minutes"].lt(SMALL_SAMPLE_MINUTES).sum())
    unknown_minutes = int(data["minutes"].isna().sum())
    st.caption(f"Sample context: {small:,} records below {SMALL_SAMPLE_MINUTES:,} minutes; "
               f"{unknown_minutes:,} with unavailable/invalid minutes. No extra minute threshold is applied. "
               "Short samples can produce unstable per-90 rates. Goalkeepers are excluded.")
    with st.expander("Data availability and validation"):
        for note in notes:
            st.write(note)
        availability = pd.DataFrame({
            "Metric": list(METRICS.values()),
            "Unavailable totals": [int(data[c].isna().sum()) for c in METRICS],
            "Unavailable per90": [int(data[c + '_per90'].isna().sum()) for c in METRICS],
        })
        st.dataframe(availability, hide_index=True, key="def_availability")
        st.caption("Unavailable values are excluded only from views requiring that value; they are never filled with zero.")

    controls = st.columns([2, 1, 1])
    metric = controls[0].selectbox("Ranking activity", list(METRICS), format_func=METRICS.get, key="def_metric")
    mode = controls[1].radio("Ranking units", ["Per 90", "Total"], horizontal=True, key="def_mode")
    top_n = controls[2].selectbox("Top N", [10, 15, 20, 30], key="def_top_n")
    column = metric + "_per90" if mode == "Per 90" else metric
    unit = "actions per 90 minutes" if mode == "Per 90" else "actions (total)"
    ranked = _rank(data, column)
    top = ranked.head(top_n)
    pooled, valid_n = _pooled_per90(data, metric)
    kpis = st.columns(4)
    kpis[0].metric("Outfield records", f"{len(data):,}", help="Player–team rows after global filters and explicit total-row exclusion.")
    kpis[1].metric("Median minutes", f"{minutes.median():,.0f}" if not minutes.empty else "N/A", help=f"Across {len(minutes):,} records with valid minutes.")
    total = data[metric].sum(min_count=1)
    kpis[2].metric(f"{METRICS[metric]} · total", f"{total:,.0f}" if pd.notna(total) else "N/A",
                   help=f"Actions across {data[metric].notna().sum():,} records with valid counts, including records without valid minutes.")
    kpis[3].metric("Pooled activity / 90", f"{pooled:.2f}" if pd.notna(pooled) else "N/A",
                   help=f"{METRICS[metric]}: sum of events × 90 / sum of minutes from the same {valid_n:,} valid records.")

    st.markdown("#### Who records the most defensive activity?")
    st.caption(f"{len(ranked):,} eligible records for {METRICS[metric].lower()} in {mode.lower()} units; "
               f"showing {len(top):,}. Ties use player, team, league and row ID order. Activity is not defensive quality.")
    if top.empty:
        st.info("This statistic is unavailable for the current population and unit. Choose another activity or Total.")
    else:
        fig = px.bar(
            top, x=column, y="label", orientation="h", text=column,
            hover_data={"minutes": ":.0f", "pos": True, "team": True, "league": True, column: ":.2f"},
            labels={column: f"{METRICS[metric]} · {unit}", "label": ""},
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig.update_traces(texttemplate="%{x:.2f}" if mode == "Per 90" else "%{x:.0f}")
        fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=top["label"].tolist())
        _show_chart(fig, "def_ranking", max(380, 30 * len(top) + 100))

    st.markdown("#### How do ball-winning activity and recoveries combine?")
    scatter = data.dropna(subset=["tackles_interceptions_per90", "recoveries_per90", "minutes"])
    if scatter.empty:
        st.info("The profile chart needs valid minutes, tackles + interceptions and recoveries.")
    else:
        fig = px.scatter(
            scatter, x="tackles_interceptions_per90", y="recoveries_per90", color="pos", symbol="pos",
            size="minutes", size_max=27, color_discrete_map=COLORS, hover_name="label",
            hover_data={"minutes": ":.0f", "team": True, "league": True,
                        "tackles_interceptions_per90": ":.2f", "recoveries_per90": ":.2f"},
            labels={"tackles_interceptions_per90": "Tackles + interceptions per 90 minutes",
                    "recoveries_per90": "Recoveries per 90 minutes", "pos": "Primary position"},
        )
        fig.update_traces(marker=dict(sizemin=5, opacity=0.75))
        fig.add_vline(x=float(scatter["tackles_interceptions_per90"].median()), line_dash="dash", line_color=COLORS["muted"])
        fig.add_hline(y=float(scatter["recoveries_per90"].median()), line_dash="dash", line_color=COLORS["muted"])
        _show_chart(fig, "def_scatter")
        st.caption(f"{len(scatter):,} complete records. Dashed lines are medians of the displayed population. "
                   "Upper-right means above-median activity on both axes, not better overall defending or a causal relationship. "
                   "Marker area reflects minutes; the smallest markers have a readability floor.")

    st.markdown("#### Where on the pitch do tackles occur?")
    zone_mode = st.radio("Tackle-zone units", ["Share (%)", "Total"], horizontal=True, key="def_zone_mode")
    zones = top.dropna(subset=list(ZONES)).copy()
    sums = zones[list(ZONES)].sum(axis=1)
    if zone_mode == "Share (%)":
        zones = zones.loc[sums.gt(0)].copy()
        zones[list(ZONES)] = zones[list(ZONES)].div(sums.loc[zones.index], axis=0) * 100
    st.caption(f"Uses the ranking's player–team records: {len(zones)} of {len(top)} have complete zones "
               "and, for shares, a positive zone sum. These are tackle zones, not touches, pressures or precise event locations.")
    if zones.empty:
        st.info("Tackle-zone distribution is unavailable. Missing zones and zero-total shares are not plotted as zero.")
    else:
        fig = go.Figure()
        for i, (col, label) in enumerate(ZONES.items()):
            fig.add_bar(x=zones[col], y=zones["label"], name=label, orientation="h",
                        marker=dict(color=COLOR_SEQ[i], pattern_shape=["", "/", "."][i]),
                        hovertemplate="%{y}<br>" + label + ": %{x:.2f}" + ("%" if zone_mode == "Share (%)" else " tackles") + "<extra></extra>")
        fig.update_layout(barmode="stack", xaxis_title="Share of recorded zone tackles (%)" if zone_mode == "Share (%)" else "Tackles (total)",
                          legend=dict(orientation="h", y=1.15))
        fig.update_yaxes(autorange="reversed")
        _show_chart(fig, "def_zones", max(380, 30 * len(zones) + 100))

    st.markdown("#### Compare player–team profiles and position peers")
    options = data.sort_values(["player", "team", "league", "record_id"])["record_id"].tolist()
    labels = data.set_index("record_id")["label"].to_dict()
    selected = st.multiselect("Player–team records (maximum 3)", options, default=top["record_id"].head(2).tolist(),
                              format_func=labels.get, max_selections=3, key="def_players")
    _comparison(data, selected)

    st.markdown("#### Analysis table and download")
    scope = st.radio("Table and CSV scope", ["Ranking Top N", "All filtered outfield records"], horizontal=True, key="def_export_scope")
    table = _table(top if scope == "Ranking Top N" else data)
    st.caption("Table displays two decimals; CSV preserves calculation precision. N/A means unavailable. "
               "Totals are action counts; _per90 columns are actions per 90 minutes. Table and CSV contain the same rows.")
    st.dataframe(table.style.format(precision=2, na_rep="N/A"), hide_index=True, key="def_table")
    st.download_button("Download displayed analysis (CSV)", table.to_csv(index=False, na_rep="N/A").encode("utf-8-sig"),
                       file_name="euroscout_defensive_analysis.csv", mime="text/csv", key="def_download")

    with st.expander("Methodology and limitations"):
        st.markdown(
            "- **Per 90:** event total × 90 / valid minutes. Rounded `nineties` is never the denominator. "
            "Pooled rates use sums of events and minutes from exactly the same valid rows; percentages are not normalized.\n"
            "- **Missing values:** remain unavailable, including percentages when there were no attempts. Non-finite, "
            "negative counts, non-positive minutes and percentages outside 0–100 are unavailable. Valid extreme values remain.\n"
            "- **Combined activity:** missing tackles + interceptions is derived only when both components are valid. "
            "Supplied inconsistencies are reported, never overwritten. Zone shares use the sum of three valid zone counts.\n"
            "- **Population:** current global filters, outfield records only. One observation is a player–team row, "
            "not a unique person. Club stints and duplicate names are retained. Explicit multi-team totals are excluded; "
            "unmarked totals cannot reliably be identified. Same-position reference rates use primary positions within this population; "
            "they change with filters. No activity percentile or universal quality score is computed.\n"
            "- **Samples:** the 900-minute flag is descriptive, not an added filter or guarantee of reliability. "
            "Short stints and small comparison groups require caution; no confidence interval is estimated.\n"
            "- **Semantics:** defensive blocks use `blocks_stats_defense`; tackle zones use `def_3rd`, `mid_3rd`, `att_3rd`. "
            "Touch zones and passing blocks are not substitutes. Dribbler-challenge success is not total tackles divided by challenges, "
            "nor tackles won divided by tackles. Ancillary challenge/aerial metrics are excluded from the main views because "
            "their full source definitions are not independently established here.\n"
            "- **Interpretation:** aggregate 2024–25 season data reflects role, tactics, league and defensive opportunities. "
            "No possession adjustment is possible without supporting data. This does not measure direct pressures, distance, "
            "sprints, stamina, positioning or complete defensive quality. Validate patterns with match video and tactical context."
        )

    if ranked.empty:
        insight = f"No valid {METRICS[metric].lower()} values in {mode.lower()} units are available across {len(data):,} outfield records. Choose another metric or inspect missing data before making comparisons."
    else:
        leader = ranked.iloc[0]
        ties = int(ranked[column].eq(leader[column]).sum())
        minute_text = f"{leader['minutes']:,.0f} minutes" if pd.notna(leader["minutes"]) else "unavailable minutes"
        insight = (f"{leader['label']} records {leader[column]:.2f} {unit} for {METRICS[metric].lower()}, "
                   f"the highest observed value among {len(ranked):,} eligible records ({ties} record(s) share this value), "
                   f"over {minute_text}. ")
        if pd.notna(pooled):
            insight += f"The pooled rate is {pooled:.2f} actions per 90 across {valid_n:,} valid records. "
        if pd.notna(leader["minutes"]) and leader["minutes"] < SMALL_SAMPLE_MINUTES:
            insight += f"This leading record is below the {SMALL_SAMPLE_MINUTES:,}-minute sample flag. "
        insight += "Review same-position profiles and match video to investigate role and opportunities; high activity alone does not establish superior defending or physical workrate."
    st.info("**Key Insight**\n\n" + insight)
