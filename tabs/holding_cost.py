# tabs/holding_cost.py

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from utils.classification import top_n_by_metric

def render(df_hc: pd.DataFrame, theme):
    st.header("💰 Holding Cost & Aging")

    # ─── Sidebar Controls ───────────────────────────────────────
    st.sidebar.subheader("Holding Cost Filters")
    proteins = sorted(df_hc["Protein"].dropna().unique())
    sel_prot = st.sidebar.multiselect("Protein", proteins, default=proteins)
    df = df_hc[df_hc["Protein"].isin(sel_prot)].copy()

    # Slider for minimum holding-cost percent
    min_hc_pct = st.sidebar.slider(
        "Min Holding-Cost %", 0.0, 100.0, 0.0, step=1.0
    )
    df = df[df["HoldingCostPercent"] >= min_hc_pct]

    # ─── Key Metrics ────────────────────────────────────────────
    total_hc = df["TotalHoldingCost"].sum()
    avg_pct   = df["HoldingCostPercent"].mean()
    over_1y   = (df["DaysInStorage"] >= 365).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total HC",       f"${total_hc:,.0f}")
    c2.metric("Avg HC %",       f"{avg_pct:.1f}%")
    c3.metric("SKUs ≥ 1 yr 🕒",  f"{over_1y}")

    # ─── 1) Distribution of Days in Storage ───────────────────
    st.subheader("Inventory Aging Distribution")
    bins = st.slider("Number of bins", 5, 60, 30, step=5, key="hc_bins")
    max_days = int(df["DaysInStorage"].quantile(0.99))
    aging = df[df["DaysInStorage"] <= max_days]

    hist_aging = (
        alt.Chart(aging)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("DaysInStorage:Q",
                    bin=alt.Bin(maxbins=bins),
                    title="Days in Storage"),
            y=alt.Y("count():Q", title="Count of SKUs"),
            tooltip=[alt.Tooltip("count():Q", title="SKUs")]
        )
        .properties(height=300)
        .interactive()
    )

    mean_rule = (
        alt.Chart(pd.DataFrame({"v":[df["DaysInStorage"].mean()]}))
        .mark_rule(color="red", size=2)
        .encode(x="v:Q")
    )
    median_rule = (
        alt.Chart(pd.DataFrame({"v":[df["DaysInStorage"].median()]}))
        .mark_rule(color="blue", strokeDash=[4,4], size=2)
        .encode(x="v:Q")
    )

    st.altair_chart(
        theme(hist_aging + mean_rule + median_rule),
        use_container_width=True
    )
    st.markdown("**Red**=mean • **Blue**=median")

    # ─── 2) Boxplot HC% by Protein ─────────────────────────────
    st.subheader("HC % Distribution by Protein")
    order = (
        df.groupby("Protein")["HoldingCostPercent"]
          .median().sort_values(ascending=False)
          .index.tolist()
    )
    box = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("Protein:N", sort=order, title="Protein"),
            y=alt.Y("HoldingCostPercent:Q", title="HC %"),
            color=alt.Color("Protein:N", legend=None)
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(theme(box), use_container_width=True)

    # ─── 3) Cumulative HC Pareto Curve ─────────────────────────
    st.subheader("Pareto: Cumulative HC by SKU (descending)")
    pareto = (
        df[["SKU_Desc","TotalHoldingCost"]]
        .sort_values("TotalHoldingCost", ascending=False)
        .assign(CumCost=lambda d: d["TotalHoldingCost"].cumsum(),
                Rank=lambda d: np.arange(1, len(d)+1))
    )
    total = pareto["TotalHoldingCost"].sum()
    pareto["CumPct"] = pareto["CumCost"] / total

    line = (
        alt.Chart(pareto)
        .mark_line(point=False)
        .encode(
            x=alt.X("Rank:Q", title="SKU Rank"),
            y=alt.Y("CumPct:Q", title="Cumulative HC %"),
            tooltip=[
                alt.Tooltip("SKU_Desc:N", title="SKU"),
                alt.Tooltip("TotalHoldingCost:Q", title="HC ($)", format=",.0f"),
                alt.Tooltip("CumPct:Q", format=".2f", title="Cum %")
            ]
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(theme(line), use_container_width=True)

    # ─── 4) Top-N SKUs by HC & % ────────────────────────────────
    st.subheader("Top-10 SKUs by HC & % of Total")
    top_cost = top_n_by_metric(df, "SKU_Desc", "TotalHoldingCost", 10)
    top_cost["PctOfTotal"] = top_cost["TotalHoldingCost"] / total

    bar1 = (
        alt.Chart(top_cost)
        .mark_bar()
        .encode(
            x=alt.X("TotalHoldingCost:Q", title="Total HC ($)"),
            y=alt.Y("SKU_Desc:N", sort="-x", title="SKU"),
            tooltip=[
                alt.Tooltip("SKU_Desc:N"),
                alt.Tooltip("TotalHoldingCost:Q", format=",.0f"),
                alt.Tooltip("PctOfTotal:Q", format=".2%", title="% Total")
            ]
        )
        .properties(height=300)
    )
    st.altair_chart(theme(bar1), use_container_width=True)

    # ─── 5) Heatmap: Protein vs AgeBucket HC% ─────────────────
    st.subheader("Heatmap: Avg HC % by Protein & Age Bucket")
    hm = (
        df
        .groupby(["Protein","AgeBucket"], observed=False)["HoldingCostPercent"]
        .mean()
        .reset_index()
    )
    heat = (
        alt.Chart(hm)
        .mark_rect()
        .encode(
            x=alt.X("AgeBucket:O", title="Age Bucket"),
            y=alt.Y("Protein:N", title="Protein"),
            color=alt.Color("HoldingCostPercent:Q", title="Avg HC %", scale=alt.Scale(scheme="magma")),
            tooltip=[
                alt.Tooltip("Protein:N"),
                alt.Tooltip("AgeBucket:O"),
                alt.Tooltip("HoldingCostPercent:Q", format=".1f")
            ]
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(theme(heat), use_container_width=True)

    # ─── 6) Aging Summary Table ────────────────────────────────
    st.subheader("Aging Summary")
    summary = (
        df
        .groupby("AgeBucket", observed=False)
        .agg(
            Count=("SKU", "nunique"),
            AvgHCpct=("HoldingCostPercent","mean"),
            TotalHC=("TotalHoldingCost","sum")
        )
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)
