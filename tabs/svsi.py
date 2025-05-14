# tabs/svsi.py

import streamlit as st
import altair as alt
import pandas as pd
from utils.classification import quadrantify, top_n_by_metric

def render(df: pd.DataFrame, theme):
    st.header("📈 Sales vs Inventory")

    # ─── Sidebar Filters ────────────────────────────────────────
    st.sidebar.subheader("Sales vs Inv Filters")
    prots = sorted(df["Protein"].dropna().unique())
    sups  = sorted(df["Supplier"].dropna().unique())

    f_prot = st.sidebar.multiselect("Protein", prots, default=prots)
    f_sup  = st.sidebar.multiselect("Supplier", sups, default=sups)

    df_f = df[
        df["Protein"].isin(f_prot) &
        df["Supplier"].isin(f_sup)
    ].copy()

    # ─── Axis Zoom Sliders ─────────────────────────────────────
    shipped_max = st.sidebar.slider(
        "Max Total Shipped (lb)", 
        float(df_f["TotalShippedLb"].min()), 
        float(df_f["TotalShippedLb"].quantile(0.99)), 
        float(df_f["TotalShippedLb"].quantile(0.99)),
        step=1.0
    )
    onhand_max  = st.sidebar.slider(
        "Max On-Hand Weight (lb)",
        float(df_f["OnHandWeightLb"].min()), 
        float(df_f["OnHandWeightLb"].quantile(0.99)), 
        float(df_f["OnHandWeightLb"].quantile(0.99)),
        step=1.0
    )

    df_f = df_f[
        (df_f["TotalShippedLb"] <= shipped_max) &
        (df_f["OnHandWeightLb"] <= onhand_max)
    ]

    # ─── 1) Scatter + 45° + trend ───────────────────────────────
    st.subheader("Shipped vs On-Hand Scatter")
    brush = alt.selection_interval()
    hover = alt.selection_single(on="mouseover", fields=["SKU_Desc"], empty="none")

    base = alt.Chart(df_f).mark_circle().encode(
        x=alt.X("TotalShippedLb:Q", title="Total Shipped (lb)", scale=alt.Scale(type="log")),
        y=alt.Y("OnHandWeightLb:Q", title="On-Hand Weight (lb)", scale=alt.Scale(type="log")),
        size=alt.Size("OnHandCost:Q", title="On-Hand Cost ($)", scale=alt.Scale(type="log")),
        color=alt.condition(hover, alt.Color("Protein:N", title="Protein"), alt.value("lightgray")),
        tooltip=[
            alt.Tooltip("SKU_Desc:N",        title="SKU"),
            alt.Tooltip("Supplier:N",        title="Supplier"),
            alt.Tooltip("TotalShippedLb:Q",  title="Shipped (lb)"),
            alt.Tooltip("OnHandWeightLb:Q",  title="On-Hand (lb)"),
            alt.Tooltip("OnHandCost:Q",      title="Cost ($)")
        ],
        opacity=alt.condition(hover, alt.value(1), alt.value(0.3))
    ).add_selection(brush, hover).properties(height=450).interactive()

    ref = alt.Chart(pd.DataFrame({"x":[1, shipped_max]})).mark_line(color="gray", strokeDash=[4,2]).encode(
        x="x:Q", y="x:Q"
    )
    trend = base.transform_regression("TotalShippedLb","OnHandWeightLb").mark_line(color="red", size=2)

    st.altair_chart(theme((ref + base + trend).properties(width=800)), use_container_width=True)

    # ─── 2) Brush-linked Table ───────────────────────────────────
    st.subheader("Selected SKUs Detail")
    df_selected = alt.Chart(df_f).transform_filter(brush).transform_fold(
        ["SKU_Desc","Supplier","TotalShippedLb","OnHandWeightLb","OnHandCost"],
        as_=["field","value"]
    )
    st.dataframe(df_f if brush is None else df_f, use_container_width=True)

    # ─── 3) Quadrant Analysis ───────────────────────────────────
    st.subheader("Quadrant Analysis")
    df_q, xm, ym = quadrantify(df_f, "TotalShippedLb","OnHandWeightLb")
    quad = alt.Chart(df_q).mark_circle().encode(
        x=alt.X("TotalShippedLb:Q", scale=alt.Scale(type="log")),
        y=alt.Y("OnHandWeightLb:Q", scale=alt.Scale(type="log")),
        color=alt.Color("Quadrant:N", title="Quadrant"),
        tooltip=["SKU_Desc","Quadrant"]
    ).properties(height=400).interactive()
    lines = alt.Chart(pd.DataFrame([{"v":xm},{"v":ym}])).mark_rule(color="black", strokeDash=[2,2]).encode(
        x="v:Q", y="v:Q"
    )
    st.altair_chart(theme(quad & lines), use_container_width=True)

    # ─── 4) Quadrant Counts & Heatmap ───────────────────────────
    counts = df_q.groupby("Quadrant").size().reset_index(name="Count")
    c1, c2 = st.columns(2)
    bar_q = (
        alt.Chart(counts)
        .mark_bar()
        .encode(x="Quadrant:N", y="Count:Q", tooltip=["Quadrant","Count"])
        .properties(height=250)
    )
    c1.altair_chart(theme(bar_q), use_container_width=True)

    heat = df_q.groupby(["Supplier","Quadrant"]).size().reset_index(name="Count")
    heatmap = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("Supplier:N", title="Supplier"),
            y=alt.Y("Quadrant:N", title="Quadrant"),
            color=alt.Color("Count:Q", title="Count", scale=alt.Scale(scheme="blues")),
            tooltip=["Supplier","Quadrant","Count"]
        )
        .properties(height=300)
    )
    c2.altair_chart(theme(heatmap), use_container_width=True)

    # ─── 5) Top-N Shipped vs Produced ──────────────────────────
    st.subheader("Top-10 SKUs by Shipped & Produced")
    t1, t2 = st.columns(2)
    ts = top_n_by_metric(df_f, "SKU_Desc","TotalShippedLb",10)
    tp = top_n_by_metric(df_f, "SKU_Desc","ProductionShippedLb",10)
    bar1 = (
        alt.Chart(ts)
        .mark_bar()
        .encode(
            y=alt.Y("SKU_Desc:N", sort="-x"), x="TotalShippedLb:Q", tooltip=["SKU_Desc","TotalShippedLb"]
        ).properties(height=250)
    )
    bar2 = (
        alt.Chart(tp)
        .mark_bar()
        .encode(
            y=alt.Y("SKU_Desc:N", sort="-x"), x="ProductionShippedLb:Q", tooltip=["SKU_Desc","ProductionShippedLb"]
        ).properties(height=250)
    )
    t1.altair_chart(theme(bar1), use_container_width=True)
    t2.altair_chart(theme(bar2), use_container_width=True)
