import os, sys
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.classification import quadrantify, classify_movement


def _get_series(df: pd.DataFrame, candidates, default=0):
    """
    Return the first existing column in `candidates` as a Series,
    or a Series of `default` if none exist.
    """
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index)


def render(df: pd.DataFrame, theme):
    st.header("📈 Sales vs Inv & 🚀 Movers")
    df = df.copy()

    # helper to pick first‐existing column
    def _get_series(df, candidates, default=0):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series(default, index=df.index)
    
    # ────────────────────────────────────────────────────────────────
    # 1) Coerce numeric columns, now accounting for ProductionUsage
    # ────────────────────────────────────────────────────────────────
    
    # ship/production usage
    df["ShippedUsage"] = pd.to_numeric(
        _get_series(df, ["TotalShipped", "TotalShippedLb", "ShippedLb"]),
        errors="coerce"
    ).fillna(0)
    
    df["ProductionUsage"] = pd.to_numeric(
        _get_series(df, ["ProductionUsage", "ProductionUsageLb"]),
        errors="coerce"
    ).fillna(0)
    
    df["TotalUsage"] = df["ShippedUsage"] + df["ProductionUsage"]
    
    # on‐hand weight: force use of the “Total” field to avoid misreads
    df["OnHandWeight"] = pd.to_numeric(
        _get_series(df, ["OnHandWeightTotal", "OnHandWeight", "OnHandWeightLb"]),
        errors="coerce"
    ).fillna(0)
    
    df["OnHandCost"] = pd.to_numeric(
        _get_series(df, ["OnHandCostTotal", "OnHandCost"]),
        errors="coerce"
    ).fillna(0)
    
    df["WeeksOnHand"] = pd.to_numeric(
        _get_series(df, ["WeeksOnHand"]), errors="coerce"
    ).fillna(0)
    
    df["AvgWeeklyUsage"] = pd.to_numeric(
        _get_series(df, ["AvgWeeklyUsage"]), errors="coerce"
    ).fillna(0)
    
    # fallback SKU description
    df["SKU_Desc"] = df.get("SKU_Desc", df.get("SKU", "").astype(str))
    
    
    # ────────────────────────────────────────────────────────────────
    # 2) Sidebar filters
    # ────────────────────────────────────────────────────────────────
    st.sidebar.subheader("Filters")
    suppliers = sorted(df["Supplier"].dropna().unique())
    f_sup = st.sidebar.multiselect("Supplier", suppliers, default=suppliers, key="svsi_sup")
    df = df[df["Supplier"].isin(f_sup)]
    
    
    # ────────────────────────────────────────────────────────────────
    # 3) Sales vs On-Hand scatter + trend
    # ────────────────────────────────────────────────────────────────
    st.subheader("🔄 Sales vs On-Hand Inventory")
    
    # slider limits
    x_max = float(df["TotalUsage"].quantile(0.99))
    y_max = float(df["OnHandWeight"].quantile(0.99))
    
    shp_max = st.sidebar.slider(
        "Max Total Usage (lb)",
        float(df["TotalUsage"].min()), x_max, x_max, step=1.0, key="svsi_usage_max"
    )
    oh_max = st.sidebar.slider(
        "Max On-Hand Wt (lb)",
        float(df["OnHandWeight"].min()), y_max, y_max, step=1.0, key="svsi_onhand_max"
    )
    
    df_svsi = df[
        (df["TotalUsage"] <= shp_max) &
        (df["OnHandWeight"] <= oh_max)
    ].copy()
    
    brush = alt.selection_interval()
    hover = alt.selection_single(on="mouseover", fields=["SKU_Desc"], empty="none")
    
    base = (
        alt.Chart(df_svsi)
        .mark_circle()
        .encode(
            x=alt.X("TotalUsage:Q", title="Total Usage (lb)", scale=alt.Scale(type="log")),
            y=alt.Y("OnHandWeight:Q", title="On-Hand Wt (lb)", scale=alt.Scale(type="log")),
            size=alt.Size("OnHandCost:Q", title="On-Hand Cost ($)", scale=alt.Scale(type="log")),
            color=alt.condition(hover, "Protein:N", alt.value("lightgray")),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("SKU_Desc:N", title="SKU"),
                alt.Tooltip("Supplier:N", title="Supplier"),
                alt.Tooltip("TotalUsage:Q", format=",.0f", title="Usage"),
                alt.Tooltip("OnHandWeight:Q", format=",.0f", title="On-Hand"),
                alt.Tooltip("OnHandCost:Q", format=",.0f", title="Cost")
            ]
        )
        .add_selection(brush, hover)
        .properties(height=450)
        .interactive()
    )
    
    ref_line = (
        alt.Chart(pd.DataFrame({"v": [1, shp_max]}))
        .mark_line(color="gray", strokeDash=[4,2])
        .encode(x="v:Q", y="v:Q")
    )
    
    trend = base.transform_regression("TotalUsage", "OnHandWeight").mark_line(color="red", size=2)
    
    st.altair_chart(
        theme((ref_line + base + trend).properties(width=800)),
        use_container_width=True
    )
    
    
    # ────────────────────────────────────────────────────────────────
    # 4) Quadrant Analysis
    # ────────────────────────────────────────────────────────────────
    st.subheader("Quadrant Analysis")
    df_svsi["OverallUsage"] = df_svsi["TotalUsage"]
    df_q, xm, ym = quadrantify(df_svsi, "OverallUsage", "OnHandWeight")
    
    # compute domains, avoid zeros
    def _pmin(s):
        v = s[s>0]
        return v.min() if not v.empty else 1
    
    x_min, x_max_q = _pmin(df_q["OverallUsage"])*0.8, df_q["OverallUsage"].max()*1.1
    y_min, y_max_q = _pmin(df_q["OnHandWeight"])*0.8, df_q["OnHandWeight"].max()*1.1
    
    quad_bg = pd.DataFrame([
        {"xmin": x_min,    "xmax": xm,    "ymin": y_min,    "ymax": ym},
        {"xmin": xm,       "xmax": x_max_q,"ymin": y_min,    "ymax": ym},
        {"xmin": x_min,    "xmax": xm,    "ymin": ym,       "ymax": y_max_q},
        {"xmin": xm,       "xmax": x_max_q,"ymin": ym,       "ymax": y_max_q},
    ])
    colors = ["#fde0dd","#fdd0a2","#a1d99b","#74c476"]
    quad_bg["color"] = colors
    
    bg = (
        alt.Chart(quad_bg)
        .mark_rect()
        .encode(
            x=alt.X("xmin:Q", scale=alt.Scale(type="log", domain=[x_min, x_max_q])),
            x2="xmax:Q",
            y=alt.Y("ymin:Q", scale=alt.Scale(type="log", domain=[y_min, y_max_q])),
            y2="ymax:Q",
            color=alt.Color("color:N", legend=None)
        )
    )
    
    vline = alt.Chart(pd.DataFrame({"x":[xm]})).mark_rule(color="gray", strokeDash=[4,4]).encode(x="x:Q")
    hline = alt.Chart(pd.DataFrame({"y":[ym]})).mark_rule(color="gray", strokeDash=[4,4]).encode(y="y:Q")
    
    sel = alt.selection_multi(fields=["Quadrant"], bind="legend")
    pts = (
        alt.Chart(df_q)
        .mark_circle(size=70)
        .encode(
            x=alt.X("OverallUsage:Q", scale=alt.Scale(type="log", domain=[x_min,x_max_q]), axis=alt.Axis(title="Total Usage", format="~s")),
            y=alt.Y("OnHandWeight:Q", scale=alt.Scale(type="log", domain=[y_min,y_max_q]), axis=alt.Axis(title="On-Hand Weight", format="~s")),
            color=alt.Color("Quadrant:N", title="Quadrant"),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.2)),
            tooltip=[
                alt.Tooltip("SKU_Desc:N"), 
                alt.Tooltip("OverallUsage:Q", format=",.0f", title="Usage"), 
                alt.Tooltip("OnHandWeight:Q", format=",.0f", title="On-Hand"), 
                alt.Tooltip("Quadrant:N")
            ]
        )
        .add_selection(sel)
    )
    
    st.altair_chart(
        theme(
            (bg + vline + hline + pts)
            .properties(title="Quadrant: Usage vs On-Hand", width=750, height=450)
            .configure_axis(grid=True)
        ),
        use_container_width=True
    )
    # 5) High-vs-Slow Movers
    st.subheader("🚀 High-vs-Slow Movers")
    q = st.slider("High-mover cutoff", 0.1, 0.9, 0.5, step=0.05, key="mv_q")
    dfm = classify_movement(df_svsi, quantile=q)
    dfm["AvgWeeklyUsage"] = dfm["AvgWeeklyUsage"].clip(lower=0.1)
    dfm["WeeksOnHand"]    = dfm["WeeksOnHand"].clip(lower=0.1)

    c1, c2 = st.columns(2)
    top_u = dfm.nlargest(10, "AvgWeeklyUsage")
    top_w = dfm.nlargest(10, "WeeksOnHand")

    c1.altair_chart(
        alt.Chart(top_u).mark_bar().encode(
            x="AvgWeeklyUsage:Q", y=alt.Y("SKU_Desc:N", sort="-x"), tooltip=["SKU_Desc","AvgWeeklyUsage"]
        ).properties(height=300), use_container_width=True
    )
    c2.altair_chart(
        alt.Chart(top_w).mark_bar().encode(
            x="WeeksOnHand:Q", y=alt.Y("SKU_Desc:N", sort="-x"), tooltip=["SKU_Desc","WeeksOnHand"]
        ).properties(height=300), use_container_width=True
    )

    # 6) SKU Count by Supplier & Class
    st.subheader("SKU Count by Supplier & Class")
    heat = (
        dfm.groupby(["Supplier","MovementClass"]).size().reset_index(name="Count")
    )
    st.altair_chart(
        theme(alt.Chart(heat).mark_rect().encode(
            x="Supplier:N", y="MovementClass:N", color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")), tooltip=["Supplier","MovementClass","Count"]
        ).properties(height=400)), use_container_width=True
    )
