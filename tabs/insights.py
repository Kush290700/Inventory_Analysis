import streamlit as st
import altair as alt
import pandas as pd
from utils.classification import top_n_by_metric


def render(df: pd.DataFrame, df_hc: pd.DataFrame, theme):
    st.header("🔎 Inventory Cost Insights")

    # ─── Sidebar Filter ─────────────────────────────────────────────
    st.sidebar.subheader("Filter by Supplier")
    suppliers = sorted(df.get("Supplier", pd.Series(dtype=str)).dropna().unique())
    f_sup = st.sidebar.multiselect(
        "Supplier", suppliers, default=suppliers, key="ins_sup"
    )
    df_f = df[df.get("Supplier", pd.Series(dtype=str)).isin(f_sup)].copy()

    # filter holding-cost frame by supplier if column exists
    if "Supplier" in df_hc.columns:
        df_hc_f = df_hc[df_hc["Supplier"].isin(f_sup)].copy()
    else:
        df_hc_f = df_hc.copy()

    st.markdown("---")

    # ─── KPI Cards (Cost Focus) ──────────────────────────────────────
    total_inv_value = df_f["OnHandCostTotal"].sum() if "OnHandCostTotal" in df_f.columns else 0
    avg_cost_per_sku = (
        df_f.groupby("SKU")["OnHandCostTotal"].sum().mean()
        if {"OnHandCostTotal","SKU"}.issubset(df_f.columns) else 0
    )
    total_hc_value = df_hc_f["TotalHoldingCost"].sum() if "TotalHoldingCost" in df_hc_f.columns else None

    cols = st.columns(3)
    cols[0].metric("Total Inv Value", f"${total_inv_value:,.0f}")
    cols[1].metric("Avg Cost per SKU", f"${avg_cost_per_sku:,.0f}")
    if total_hc_value is not None:
        cols[2].metric("Total Holding Cost", f"${total_hc_value:,.0f}")
    st.markdown("---")

    # ─── 1) Top Suppliers by Inventory Cost ──────────────────────────
    st.subheader("Top Suppliers by Inventory Cost")
    n_sup = st.slider("Number of suppliers", 5, 20, 10, key="ins_nsup")
    if {"OnHandCostTotal","Supplier"}.issubset(df_f.columns):
        try:
            top_sup = top_n_by_metric(df_f, "Supplier", "OnHandCostTotal", n_sup)
            chart1 = (
                alt.Chart(top_sup)
                .mark_bar()
                .encode(
                    y=alt.Y("Supplier:N", sort="-x", title="Supplier"),
                    x=alt.X("OnHandCostTotal:Q", title="Inv Value ($)", axis=alt.Axis(format=",.0f")),
                    color=alt.Color("OnHandCostTotal:Q", legend=None, scale=alt.Scale(scheme="greens")),
                    tooltip=[alt.Tooltip("Supplier:N"), alt.Tooltip("OnHandCostTotal:Q", format=",.0f", title="Value")]
                )
                .properties(height=300)
            )
            st.altair_chart(theme(chart1), use_container_width=True)
        except Exception:
            st.info("Could not render top suppliers chart.")
    else:
        st.info("No cost/supplier columns found – skipping top suppliers.")

    # ─── 2) ABC Class Breakdown by Inventory Value ───────────────────
    if {"ABC","InventoryValue"}.issubset(df_hc_f.columns):
        st.subheader("ABC Class Breakdown by Inventory Value")
        abc_df = (
            df_hc_f.groupby("ABC", as_index=False).agg(Value=("InventoryValue","sum"))
        )
        chart2 = (
            alt.Chart(abc_df)
            .mark_bar()
            .encode(
                x=alt.X("ABC:N", sort=["A","B","C"], title="Class"),
                y=alt.Y("Value:Q", title="Inv Value ($)", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Value:Q", legend=None, scale=alt.Scale(scheme="purples")),
                tooltip=[alt.Tooltip("ABC:N", title="Class"), alt.Tooltip("Value:Q", format=",.0f", title="Value")]
            )
            .properties(height=300)
        )
        st.altair_chart(theme(chart2), use_container_width=True)
    else:
        st.info("ABC or InventoryValue missing – skipping class breakdown.")

    # ─── 3) Inventory Cost Distribution ──────────────────────────────
    st.subheader("Inventory Cost Distribution")
    bins_cost = st.slider("Cost histogram bins", 10, 100, 30, key="ins_cost_bins")
    hist_cost = (
        alt.Chart(df_f)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("OnHandCostTotal:Q", bin=alt.Bin(maxbins=bins_cost), title="Inv Cost ($)"),
            y=alt.Y("count():Q", title="Number of SKUs"),
            tooltip=[alt.Tooltip("count():Q", title="SKUs")]
        )
        .properties(height=300)
    )
    density_cost = (
        alt.Chart(df_f)
        .transform_density("OnHandCostTotal", bandwidth=(df_f["OnHandCostTotal"].std() or 1)/2, as_=["Cost","Density"])
        .mark_area(opacity=0.3, color="blue")
        .encode(x="Cost:Q", y="Density:Q")
    )
    st.altair_chart(theme(hist_cost + density_cost), use_container_width=True)

    # ─── 4) Inventory Cost by Protein ───────────────────────────────
    if "Protein" in df_f.columns and "OnHandCostTotal" in df_f.columns:
        st.subheader("Inventory Cost by Protein")
        prot_cost = df_f.groupby("Protein", as_index=False).agg(TotalCost=("OnHandCostTotal","sum"))
        chart3 = (
            alt.Chart(prot_cost)
            .mark_bar()
            .encode(
                x=alt.X("Protein:N", sort="-y", title="Protein"),
                y=alt.Y("TotalCost:Q", title="Inv Cost ($)", axis=alt.Axis(format=",.0f")),
                color=alt.Color("TotalCost:Q", legend=None, scale=alt.Scale(scheme="greens")),
                tooltip=[alt.Tooltip("Protein:N"), alt.Tooltip("TotalCost:Q", format=",.0f", title="Value")]
            )
            .properties(height=300)
        )
        st.altair_chart(theme(chart3), use_container_width=True)
    else:
        st.info("Protein or cost column missing – skipping cost by protein.")

    # ─── 5) Cost vs Weeks On-Hand ───────────────────────────────────
    if {"WeeksOnHand","OnHandCostTotal"}.issubset(df_f.columns):
        st.subheader("Cost vs Weeks On-Hand")
        scatter = (
            alt.Chart(df_f)
            .mark_circle(size=60)
            .encode(
                x=alt.X("WeeksOnHand:Q", title="Weeks On-Hand"),
                y=alt.Y("OnHandCostTotal:Q", title="Inv Cost ($)", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Protein:N", title="Protein"),
                tooltip=[
                    alt.Tooltip("SKU_Desc:N", title="SKU"),
                    alt.Tooltip("Supplier:N", title="Supplier"),
                    alt.Tooltip("WeeksOnHand:Q", title="Weeks On-Hand"),
                    alt.Tooltip("OnHandCostTotal:Q", format=",.0f", title="Inv Cost")
                ]
            )
            .properties(height=400)
            .interactive()
        )
        st.altair_chart(theme(scatter), use_container_width=True)
    else:
        st.info("WeeksOnHand or cost column missing – skipping cost vs weeks.")

    st.markdown("---")

    # ─── Download Filtered Cost Data ───────────────────────────────────
    to_keep = [c for c in ["SKU","Supplier","OnHandCostTotal"] if c in df_f.columns]
    if to_keep:
        csv = df_f[to_keep].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Inventory Cost CSV", csv, "inventory_cost.csv", "text/csv"
        )
    else:
        st.info("No cost columns available for download.")
