import io
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

def render(df: pd.DataFrame, df_hc: pd.DataFrame, theme):
    """
    📈 Advanced Inventory KPIs & Insights (applies global filters)
    """
    st.header("📈 Inventory Management Dashboard")

    # ─── Core series ─────────────────────────────────────────────────
    weeks        = df.get("WeeksOnHand",        pd.Series(dtype=float))
    turns        = df.get("AnnualTurns",        pd.Series(dtype=float))
    weight_lb    = df.get("OnHandWeightTotal",  pd.Series(dtype=float))
    cost         = df.get("OnHandCostTotal",    pd.Series(dtype=float))
    skus         = df.get("SKU",                pd.Series(dtype=object))
    packs        = df.get("PackCount",          pd.Series(dtype=int))
    avg_wpp      = df.get("AvgWeightPerPack",   pd.Series(dtype=float))

    # ─── Summary values ──────────────────────────────────────────────
    total_skus     = skus.nunique()
    total_weight   = weight_lb.sum()
    total_cost     = cost.sum()
    avg_woh        = weeks.mean()
    med_woh        = weeks.median()
    avg_turns      = turns.mean()
    med_turns      = turns.median()
    at_risk        = int((weeks < 1).sum())
    healthy        = total_skus - at_risk
    total_packs    = int(packs.sum())
    avg_wpp_overall = avg_wpp.mean()

    # ─── KPI Banner ───────────────────────────────────────────────────
    cols = st.columns([1,1,1,1,1,1,1,1,1])
    cols[0].metric("SKUs",             total_skus)
    cols[1].metric("Avg WOH",          f"{avg_woh:.1f} wks", delta=f"med {med_woh:.1f}")
    cols[2].metric("Avg Turns",        f"{avg_turns:.1f}/yr", delta=f"med {med_turns:.1f}")
    cols[3].metric("Total Wt",         f"{total_weight:,.0f} lb")
    cols[4].metric("Total Cost",       f"${total_cost:,.0f}")
    cols[5].metric("At-Risk SKUs",     at_risk, delta=f"Healthy {healthy}")
    cols[6].metric("Total Packs",      f"{total_packs:,}")
    cols[7].metric("Avg Wt/Pack",      f"{avg_wpp_overall:.2f} lb")
    cols[8].metric("Avg Cost per lb",  f"${(total_cost/total_weight if total_weight else 0):.2f}")

    # ─── Mini-sparks for WOH & Turns ────────────────────────────────
    spark_woh = (
        alt.Chart(df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("WeeksOnHand:Q", bin=alt.Bin(maxbins=20), title=None),
            y=alt.Y("count():Q", title=None)
        )
        .properties(width=100, height=50)
    )
    cols[1].altair_chart(theme(spark_woh), use_container_width=True)

    spark_turns = (
        alt.Chart(df)
        .transform_density("AnnualTurns", as_=["Turns","Density"], bandwidth=1.0)
        .mark_area(opacity=0.3)
        .encode(x="Turns:Q", y="Density:Q")
        .properties(width=100, height=50)
    )
    cols[2].altair_chart(theme(spark_turns), use_container_width=True)

    st.markdown("---")

    # ─── At-Risk vs Healthy Donut ───────────────────────────────────
    risk_df = pd.DataFrame({
        "Status": ["At-Risk","Healthy"],
        "Count":  [at_risk, healthy]
    })
    risk_df["Pct"] = risk_df["Count"] / risk_df["Count"].sum()
    risk_pct = risk_df.loc[risk_df.Status=="At-Risk","Pct"].iloc[0]

    donut = (
        alt.Chart(risk_df)
        .mark_arc(innerRadius=60, outerRadius=100, stroke="#fff", strokeWidth=2)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Status:N",
                            scale=alt.Scale(domain=["At-Risk","Healthy"],
                                            range=["#d62728","#2ca02c"])),
            tooltip=[
                alt.Tooltip("Status:N"),
                alt.Tooltip("Count:Q", format=",d"),
                alt.Tooltip("Pct:Q", format=".1%")
            ]
        )
    )
    center = (
        alt.Chart(pd.DataFrame([{}]))
        .mark_text(size=20, align="center", baseline="middle")
        .encode(text=alt.value(f"At-Risk\n{risk_pct:.1%}"))
    )
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.altair_chart(
            theme(
                alt.layer(donut, center)
                   .properties(width=300, height=300, title="At-Risk vs Healthy")
                   .configure_title(anchor="middle", fontSize=16)
            ),
            use_container_width=True
        )
        st.markdown(
            f"**Healthy**: {healthy} SKUs  |  **At-Risk**: {at_risk} SKUs  |  **% At-Risk**: {risk_pct:.1%}"
        )

    st.markdown("---")

    # ─── Holding-Cost % Distribution ────────────────────────────────
    st.subheader("Holding-Cost % Distribution")
    if "HoldingCostPercent" in df_hc.columns:
        hc_dist = (
            alt.Chart(df_hc)
            .transform_density("HoldingCostPercent", as_=["HC","Density"], bandwidth=0.5)
            .mark_area(color="orange", opacity=0.3)
            .encode(x="HC:Q", y="Density:Q")
        )
        hc_hist = (
            alt.Chart(df_hc)
            .mark_bar(opacity=0.6)
            .encode(
                x=alt.X("HoldingCostPercent:Q", bin=alt.Bin(maxbins=30), title="HC %"),
                y=alt.Y("count():Q", title="SKU Count")
            )
        )
        st.altair_chart(
            theme((hc_hist + hc_dist).properties(height=300).interactive()),
            use_container_width=True
        )
    else:
        st.info("No `HoldingCostPercent` column found.")

    # ─── Top-5 SKUs by Holding Cost ────────────────────────────────
    st.subheader("Top-5 SKUs by Holding Cost")
    if "TotalHoldingCost" in df_hc.columns:
        top5 = (
            df_hc.groupby("SKU_Desc", as_index=False)["TotalHoldingCost"]
                 .sum()
                 .nlargest(5, "TotalHoldingCost")
        )
        bar = (
            alt.Chart(top5)
            .mark_bar()
            .encode(
                y=alt.Y("SKU_Desc:N", sort="-x"),
                x=alt.X("TotalHoldingCost:Q", title="HC ($)", axis=alt.Axis(format=",$.0f")),
                color=alt.Color("TotalHoldingCost:Q", scale=alt.Scale(scheme="reds")),
                tooltip=[alt.Tooltip("SKU_Desc:N"), alt.Tooltip("TotalHoldingCost:Q", format=",$.0f")]
            )
            .properties(height=200)
        )
        st.altair_chart(theme(bar), use_container_width=True)
    else:
        st.info("No `TotalHoldingCost` column found.")

    # ─── Download full report ───────────────────────────────────────
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer,   sheet_name="Inventory",    index=False)
            df_hc.to_excel(writer, sheet_name="HoldingCost",  index=False)
        buf.seek(0)
        st.download_button(
            "📥 Download KPI Report (Excel)",
            buf.getvalue(),
            file_name="Inventory_KPIs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
