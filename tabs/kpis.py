import io
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

def render(df: pd.DataFrame, df_hc: pd.DataFrame, theme):
    """
    📈 Advanced Inventory KPIs & Insights (no filters)
    """
    st.header("📈 Inventory Management Dashboard")

    # ─── Core KPI computations ──────────────────────────────────
    weeks     = df.get("WeeksOnHand",     pd.Series(dtype=float))
    turns     = df.get("AnnualTurns",     pd.Series(dtype=float))
    weight_lb = df.get("OnHandWeightTotal", pd.Series(dtype=float))
    cost      = df.get("OnHandCostTotal",   pd.Series(dtype=float))
    skus      = df.get("SKU",             pd.Series(dtype=object))

    total_skus = skus.nunique()
    avg_woh    = weeks.mean()
    med_woh    = weeks.median()
    avg_turn   = turns.mean()
    med_turn   = turns.median()
    total_wt   = weight_lb.sum()
    total_cost = cost.sum()
    at_risk    = (weeks < 1).sum()
    healthy    = total_skus - at_risk

    # ─── KPI Banner ────────────────────────────────────────────
    cols = st.columns([1,1,1,1,1,1,1])
    cols[0].metric("SKUs",              total_skus)
    cols[1].metric("Avg WOH",           f"{avg_woh:.1f} wks", delta=f"med {med_woh:.1f}")
    cols[2].metric("Avg Turns",         f"{avg_turn:.1f}/yr", delta=f"med {med_turn:.1f}")
    cols[3].metric("Total Wt",          f"{total_wt:,.0f} lb")
    cols[4].metric("Total Cost",        f"${total_cost:,.0f}")
    cols[5].metric("At-Risk SKUs",      at_risk, delta=f"healthy {healthy}")
    cols[6].metric("Avg Cost per lb",   f"${(total_cost/total_wt if total_wt else 0):.2f}")

    # small WOH distribution spark
    spark_woh = (
        alt.Chart(df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("WeeksOnHand:Q", bin=alt.Bin(maxbins=20), title=None),
            y=alt.Y("count():Q",    title=None)
        )
        .properties(width=100, height=50)
    )
    cols[1].altair_chart(theme(spark_woh), use_container_width=True)

    # small Turns distribution spark
    spark_turn = (
        alt.Chart(df)
        .transform_density("AnnualTurns", as_=["Turns","Density"], bandwidth=1.0)
        .mark_area(opacity=0.3)
        .encode(x="Turns:Q", y="Density:Q")
        .properties(width=100, height=50)
    )
    cols[2].altair_chart(theme(spark_turn), use_container_width=True)

    st.markdown("---")

    # ─── At-Risk vs Healthy Donut ────────────────────────────────
    risk_df = pd.DataFrame({
        "Status": ["At-Risk", "Healthy"],
        "Count":  [at_risk, healthy]
    })
    risk_df['Pct'] = risk_df.Count / risk_df.Count.sum()
    risk_pct = risk_df.loc[risk_df.Status=='At-Risk','Pct'].iloc[0]

    donut = (
        alt.Chart(risk_df)
        .mark_arc(innerRadius=60, outerRadius=100, stroke='#fff', strokeWidth=2)
        .encode(
            theta=alt.Theta("Count:Q", title=None),
            color=alt.Color("Status:N", scale=alt.Scale(domain=["At-Risk","Healthy"], range=["#d62728","#2ca02c"])),
            tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("Count:Q", format=",d"), alt.Tooltip("Pct:Q", format=".1%")]  
        )
        .properties(width=500, height=500)
    )
    center_text = (
        alt.Chart(pd.DataFrame([{}]))
        .mark_text(size=20, align='center', baseline='middle')
        .encode(text=alt.value(f"At-Risk\n{risk_pct:.1%}"))
        .properties(width=500, height=500)
    )
    st.altair_chart(theme(alt.layer(donut, center_text).properties(title="At-Risk vs Healthy SKUs")), use_container_width=False)

    st.markdown("---")

    # ─── Holding-Cost % Distribution ─────────────────────────────
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
                y=alt.Y("count():Q", title="Count of SKUs")
            )
        )
        st.altair_chart(theme((hc_hist + hc_dist).properties(height=300).interactive()), use_container_width=True)
    else:
        st.info("No HoldingCostPercent column found.")

    st.subheader("Top-5 SKUs by Holding Cost")
    if "TotalHoldingCost" in df_hc.columns:
        top5 = (
            df_hc.groupby("SKU_Desc", as_index=False)["TotalHoldingCost"]
                 .sum().nlargest(5, "TotalHoldingCost")
        )
        bar_hc = (
            alt.Chart(top5)
            .mark_bar()
            .encode(
                y=alt.Y("SKU_Desc:N", sort="-x", title="SKU"),
                x=alt.X("TotalHoldingCost:Q", title="HC ($)", axis=alt.Axis(format=",.0f")),
                color=alt.Color("TotalHoldingCost:Q", scale=alt.Scale(scheme="reds")),
                tooltip=[alt.Tooltip("SKU_Desc:N"), alt.Tooltip("TotalHoldingCost:Q", format="$,.0f")]
            )
            .properties(height=200)
        )
        st.altair_chart(theme(bar_hc), use_container_width=True)
    else:
        st.info("No TotalHoldingCost column found.")

    # ─── Download Full KPI Report ──────────────────────────────
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Inventory", index=False)
            df_hc.to_excel(writer, sheet_name="HoldingCost", index=False)
        buf.seek(0)
        st.download_button(
            "📥 Download KPI Report (Excel)",
            buf.getvalue(),
            file_name="Inventory_KPIs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
