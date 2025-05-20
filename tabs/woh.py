import io
import streamlit as st
import altair as alt
import pandas as pd
import sys, os
import numpy as np
from prophet import Prophet

# ensure the project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.classification import compute_threshold_move

def render(df: pd.DataFrame, df_hc: pd.DataFrame, theme):
    """
    📊 Weeks-On-Hand Analysis
    df: aggregated SKU-level DataFrame with columns:
        SKU, SKU_Desc, ProductState, Supplier,
        WeeksOnHand, AvgWeeklyUsage,
        OnHandWeightTotal, OnHandCostTotal
    df_hc: holding-cost DataFrame (for threshold logic)
    theme: Altair theme function
    """
    st.header("📊 Weeks-On-Hand Analysis")

    # ── Precompute FZ WeeksOnHand lookup & exclude zero-usage ────────────────
    fz = df[
        (df["ProductState"].str.upper().str.startswith("FZ")) &
        (df["AvgWeeklyUsage"] > 0)
    ].copy()
    fz["Status"] = np.where(fz["AvgWeeklyUsage"] == 0, "Frozen", "Active")
    
    # lookup: SKU → FZ WeeksOnHand & FZ OnHandWeight
    fz_woh = fz.set_index("SKU")["WeeksOnHand"]
    fz_weight = fz.set_index("SKU")["OnHandWeightTotal"]
    fz_cost   = fz.set_index("SKU")["OnHandCostTotal"]
    
    
    # ── Move FZ → EXT ────────────────────────────────────────────
    st.subheader("🔄 Move FZ → EXT")
    
    thr1 = st.slider(
        "Desired FZ WOH (weeks)",
        0.0, 52.0, 4.0, 0.5,
        key="w2e_thr"
    )
    
    # only those above threshold
    sel = fz["WeeksOnHand"] > thr1
    to_move = fz[sel].copy()
    
    # compute how much weight to send: current_weight - desired_weight
    to_move["DesiredFZ_Weight"] = to_move["AvgWeeklyUsage"] * thr1
    to_move["WeightToMove"]    = to_move["OnHandWeightTotal"] - to_move["DesiredFZ_Weight"]
    # prorate cost
    to_move["CostToMove"]      = (
        to_move["WeightToMove"] /
        to_move["OnHandWeightTotal"]
    ) * to_move["OnHandCostTotal"]
    
    # metrics (exclude any tiny negatives)
    total_wt_move = to_move.loc[to_move["WeightToMove"]>0, "WeightToMove"].sum()
    total_cost_move = to_move.loc[to_move["WeightToMove"]>0, "CostToMove"].sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs to Move",         to_move["SKU"].nunique())
    c2.metric("Total Weight to Move", f"{total_wt_move:,.0f} lb")
    c3.metric("Total Cost to Move",   f"${total_cost_move:,.0f}")
    
    # supplier filter
    suppliers = sorted(to_move["Supplier"].dropna().unique())
    sel_sup   = st.multiselect("Filter Suppliers", suppliers, default=suppliers, key="mv1_sups")
    mv1 = to_move[to_move["Supplier"].isin(sel_sup) & (to_move["WeightToMove"]>0)]
    
    if mv1.empty:
        st.info("No items match the current filters.")
    else:
        sel2 = alt.selection_multi(fields=["Supplier"], bind="legend")
        chart1 = (
            alt.Chart(mv1)
            .mark_bar()
            .encode(
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                x=alt.X("WeightToMove:Q", title="Weight to Move (lb)"),
                color="Supplier:N",
                opacity=alt.condition(sel2, alt.value(1), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("SKU_Desc:N", title="SKU"),
                    alt.Tooltip("Supplier:N", title="Supplier"),
                    alt.Tooltip("WeeksOnHand:Q", title="Current FZ WOH"),
                    alt.Tooltip("DesiredFZ_Weight:Q", format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToMove:Q", format=",.0f", title="Weight to Move"),
                    alt.Tooltip("CostToMove:Q",   format=",.0f", title="Cost to Move"),
                ]
            )
            .add_selection(sel2)
            .properties(width=800, height=alt.Step(25))
            .resolve_scale(y="independent")
            .interactive()
        )
        st.altair_chart(theme(chart1), use_container_width=True)
    
        buf1 = io.BytesIO()
        mv1.to_excel(buf1, index=False, sheet_name="FZ2EXT")
        buf1.seek(0)
        st.download_button(
            "Download FZ→EXT List",
            buf1.getvalue(),
            file_name="FZ2EXT.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    
    # ── Move EXT → FZ ────────────────────────────────────────────
    st.subheader("🔄 Move EXT → FZ")
    
    ext = df[
        (df["ProductState"].str.upper().str.startswith("EXT")) &
        (df["AvgWeeklyUsage"] > 0)
    ].copy()
    ext["Status"] = np.where(ext["AvgWeeklyUsage"] == 0, "Frozen", "Active")
    
    # map in the existing FZ weight and WOH
    ext["FZ_WOH"]    = ext["SKU"].map(fz_woh).fillna(0)
    ext["FZ_Weight"] = ext["SKU"].map(fz_weight).fillna(0)
    ext["FZ_Cost"]   = ext["SKU"].map(fz_cost).fillna(0)
    
    # threshold slider
    thr2 = st.slider(
        "Desired FZ WOH to achieve",
        0.0,
        float(ext["FZ_WOH"].max()),
        1.0,
        step=0.25,
        key="e2f_thr"
    )
    
    # only those below threshold
    back = ext[ext["FZ_WOH"] < thr2].copy()
    
    # compute how much to bring back to FZ
    back["DesiredFZ_Weight"] = back["AvgWeeklyUsage"] * thr2
    back["WeightToReturn"]   = back["DesiredFZ_Weight"] - back["FZ_Weight"]
    # cap at what's actually at EXT
    back["WeightToReturn"]   = back[ "WeightToReturn"].clip(lower=0, upper=back["OnHandWeightTotal"])
    back["CostToReturn"]     = (
        back["WeightToReturn"] /
        back["OnHandWeightTotal"]
    ) * back["OnHandCostTotal"]
    
    # metrics
    total_wt_return = back["WeightToReturn"].sum()
    total_cost_return = back["CostToReturn"].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("SKUs to Return",         back["SKU"].nunique())
    col2.metric("Total Weight to Return", f"{total_wt_return:,.0f} lb")
    col3.metric("Total Cost to Return",   f"${total_cost_return:,.0f}")
    
    # supplier filter
    sup2 = sorted(back["Supplier"].dropna().unique())
    chosen2 = st.multiselect("Filter Suppliers", sup2, default=sup2, key="mv2_sups")
    back = back[back["Supplier"].isin(chosen2)]
    
    if back.empty:
        st.info("No items match the current filters.")
    else:
        sel3 = alt.selection_multi(fields=["Supplier"], bind="legend")
        chart2 = (
            alt.Chart(back)
            .mark_bar()
            .encode(
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                x=alt.X("WeightToReturn:Q", title="Weight to Return (lb)"),
                color="Supplier:N",
                opacity=alt.condition(sel3, alt.value(1), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("SKU_Desc:N",       title="SKU"),
                    alt.Tooltip("Supplier:N",       title="Supplier"),
                    alt.Tooltip("FZ_WOH:Q",         title="Current FZ WOH"),
                    alt.Tooltip("DesiredFZ_Weight:Q", format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToReturn:Q", format=",.0f", title="Weight to Return"),
                    alt.Tooltip("CostToReturn:Q",   format=",.0f", title="Cost to Return"),
                ]
            )
            .add_selection(sel3)
            .properties(width=800, height=alt.Step(25))
            .resolve_scale(y="independent")
            .interactive()
        )
        st.altair_chart(theme(chart2), use_container_width=True)
    
        buf2 = io.BytesIO()
        back.to_excel(buf2, index=False, sheet_name="EXT2FZ")
        buf2.seek(0)
        st.download_button(
            "Download EXT→FZ List",
            buf2.getvalue(),
            file_name="EXT2FZ.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # ── Forecast: Weight On-Hand Needed ────────────────────────────────────────
    st.subheader("🔮 Forecasted On-Hand Weight Needed")
    
    # SKU & horizon selectors
    skus    = sorted(df["SKU"].unique())
    sku_sel = st.selectbox("SKU to forecast", skus, key="fc_sku")
    horizon = st.slider("Forecast horizon (weeks)", 1, 12, 4, key="fc_horizon")
    
    # build weekly usage series from cleaned df
    ts = (
        df[df["SKU"] == sku_sel]
        .dropna(subset=["Date"])               # drop any rows where date parse failed
        .set_index("Date")                     # use Date as index
        .resample("W")                         # weekly frequency
        [["TotalUsage"]]                       # keep only TotalUsage
        .sum()                                 # sum usage per week
        .reset_index()
        .rename(columns={"Date": "ds", "TotalUsage": "y"})
    )
    
    # check we have at least a few points
    if len(ts) < 2:
        st.warning(f"Not enough historical data for SKU {sku_sel} to forecast.")
    else:
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        m.fit(ts)
    
        future = m.make_future_dataframe(periods=horizon, freq="W")
        fcast  = m.predict(future)
        # calculate weight needed = forecast × desired WOH
        desired_woh = st.session_state.w2e_thr
        fcast["OnHandNeeded"] = fcast["yhat"] * desired_woh
    
        chart_fc = (
            alt.Chart(fcast)
            .mark_line()
            .encode(
                x=alt.X("ds:T", title="Week"),
                y=alt.Y("OnHandNeeded:Q", title="Forecasted Weight Needed (lb)")
            )
            .properties(
                title=f"SKU {sku_sel}: Forecasted On-Hand Weight for {desired_woh} WOH",
                width=800, height=400
            )
        )
        st.altair_chart(chart_fc, use_container_width=True)
    
    # ── Distribution of WOH ─────────────────────────────────────
    st.subheader("Distribution of WOH")

    bin_count = st.slider(
        "Number of bins (WOH dist)",
        min_value=10, max_value=100, value=40, step=5,
        key="woh_bins"
    )
    wo_max = st.slider(
        "Max WOH (weeks) to display",
        float(df["WeeksOnHand"].min()),
        float(df["WeeksOnHand"].quantile(0.99)),
        float(df["WeeksOnHand"].quantile(0.99)),
        step=1.0,
        key="woh_max"
    )

    filtered = df[df["WeeksOnHand"] <= wo_max]

    hist = (
        alt.Chart(filtered)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("WeeksOnHand:Q", bin=alt.Bin(maxbins=bin_count), title="WOH (weeks)"),
            y=alt.Y("count():Q", title="Count of SKUs"),
            tooltip=[
                alt.Tooltip("count():Q", title="Count"),
                alt.Tooltip("bin_maxbins(WeeksOnHand):Q", title="Bin end")
            ]
        )
    )
    dens = (
        alt.Chart(filtered)
        .transform_density(
            "WeeksOnHand",
            as_=["WeeksOnHand","density"],
            extent=[0, wo_max],
            counts=True,
            steps=bin_count
        )
        .mark_line(color="orange", size=3)
        .encode(
            x="WeeksOnHand:Q",
            y=alt.Y("density:Q", title="Density (counts)"),
            tooltip=[alt.Tooltip("density:Q", title="Density")]
        )
    )

    mean_val   = float(filtered["WeeksOnHand"].mean())
    median_val = float(filtered["WeeksOnHand"].median())

    mean_rule   = alt.Chart(pd.DataFrame({"value":[mean_val]})).mark_rule(color="red", size=2).encode(x="value:Q")
    median_rule = alt.Chart(pd.DataFrame({"value":[median_val]})).mark_rule(color="blue", strokeDash=[4,4], size=2).encode(x="value:Q")

    st.altair_chart(theme((hist + dens + mean_rule + median_rule).properties(height=350).interactive()),
                     use_container_width=True)
    st.markdown("**Red** = mean • **Blue (dashed)** = median • **Orange** = density")

    # ── Annual Turns Distribution ───────────────────────────────
    st.subheader("Annual Turns Distribution")

    turn_bins = st.slider(
        "Number of bins (Annual Turns)",
        min_value=10, max_value=100, value=30, step=5,
        key="turn_bins"
    )

    mean_at   = float(df["AnnualTurns"].mean())
    median_at = float(df["AnnualTurns"].median())

    hist2 = (
        alt.Chart(df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("AnnualTurns:Q", bin=alt.Bin(maxbins=turn_bins),
                    title="Annual Turns"),
            y=alt.Y("count():Q", title="Count of SKUs"),
            tooltip=[alt.Tooltip("count():Q", title="Count")]
        )
    )
    mean_line   = alt.Chart(pd.DataFrame({"value":[mean_at]})).mark_rule(color="red", size=2).encode(x="value:Q")
    median_line = alt.Chart(pd.DataFrame({"value":[median_at]})).mark_rule(color="blue", strokeDash=[4,4], size=2).encode(x="value:Q")
    chart2 = alt.layer(hist2, mean_line, median_line).properties(height=350).interactive()
    st.altair_chart(theme(chart2), use_container_width=True)
    st.markdown("**Red** = mean • **Blue (dashed)** = median")

    # ── Avg WOH by State ───────────────────────────────────────
    st.subheader("Average Weeks-On-Hand by State")
    avg_state = (
        df.groupby("ProductState", as_index=False)["WeeksOnHand"]
          .mean().rename(columns={"ProductState":"State","WeeksOnHand":"AvgWOH"})
    )
    avg_state["State"] = avg_state["State"].fillna("Unknown")

    order = avg_state.sort_values("AvgWOH")["State"].tolist()
    min_wo = st.slider(
        "Hide states with Avg WOH below",
        float(avg_state["AvgWOH"].min()),
        float(avg_state["AvgWOH"].max()),
        float(avg_state["AvgWOH"].min()),
        step=0.5,
        key="state_min"
    )
    filtered_state = avg_state[avg_state["AvgWOH"] >= min_wo]

    hover = alt.selection_point(fields=["State"], on="mouseover", empty="none")
    bars = (
        alt.Chart(filtered_state)
        .mark_bar()
        .encode(
            y=alt.Y("State:N", sort=order, title="State"),
            x=alt.X("AvgWOH:Q", title="Avg Weeks-On-Hand"),
            color=alt.condition(hover, alt.value("#4C78A8"), alt.value("#AAA")),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.7)),
            tooltip=["State","AvgWOH"]
        )
        .add_selection(hover)
    )
    labels = bars.mark_text(align="left", baseline="middle", dx=3).encode(text=alt.Text("AvgWOH:Q", format=".1f"))
    st.altair_chart(theme((bars + labels).properties(width=700, height=alt.Step(25))), use_container_width=True)

    # ── WOH Distribution by Protein ───────────────────────────
    st.subheader("WOH Distribution by Protein")
    order_p = (
        df.groupby("Protein")["WeeksOnHand"].median()
          .sort_values(ascending=False)
          .index.tolist()
    )
    sel = alt.selection_point(fields=["Protein"], bind="legend")

    box = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x="WeeksOnHand:Q", y=alt.Y("Protein:N", sort=order_p),                                
            color="Protein:N",
            opacity=alt.condition(sel, alt.value(1), alt.value(0.2))
        )
        .add_selection(sel)
    )

    jitter = (
        alt.Chart(df)
        .transform_calculate(y_jitter="(random() - 0.5) * 0.6")
        .mark_circle(size=18)
        .encode(
            x="WeeksOnHand:Q",
            y=alt.Y("Protein:N", sort=order_p),
            yOffset="y_jitter:Q",
            color="Protein:N",
            opacity=alt.condition(sel, alt.value(0.6), alt.value(0.1)),
            tooltip=["SKU_Desc","WeeksOnHand"]
        )
    )

    st.altair_chart(theme((box + jitter).properties(width=800, height=400).interactive()), use_container_width=True)
