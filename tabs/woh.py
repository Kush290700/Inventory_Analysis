import io
import streamlit as st
import altair as alt
import pandas as pd
import sys, os
import numpy as np
from prophet import Prophet
from math import ceil
import scipy.stats as ss

# ensure the project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.classification import compute_threshold_move

def render(df: pd.DataFrame, df_hc: pd.DataFrame, cost_df: pd.DataFrame, theme):
    st.header("📊 Weeks-On-Hand Analysis")

    # ── Compute PackCount & AvgWeightPerPack ───────────────────────────────
    # Map NumPacks from cost_df by SKU
    if "NumPacks" in cost_df.columns:
        packs_series = (
            pd.to_numeric(cost_df["NumPacks"], errors="coerce")
              .fillna(0)
              .astype(int)
        )
        pack_map = pd.Series(packs_series.values, index=cost_df["SKU"].astype(str))
        packs = df["SKU"].astype(str).map(pack_map)
    else:
        packs = pd.Series(np.nan, index=df.index)

    # fallback to ItemCount if available, else to 1
    if "ItemCount" in df.columns:
        item_counts = pd.to_numeric(df["ItemCount"], errors="coerce").fillna(1).astype(int)
    else:
        item_counts = pd.Series(1, index=df.index)

    df["PackCount"] = packs.fillna(item_counts).astype(int)
    df["AvgWeightPerPack"] = (
        df["OnHandWeightTotal"] /
        df["PackCount"].replace(0, np.nan)
    )

    # ── Precompute FZ & EXT, exclude zero-usage ─────────────────────────────
    fz  = df[
        (df["ProductState"].str.upper().str.startswith("FZ")) &
        (df["AvgWeeklyUsage"] > 0)
    ].copy()
    ext = df[
        (df["ProductState"].str.upper().str.startswith("EXT")) &
        (df["AvgWeeklyUsage"] > 0)
    ].copy()

    fz_woh            = fz.set_index("SKU")["WeeksOnHand"]
    fz_weight         = fz.set_index("SKU")["OnHandWeightTotal"]
    ext_weight_lookup = ext.set_index("SKU")["OnHandWeightTotal"]

    # ── Move FZ → EXT ───────────────────────────────────────────────────────
    st.subheader("🔄 Move FZ → EXT")
    thr1 = st.slider("Desired FZ WOH (weeks)", 0.0, 52.0, 4.0, 0.5, key="w2e_thr")

    to_move = fz[fz["WeeksOnHand"] > thr1].copy()
    to_move["DesiredFZ_Weight"] = to_move["AvgWeeklyUsage"] * thr1
    to_move["WeightToMove"]     = to_move["OnHandWeightTotal"] - to_move["DesiredFZ_Weight"]
    to_move["CostToMove"]       = (
        to_move["WeightToMove"] / to_move["OnHandWeightTotal"]
    ) * to_move["OnHandCostTotal"]
    to_move["EXT_Weight"]       = to_move["SKU"].map(ext_weight_lookup).fillna(0)
    to_move["TotalOnHand"]      = to_move["OnHandWeightTotal"] + to_move["EXT_Weight"]

    mv_pos        = to_move["WeightToMove"] > 0
    total_wt_move = to_move.loc[mv_pos, "WeightToMove"].sum()
    total_cost    = to_move.loc[mv_pos, "CostToMove"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs to Move",         to_move["SKU"].nunique())
    c2.metric("Total Weight to Move", f"{total_wt_move:,.0f} lb")
    c3.metric("Total Cost to Move",   f"${total_cost:,.0f}")

    suppliers = sorted(to_move["Supplier"].dropna().unique())
    sel_sup   = st.multiselect("Filter Suppliers", suppliers, default=suppliers, key="mv1_sups")
    mv1       = to_move[mv_pos & to_move["Supplier"].isin(sel_sup)]

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
                    alt.Tooltip("SKU_Desc:N",          title="SKU"),
                    alt.Tooltip("Supplier:N",          title="Supplier"),
                    alt.Tooltip("WeeksOnHand:Q",       title="Current FZ WOH"),
                    alt.Tooltip("OnHandWeightTotal:Q", format=",.0f", title="FZ On-Hand Wt"),
                    alt.Tooltip("EXT_Weight:Q",        format=",.0f", title="EXT On-Hand Wt"),
                    alt.Tooltip("TotalOnHand:Q",       format=",.0f", title="Total On-Hand Wt"),
                    alt.Tooltip("PackCount:Q",         title="Case or Packs Available"),
                    alt.Tooltip("AvgWeightPerPack:Q",  format=",.1f", title="Avg Wt/Case or Pack"),
                    alt.Tooltip("DesiredFZ_Weight:Q",  format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToMove:Q",      format=",.0f", title="Weight to Move"),
                    alt.Tooltip("CostToMove:Q",        format=",.0f", title="Cost to Move"),
                ]
            )
            .add_selection(sel2)
            .properties(width=800, height=alt.Step(25))
            .resolve_scale(y="independent")
            .interactive()
        )
        st.altair_chart(theme(chart1), use_container_width=True)

      # 1) Define exactly the columns you want
        export_cols_fz2ext = [
            "SKU_Desc", "ProductState", "Supplier",
            "OnHandWeightTotal", "TotalShippedLb", "TotalProductionLb",
            "TotalUsage", "AvgWeeklyUsage", "WeeksOnHand",
            "PackCount", "DesiredFZ_Weight", "WeightToMove", "EXT_Weight"
        ]
    
        # 2) Build a pared‐down DataFrame (you may need to rename your columns if they differ)
        mv1_export = mv1.rename(columns={"ProductState":"ProductState",
                                         "EXT_Weight":"EXT_Weight"})  # ensure column names match
        mv1_export = mv1_export[export_cols_fz2ext]
    
        # 3) Write that to Excel & expose download
        buf1 = io.BytesIO()
        mv1_export.to_excel(buf1, index=False, sheet_name="FZ2EXT")
        buf1.seek(0)
        st.download_button(
            "Download FZ→EXT List",
            buf1.getvalue(),
            file_name="FZ2EXT_SelectedCols.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ── Move EXT → FZ ───────────────────────────────────────────────────────
    st.subheader("🔄 Move EXT → FZ")
    thr2_default = 1.0
    try:
        if not df_hc.empty:
            thr2_default = float(compute_threshold_move(ext, df_hc))
    except:
        pass

    thr2 = st.slider(
        "Desired FZ WOH to achieve",
        0.0, float(fz_woh.max()), thr2_default, step=0.25, key="e2f_thr"
    )

    back = ext[ext["SKU"].map(fz_woh).fillna(0) < thr2].copy()
    back["FZ_Weight"]        = back["SKU"].map(fz_weight).fillna(0)
    back["FZ_WOH"]           = back["SKU"].map(fz_woh).fillna(0)
    back["DesiredFZ_Weight"] = back["AvgWeeklyUsage"] * thr2
    back["WeightToReturn"]   = back["DesiredFZ_Weight"].sub(back["FZ_Weight"]).clip(0, back["OnHandWeightTotal"])
    back["CostToReturn"]     = (back["WeightToReturn"] / back["OnHandWeightTotal"]) * back["OnHandCostTotal"]
    back["TotalOnHand"]      = back["OnHandWeightTotal"] + back["FZ_Weight"]

    total_wt_return   = back["WeightToReturn"].sum()
    total_cost_return = back["CostToReturn"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("SKUs to Return",         back["SKU"].nunique())
    col2.metric("Total Weight to Return", f"{total_wt_return:,.0f} lb")
    col3.metric("Total Cost to Return",   f"${total_cost_return:,.0f}")

    sup2    = sorted(back["Supplier"].dropna().unique())
    chosen2 = st.multiselect("Filter Suppliers", sup2, default=sup2, key="mv2_sups")
    back    = back[back["Supplier"].isin(chosen2)]

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
                    alt.Tooltip("SKU_Desc:N",           title="SKU"),
                    alt.Tooltip("Supplier:N",           title="Supplier"),
                    alt.Tooltip("OnHandWeightTotal:Q",  format=",.0f", title="EXT On-Hand Wt"),
                    alt.Tooltip("FZ_Weight:Q",          format=",.0f", title="FZ On-Hand Wt"),
                    alt.Tooltip("TotalOnHand:Q",        format=",.0f", title="Total On-Hand Wt"),
                    alt.Tooltip("PackCount:Q",          title="Case or Packs Available"),
                    alt.Tooltip("AvgWeightPerPack:Q",   format=",.1f", title="Avg Wt/Case or Pack"),
                    alt.Tooltip("FZ_WOH:Q",             title="Current FZ WOH"),
                    alt.Tooltip("DesiredFZ_Weight:Q",   format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToReturn:Q",     format=",.0f", title="Weight to Return"),
                    alt.Tooltip("CostToReturn:Q",       format=",.0f", title="Cost to Return"),
                ]
            )
            .add_selection(sel3)
            .properties(width=800, height=alt.Step(25))
            .resolve_scale(y="independent")
            .interactive()
        )
        st.altair_chart(theme(chart2), use_container_width=True)

        export_cols_ext2fz = [
                "SKU_Desc", "ProductState", "Supplier",
                "OnHandWeightTotal", "TotalShippedLb", "TotalProductionLb",
                "TotalUsage", "AvgWeeklyUsage", "WeeksOnHand",
                "PackCount", "DesiredFZ_Weight", "WeightToReturn", "FZ_Weight"
            ]
        
            back_export = back.rename(columns={"ProductState":"ProductState",
                                               "FZ_Weight":"FZ_Weight"})
            back_export = back_export[export_cols_ext2fz]
        
            buf2 = io.BytesIO()
            back_export.to_excel(buf2, index=False, sheet_name="EXT2FZ")
            buf2.seek(0)
            st.download_button(
                "Download EXT→FZ List",
                buf2.getvalue(),
                file_name="EXT2FZ_SelectedCols.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    
    # ── Purchase Recommendations by Desired WOH ──────────────────────────────
    st.subheader("🛒 Purchase Recommendations by Desired WOH")
    
    # 1) User input: target Weeks‐On‐Hand
    desired_woh = st.slider(
        "Desired Weeks-On-Hand",
        0.0, 52.0, 4.0, 0.5,
        help="How many weeks’ worth of stock you want on hand"
    )
    
    # 2) Roll up across all states
    inv = (
        df
        .groupby(["SKU", "SKU_Desc"], as_index=False)
        .agg({
            "AvgWeeklyUsage":    "mean",
            "OnHandWeightTotal": "sum",
            "OnHandCostTotal":   "sum"
        })
        .rename(columns={
            "AvgWeeklyUsage":    "MeanUse",
            "OnHandWeightTotal": "InvWt",
            "OnHandCostTotal":   "InvCost"
        })
    )
    
    # 3) Map pack‐count from cost_df
    if "NumPacks" in cost_df.columns:
        packs_series = (
            pd.to_numeric(cost_df["NumPacks"], errors="coerce")
              .fillna(1)
              .astype(int)
        )
        pack_map = pd.Series(packs_series.values, index=cost_df["SKU"].astype(str))
        inv["PackCount"] = inv["SKU"].astype(str).map(pack_map).fillna(1).astype(int)
    else:
        inv["PackCount"] = 1
    
    # 4) Compute weight per pack
    inv["PackWt"] = inv["InvWt"] / inv["PackCount"]
    
    # 5) Pounds needed to hit desired_woh
    inv["DesiredWt"] = inv["MeanUse"] * desired_woh
    inv["ToBuyWt"]   = (inv["DesiredWt"] - inv["InvWt"]).clip(lower=0)
    
    # 6) Packs to order (round up to whole packs) & actual order weight
    inv["PacksToOrder"] = np.ceil(inv["ToBuyWt"] / inv["PackWt"]).astype(int)
    inv["OrderWt"]      = inv["PacksToOrder"] * inv["PackWt"]
    
    # 7) Cost calculations
    inv["CostPerLb"] = inv["InvCost"] / inv["InvWt"]
    inv["EstCost"]   = inv["OrderWt"] * inv["CostPerLb"]
    
    # 8) Sort high‐usage first
    inv = inv.sort_values(["MeanUse","EstCost"], ascending=[False, True])
    
    # 9) Display essentials
    display = (
        inv[[
          "SKU","SKU_Desc",
          "InvWt","DesiredWt",
          "PackCount","PacksToOrder","OrderWt","EstCost"
        ]]
        .assign(
          InvWt=lambda x: x["InvWt"].map("{:,.0f} lb".format),
          DesiredWt=lambda x: x["DesiredWt"].map("{:,.0f} lb".format),
          OrderWt=lambda x: x["OrderWt"].map("{:,.0f} lb".format),
          EstCost=lambda x: x["EstCost"].map("${:,.2f}".format)
        )
    )
    st.dataframe(display, use_container_width=True)
    
    # 10) Downloadable order plan
    buf = io.BytesIO()
    inv.to_excel(buf, index=False, sheet_name="PurchasePlan")
    buf.seek(0)
    st.download_button(
        "📥 Download Purchase Plan",
        data=buf.getvalue(),
        file_name="Purchase_Plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    # ── Distribution of WOH ─────────────────────────────────────
    st.subheader("Distribution of Weeks-On-Hand")

    # 1) Summary metrics
    p25, p50, p75, p90 = (
        df["WeeksOnHand"].quantile(q) for q in (0.25, 0.50, 0.75, 0.90)
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("25th percentile", f"{p25:.1f} weeks")
    m2.metric("Median (50th)",   f"{p50:.1f} weeks")
    m3.metric("75th percentile", f"{p75:.1f} weeks")
    m4.metric("90th percentile", f"{p90:.1f} weeks")

    # 2) Interactive threshold
    thresh = st.slider(
        "Highlight SKUs with WOH ≤ …",
        float(df["WeeksOnHand"].min()),
        float(df["WeeksOnHand"].max()),
        float(p50),
        key="woh_threshold"
    )
    below_count = int((df["WeeksOnHand"] <= thresh).sum())
    st.markdown(f"**{below_count:,}** SKUs have WOH ≤ {thresh:.1f} weeks")

    # 3) Histogram + density + CDF overlay
    filtered = df[df["WeeksOnHand"] <= float(df["WeeksOnHand"].quantile(0.99))]

    bins = st.slider(
        "Number of bins",
        10, 100, 40, step=5,
        key="woh_hist_bins"
    )

    base = alt.Chart(filtered).encode(
        x=alt.X("WeeksOnHand:Q", title="Weeks-On-Hand")
    )

    hist = base.mark_bar(opacity=0.6).encode(
        y=alt.Y("count():Q", title="SKU Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")]
    )

    density = base.transform_density(
        "WeeksOnHand",
        as_=["WeeksOnHand","density"],
        extent=[0, float(filtered["WeeksOnHand"].max())],
        counts=True,
        steps=bins
    ).mark_line(color="orange", size=2).encode(
        y=alt.Y("density:Q", title="Density")
    )

    # cumulative distribution
    cdf = (
        base.transform_window(
            cumulative="count()",
            sort=[{"field":"WeeksOnHand"}]
        )
        .transform_joinaggregate(
            total="count()"
        )
        .transform_calculate(
            cum_pct="datum.cumulative / datum.total"
        )
        .mark_line(color="green", strokeDash=[4,2])
        .encode(
            y=alt.Y("cum_pct:Q", title="Cumulative %", axis=alt.Axis(format="%")),
        )
    )

    # combine
    chart = alt.layer(hist, density, cdf).resolve_scale(
        y="independent"
    ).properties(height=350)

    st.altair_chart(theme(chart).interactive(), use_container_width=True)


    # ── Annual Turns Distribution ───────────────────────────────
    st.subheader("Annual Turns Distribution")

    # Summary percentiles
    at25, at50, at75 = (
        df["AnnualTurns"].quantile(q) for q in (0.25, 0.50, 0.75)
    )
    a1, a2, a3 = st.columns(3)
    a1.metric("25th percentile", f"{at25:.1f}")
    a2.metric("Median (50th)",   f"{at50:.1f}")
    a3.metric("75th percentile", f"{at75:.1f}")

    turn_bins = st.slider(
        "Number of bins (Annual Turns)",
        10, 100, 30, step=5,
        key="turn_bins"
    )

    hist2 = alt.Chart(df).mark_bar(opacity=0.6).encode(
        x=alt.X("AnnualTurns:Q", bin=alt.Bin(maxbins=turn_bins), title="Annual Turns"),
        y=alt.Y("count():Q", title="SKU Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")]
    )

    # add median & mean lines
    mean_at   = float(df["AnnualTurns"].mean())
    median_at = float(at50)
    line_mean   = alt.Chart(pd.DataFrame({"v":[mean_at]})).mark_rule(color="red").encode(x="v:Q")
    line_median = alt.Chart(pd.DataFrame({"v":[median_at]})).mark_rule(color="blue", strokeDash=[4,4]).encode(x="v:Q")

    st.altair_chart(theme(alt.layer(hist2, line_mean, line_median).properties(height=350)).interactive(),
                     use_container_width=True)
    st.markdown("**Red** = mean • **Blue (dashed)** = median")


    # ── Avg WOH by State ───────────────────────────────────────
    st.subheader("Average WOH by State")

    avg_state = (
        df.groupby("ProductState", as_index=False)
          .agg(
             AvgWOH=("WeeksOnHand","mean"),
             Count = ("SKU","nunique")
          )
          .sort_values("AvgWOH")
    )
    state_min = st.slider(
        "Hide states with Avg WOH below",
        float(avg_state["AvgWOH"].min()),
        float(avg_state["AvgWOH"].max()),
        float(avg_state["AvgWOH"].min()),
        step=0.5,
        key="state_min"
    )
    filtered_state = avg_state[avg_state["AvgWOH"] >= state_min]

    bars = alt.Chart(filtered_state).mark_bar().encode(
        y=alt.Y("ProductState:N", sort=filtered_state["ProductState"].tolist(), title="State"),
        x=alt.X("AvgWOH:Q", title="Avg Weeks-On-Hand"),
        tooltip=["ProductState","AvgWOH","Count"]
    )

    labels = bars.mark_text(
        align="left",
        baseline="middle",
        dx=5
    ).encode(text="Count:Q")

    st.altair_chart(theme((bars + labels).properties(width=700, height=alt.Step(25))).interactive(),
                     use_container_width=True)


    # ── WOH Distribution by Protein ───────────────────────────
    st.subheader("WOH Distribution by Protein")

    # overall counts
    total_skus = df.shape[0]
    total_prots= df["Protein"].nunique()
    st.markdown(f"**{total_skus:,}** SKUs across **{total_prots}** Proteins")

    order_p = (
        df.groupby("Protein")["WeeksOnHand"]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )

    sel = alt.selection_point(fields=["Protein"], bind="legend")

    box = alt.Chart(df).mark_boxplot(extent="min-max").encode(
        x="WeeksOnHand:Q",
        y=alt.Y("Protein:N", sort=order_p),
        color="Protein:N",
        opacity=alt.condition(sel, alt.value(1), alt.value(0.2)),
        tooltip=[
            alt.Tooltip("Protein:N", title="Protein"),
            alt.Tooltip("count():Q",   title="SKUs", aggregate="count"),
            alt.Tooltip("median(WeeksOnHand):Q", title="Median WOH")
        ]
    ).add_selection(sel)

    jitter = alt.Chart(df).transform_calculate(
        y_jitter="(random() - 0.5) * 0.6"
    ).mark_circle(size=18).encode(
        x="WeeksOnHand:Q",
        y=alt.Y("Protein:N", sort=order_p),
        yOffset="y_jitter:Q",
        color="Protein:N",
        opacity=alt.condition(sel, alt.value(0.6), alt.value(0.1)),
        tooltip=["SKU_Desc","Protein","WeeksOnHand"]
    )

    st.altair_chart(theme((box + jitter).properties(width=800, height=400)).interactive(),
                     use_container_width=True)
