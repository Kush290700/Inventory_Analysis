import io
import os
import sys
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Ensure the project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.classification import compute_threshold_move

@st.cache_data(show_spinner=False)
def compute_parent_purchase_plan(
    df_inv: pd.DataFrame,
    pd_detail: pd.DataFrame,
    cost_df: pd.DataFrame,
    desired_woh: float
) -> pd.DataFrame:
    # --- Input validation as before ---
    required_cols = {
        'df_inv': ['SKU', 'AvgWeeklyUsage', 'OnHandWeightTotal', 'OnHandCostTotal', 'SKU_Desc'],
        'pd_detail': ['Product Code', 'Velocity Parent', 'Description'],
        'cost_df': ['SKU']
    }
    for df_name, cols in required_cols.items():
        df = locals()[df_name]
        if df_name == 'cost_df' and df.empty:
            continue
        if df.empty:
            st.error(f"{df_name} is empty")
            return pd.DataFrame()
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in {df_name}: {missing_cols}")
            return pd.DataFrame()

    # --- Clean and prepare detail sheet ---
    d = pd_detail.copy()
    d.columns = [str(c).strip() for c in d.columns]
    d = d.rename(columns={
        "Product Code": "SKU",
        "Velocity Parent": "ParentSKU",
        "Description": "ParentDesc"
    })
    for col in ["SKU", "ParentSKU", "ParentDesc"]:
        if col not in d.columns:
            d[col] = ""
    d["SKU"] = d["SKU"].fillna("").astype(str).str.strip()
    d["ParentSKU"] = d["ParentSKU"].fillna("").astype(str).str.strip()
    mask = (d["ParentSKU"].str.lower().isin(["", "nan", "none", "null"])) | (d["ParentSKU"].isna())
    d.loc[mask, "ParentSKU"] = d.loc[mask, "SKU"]

    # --- Build child->parent and parent description maps ---
    child_to_parent = dict(zip(d["SKU"], d["ParentSKU"]))
    parent_desc_map = dict(zip(d["ParentSKU"], d["ParentDesc"]))

    # --- Map each SKU to its parent ---
    inv = df_inv.copy()
    inv["SKU"] = inv["SKU"].astype(str).str.strip()
    inv["SKU_Desc"] = inv["SKU_Desc"].astype(str).str.strip()
    inv["ParentSKU"] = inv["SKU"].map(child_to_parent).fillna(inv["SKU"])

    # --- Aggregate all stats to parent SKU level ---
    parent_df = (
        inv.groupby("ParentSKU", as_index=False)
           .agg(
               MeanUse=("AvgWeeklyUsage", "sum"),
               InvWt=("OnHandWeightTotal", "sum"),
               InvCost=("OnHandCostTotal", "sum")
           )
    )

    parent_df["DesiredWt"] = parent_df["MeanUse"] * desired_woh
    parent_df["ToBuyWt"] = (parent_df["DesiredWt"] - parent_df["InvWt"]).clip(lower=0)

    # --- Assign pack count (average across children if available, or 1) ---
    if not cost_df.empty and "NumPacks" in cost_df.columns:
        pack_map = pd.Series(
            pd.to_numeric(cost_df["NumPacks"], errors="coerce").fillna(1).astype(int).clip(lower=1).values,
            index=cost_df["SKU"].astype(str)
        ).to_dict()
        parent_df["PackCount"] = parent_df["ParentSKU"].map(pack_map).fillna(1).astype(int)
    else:
        parent_df["PackCount"] = 1

    # --- Order weight logic: ---
    parent_df["PackWt"] = np.where(
        (parent_df["InvWt"] > 0) & (parent_df["PackCount"] > 0),
        parent_df["InvWt"] / parent_df["PackCount"],
        0
    )
    parent_df["PacksToOrder"] = np.where(
        parent_df["PackWt"] > 0,
        np.ceil(parent_df["ToBuyWt"] / parent_df["PackWt"]),
        0
    ).astype(int)
    parent_df["OrderWt"] = parent_df["PacksToOrder"] * parent_df["PackWt"]

    # --- Add descriptions ---
    parent_df["SKU"] = parent_df["ParentSKU"]
    parent_df["SKU_Desc"] = parent_df["SKU"].map(parent_desc_map).fillna(parent_df["SKU"])

    # --- Cost metrics ---
    parent_df["CostPerLb"] = np.where(
        parent_df["InvWt"] > 0,
        parent_df["InvCost"] / parent_df["InvWt"],
        0
    )
    parent_df["EstCost"] = parent_df["OrderWt"] * parent_df["CostPerLb"]

    # --- Only return parents that actually need to be ordered ---
    # (If total parent+children inv is enough, ToBuyWt=0)
    plan = parent_df[parent_df["PacksToOrder"] > 0][
        ["SKU", "SKU_Desc", "InvWt", "DesiredWt", "PackCount", "PacksToOrder", "OrderWt", "EstCost"]
    ].copy()

    # Reset index for clean display/export
    plan.reset_index(drop=True, inplace=True)
    return plan

def render(df, df_hc, cost_df, theme, sheets):
    core_cols = [
        "SKU", "WeeksOnHand", "AvgWeeklyUsage", "OnHandWeightTotal",
        "OnHandCostTotal", "SKU_Desc", "ProductState", "Supplier"
    ]
    if df.empty:
        st.warning("No inventory data available.")
        return
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        st.error(f"Missing core columns in inventory data: {missing_core}")
        return
    df = df.copy()
    st.header("📊 Weeks-On-Hand Analysis")
    if not cost_df.empty and "NumPacks" in cost_df.columns and "SKU" in cost_df.columns:
        packs_series = (
            pd.to_numeric(cost_df["NumPacks"], errors="coerce")
              .fillna(1)
              .astype(int)
              .clip(lower=1)
        )
        pack_map = pd.Series(packs_series.values, index=cost_df["SKU"].astype(str))
        packs = df["SKU"].astype(str).map(pack_map)
    else:
        packs = pd.Series(1, index=df.index)
    if "ItemCount" in df.columns:
        item_counts = pd.to_numeric(df["ItemCount"], errors="coerce").fillna(1).astype(int).clip(lower=1)
    else:
        item_counts = pd.Series(1, index=df.index)
    df["PackCount"] = packs.fillna(item_counts).astype(int)
    df["AvgWeightPerPack"] = np.where(
        df["PackCount"] > 0,
        df["OnHandWeightTotal"] / df["PackCount"],
        df["OnHandWeightTotal"]
    )
    # FZ & EXT logic
    state = df["ProductState"].fillna("").str.upper()
    fz = df[(state.str.startswith("FZ")) & (df["AvgWeeklyUsage"] > 0)].copy()
    ext = df[(state.str.startswith("EXT")) & (df["AvgWeeklyUsage"] > 0)].copy()
    fz_woh = fz.set_index("SKU")["WeeksOnHand"]
    ext_weight_lookup = ext.set_index("SKU")["OnHandWeightTotal"]

    # FZ→EXT
    st.subheader("🔄 Move FZ → EXT")
    thr1 = st.slider("Desired FZ WOH (weeks)", 0.0, 4.0, 1.5, 0.25, key="w2e_thr")
    to_move = fz[fz["WeeksOnHand"] > thr1].copy()
    to_move["DesiredFZ_Weight"] = to_move["AvgWeeklyUsage"] * thr1
    to_move["WeightToMove"] = (to_move["OnHandWeightTotal"] - to_move["DesiredFZ_Weight"]).clip(lower=0)
    to_move["EXT_Weight"] = to_move["SKU"].map(ext_weight_lookup).fillna(0)
    to_move["TotalOnHand"] = to_move["OnHandWeightTotal"] + to_move["EXT_Weight"]
    mv1 = to_move[to_move["WeightToMove"] > 0]
    total_wt_move = mv1["WeightToMove"].sum()
    total_cost_move = np.where(
        mv1["OnHandWeightTotal"] > 0,
        (mv1["WeightToMove"] / mv1["OnHandWeightTotal"]) * mv1["OnHandCostTotal"],
        0
    ).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs to Move", mv1["SKU"].nunique())
    c2.metric("Total Weight to Move", f"{total_wt_move:,.0f} lb")
    c3.metric("Total Cost to Move", f"${total_cost_move:,.0f}")
    suppliers = sorted(mv1["Supplier"].dropna().unique())
    sel_sup = st.multiselect("Filter Suppliers", suppliers, default=suppliers, key="mv1_sups")
    mv1 = mv1[mv1["Supplier"].isin(sel_sup)]
    if not mv1.empty:
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
                    alt.Tooltip("WeeksOnHand:Q", title="Current FZ WOH", format=".1f"),
                    alt.Tooltip("OnHandWeightTotal:Q", format=",.0f", title="FZ On-Hand Wt"),
                    alt.Tooltip("EXT_Weight:Q", format=",.0f", title="EXT On-Hand Wt"),
                    alt.Tooltip("TotalOnHand:Q", format=",.0f", title="Total On-Hand Wt"),
                    alt.Tooltip("PackCount:Q", title="Case or Packs Available"),
                    alt.Tooltip("AvgWeightPerPack:Q", format=",.1f", title="Avg Wt/Pack"),
                    alt.Tooltip("DesiredFZ_Weight:Q", format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToMove:Q", format=",.0f", title="Weight to Move"),
                ]
            )
            .add_selection(sel2)
            .properties(height=alt.Step(25))
        )
        st.altair_chart(theme(chart1).interactive(), use_container_width=True)
    if not mv1.empty:
        export_cols = [
            "SKU_Desc", "ProductState", "Supplier", "OnHandWeightTotal",
            "TotalShippedLb", "TotalProductionLb", "TotalUsage", "AvgWeeklyUsage",
            "WeeksOnHand", "PackCount", "DesiredFZ_Weight", "WeightToMove", "EXT_Weight"
        ]
        export_cols = [c for c in export_cols if c in mv1.columns]
        buf1 = io.BytesIO()
        mv1[export_cols].to_excel(buf1, index=False, sheet_name="FZ2EXT")
        buf1.seek(0)
        st.download_button(
            "Download FZ→EXT List",
            buf1.getvalue(),
            file_name="FZ2EXT_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # EXT→FZ
    st.subheader("🔄 Move EXT → FZ")
    thr2_default = 1.0
    try:
        if not df_hc.empty:
            thr2_default = float(compute_threshold_move(ext, df_hc))
    except Exception as e:
        st.warning(f"Error computing threshold for EXT → FZ: {str(e)}")
    thr2 = st.slider(
        "Desired FZ WOH to achieve",
        0.0, float(fz_woh.max()) if not fz_woh.empty else 52.0, thr2_default, step=0.25,
        key="e2f_thr"
    )
    back = ext[ext["SKU"].map(fz_woh).fillna(0) < thr2].copy()
    back["FZ_Weight"] = back["SKU"].map(fz_woh).fillna(0)
    back["DesiredFZ_Weight"] = back["AvgWeeklyUsage"] * thr2
    back["WeightToReturn"] = back["DesiredFZ_Weight"].sub(back["FZ_Weight"]).clip(lower=0)
    back["TotalOnHand"] = back["OnHandWeightTotal"] + back["FZ_Weight"]
    total_wt_ret = back["WeightToReturn"].sum()
    total_cost_ret = np.where(
        back["OnHandWeightTotal"] > 0,
        (back["WeightToReturn"] / back["OnHandWeightTotal"]) * back["OnHandCostTotal"],
        0
    ).sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("SKUs to Return", back["SKU"].nunique())
    col2.metric("Total Weight to Return", f"{total_wt_ret:,.0f} lb")
    col3.metric("Total Cost to Return", f"${total_cost_ret:,.0f}")
    sup2 = sorted(back["Supplier"].dropna().unique())
    chosen2 = st.multiselect("Filter Suppliers", sup2, default=sup2, key="mv2_sups")
    back = back[back["Supplier"].isin(chosen2)]
    if not back.empty:
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
                    alt.Tooltip("SKU_Desc:N", title="SKU"),
                    alt.Tooltip("Supplier:N", title="Supplier"),
                    alt.Tooltip("WeeksOnHand:Q", title="Current WOH", format=".1f"),
                    alt.Tooltip("OnHandWeightTotal:Q", format=",.0f", title="EXT On-Hand Wt"),
                    alt.Tooltip("FZ_Weight:Q", format=",.0f", title="FZ On-Hand Wt"),
                    alt.Tooltip("TotalOnHand:Q", format=",.0f", title="Total On-Hand Wt"),
                    alt.Tooltip("PackCount:Q", title="Case or Packs Available"),
                    alt.Tooltip("AvgWeightPerPack:Q", format=",.1f", title="Avg Wt/Pack"),
                    alt.Tooltip("DesiredFZ_Weight:Q", format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToReturn:Q", format=",.0f", title="Weight to Return"),
                ]
            )
            .add_selection(sel3)
            .properties(height=alt.Step(25))
        )
        st.altair_chart(theme(chart2).interactive(), use_container_width=True)
    if not back.empty:
        export_cols = [
            "SKU_Desc", "ProductState", "Supplier", "OnHandWeightTotal",
            "TotalShippedLb", "TotalProductionLb", "TotalUsage", "AvgWeeklyUsage",
            "WeeksOnHand", "PackCount", "DesiredFZ_Weight", "WeightToReturn", "FZ_Weight"
        ]
        export_cols = [c for c in export_cols if c in back.columns]
        buf2 = io.BytesIO()
        back[export_cols].to_excel(buf2, index=False, sheet_name="EXT2FZ")
        buf2.seek(0)
        st.download_button(
            "Download EXT→FZ List",
            buf2.getvalue(),
            file_name="EXT2FZ_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # --- Purchase Recommendations by Desired WOH ---
    st.subheader("🛒 Purchase Recommendations by Desired WOH")
    supplier_opts = ["All"] + sorted(df["Supplier"].astype(str).unique())
    selected_supplier = st.selectbox("Supplier", supplier_opts, key="pr_supplier")
    df_pr = df if selected_supplier == "All" else df[df["Supplier"] == selected_supplier]
    desired_woh = st.slider(
        "Desired Weeks-On-Hand", 0.0, 12.0, 4.0, 0.5,
        help="How many weeks’ worth of stock you want on hand"
    )
    pd_detail = sheets.get("Product Detail", pd.DataFrame())
    plan_df = compute_parent_purchase_plan(df_pr, pd_detail, cost_df, desired_woh)
    # Display parent-level plan
    if not plan_df.empty:
        display = (
            plan_df[[
                "SKU", "SKU_Desc", "InvWt", "DesiredWt",
                "PackCount", "PacksToOrder", "OrderWt", "EstCost"
            ]]
            .assign(
                InvWt=lambda d: d["InvWt"].map("{:,.0f} lb".format),
                DesiredWt=lambda d: d["DesiredWt"].map("{:,.0f} lb".format),
                OrderWt=lambda d: d["OrderWt"].map("{:,.0f} lb".format),
                EstCost=lambda d: d["EstCost"].map("${:,.2f}".format)
            )
        )
        st.dataframe(display, use_container_width=True)
        buf = io.BytesIO()
        plan_df.to_excel(buf, index=False, sheet_name="PurchasePlan")
        buf.seek(0)
        st.download_button(
            "📥 Download Purchase Plan",
            data=buf.getvalue(),
            file_name="Purchase_Plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No purchase plan generated due to invalid or missing data.")

    # --- Distribution of WOH ---
    st.subheader("Distribution of Weeks-On-Hand")
    if df["WeeksOnHand"].isna().all() or df["WeeksOnHand"].empty:
        st.warning("Cannot compute Weeks-On-Hand metrics: all values are missing.")
    else:
        p25, p50, p75, p90 = (
            df["WeeksOnHand"].quantile(q) for q in (0.25, 0.50, 0.75, 0.90)
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("25th percentile", f"{p25:.1f} weeks")
        m2.metric("Median (50th)",   f"{p50:.1f} weeks")
        m3.metric("75th percentile", f"{p75:.1f} weeks")
        m4.metric("90th percentile", f"{p90:.1f} weeks")
        thresh = st.slider(
            "Highlight SKUs with WOH ≤ …",
            float(df["WeeksOnHand"].min()) if not df["WeeksOnHand"].empty else 0.0,
            float(df["WeeksOnHand"].max()) if not df["WeeksOnHand"].empty else 52.0,
            float(p50) if not df["WeeksOnHand"].empty else 4.0,
            key="woh_threshold"
        )
        below_count = int((df["WeeksOnHand"] <= thresh).sum())
        st.markdown(f"**{below_count:,}** SKUs have WOH ≤ {thresh:.1f} weeks")
        q99 = df["WeeksOnHand"].quantile(0.99)
        if pd.isna(q99) or df["WeeksOnHand"].empty:
            st.warning("Cannot draw histogram: all WeeksOnHand values are missing.")
        else:
            filtered = df[df["WeeksOnHand"] <= q99]
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
                tooltip=[alt.Tooltip("count():Q", title="Count", format=",.0f")]
            )
            density = base.transform_density(
                "WeeksOnHand",
                as_=["WeeksOnHand", "density"],
                extent=[0, float(filtered["WeeksOnHand"].max())] if not filtered.empty else [0, 52.0],
                counts=True,
                steps=bins
            ).mark_line(color="orange", size=2).encode(
                y=alt.Y("density:Q", title="Density")
            )
            cdf = (
                base.transform_window(
                    cumulative="count()",
                    sort=[{"field": "WeeksOnHand"}]
                )
                .transform_joinaggregate(
                    total="count()"
                )
                .transform_calculate(
                    cum_pct="datum.cumulative / datum.total"
                )
                .mark_line(color="green", strokeDash=[4, 2])
                .encode(
                    y=alt.Y("cum_pct:Q", title="Cumulative %", axis=alt.Axis(format="%"))
                )
            )
            chart = alt.layer(hist, density, cdf).resolve_scale(
                y="independent"
            ).properties(height=350)
            st.altair_chart(theme(chart).interactive(), use_container_width=True)

    # --- Annual Turns Distribution ---
    if "AnnualTurns" in df.columns and not df["AnnualTurns"].isna().all():
        st.subheader("Annual Turns Distribution")
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
            tooltip=[alt.Tooltip("count():Q", title="Count", format=",.0f")]
        )
        mean_at = float(df["AnnualTurns"].mean())
        median_at = float(at50)
        line_mean = alt.Chart(pd.DataFrame({"v": [mean_at]})).mark_rule(color="red").encode(x="v:Q")
        line_median = alt.Chart(pd.DataFrame({"v": [median_at]})).mark_rule(color="blue", strokeDash=[4, 4]).encode(x="v:Q")
        combo = alt.layer(hist2, line_mean, line_median).properties(height=350)
        st.altair_chart(theme(combo).interactive(), use_container_width=True)
        st.markdown("**Red** = mean • **Blue (dashed)** = median")
    else:
        st.warning("Cannot display Annual Turns Distribution: AnnualTurns column missing or all values are NaN.")

    # --- Avg WOH by State ---
    st.subheader("Average WOH by State")
    avg_state = (
        df.groupby("ProductState", as_index=False)
        .agg(AvgWOH=("WeeksOnHand", "mean"), Count=("SKU", "nunique"))
        .sort_values("AvgWOH")
    )
    state_min = st.slider(
        "Hide states with Avg WOH below",
        float(avg_state["AvgWOH"].min()) if not avg_state.empty else 0.0,
        float(avg_state["AvgWOH"].max()) if not avg_state.empty else 52.0,
        float(avg_state["AvgWOH"].min()) if not avg_state.empty else 0.0,
        step=0.5,
        key="state_min"
    )
    filtered_state = avg_state[avg_state["AvgWOH"] >= state_min]
    bars = alt.Chart(filtered_state).mark_bar().encode(
        y=alt.Y("ProductState:N", sort=filtered_state["ProductState"].tolist(), title="State"),
        x=alt.X("AvgWOH:Q", title="Avg Weeks-On-Hand"),
        tooltip=[
            alt.Tooltip("ProductState:N", title="State"),
            alt.Tooltip("AvgWOH:Q", title="Avg WOH", format=".1f"),
            alt.Tooltip("Count:Q", title="SKU Count", format=",.0f")
        ]
    )
    labels = bars.mark_text(align="left", baseline="middle", dx=5).encode(text=alt.Text("Count:Q", format=",.0f"))
    st.altair_chart(theme((bars + labels).properties(height=alt.Step(25))).interactive(),
                    use_container_width=True)

    # --- WOH Distribution by Protein ---
    if "Protein" in df.columns and not df["Protein"].isna().all():
        st.subheader("WOH Distribution by Protein")
        total_skus = df.shape[0]
        total_prots = df["Protein"].nunique()
        st.markdown(f"**{total_skus:,}** SKUs across **{total_prots}** Proteins")
        order_p = (
            df.groupby("Protein")["WeeksOnHand"]
              .median()
              .sort_values(ascending=False)
              .index.tolist()
        )
        sel = alt.selection_point(fields=["Protein"], bind="legend")
        box = alt.Chart(df).mark_boxplot(extent="min-max").encode(
            x=alt.X("WeeksOnHand:Q", title="Weeks-On-Hand"),
            y=alt.Y("Protein:N", sort=order_p, title="Protein"),
            color="Protein:N",
            opacity=alt.condition(sel, alt.value(1), alt.value(0.2)),
            tooltip=[
                alt.Tooltip("Protein:N", title="Protein"),
                alt.Tooltip("count():Q", title="SKUs", format=",.0f"),
                alt.Tooltip("median(WeeksOnHand):Q", title="Median WOH", format=".1f")
            ]
        ).add_selection(sel)
        jitter = alt.Chart(df).transform_calculate(
            y_jitter="(random() - 0.5) * 0.6"
        ).mark_circle(size=18).encode(
            x=alt.X("WeeksOnHand:Q", title="Weeks-On-Hand"),
            y=alt.Y("Protein:N", sort=order_p, title="Protein"),
            yOffset="y_jitter:Q",
            color="Protein:N",
            opacity=alt.condition(sel, alt.value(0.6), alt.value(0.1)),
            tooltip=[
                alt.Tooltip("SKU_Desc:N", title="SKU"),
                alt.Tooltip("Protein:N", title="Protein"),
                alt.Tooltip("WeeksOnHand:Q", title="WOH", format=".1f")
            ]
        )
        st.altair_chart(theme((box + jitter).properties(height=400)).interactive(),
                        use_container_width=True)
    else:
        st.warning("Cannot display WOH Distribution by Protein: Protein column missing or all values are NaN.")
