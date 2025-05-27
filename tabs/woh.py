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
    """
    Builds a purchase plan aggregated at the Velocity Parent SKU level.

    Args:
        df_inv: DataFrame with inventory data (SKU, AvgWeeklyUsage, OnHandWeightTotal, OnHandCostTotal, SKU_Desc)
        pd_detail: DataFrame with product details (Product Code, Velocity Parent, Description)
        cost_df: DataFrame with cost data (SKU, NumPacks). If empty or missing NumPacks, defaults to PackCount=1.
        desired_woh: Desired weeks-on-hand for inventory

    Returns:
        DataFrame with parent-level purchase plan
    """
    # Input validation
    if df_inv.empty:
        st.error("Inventory data is empty")
        return pd.DataFrame()
    if pd_detail.empty:
        st.error("Product detail sheet is empty")
        return pd.DataFrame()
    if not cost_df.empty and "SKU" not in cost_df.columns:
        st.error("Cost sheet provided without 'SKU' column")
        return pd.DataFrame()

    # Prepare product detail sheet
    d = pd_detail.rename(columns={
        "Product Code": "SKU",
        "Velocity Parent": "ParentSKU",
        "Description": "ParentDesc"
    }).copy()
    for col in ["SKU","ParentSKU","ParentDesc"]:
        d[col] = d.get(col, pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    # Default null ParentSKU to child SKU
    null_parent = d["ParentSKU"].str.lower().isin(["","nan","null"])
    d.loc[null_parent, "ParentSKU"] = d.loc[null_parent, "SKU"]

    child_to_parent = dict(zip(d["SKU"], d["ParentSKU"]))
    parent_desc_map = dict(zip(d["ParentSKU"], d["ParentDesc"]))

    # Compute child-level demand
    inv = df_inv.copy()
    inv["SKU"] = inv["SKU"].astype(str).str.strip()
    inv["SKU_Desc"] = inv["SKU_Desc"].astype(str).str.strip()
    child = (inv.groupby(["SKU","SKU_Desc"], as_index=False)
               .agg(
                   MeanUse=("AvgWeeklyUsage","mean"),
                   InvWt=("OnHandWeightTotal","sum"),
                   InvCost=("OnHandCostTotal","sum")
               ))
    child["DesiredWt"] = child["MeanUse"] * desired_woh
    child["ToBuyWt"] = (child["DesiredWt"] - child["InvWt"]).clip(lower=0)

    # Map each child to its parent
    child["ParentSKU"] = child["SKU"].map(child_to_parent).fillna(child["SKU"])

    # Pack counts
    if not cost_df.empty and "NumPacks" in cost_df.columns:
        packs = (pd.to_numeric(cost_df["NumPacks"], errors="coerce").fillna(1)
                 .astype(int).clip(lower=1))
        pack_map = dict(zip(cost_df["SKU"].astype(str), packs))
        child["PackCount"] = child["SKU"].map(pack_map).fillna(1).astype(int)
    else:
        child["PackCount"] = 1

    # Prevent division by zero
    child["PackWt"] = np.where(
        child["PackCount"]>0,
        child["InvWt"]/child["PackCount"],
        0
    )
    child["PacksToOrder"] = np.ceil(
        child["ToBuyWt"]/child["PackWt"].replace(0,np.nan)
    ).fillna(0).astype(int)
    child["OrderWt"] = child["PacksToOrder"] * child["PackWt"]

    # Aggregate to parent
    parent = (child.groupby("ParentSKU",as_index=False)
                   .agg(
                       MeanUse=("MeanUse","sum"),
                       InvWt=("InvWt","sum"),
                       InvCost=("InvCost","sum"),
                       ToBuyWt=("ToBuyWt","sum"),
                       PackCount=("PackCount","mean"),
                       PacksToOrder=("PacksToOrder","sum"),
                       OrderWt=("OrderWt","sum")
                   ))
    parent = parent.rename(columns={"ParentSKU":"SKU"})
    parent["SKU_Desc"] = parent["SKU"].map(parent_desc_map).fillna(parent["SKU"])

    # Cost estimates
    parent["CostPerLb"] = np.where(parent["InvWt"]>0,
                                      parent["InvCost"]/parent["InvWt"],
                                      0)
    parent["EstCost"] = parent["OrderWt"] * parent["CostPerLb"]
    parent["DesiredWt"] = parent["MeanUse"] * desired_woh

    return parent[parent["PacksToOrder"]>0]


def render(df, df_hc, cost_df, theme, sheets):
    # Validate core columns
    core = ["SKU","WeeksOnHand","AvgWeeklyUsage","OnHandWeightTotal",
            "OnHandCostTotal","SKU_Desc","ProductState","Supplier"]
    missing = [c for c in core if c not in df.columns]
    if missing:
        st.error(f"Missing core columns: {missing}")
        return
    df = df.copy()

    st.header("📊 Weeks-On-Hand Analysis")

    # PackCount & AvgWeightPerPack
    if not cost_df.empty and set(["SKU","NumPacks"]).issubset(cost_df.columns):
        packs = pd.to_numeric(cost_df["NumPacks"], errors="coerce").fillna(1).astype(int).clip(lower=1)
        pmap = dict(zip(cost_df["SKU"].astype(str), packs))
        df["PackCount"] = df["SKU"].astype(str).map(pmap).fillna(1).astype(int)
    else:
        df["PackCount"] = 1
    df["AvgWeightPerPack"] = np.where(
        df["PackCount"]>0,
        df["OnHandWeightTotal"]/df["PackCount"],
        df["OnHandWeightTotal"]
    )

    # Split FZ & EXT
    state = df["ProductState"].fillna("").str.upper()
    fz = df[state.str.startswith("FZ") & df["AvgWeeklyUsage"].gt(0)].copy()
    ext = df[state.str.startswith("EXT") & df["AvgWeeklyUsage"].gt(0)].copy()
    fz_woh = fz.set_index("SKU")["WeeksOnHand"]
    ext_wt = ext.set_index("SKU")["OnHandWeightTotal"]

    # --- Move FZ → EXT ---
    st.subheader("🔄 Move FZ → EXT")
    thr1 = st.slider("Desired FZ WOH (weeks)", 0.0, 52.0, 4.0, 0.5)
    mv1 = fz[fz["WeeksOnHand"]>thr1].copy()
    mv1["DesiredFZ_Weight"] = mv1["AvgWeeklyUsage"]*thr1
    mv1["WeightToMove"] = (mv1["OnHandWeightTotal"]-mv1["DesiredFZ_Weight"]).clip(lower=0)
    mv1["EXT_Weight"] = mv1["SKU"].map(ext_wt).fillna(0)
    mv1["TotalOnHand"] = mv1["OnHandWeightTotal"]+mv1["EXT_Weight"]
    mv1 = mv1[mv1["WeightToMove"]>0]

    c1,c2,c3 = st.columns(3)
    c1.metric("SKUs to Move", mv1["SKU"].nunique())
    c2.metric("Total Weight to Move", f"{mv1['WeightToMove'].sum():,.0f} lb")
    cost_move = ((mv1['WeightToMove']/mv1['OnHandWeightTotal'].replace(0,np.nan))*mv1['OnHandCostTotal']).sum()
    c3.metric("Total Cost to Move", f"${cost_move:,.0f}")

    # Supplier filter and chart
    supps = sorted(mv1['Supplier'].dropna().unique())
    sel = st.multiselect("Filter Suppliers", supps, default=supps)
    mv1 = mv1[mv1['Supplier'].isin(sel)] if sel else mv1
    if not mv1.empty:
        sel2 = alt.selection_multi(fields=["Supplier"], bind="legend")
        chart1 = (alt.Chart(mv1)
                  .mark_bar()
                  .encode(
                      y=alt.Y("SKU_Desc:N",sort='-x'),
                      x="WeightToMove:Q",
                      color="Supplier:N",
                      opacity=alt.condition(sel2, alt.value(1), alt.value(0.2)),
                      tooltip=["SKU_Desc","Supplier","WeightToMove"]
                  )
                  .add_selection(sel2)
                  .properties(height=300))
        st.altair_chart(theme(chart1), use_container_width=True)

        # Download
        exp = [c for c in ["SKU_Desc","Supplier","WeightToMove","EXT_Weight"] if c in mv1.columns]
        buf = io.BytesIO()
        mv1[exp].to_excel(buf, index=False, sheet_name="FZ2EXT")
        buf.seek(0)
        st.download_button("Download FZ→EXT", buf.getvalue(), "FZ2EXT.xlsx")

    # --- Move EXT → FZ ---
    st.subheader("🔄 Move EXT → FZ")
    thr2 = 1.0
    try:
        thr2 = float(compute_threshold_move(ext, df_hc)) if not df_hc.empty else thr2
    except Exception:
        st.warning("EXT→FZ threshold compute failed; using default 1.0")
    thr2 = st.slider("Desired FZ WOH to achieve", 0.0,_float(fz_woh.max()) if not fz_woh.empty else 52.0, thr2, step=0.25)
    back = ext[ext["SKU"].map(fz_woh).fillna(0)<thr2].copy()
    back["FZ_Weight"] = back["SKU"].map(fz_woh).fillna(0)
    back["DesiredFZ_Weight"] = back["AvgWeeklyUsage"]*thr2
    back["WeightToReturn"] = (back["DesiredFZ_Weight"]-back["FZ_Weight"]).clip(lower=0)
    back["TotalOnHand"] = back["OnHandWeightTotal"]+back["FZ_Weight"]

    d1,d2,d3 = st.columns(3)
    d1.metric("SKUs to Return", back["SKU"].nunique())
    d2.metric("Total Weight to Return", f"{back['WeightToReturn'].sum():,.0f} lb")
    cost_ret = ((back['WeightToReturn']/back['OnHandWeightTotal'].replace(0,np.nan))*back['OnHandCostTotal']).sum()
    d3.metric("Total Cost to Return", f"${cost_ret:,.0f}")

    # Supplier filter and chart for return
    sup2 = sorted(back['Supplier'].dropna().unique())
    chosen = st.multiselect("Filter Suppliers", sup2, default=sup2)
    back = back[back['Supplier'].isin(chosen)] if chosen else back
    if not back.empty:
        sel3 = alt.selection_multi(fields=["Supplier"], bind="legend")
        chart2 = (alt.Chart(back)
                  .mark_bar()
                  .encode(
                      y=alt.Y("SKU_Desc:N",sort='-x'),
                      x="WeightToReturn:Q",
                      color="Supplier:N",
                      opacity=alt.condition(sel3, alt.value(1), alt.value(0.2)),
                      tooltip=["SKU_Desc","Supplier","WeightToReturn"]
                  )
                  .add_selection(sel3)
                  .properties(height=300))
        st.altair_chart(theme(chart2), use_container_width=True)

        buf2 = io.BytesIO()
        cols2 = [c for c in ["SKU_Desc","Supplier","WeightToReturn","FZ_Weight"] if c in back.columns]
        back[cols2].to_excel(buf2,index=False, sheet_name="EXT2FZ")
        buf2.seek(0)
        st.download_button("Download EXT→FZ", buf2.getvalue(), "EXT2FZ.xlsx")

    # --- Purchase Recommendations ---
    st.subheader("🛒 Purchase Recommendations by Desired WOH")
    sup_opts = ["All"]+sorted(df['Supplier'].astype(str).unique())
    sel_sup = st.selectbox("Supplier", sup_opts)
    df_pr = df if sel_sup=="All" else df[df['Supplier']==sel_sup]
    des = st.slider("Desired Weeks-On-Hand",0.0,52.0,4.0,0.5)
    details = sheets.get("Product Detail",pd.DataFrame())
    plan = compute_parent_purchase_plan(df_pr,details,cost_df,des)
    if not plan.empty:
        st.dataframe(plan)
        buf3 = io.BytesIO(); plan.to_excel(buf3,index=False); buf3.seek(0)
        st.download_button("Download Purchase Plan",buf3.getvalue(),"PurchasePlan.xlsx")

    # --- Distribution of WOH ---
    st.subheader("Distribution of Weeks-On-Hand")
    if df["WeeksOnHand"].notna().any():
        q25,q50,q75,q90 = [df["WeeksOnHand"].quantile(q) for q in (0.25,0.5,0.75,0.90)]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("25th pct",f"{q25:.1f}w")
        m2.metric("Median",f"{q50:.1f}w")
        m3.metric("75th pct",f"{q75:.1f}w")
        m4.metric("90th pct",f"{q90:.1f}w")
        bins = st.slider("Histogram bins",10,100,40,5)
        filt = df[df["WeeksOnHand"]<=df["WeeksOnHand"].quantile(0.99)]
        base = alt.Chart(filt).encode(x="WeeksOnHand:Q")
        hist = base.mark_bar(opacity=0.6).encode(y="count():Q")
        dens = base.transform_density("WeeksOnHand",as_=["WeeksOnHand","density"],counts=True,steps=bins)
        dens = dens.mark_line().encode(y="density:Q")
        cdf = (base.transform_window(cumulative="count()",sort=[{"field":"WeeksOnHand"}])
               .transform_joinaggregate(total="count()")
               .transform_calculate(cum_pct="datum.cumulative/datum.total")
               .mark_line(strokeDash=[4,2])
               .encode(y=alt.Y("cum_pct:Q",axis=alt.Axis(format="%")))
        st.altair_chart(theme(alt.layer(hist,dens,cdf).resolve_scale(y="independent").properties(height=300)),use_container_width=True)

    # --- Annual Turns ---
    if "AnnualTurns" in df.columns and df["AnnualTurns"].notna().any():
        st.subheader("Annual Turns Distribution")
        t25,t50,t75 = [df["AnnualTurns"].quantile(q) for q in (0.25,0.5,0.75)]
        a1,a2,a3 = st.columns(3)
        a1.metric("25th pct",f"{t25:.1f}")
        a2.metric("Median",f"{t50:.1f}")
        a3.metric("75th pct",f"{t75:.1f}")
        tbins = st.slider("Turns bins",10,100,30)
        hist2 = alt.Chart(df).mark_bar(opacity=0.6).encode(x=alt.X("AnnualTurns:Q",bin=alt.Bin(maxbins=tbins)),y="count():Q")
        mean_t = df["AnnualTurns"].mean()
        med_t  = df["AnnualTurns"].median()
        line_m = alt.Chart(pd.DataFrame({"v":[mean_t]})).mark_rule(color="red").encode(x="v:Q")
        line_d = alt.Chart(pd.DataFrame({"v":[med_t]})).mark_rule(color="blue",strokeDash=[4,4]).encode(x="v:Q")
        st.altair_chart(theme(alt.layer(hist2,line_m,line_d).properties(height=300)),use_container_width=True)

    # --- Avg WOH by State ---
    st.subheader("Average WOH by State")
    state_df = df.groupby("ProductState",as_index=False).agg(AvgWOH=("WeeksOnHand","mean"),Count=("SKU","nunique")).sort_values("AvgWOH")
    minv, maxv = (state_df["AvgWOH"].min() if not state_df.empty else 0), (state_df["AvgWOH"].max() if not state_df.empty else 52)
    thr = st.slider("Hide states with Avg WOH below",minv,maxv,minv,0.5)
    sd = state_df[state_df["AvgWOH"]>=thr]
    bars = alt.Chart(sd).mark_bar().encode(y=alt.Y("ProductState:N",sort=sd["ProductState"].tolist()),x="AvgWOH:Q",tooltip=["ProductState","AvgWOH","Count"])
    txt = bars.mark_text(dx=5,align="left").encode(text=alt.Text("Count:Q"))
    st.altair_chart(theme((bars+txt).properties(height=alt.Step(30))),use_container_width=True)

    # --- WOH by Protein ---
    if "Protein" in df.columns and df["Protein"].notna().any():
        st.subheader("WOH Distribution by Protein")
        order_p = df.groupby("Protein")["WeeksOnHand"].median().sort_values(ascending=False).index.tolist()
        selp = alt.selection_point(fields=["Protein"],bind="legend")
        box = alt.Chart(df).mark_boxplot(extent="min-max").encode(y=alt.Y("Protein:N",sort=order_p),x="WeeksOnHand:Q",color="Protein:N",opacity=alt.condition(selp,alt.value(1),alt.value(0.2)))
        jitter = alt.Chart(df).transform_calculate(y_jitter="(random()-0.5)*0.6").mark_circle(size=10).encode(y=alt.Y("Protein:N",sort=order_p),yOffset="y_jitter:Q",x="WeeksOnHand:Q",color="Protein:N",opacity=alt.condition(selp,alt.value(0.6),alt.value(0.1)))
        st.altair_chart(theme((box+jitter).add_selection(selp).properties(height=400)),use_container_width=True)
    else:
        st.warning("Cannot display WOH Distribution by Protein: column missing or all NaN.")
