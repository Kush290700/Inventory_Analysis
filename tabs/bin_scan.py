# tabs/bin_scan.py
import io
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

def _prepare_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # 1) Build a unified ProductDesc
    if {"SKU1","SKU"}.issubset(df.columns):
        df["ProductDesc"] = (
            df["SKU1"].astype(str).str.strip()
          + " – "
          + df["SKU"].astype(str).str.strip()
        )
    elif "SKU" in df.columns:
        df["ProductDesc"] = df["SKU"].astype(str).str.strip()
    elif "SKU1" in df.columns:
        df["ProductDesc"] = df["SKU1"].astype(str).str.strip()
    else:
        df["ProductDesc"] = "<unknown>"

    # 2) Numeric cleanup
    df["ItemCount"] = (
        pd.to_numeric(df.get("ItemCount", 1), errors="coerce")
          .fillna(1)
          .astype(int)
    )
    df["WeightLb"] = (
        pd.to_numeric(df.get("WeightLb", 0), errors="coerce")
          .fillna(0.0)
    )

    # WeightLb is already per-pack
    df["TotalWeight"] = df["WeightLb"]

    # 3) Parse CreatedAt
    df["CreatedAt"] = pd.to_datetime(df.get("CreatedAt"), errors="coerce")

    # 4) If no LastKnownBin column (Mikuni), use PalletId1
    if "LastKnownBin" not in df.columns:
        df["LastKnownBin"] = df.get("PalletId1").astype(str)

    # 5) Keep only valid packs
    df = df[
        df["PackId1"].notna() &
        (df["TotalWeight"] > 0) &
        df["CreatedAt"].notna()
    ]

    # 6) Dedupe: most recent per PackId1
    df = (
        df
        .sort_values("CreatedAt", ascending=False)
        .drop_duplicates(subset="PackId1", keep="first")
        .reset_index(drop=True)
    )

    return df


def render(inv1_df: pd.DataFrame, mikuni_df: pd.DataFrame, theme):
    st.header("🗺️ Bin & Location Inventory")

    # ─── 1) Clean + remove Mikuni rows from inv1 ───────────────
    df1 = _prepare_df(inv1_df)
    # Ensure ProductLocation exists for inv1
    if "ProductLocation" not in df1.columns:
        df1["ProductLocation"] = "Main"
    # Drop any that really belong to Mikuni (so they only come from mikuni_df)
    df1 = df1[df1["ProductLocation"].str.lower() != "mikuni"]

    # ─── 1b) Prepare the Mikuni sheet ──────────────────────────
    if not mikuni_df.empty:
        df2 = _prepare_df(mikuni_df)
        # Label all Mikuni rows
        df2["ProductLocation"] = "Mikuni"
    else:
        df2 = pd.DataFrame(columns=df1.columns)

    # ─── 2) Combine and assign Source ─────────────────────────
    df = pd.concat([df1, df2], ignore_index=True)
    df["Source"] = df["ProductLocation"].str.lower().eq("mikuni") \
                .map({True: "Mikuni", False: "Main"})

    # ─── 3) Data Source filter ─────────────────────────────────
    st.sidebar.subheader("Data Source")
    source_opts = ["Main", "Mikuni"]
    sel_src = st.sidebar.multiselect(
        "Show data from…",
        source_opts,
        default=["Main", "Mikuni"]
    )
    df = df[df["Source"].isin(sel_src)]

    # ─── 4) Sidebar filters ────────────────────────────────────
    st.sidebar.subheader("Filters")
    locs  = ["All"] + sorted(df["ProductLocation"].dropna().unique())
    bins  = ["All"] + sorted(df["LastKnownBin"].dropna().unique())
    prots = ["All"] + sorted(df["Protein"].dropna().unique())

    sel_loc  = st.sidebar.selectbox("Product Location", locs)
    sel_bin  = st.sidebar.selectbox("Bin", bins)
    sel_prot = st.sidebar.selectbox("Protein", prots)

    if sel_loc  != "All":
        df = df[df["ProductLocation"] == sel_loc]
    if sel_bin  != "All":
        df = df[df["LastKnownBin"]     == sel_bin]
    if sel_prot != "All":
        df = df[df["Protein"]          == sel_prot]

    # ─── 5) High-level KPIs ────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Packs",      df["PackId1"].nunique())
    c2.metric("Total Weight",     f"{df['TotalWeight'].sum():,.0f} lb")
    c3.metric("Unique Products",  df["ProductDesc"].nunique())
    c4.metric("Unique Bins",      df["LastKnownBin"].nunique())
    c5.metric("Unique Locations", df["ProductLocation"].nunique())
    c6.metric("Unique Proteins",  df["Protein"].nunique())

    # ─── 6) Total Weight by Protein ───────────────────────────
    st.subheader("Total Weight by Protein")
    prot_df = df.groupby("Protein", as_index=False)["TotalWeight"].sum()
    chart_prot = (
        alt.Chart(prot_df)
        .mark_bar()
        .encode(
            x=alt.X("TotalWeight:Q", title="Weight (lb)", axis=alt.Axis(format=",.0f")),
            y=alt.Y("Protein:N", title="Protein"),
            color=alt.Color("TotalWeight:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Protein","TotalWeight"]
        )
        .properties(height=300)
    )
    st.altair_chart(theme(chart_prot), use_container_width=True)

    # ─── 7) Total Weight by Location ──────────────────────────
    st.subheader("Total Weight by Location")
    loc_df = df.groupby("ProductLocation", as_index=False)["TotalWeight"].sum()
    chart_loc = (
        alt.Chart(loc_df)
        .mark_bar()
        .encode(
            x=alt.X("TotalWeight:Q", title="Weight (lb)", axis=alt.Axis(format=",.0f")),
            y=alt.Y("ProductLocation:N", title="Location"),
            color=alt.Color("TotalWeight:Q", scale=alt.Scale(scheme="greens")),
            tooltip=["ProductLocation","TotalWeight"]
        )
        .properties(height=300)
    )
    st.altair_chart(theme(chart_loc), use_container_width=True)

    # ─── 8) Inventory Distribution by Location & Protein ──────
    st.subheader("Inventory Distribution by Location & Protein")
    loc_prot = (
        df
        .groupby(["ProductLocation","Protein"], as_index=False)["TotalWeight"]
        .sum()
    )
    chart = (
        alt.Chart(loc_prot)
        .mark_bar()
        .encode(
            x=alt.X("ProductLocation:N", title="Location"),
            y=alt.Y("TotalWeight:Q", title="Weight (lb)"),
            color=alt.Color("Protein:N", title="Protein"),
            tooltip=[
                "ProductLocation",
                "Protein",
                alt.Tooltip("TotalWeight:Q", format=",.0f", title="Weight")
            ]
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(theme(chart), use_container_width=True)

    # ─── 9) Download detailed list ─────────────────────────────
    st.markdown("**Download detailed list** for any Location + Protein combination below:")
    sel_loc  = st.selectbox("→ Location", sorted(df["ProductLocation"].dropna().unique()))
    sel_prot = st.selectbox("→ Protein",  sorted(df["Protein"].dropna().unique()))
    filtered = df[(df["ProductLocation"] == sel_loc) & (df["Protein"] == sel_prot)]
    st.subheader(f"Preview: {sel_loc} / {sel_prot}")
    if not filtered.empty:
        st.dataframe(filtered.reset_index(drop=True))
        buf = io.BytesIO()
        filtered.to_excel(buf, index=False, sheet_name=f"{sel_loc}-{sel_prot}")
        buf.seek(0)
        st.download_button(
            "📥 Download Excel",
            buf,
            file_name=f"{sel_loc}_{sel_prot}_inventory.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info(f"No products found for **{sel_loc}** + **{sel_prot}**.")

    # ─── 10) Top-N Products by On-Hand Weight ─────────────────
    st.subheader("Top-N Products by On-Hand Weight")
    top_n = st.slider("Show Top N", min_value=5, max_value=30, value=10)
    prod_totals = (
        df.groupby(["ProductDesc","Source","ProductLocation"], as_index=False)
          .agg(TotalWeight=("TotalWeight","sum"))
          .sort_values("TotalWeight", ascending=False)
          .head(top_n)
    )
    chart_top = (
        alt.Chart(prod_totals)
        .mark_bar()
        .encode(
            y=alt.Y("ProductDesc:N", sort="-x", title="Product"),
            x=alt.X("TotalWeight:Q", title="Weight (lb)", axis=alt.Axis(format=",.0f")),
            color=alt.Color("Source:N", title="Source"),
            tooltip=[
                alt.Tooltip("ProductDesc:N", title="Product"),
                alt.Tooltip("TotalWeight:Q", format=",.0f", title="Weight"),
                alt.Tooltip("Source:N"),
                alt.Tooltip("ProductLocation:N", title="Location"),
            ]
        )
        .properties(height=25*top_n)
    )
    st.altair_chart(theme(chart_top), use_container_width=True)

    # ─── 11) Bin Utilization Summary ──────────────────────────
    st.subheader("Bin Utilization Summary")

    # Fill missing bins & locations
    df["LastKnownBin"] = df["LastKnownBin"].fillna("Unknown")
    df["ProductLocation"] = df["ProductLocation"].fillna("Unknown")

    # Dropdown to choose Top N bins or All
    bin_opts = ["Top 5", "Top 10", "Top 20", "All"]
    choice = st.selectbox("Show bins by total weight:", bin_opts, index=1, key="bin_select")
    top_n = int(choice.split()[1]) if choice != "All" else None

    # Aggregate by bin + location
    util = (
        df
        .groupby(["LastKnownBin","ProductLocation"], as_index=False)
        .agg(
            PackCount   = ("PackId1",     "nunique"),
            TotalWeight = ("TotalWeight", "sum"),
            UniqueItems = ("ProductDesc", "nunique"),
        )
    )

    # Determine overall bin order by total weight
    bin_totals = (
        util.groupby("LastKnownBin")["TotalWeight"]
            .sum()
            .reset_index()
            .sort_values("TotalWeight", ascending=False)
    )
    all_bins = bin_totals["LastKnownBin"].tolist()
    bins_to_show = all_bins[:top_n] if top_n else all_bins

    # Build full cartesian of bins x locations to show zeros
    locations = sorted(df["ProductLocation"].unique())
    bins_df = pd.DataFrame({"LastKnownBin": bins_to_show})
    loc_df  = pd.DataFrame({"ProductLocation": locations})
    full = bins_df.merge(loc_df, how="cross")
    util_f = (
        full
        .merge(util, on=["LastKnownBin","ProductLocation"], how="left")
        .fillna({"PackCount":0, "TotalWeight":0, "UniqueItems":0})
    )

    # Final chart: stacked bars showing Mikuni, Main, Unknown
    chart_util = (
        alt.Chart(util_f)
        .mark_bar()
        .encode(
            y=alt.Y("LastKnownBin:N", sort=bins_to_show, title="Bin"),
            x=alt.X("TotalWeight:Q", title="Total Weight (lb)",
                    axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "ProductLocation:N", title="Location",
                scale=alt.Scale(domain=locations)
            ),
            tooltip=[
                alt.Tooltip("LastKnownBin:N",    title="Bin"),
                alt.Tooltip("ProductLocation:N", title="Location"),
                alt.Tooltip("PackCount:Q",       title="Packs"),
                alt.Tooltip("TotalWeight:Q",     format=",.0f", title="Weight"),
                alt.Tooltip("UniqueItems:Q",     title="Unique Products"),
            ]
        )
        .properties(height=25 * len(bins_to_show))
    )
    st.altair_chart(theme(chart_util), use_container_width=True)

    # Optional: raw numbers
    st.dataframe(
        util_f.pivot_table(
            index="LastKnownBin",
            columns="ProductLocation",
            values=["TotalWeight","PackCount","UniqueItems"],
            fill_value=0
        ), use_container_width=True
    )

    # ─── 12) Distribution of On-Hand Weights ─────────────────
    st.subheader("Distribution of On-Hand Weights")
    mean_w = df["TotalWeight"].mean()
    med_w  = df["TotalWeight"].median()
    base = alt.Chart(df)
    hist = base.mark_bar(opacity=0.6).encode(
        x=alt.X("TotalWeight:Q", bin=alt.Bin(maxbins=40), title="Weight (lb)"),
        y=alt.Y("count():Q", title="Count of Packs")
    )
    dens = base.transform_density("TotalWeight", as_=["Weight","Density"]).mark_area(
        opacity=0.3, color="orange"
    ).encode(x="Weight:Q", y="Density:Q")
    mean_rule = alt.Chart(pd.DataFrame({"v":[mean_w]})).mark_rule(color="red").encode(x="v:Q")
    med_rule  = alt.Chart(pd.DataFrame({"v":[med_w]})).mark_rule(color="blue", strokeDash=[4,2]).encode(x="v:Q")
    chart_dist = (hist + dens + mean_rule + med_rule).properties(height=300).interactive()
    st.altair_chart(theme(chart_dist), use_container_width=True)
    st.markdown("**Red**=mean &nbsp;&nbsp; **Blue**=median &nbsp;&nbsp; (orange=density)")

    # ─── 13) Search & Download Products ───────────────────────
    st.subheader("🔍 Search & Download Products")
    query = st.text_input("Enter part of a product name:")
    if query:
        results = df[df["ProductDesc"].str.contains(query, case=False, na=False)]
        st.write(f"Found **{len(results)}** matching packs.")
        st.dataframe(
            results[[
                "ProductDesc","PackId1","ProductLocation","LastKnownBin","TotalWeight","Source"
            ]].reset_index(drop=True),
            use_container_width=True
        )
        buf = io.BytesIO()
        results.to_excel(buf, index=False, sheet_name="SearchResults")
        buf.seek(0)
        st.download_button(
            "📥 Download Search Results",
            buf,
            file_name="search_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Type in the box above to preview and download.")
