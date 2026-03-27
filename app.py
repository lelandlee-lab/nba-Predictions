import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Awards Predictor",
    page_icon="🏀",
    layout="wide",
)

# ── Load & Process Data ──────────────────────────────────────────────────────
@st.cache_data
def load_and_process():
    """Replicate the full data processing + model construction pipeline."""
    raw = pd.read_csv("database_24_25.csv")

    # Aggregate game-level data to per-game averages per player-team combo
    stat_cols = ["MP", "FG", "FGA", "3P", "3PA", "FT", "FTA",
                 "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "GmSc"]

    agg = raw.groupby(["Player", "Tm"]).agg(
        G=("PTS", "count"),
        **{col: (col, "mean") for col in stat_cols}
    ).reset_index()

    # Compute shooting percentages from aggregated totals
    agg["FG%"] = np.where(agg["FGA"] > 0, agg["FG"] / agg["FGA"], 0)
    agg["3P%"] = np.where(agg["3PA"] > 0, agg["3P"] / agg["3PA"], 0)
    agg["FT%"] = np.where(agg["FTA"] > 0, agg["FT"] / agg["FTA"], 0)

    # Count games started (approximate: use games where MP >= 20 as proxy)
    gs_approx = raw[raw["MP"] >= 20].groupby(["Player", "Tm"]).size().reset_index(name="GS")
    agg = agg.merge(gs_approx, on=["Player", "Tm"], how="left")
    agg["GS"] = agg["GS"].fillna(0).astype(int)

    # Filter to rotation players
    MIN_GAMES = 10
    MIN_MINUTES = 15.0
    players = agg[(agg["G"] >= MIN_GAMES) & (agg["MP"] >= MIN_MINUTES)].copy()

    # Per-36 minute stats
    for stat in ["PTS", "TRB", "AST", "STL", "BLK", "TOV"]:
        players[stat + "36"] = players[stat] * 36.0 / players["MP"]

    # Starter / bench classification
    players["start_rate"] = players["GS"] / players["G"]
    players["is_starter"] = players["start_rate"] >= 0.5
    players["is_bench"] = players["start_rate"] <= 0.4

    # ── Feature Engineering & Clustering ─────────────────────────────────
    offense_features = ["PTS", "PTS36", "AST", "AST36", "FG%",
                        "3P", "3PA", "3P%", "FT", "FTA", "FT%", "MP"]
    defense_features = ["TRB", "TRB36", "STL", "STL36", "BLK", "BLK36",
                        "DRB", "ORB", "PF"]

    offense_features = [c for c in offense_features if c in players.columns]
    defense_features = [c for c in defense_features if c in players.columns]

    # Z-scores
    def compute_z_scores(df, feature_list, prefix):
        X = df[feature_list].copy().fillna(df[feature_list].mean())
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
        for i, col in enumerate(feature_list):
            df[f"{prefix}_{col}_z"] = Z[:, i]
        return scaler, df

    off_scaler, players = compute_z_scores(players, offense_features, "off")
    def_scaler, players = compute_z_scores(players, defense_features, "def")

    # Composite scores
    off_z_cols = [c for c in players.columns if c.startswith("off_") and c.endswith("_z")]
    def_z_cols = [c for c in players.columns if c.startswith("def_") and c.endswith("_z")]
    players["off_score"] = players[off_z_cols].mean(axis=1)
    players["def_score"] = players[def_z_cols].mean(axis=1)
    players["overall_score"] = 0.6 * players["off_score"] + 0.4 * players["def_score"]

    # K-Means clustering
    kmeans_off = KMeans(n_clusters=3, random_state=0, n_init=10)
    players["off_tier"] = kmeans_off.fit_predict(players[off_z_cols].fillna(0))
    kmeans_def = KMeans(n_clusters=3, random_state=0, n_init=10)
    players["def_tier"] = kmeans_def.fit_predict(players[def_z_cols].fillna(0))

    # Re-label tiers so 0 = best
    for tier_col, score_col in [("off_tier", "off_score"), ("def_tier", "def_score")]:
        tier_means = players.groupby(tier_col)[score_col].mean().sort_values(ascending=False)
        mapping = {old: new for new, old in enumerate(tier_means.index)}
        players[tier_col] = players[tier_col].map(mapping)

    # MP z-score for award weighting
    mp_scaler = StandardScaler()
    players["MP_z"] = mp_scaler.fit_transform(players[["MP"]].fillna(players["MP"].mean()))

    # ── Award Scores ─────────────────────────────────────────────────────
    players["mvp_score"] = 0.7 * players["off_score"] + 0.3 * players["def_score"] + 0.2 * players["MP_z"]
    players["dpoy_score"] = players["def_score"] + 0.2 * players["MP_z"]
    players["sixth_score"] = players["off_score"] + 0.1 * players["def_score"]

    # Conference labels
    east_teams = ['ATL','BOS','BRK','CHO','CHI','CLE','DET','IND',
                  'MIA','MIL','NYK','ORL','PHI','TOR','WAS']
    west_teams = ['DAL','DEN','GSW','HOU','LAC','LAL','MEM','MIN',
                  'NOP','OKC','PHO','POR','SAC','SAS','UTA']
    players["Conference"] = np.where(
        players["Tm"].isin(east_teams), "East",
        np.where(players["Tm"].isin(west_teams), "West", "Unknown")
    )

    return players, off_scaler, def_scaler, offense_features, defense_features


players, off_scaler, def_scaler, off_feats, def_feats = load_and_process()


# ── Sidebar Navigation ───────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["🏆 Award Predictions", "📊 Player Explorer", "🔮 Custom Prediction", "📈 Clustering Viz"]
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Award Predictions
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏆 Award Predictions":
    st.title("🏆 NBA End-of-Season Award Predictions")
    st.caption("Powered by K-Means clustering, z-score normalization, and custom scoring formulas on 2024-25 player data.")

    col1, col2, col3 = st.columns(3)

    # MVP
    mvp_pool = players[(players["is_starter"]) & (players["off_tier"] <= 1) & (players["MP"] >= 20)]
    if mvp_pool.empty:
        mvp_pool = players[players["MP"] >= 20]
    mvp_pool = mvp_pool.sort_values("mvp_score", ascending=False)
    mvp_winner = mvp_pool.iloc[0]

    with col1:
        st.subheader("🥇 MVP")
        st.metric(mvp_winner["Player"], f"{mvp_winner['Tm']}")
        st.write(f"**Score:** {mvp_winner['mvp_score']:.3f}")
        st.write(f"PPG: {mvp_winner['PTS']:.1f} | APG: {mvp_winner['AST']:.1f} | RPG: {mvp_winner['TRB']:.1f}")

    # DPOY
    dpoy_pool = players[players["MP"] >= 20].sort_values("dpoy_score", ascending=False)
    dpoy_winner = dpoy_pool.iloc[0]

    with col2:
        st.subheader("🛡️ DPOY")
        st.metric(dpoy_winner["Player"], f"{dpoy_winner['Tm']}")
        st.write(f"**Score:** {dpoy_winner['dpoy_score']:.3f}")
        st.write(f"RPG: {dpoy_winner['TRB']:.1f} | SPG: {dpoy_winner['STL']:.1f} | BPG: {dpoy_winner['BLK']:.1f}")

    # Sixth Man
    sixth_pool = players[(players["is_bench"]) & (players["MP"] >= 18)]
    if sixth_pool.empty:
        sixth_pool = players[players["is_bench"]]
    sixth_pool = sixth_pool.sort_values("sixth_score", ascending=False)
    sixth_winner = sixth_pool.iloc[0]

    with col3:
        st.subheader("🔥 Sixth Man")
        st.metric(sixth_winner["Player"], f"{sixth_winner['Tm']}")
        st.write(f"**Score:** {sixth_winner['sixth_score']:.3f}")
        st.write(f"PPG: {sixth_winner['PTS']:.1f} | APG: {sixth_winner['AST']:.1f}")

    st.divider()

    # Top 10 tables
    st.subheader("Top 10 MVP Candidates")
    mvp_display = mvp_pool.head(10)[["Player", "Tm", "G", "MP", "PTS", "AST", "TRB", "off_score", "def_score", "mvp_score"]].reset_index(drop=True)
    mvp_display.index += 1
    st.dataframe(mvp_display, use_container_width=True)

    st.subheader("Top 10 DPOY Candidates")
    dpoy_display = dpoy_pool.head(10)[["Player", "Tm", "G", "MP", "TRB", "STL", "BLK", "def_score", "dpoy_score"]].reset_index(drop=True)
    dpoy_display.index += 1
    st.dataframe(dpoy_display, use_container_width=True)

    st.subheader("Top 10 Sixth Man Candidates")
    sixth_display = sixth_pool.head(10)[["Player", "Tm", "G", "MP", "PTS", "AST", "off_score", "sixth_score"]].reset_index(drop=True)
    sixth_display.index += 1
    st.dataframe(sixth_display, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Player Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Player Explorer":
    st.title("📊 Player Explorer")

    # Team filter
    teams = sorted(players["Tm"].unique())
    selected_teams = st.multiselect("Filter by Team", teams, default=[])

    filtered = players if not selected_teams else players[players["Tm"].isin(selected_teams)]

    # Sort option
    sort_col = st.selectbox("Sort by", ["overall_score", "off_score", "def_score", "mvp_score", "PTS", "AST", "TRB", "STL", "BLK"], index=0)
    filtered = filtered.sort_values(sort_col, ascending=False)

    display_cols = ["Player", "Tm", "G", "MP", "PTS", "AST", "TRB", "STL", "BLK",
                    "off_score", "def_score", "overall_score", "off_tier", "def_tier"]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=600)

    # Player comparison
    st.divider()
    st.subheader("Compare Players")
    all_players = sorted(players["Player"].unique())
    compare = st.multiselect("Select players to compare", all_players, default=all_players[:2])

    if len(compare) >= 2:
        comp_df = players[players["Player"].isin(compare)]
        radar_stats = ["PTS", "AST", "TRB", "STL", "BLK", "FG%"]

        fig = go.Figure()
        for _, row in comp_df.iterrows():
            vals = [row[s] for s in radar_stats]
            # Normalize to 0-1 for radar
            maxes = [players[s].max() for s in radar_stats]
            normed = [v / m if m > 0 else 0 for v, m in zip(vals, maxes)]
            fig.add_trace(go.Scatterpolar(
                r=normed + [normed[0]],
                theta=radar_stats + [radar_stats[0]],
                name=f"{row['Player']} ({row['Tm']})",
                fill='toself',
                opacity=0.6,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Player Comparison (Normalized)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Custom Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Custom Prediction":
    st.title("🔮 Where Would a Player Rank?")
    st.write("Enter hypothetical per-game stats and see how a player would score in our model.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Scoring")
        pts = st.number_input("PPG", 0.0, 50.0, 25.0, 0.5)
        fg_pct = st.number_input("FG%", 0.0, 1.0, 0.47, 0.01)
        three_p = st.number_input("3PM", 0.0, 10.0, 2.5, 0.1)
        three_pa = st.number_input("3PA", 0.0, 15.0, 6.5, 0.1)
        three_pct = st.number_input("3P%", 0.0, 1.0, 0.38, 0.01)
        ft = st.number_input("FTM", 0.0, 15.0, 5.0, 0.1)
        fta = st.number_input("FTA", 0.0, 18.0, 6.0, 0.1)
        ft_pct = st.number_input("FT%", 0.0, 1.0, 0.83, 0.01)

    with col2:
        st.subheader("Playmaking & Boards")
        ast = st.number_input("APG", 0.0, 15.0, 5.0, 0.1)
        trb = st.number_input("RPG", 0.0, 20.0, 7.0, 0.1)
        orb = st.number_input("OREB", 0.0, 8.0, 1.0, 0.1)
        drb = st.number_input("DREB", 0.0, 15.0, 6.0, 0.1)

    with col3:
        st.subheader("Defense & Minutes")
        stl = st.number_input("SPG", 0.0, 5.0, 1.2, 0.1)
        blk = st.number_input("BPG", 0.0, 5.0, 0.8, 0.1)
        tov = st.number_input("TOV", 0.0, 10.0, 3.0, 0.1)
        pf = st.number_input("PF", 0.0, 6.0, 2.5, 0.1)
        mp = st.number_input("MPG", 10.0, 48.0, 34.0, 0.5)

    if st.button("🏀 Predict Rankings", type="primary"):
        # Build per-36 stats
        pts36 = pts * 36.0 / mp
        trb36 = trb * 36.0 / mp
        ast36 = ast * 36.0 / mp
        stl36 = stl * 36.0 / mp
        blk36 = blk * 36.0 / mp

        # Build feature arrays in the same order used during training
        off_vals = []
        off_feat_map = {
            "PTS": pts, "PTS36": pts36, "AST": ast, "AST36": ast36, "FG%": fg_pct,
            "3P": three_p, "3PA": three_pa, "3P%": three_pct,
            "FT": ft, "FTA": fta, "FT%": ft_pct, "MP": mp
        }
        for f in ["PTS", "PTS36", "AST", "AST36", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "FT%", "MP"]:
            off_vals.append(off_feat_map.get(f, 0))

        def_vals = []
        def_feat_map = {
            "TRB": trb, "TRB36": trb36, "STL": stl, "STL36": stl36,
            "BLK": blk, "BLK36": blk36, "DRB": drb, "ORB": orb, "PF": pf
        }
        for f in ["TRB", "TRB36", "STL", "STL36", "BLK", "BLK36", "DRB", "ORB", "PF"]:
            def_vals.append(def_feat_map.get(f, 0))

        off_z = off_scaler.transform([off_vals])[0]
        def_z = def_scaler.transform([def_vals])[0]

        off_score = off_z.mean()
        def_score = def_z.mean()
        overall = 0.6 * off_score + 0.4 * def_score

        mp_mean = players["MP"].mean()
        mp_std = players["MP"].std()
        mp_z = (mp - mp_mean) / mp_std if mp_std > 0 else 0

        mvp_score = 0.7 * off_score + 0.3 * def_score + 0.2 * mp_z

        st.divider()
        st.subheader("Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Offensive Score", f"{off_score:.3f}")
        m2.metric("Defensive Score", f"{def_score:.3f}")
        m3.metric("Overall Score", f"{overall:.3f}")
        m4.metric("MVP Score", f"{mvp_score:.3f}")

        # Rank among real players
        off_rank = (players["off_score"] > off_score).sum() + 1
        def_rank = (players["def_score"] > def_score).sum() + 1
        overall_rank = (players["overall_score"] > overall).sum() + 1
        mvp_rank = (players["mvp_score"] > mvp_score).sum() + 1
        total = len(players)

        st.write(f"**Offensive rank:** #{off_rank} out of {total} players")
        st.write(f"**Defensive rank:** #{def_rank} out of {total} players")
        st.write(f"**Overall rank:** #{overall_rank} out of {total} players")
        st.write(f"**MVP rank:** #{mvp_rank} out of {total} players")

        # Show nearby players
        st.subheader("Similar Players by Overall Score")
        players_sorted = players.sort_values("overall_score", ascending=False).reset_index(drop=True)
        insert_idx = overall_rank - 1
        nearby = players_sorted.iloc[max(0, insert_idx-3):insert_idx+3]
        st.dataframe(nearby[["Player", "Tm", "PTS", "AST", "TRB", "overall_score"]].reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Clustering Visualization
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Clustering Viz":
    st.title("📈 K-Means Clustering Visualization")
    st.write("Offensive and defensive player tiers identified through unsupervised learning.")

    tab1, tab2 = st.tabs(["Offensive Tiers", "Defensive Tiers"])

    with tab1:
        fig = px.scatter(
            players,
            x="PTS",
            y="AST",
            color=players["off_tier"].astype(str),
            hover_data=["Player", "Tm", "off_score"],
            title="Offensive Clusters: Points vs Assists",
            labels={"color": "Offensive Tier"},
            color_discrete_sequence=px.colors.qualitative.Set1,
            height=600,
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            players,
            x="PTS",
            y="FG%",
            color=players["off_tier"].astype(str),
            hover_data=["Player", "Tm", "off_score"],
            title="Offensive Clusters: Points vs FG%",
            labels={"color": "Offensive Tier"},
            color_discrete_sequence=px.colors.qualitative.Set1,
            height=600,
        )
        fig2.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig3 = px.scatter(
            players,
            x="TRB",
            y="BLK",
            color=players["def_tier"].astype(str),
            hover_data=["Player", "Tm", "def_score"],
            title="Defensive Clusters: Rebounds vs Blocks",
            labels={"color": "Defensive Tier"},
            color_discrete_sequence=px.colors.qualitative.Dark2,
            height=600,
        )
        fig3.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(
            players,
            x="STL",
            y="BLK",
            color=players["def_tier"].astype(str),
            hover_data=["Player", "Tm", "def_score"],
            title="Defensive Clusters: Steals vs Blocks",
            labels={"color": "Defensive Tier"},
            color_discrete_sequence=px.colors.qualitative.Dark2,
            height=600,
        )
        fig4.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Tier Distribution")
    col1, col2 = st.columns(2)
    with col1:
        off_counts = players["off_tier"].value_counts().sort_index()
        fig5 = px.bar(x=off_counts.index.astype(str), y=off_counts.values,
                       labels={"x": "Tier", "y": "Players"}, title="Offensive Tier Distribution")
        st.plotly_chart(fig5, use_container_width=True)
    with col2:
        def_counts = players["def_tier"].value_counts().sort_index()
        fig6 = px.bar(x=def_counts.index.astype(str), y=def_counts.values,
                       labels={"x": "Tier", "y": "Players"}, title="Defensive Tier Distribution")
        st.plotly_chart(fig6, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Built by Haitham Assaf | SJSU Data Science")
st.sidebar.caption("Data: Basketball Reference 2024-25 Season")
