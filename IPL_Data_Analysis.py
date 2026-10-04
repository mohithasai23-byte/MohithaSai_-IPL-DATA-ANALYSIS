# ============================================================
# IPL DATA ANALYSIS
# ============================================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

# ============================================================
# STEP 2: LOAD CLEANED IPL DATA
# ============================================================

# Keep IPL_Cleaned.csv in the same folder as this Python file.
df = pd.read_csv("IPL_Data_Raw.csv")

print("First 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# ============================================================
# PART 2: DATA EXPLORATION & CLEANING CHECK
# ============================================================

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

# Remove duplicate records if any
df = df.drop_duplicates().copy()

# Convert date column
df["Match_Date"] = pd.to_datetime(df["Match_Date"], errors="coerce")

# Fill categorical missing values
if df["City"].isnull().any():
    df["City"] = df["City"].fillna("Unknown")

if df["Player_of_Match"].isnull().any():
    df["Player_of_Match"] = df["Player_of_Match"].fillna("Unknown")

# Make sure numeric columns are numeric
numeric_columns = [
    "Match_ID", "Season", "Win_By_Runs", "Win_By_Wickets",
    "Innings", "Runs_Scored", "Extra_Runs", "Total_Runs",
    "Wickets", "Over", "Ball"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nUnique values in important columns:")

for col in ["Team1", "Team2", "Toss_Decision", "Winner", "Venue"]:
    print(f"\n{col}:")
    print(df[col].unique())

# All teams
all_teams = sorted(set(df["Team1"].dropna()) | set(df["Team2"].dropna()))

print("\nAll teams:")
print(all_teams)

# ============================================================
# IMPORTANT:
# This is ball-level data. Match information is repeated
# for every ball. Therefore use one row per Match_ID for
# match-level calculations.
# ============================================================

matches = df.drop_duplicates("Match_ID").copy()

print("\nNumber of unique matches:")
print(matches["Match_ID"].nunique())

# ============================================================
# PART 3: DATA ANALYSIS
# ============================================================

# 1. Total number of matches
total_matches = matches["Match_ID"].nunique()

# 2. Matches played in each season
matches_by_season = matches.groupby("Season")["Match_ID"].nunique()

# 3. Matches won by each team
wins_by_team = matches["Winner"].value_counts()

# 4. Highest number of wins
highest_wins_team = wins_by_team.idxmax()
highest_wins = wins_by_team.max()

# 5. Lowest number of wins
lowest_wins_team = wins_by_team.idxmin()
lowest_wins = wins_by_team.min()

# 6. Most successful Player of the Match
pom_counts = matches["Player_of_Match"].value_counts()
top_pom_player = pom_counts.idxmax()
top_pom_awards = pom_counts.max()

# 7. Most frequently used venue
venue_counts = matches["Venue"].value_counts()
top_venue = venue_counts.idxmax()
top_venue_matches = venue_counts.max()

# 8. Toss decision analysis
toss_decisions = matches["Toss_Decision"].value_counts()

# 9. Percentage of matches won after winning toss
toss_winner_also_won = (
    matches["Toss_Winner"] == matches["Winner"]
).sum()

toss_win_percentage = (
    toss_winner_also_won / total_matches * 100
)

# 10. Highest winning margin by runs
highest_win_by_runs = matches["Win_By_Runs"].max()

# 11. Highest winning margin by wickets
highest_win_by_wickets = matches["Win_By_Wickets"].max()

# 12. Highest individual run scorer in the data
batsman_runs = (
    df.groupby("Batsman")["Runs_Scored"]
    .sum()
    .sort_values(ascending=False)
)

highest_run_scorer = batsman_runs.idxmax()
highest_runs = batsman_runs.max()

# 13. Highest wicket taker
bowler_wickets = (
    df.groupby("Bowler")["Wickets"]
    .sum()
    .sort_values(ascending=False)
)

highest_wicket_taker = bowler_wickets.idxmax()
highest_wickets = bowler_wickets.max()

# 14. Top 10 batsmen
top10_batsmen = batsman_runs.head(10)

# 15. Top 10 bowlers
top10_bowlers = bowler_wickets.head(10)

# 16. Season-wise total runs
season_runs = df.groupby("Season")["Total_Runs"].sum()

# 17. Season-wise total wickets
season_wickets = df.groupby("Season")["Wickets"].sum()

# 18. Average runs per match
runs_per_match = df.groupby("Match_ID")["Total_Runs"].sum()
average_runs_per_match = runs_per_match.mean()

# 19. Team-wise average runs
team_avg_runs = (
    df.groupby("Batting_Team")["Total_Runs"]
    .mean()
    .sort_values(ascending=False)
)

# 20. Season-wise team performance
season_team_wins = pd.crosstab(
    matches["Season"],
    matches["Winner"]
)

# 21. Most successful venue for each team
team_venue_wins = (
    matches.groupby(["Winner", "Venue"])
    .size()
    .reset_index(name="Wins")
    .sort_values(
        ["Winner", "Wins"],
        ascending=[True, False]
    )
)

most_successful_venue = team_venue_wins.drop_duplicates(
    "Winner"
)

# 22. Toss outcome vs match result
matches["Toss_Result"] = np.where(
    matches["Toss_Winner"] == matches["Winner"],
    "Toss Winner Won",
    "Toss Winner Lost"
)

toss_result = matches["Toss_Result"].value_counts()

# ============================================================
# PRINT ANALYSIS RESULTS
# ============================================================

print("\n============================================================")
print("IPL DATA ANALYSIS RESULTS")
print("============================================================")

print("\n1. Total number of matches:")
print(total_matches)

print("\n2. Matches played in each season:")
print(matches_by_season)

print("\n3. Matches won by each team:")
print(wins_by_team)

print("\n4. Team with highest number of wins:")
print(highest_wins_team, "->", highest_wins)

print("\n5. Team with lowest number of wins:")
print(lowest_wins_team, "->", lowest_wins)

print("\n6. Player of the Match awards:")
print(pom_counts.head(10))

print("\n7. Most frequently used venues:")
print(venue_counts.head(10))

print("\n8. Toss decisions:")
print(toss_decisions)

print("\n9. Percentage of matches won after winning toss:")
print("{:.2f}%".format(toss_win_percentage))

print("\n10. Highest winning margin by runs:")
print(highest_win_by_runs)

print("\n11. Highest winning margin by wickets:")
print(highest_win_by_wickets)

print("\n12. Top 10 batsmen based on total runs:")
print(top10_batsmen)

print("\n13. Top 10 bowlers based on wickets:")
print(top10_bowlers)

print("\n14. Season-wise total runs:")
print(season_runs)

print("\n15. Season-wise total wickets:")
print(season_wickets)

print("\n16. Average runs per match:")
print(round(average_runs_per_match, 2))

print("\n17. Team-wise average runs:")
print(team_avg_runs)

print("\n18. Season-wise team performance:")
print(season_team_wins)

print("\n19. Most successful venue for each team:")
print(most_successful_venue)

print("\n20. Toss outcome vs match result:")
print(toss_result)

# ============================================================
# PART 4: PYTHON VISUALIZATIONS
# ============================================================

sns.set_theme(style="whitegrid")

# 1. Matches Played by Season
plt.figure(figsize=(10, 5))
sns.barplot(
    x=matches_by_season.index,
    y=matches_by_season.values
)
plt.title("Matches Played by Season")
plt.xlabel("Season")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.show()

# 2. Matches Won by Team
plt.figure(figsize=(10, 6))
wins_by_team.sort_values().plot(kind="barh")
plt.title("Matches Won by Team")
plt.xlabel("Number of Wins")
plt.ylabel("Team")
plt.tight_layout()
plt.show()

# 3. Top 10 Run Scorers
plt.figure(figsize=(10, 6))
top10_batsmen.sort_values().plot(kind="barh")
plt.title("Top 10 Run Scorers")
plt.xlabel("Total Runs")
plt.ylabel("Batsman")
plt.tight_layout()
plt.show()

# 4. Top 10 Wicket Takers
plt.figure(figsize=(10, 6))
top10_bowlers.sort_values().plot(kind="barh")
plt.title("Top 10 Wicket Takers")
plt.xlabel("Total Wickets")
plt.ylabel("Bowler")
plt.tight_layout()
plt.show()

# 5. Player of the Match Awards
plt.figure(figsize=(10, 6))
pom_counts.head(10).sort_values().plot(kind="barh")
plt.title("Top Player of the Match Award Winners")
plt.xlabel("Number of Awards")
plt.ylabel("Player")
plt.tight_layout()
plt.show()

# 6. Toss Decision Distribution
plt.figure(figsize=(7, 7))
plt.pie(
    toss_decisions,
    labels=toss_decisions.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Toss Decision Distribution")
plt.show()

# 7. Matches Won After Winning Toss
plt.figure(figsize=(8, 5))
sns.countplot(
    data=matches,
    x="Toss_Result",
    order=["Toss Winner Won", "Toss Winner Lost"]
)
plt.title("Matches Won After Winning Toss")
plt.xlabel("Match Result")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.show()

# 8. Season-wise Total Runs
plt.figure(figsize=(10, 5))
sns.lineplot(
    x=season_runs.index,
    y=season_runs.values,
    marker="o"
)
plt.title("Season-wise Total Runs")
plt.xlabel("Season")
plt.ylabel("Total Runs")
plt.tight_layout()
plt.show()

# 9. Season-wise Total Wickets
plt.figure(figsize=(10, 5))
sns.lineplot(
    x=season_wickets.index,
    y=season_wickets.values,
    marker="o"
)
plt.title("Season-wise Total Wickets")
plt.xlabel("Season")
plt.ylabel("Total Wickets")
plt.tight_layout()
plt.show()

# 10. Runs Distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Runs_Scored",
    bins=10,
    kde=True
)
plt.title("Runs Distribution")
plt.xlabel("Runs Scored")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 11. Team Performance Comparison
plt.figure(figsize=(10, 6))
team_avg_runs.sort_values().plot(kind="barh")
plt.title("Team Performance Comparison - Average Runs")
plt.xlabel("Average Runs")
plt.ylabel("Team")
plt.tight_layout()
plt.show()

# 12. Venue-wise Match Count
plt.figure(figsize=(10, 7))
venue_counts.head(10).sort_values().plot(kind="barh")
plt.title("Top Venues by Match Count")
plt.xlabel("Number of Matches")
plt.ylabel("Venue")
plt.tight_layout()
plt.show()

# 13. Runs vs Wickets
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Runs_Scored",
    y="Wickets"
)
plt.title("Runs vs Wickets")
plt.xlabel("Runs Scored")
plt.ylabel("Wickets")
plt.tight_layout()
plt.show()

# 14. Correlation Heatmap
correlation_columns = [
    "Season",
    "Runs_Scored",
    "Extra_Runs",
    "Total_Runs",
    "Wickets",
    "Over",
    "Ball"
]

plt.figure(figsize=(10, 7))
sns.heatmap(
    df[correlation_columns].corr(),
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# PART 5: BUSINESS / SPORTS INSIGHTS
# ============================================================

print("\n============================================================")
print("BUSINESS / SPORTS ANALYTICS INSIGHTS")
print("============================================================")

print(
    "\n1. Team with the highest number of wins:",
    highest_wins_team,
    "with",
    highest_wins,
    "wins."
)

print(
    "\n2. Season with the highest number of matches:",
    matches_by_season.idxmax(),
    "with",
    matches_by_season.max(),
    "matches."
)

print(
    "\n3. Player with the highest Player of the Match awards:",
    top_pom_player,
    "with",
    top_pom_awards,
    "awards."
)

print(
    "\n4. Highest run scorer:",
    highest_run_scorer,
    "with",
    highest_runs,
    "runs."
)

print(
    "\n5. Highest wicket taker:",
    highest_wicket_taker,
    "with",
    highest_wickets,
    "wickets."
)

print(
    "\n6. Venue with the highest number of matches:",
    top_venue,
    "with",
    top_venue_matches,
    "matches."
)

print(
    "\n7. Most common toss decision:",
    toss_decisions.idxmax()
)

print(
    "\n8. Percentage of matches where toss winner also won:",
    round(toss_win_percentage, 2),
    "%"
)

print(
    "\n9. Season with highest total runs:",
    season_runs.idxmax()
)

print(
    "\n10. Team with highest average runs:",
    team_avg_runs.idxmax()
)

print("\n11. Major performance trends:")
print("- Match participation varies by season.")
print("- Team wins show differences in team performance.")
print("- Player run totals identify leading batsmen.")
print("- Wicket totals identify leading bowlers.")
print("- Toss decisions can be compared with final match results.")
print("- Season-wise runs show changes in scoring patterns.")
print("- Venue statistics show where matches were concentrated.")

# ============================================================
# PART 6: SAVE CLEANED DATASET
# ============================================================

df.to_csv("IPL_Cleaned.csv", index=False)

print("\n============================================================")
print("Cleaned IPL dataset saved as: IPL_Cleaned.csv")
print("IPL analysis completed successfully!")
print("============================================================")
