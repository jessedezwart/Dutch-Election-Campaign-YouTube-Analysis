import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy.stats import spearmanr, pearsonr


INPUT_FILE = Path("dataset_processed/videos.jsonl")
COMMENTS_FILE = Path("dataset_processed/comments.jsonl")

DAYS_PER_YEAR = 365.25

# Een account telt als "nieuw" als het jonger is dan dit aantal jaar op de referentiedatum.
NEW_ACCOUNT_MAX_YEARS = 1.0

# Jaarcohorten: 0-1, 1-2, ... tot een verzamelbak voor oudere accounts.
MAX_COHORT_YEARS = 6
COHORT_ORDER = [f"{i}-{i + 1} jaar" for i in range(MAX_COHORT_YEARS)] + [f"{MAX_COHORT_YEARS}+ jaar"]

RESULTS_DIR = Path("results/")
TABLES_DIR = RESULTS_DIR / "tabellen"
IMAGES_DIR = RESULTS_DIR / "images"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# Partijnaam-zoektermen (afkorting + volledige naam)
PARTY_NAME_KEYWORDS = {
    "PVV": ["PVV", "Partij voor de Vrijheid"],
    "GroenLinks-PvdA": ["GroenLinks PvdA", "GroenLinks-PvdA"],
    "VVD": ["VVD", "Volkspartij voor Vrijheid en Democratie"],
    "NSC": ["NSC", "Nieuw Sociaal Contract"],
    "D66": ["D66", "Democraten 66"],
    "BBB": ["BBB", "BoerBurgerBeweging"],
    "CDA": ["CDA", "Christen-Democratisch Appèl"],
    "SP": ["SP", "Socialistische Partij"],
    "PvdD": ["PvdD", "Partij voor de Dieren"],
    "FVD": ["FVD", "Forum voor Democratie"],
    "SGP": ["SGP", "Staatkundig Gereformeerde Partij"],
    "ChristenUnie": ["ChristenUnie"],
    "Volt": ["Volt", "Volt Nederland"],
    "JA21": ["JA21"],
    "DENK": ["DENK", "DENK Nederland"],
    "50PLUS": ["50PLUS"],
    "BIJ1": ["BIJ1"],
}

# Lijsttrekker(s) per partij
PARTY_PERSON_KEYWORDS = {
    "PVV": ["Geert Wilders"],
    "GroenLinks-PvdA": ["Frans Timmermans"],
    "VVD": ["Dilan Yeşilgöz"],
    "NSC": ["Pieter Omtzigt"],
    "D66": ["Rob Jetten"],
    "BBB": ["Caroline van der Plas"],
    "CDA": ["Henri Bontenbal"],
    "SP": ["Jimmy Dijk"],
    "PvdD": ["Esther Ouwehand"],
    "FVD": ["Thierry Baudet", "Lidewij de Vos"],
    "SGP": ["Chris Stoffer"],
    "ChristenUnie": ["Mirjam Bikker"],
    "Volt": ["Laurens Dassen"],
    "JA21": ["Joost Eerdmans"],
    "DENK": ["Stephan van Baarle"],
    "50PLUS": [],
    "BIJ1": [],
}

# Unie van naam- en persoon-zoektermen per partij
PARTY_KEYWORDS = {
    party: PARTY_NAME_KEYWORDS[party] + PARTY_PERSON_KEYWORDS[party]
    for party in PARTY_NAME_KEYWORDS
}

THEME_KEYWORDS = {
    "Migratie": ["migratie verkiezingen 2025", "asiel verkiezingen 2025"],
    "Wonen": ["woningcrisis verkiezingen 2025"],
    "Zorg": ["zorg verkiezingen 2025"],
    "Klimaat": ["klimaat verkiezingen 2025", "stikstof verkiezingen 2025"],
    "Economie": ["economie verkiezingen 2025", "koopkracht verkiezingen 2025"],
    "Veiligheid": ["veiligheid verkiezingen 2025", "defensie verkiezingen 2025"],
    "Oekraïne": ["Oekraïne verkiezingen 2025"],
    "Europa": ["Europa verkiezingen 2025"],
    "Onderwijs": ["onderwijs verkiezingen 2025"],
}

SEGMENTS = ["Partij", "Persoon"]

# (sleutel, label voor as/titel). Kolommen heten total_<key> / median_<key>.
METRICS = [
    ("views", "weergaven"),
    ("comments", "reacties"),
    ("likes", "likes"),
]


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records

# Classificatie van zoektermen naar partij
def classify(value, keyword_map: dict[str, list[str]]) -> str | None:
    if isinstance(value, list):
        terms = value
    else:
        terms = [value]

    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in terms:
                return label

    return None

# Classificatie van zoektermen naar partij
def classify_party(search_terms) -> str | None:
    return classify(search_terms, PARTY_KEYWORDS)


def classify_segment(search_terms, party: str) -> str:
    terms = search_terms if isinstance(search_terms, list) else [search_terms]

    for keyword in PARTY_PERSON_KEYWORDS.get(party, []):
        if keyword in terms:
            return "Persoon"

    return "Partij"


def classify_theme(search_terms):
    return classify(search_terms, THEME_KEYWORDS)


def cohort_label(age_years: float) -> str:
    bucket = int(age_years) if age_years > 0 else 0
    if bucket >= MAX_COHORT_YEARS:
        return f"{MAX_COHORT_YEARS}+ jaar"
    return f"{bucket}-{bucket + 1} jaar"


def account_age_breakdown(df_comments: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per groep, per uniek reageerder-account geteld:
    - summary: n_accounts en aandeel nieuwe accounts (%) (< NEW_ACCOUNT_MAX_YEARS jaar)
    - cohort_pct: pivot groep x leeftijdscohort als percentage van de groep (rijen sommeren tot 100)."""
    valid = df_comments.dropna(subset=["commenter_account_age_days", "commenter_hash"]).copy()
    unique = valid.drop_duplicates(subset=[group_col, "commenter_hash"]).copy()
    unique["age_years"] = unique["commenter_account_age_days"].clip(lower=0) / DAYS_PER_YEAR
    unique["cohort"] = unique["age_years"].apply(cohort_label)
    unique["is_new"] = unique["age_years"] < NEW_ACCOUNT_MAX_YEARS

    summary = (
        unique.groupby(group_col)
        .agg(
            n_accounts=("commenter_hash", "nunique"),
            pct_nieuw=("is_new", lambda s: 100 * s.mean()),
        )
        .reset_index()
        .sort_values(group_col)
    )

    counts = (
        unique.pivot_table(index=group_col, columns="cohort", values="commenter_hash", aggfunc="count")
        .reindex(columns=COHORT_ORDER, fill_value=0)
        .fillna(0)
    )
    cohort_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    return summary, cohort_pct


def aggregate_interactions(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols)
        .agg(
            video_count=("video_hash", "count"),
            total_views=("view_count", "sum"),
            total_likes=("like_count", "sum"),
            total_comments=("comment_count", "sum"),
            median_views=("view_count", "median"),
            median_likes=("like_count", "median"),
            median_comments=("comment_count", "median"),
            avg_views=("view_count", "mean"),
            avg_likes=("like_count", "mean"),
            avg_comments=("comment_count", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def _dutch_thousands(value, _pos) -> str:
    """Formatteer grote getallen met punt als duizendtalscheiding (NL-conventie)."""
    return f"{int(value):,}".replace(",", ".")


def _style_value_axis(ax, decimals: int | None = None) -> None:
    """Academische opmaak voor de waarde-as: horizontale gridlijnen + NL-getalnotatie.
    decimals=None -> hele getallen met duizendtalscheiding; anders zoveel decimalen (komma)."""
    if decimals is None:
        formatter = _dutch_thousands
    else:
        def formatter(value, _pos):
            return f"{value:.{decimals}f}".replace(".", ",")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def _format_date_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    ax.figure.autofmt_xdate(rotation=45, ha="right")


def save_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    decimals: int | None = None,
) -> None:
    if df.empty:
        return

    df = df.sort_values(y_col, ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df[x_col].astype(str), df[y_col], color="#4878a8")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df[x_col].astype(str), rotation=45, ha="right")
    _style_value_axis(ax, decimals=decimals)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_stacked_bar_chart(
    pivot: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """x = partij, gestapeld op Partij/Persoon. Gesorteerd op rijtotaal laag->hoog."""
    if pivot.empty:
        return

    pivot = pivot.reindex(columns=SEGMENTS, fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    parties = pivot.index.astype(str).tolist()
    x = np.arange(len(parties))

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(parties))

    for segment in SEGMENTS:
        values = pivot[segment].to_numpy()
        ax.bar(x, values, bottom=bottom, label=segment)
        bottom += values

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(parties, rotation=45, ha="right")
    ax.legend(title="Type zoekterm")
    _style_value_axis(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_cohort_stack_chart(
    cohort_pct: pd.DataFrame,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    """x = partij/thema; 100%-gestapelde balken per leeftijdscohort.
    Gesorteerd op het aandeel nieuwste accounts (0-1 jaar) laag->hoog."""
    if cohort_pct.empty:
        return

    cohort_pct = cohort_pct.reindex(columns=COHORT_ORDER, fill_value=0)
    cohort_pct = cohort_pct.loc[cohort_pct[COHORT_ORDER[0]].sort_values(ascending=True).index]

    groups = cohort_pct.index.astype(str).tolist()
    x = np.arange(len(groups))
    cmap = plt.get_cmap("YlGnBu")

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(groups))

    for i, cohort in enumerate(COHORT_ORDER):
        values = cohort_pct[cohort].to_numpy()
        color = cmap(0.15 + 0.8 * i / (len(COHORT_ORDER) - 1))
        ax.bar(x, values, bottom=bottom, label=cohort, color=color)
        bottom += values

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Aandeel accounts (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(title="Accountleeftijd", loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_weekly_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str | None,
    title: str,
    y_label: str,
    legend_title: str,
    output_path: Path,
) -> None:
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    if group_col:
        pivot = (
            df.pivot_table(index=x_col, columns=group_col, values=y_col, aggfunc="sum")
            .fillna(0)
        )
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()
        cumulative = pivot.cumsum()
        # stackplot gebruikt de echte datetime-index, zodat de DateFormatter klopt
        # (pandas .plot(kind="area") gebruikt ordinale posities -> verkeerde datums).
        ax.stackplot(
            cumulative.index,
            [cumulative[col].to_numpy() for col in cumulative.columns],
            labels=[str(col) for col in cumulative.columns],
            alpha=0.75,
        )
        ax.legend(title=legend_title, loc="upper left", fontsize=8)
    else:
        df = df.sort_values(x_col).copy()
        df["cumsum"] = df[y_col].cumsum()
        x_dt = pd.to_datetime(df[x_col])
        ax.fill_between(x_dt, df["cumsum"], alpha=0.4)
        ax.plot(x_dt, df["cumsum"], marker="o")

    ax.set_title(title)
    ax.set_xlabel("Week (begin van de week)")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dutch_thousands))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    _format_date_axis(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_party_median_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter: grijze wolk van alle video's + één gelabeld mediaan-punt per partij
    op (mediaan weergaven, mediaan interactie per video). Log-log, met referentielijn."""
    d = df.copy()
    d["interactie"] = d["like_count"] + d["comment_count"]
    d = d[(d["view_count"] > 0) & (d["interactie"] > 0)]
    if d.empty:
        return

    spearman, p_spear = spearmanr(d["view_count"], d["interactie"])
    pearson_log, p_pear = pearsonr(np.log10(d["view_count"]), np.log10(d["interactie"]))
    ratio = (d["interactie"] / d["view_count"]).median()

    med = (
        d.groupby("party")
        .agg(med_views=("view_count", "median"), med_inter=("interactie", "median"), n=("view_count", "size"))
        .reset_index()
        .sort_values("party")
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # Grijze achtergrondwolk: alle video's
    ax.scatter(d["view_count"], d["interactie"], s=8, c="lightgrey",
               alpha=0.35, linewidths=0, label="Video's (individueel)")

    # Eén mediaan-punt per partij, gekleurd en gelabeld
    cmap = plt.get_cmap("tab20", len(med))
    for i, row in enumerate(med.itertuples()):
        ax.scatter(row.med_views, row.med_inter, s=130, color=cmap(i),
                   edgecolor="black", linewidths=0.7, zorder=5)
        ax.annotate(row.party, (row.med_views, row.med_inter), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, zorder=6)

    # Referentielijn: constante mediane engagement-ratio
    x_line = np.array([d["view_count"].min(), d["view_count"].max()])
    ratio_label = f"Mediane ratio ({ratio * 100:.1f}% interactie/weergave)".replace(".", ",")
    ax.plot(x_line, ratio * x_line, color="black", linestyle="--", linewidth=1.3, label=ratio_label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Vast bereik: weergaven 10^2-10^5, interactie 10^0-10^4
    ax.set_xlim(100, 100_000)
    ax.set_ylim(1, 10_000)
    ax.set_xlabel("Weergaven per video (mediaan per partij; wolk = losse video's, log)")
    ax.set_ylabel("Interactie per video — likes + reacties (log)")
    ax.set_title("Mediane weergaven vs interactie per video, per partij")

    def p_text(p):
        return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")

    def r_text(r):
        return f"{r:.2f}".replace(".", ",")

    stats_text = (
        f"Spearman ρ = {r_text(spearman)} ({p_text(p_spear)})\n"
        f"Pearson r (log-log) = {r_text(pearson_log)} ({p_text(p_pear)})\n"
        + f"n = {len(d):,}".replace(",", ".") + " video's"
    )
    ax.text(
        0.98, 0.02, stats_text, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
    )

    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_views_engagement_scatter(
    df: pd.DataFrame,
    color_col: str,
    label_singular: str,
    legend_title: str,
    output_path: Path,
    cmap_name: str = "tab10",
) -> None:
    """Scatter per video: weergaven (x) vs interactie=likes+reacties (y), log-log.
    Punten gekleurd per `color_col` (partij of thema); video's zonder waarde grijs.
    Referentielijn = mediane ratio."""
    d = df.copy()
    d["interactie"] = d["like_count"] + d["comment_count"]
    d = d[(d["view_count"] > 0) & (d["interactie"] > 0)]
    if d.empty:
        return

    # Spearman (rangen, robuust tegen uitschieters) als hoofdmaat;
    # Pearson op de log-log-waarden als maat voor de sterkte van het machtsverband.
    spearman, p_spear = spearmanr(d["view_count"], d["interactie"])
    pearson_log, p_pear = pearsonr(np.log10(d["view_count"]), np.log10(d["interactie"]))
    ratio = (d["interactie"] / d["view_count"]).median()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Video's zonder categorie (bijv. geen thema) als grijze achtergrond
    missing = d[d[color_col].isna()]
    if not missing.empty:
        ax.scatter(missing["view_count"], missing["interactie"], s=10, c="lightgrey",
                   alpha=0.4, linewidths=0, label=f"Overig (geen {label_singular})")

    # Gekleurde groepen er bovenop
    groups = sorted(d[color_col].dropna().unique())
    cmap = plt.get_cmap(cmap_name, len(groups))
    for i, group in enumerate(groups):
        sub = d[d[color_col] == group]
        ax.scatter(sub["view_count"], sub["interactie"], s=14,
                   color=cmap(i), alpha=0.7, linewidths=0, label=str(group))

    # Referentielijn: constante mediane engagement-ratio (op log-log een rechte met helling 1)
    x_line = np.array([d["view_count"].min(), d["view_count"].max()])
    ratio_label = f"Mediane ratio ({ratio * 100:.1f}% interactie/weergave)".replace(".", ",")
    ax.plot(x_line, ratio * x_line, color="black", linestyle="--", linewidth=1.3, label=ratio_label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Vast bereik: weergaven 10^2-10^5, interactie 10^0-10^4
    ax.set_xlim(100, 100_000)
    ax.set_ylim(1, 10_000)
    ax.set_xlabel("Weergaven per video (log)")
    ax.set_ylabel("Interactie per video — likes + reacties (log)")
    ax.set_title("Samenhang weergaven en interactie per video")

    def p_text(p):
        return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")

    def r_text(r):
        return f"{r:.2f}".replace(".", ",")

    stats_text = (
        f"Spearman ρ = {r_text(spearman)} ({p_text(p_spear)})\n"
        f"Pearson r (log-log) = {r_text(pearson_log)} ({p_text(p_pear)})\n"
        + f"n = {len(d):,}".replace(",", ".") + " video's"
    )
    ax.text(
        0.98, 0.02, stats_text, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
    )

    ncol = 2 if len(groups) > 10 else 1
    ax.legend(title=legend_title, fontsize=7, loc="upper left", ncol=ncol)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    records = read_jsonl(INPUT_FILE)
    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(f"Geen data gevonden in {INPUT_FILE}")

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"])

    df["week"] = df["published_at"].dt.to_period("W").dt.start_time

    for col in ["view_count", "like_count", "comment_count"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["party"] = df["search_terms"].apply(classify_party)
    df["segment"] = df.apply(lambda row: classify_segment(row["search_terms"], row["party"]), axis=1)
    df["theme"] = df["search_terms"].apply(classify_theme)

    df_with_theme = df.dropna(subset=["theme"])

    df.to_csv(TABLES_DIR / "videos_geclassificeerd.csv", index=False)

    # --- Aggregaties ---
    by_party = aggregate_interactions(df, ["party"])
    by_party_segment = aggregate_interactions(df, ["party", "segment"])
    by_theme = aggregate_interactions(df_with_theme, ["theme"])
    weekly_by_party = aggregate_interactions(df, ["week", "party"])
    weekly_by_theme = aggregate_interactions(df_with_theme, ["week", "theme"])

    by_party.to_csv(TABLES_DIR / "per_partij.csv", index=False)
    by_party_segment.to_csv(TABLES_DIR / "per_partij_per_segment.csv", index=False)
    by_theme.to_csv(TABLES_DIR / "per_thema.csv", index=False)
    weekly_by_party.to_csv(TABLES_DIR / "per_week_per_partij.csv", index=False)
    weekly_by_theme.to_csv(TABLES_DIR / "per_week_per_thema.csv", index=False)

    def party_pivot(metric: str) -> pd.DataFrame:
        return by_party_segment.pivot_table(
            index="party", columns="segment", values=metric, aggfunc="sum"
        ).fillna(0)

    # --- Per partij: aantal video's gestapeld (tellingen tel je wél op) ---
    save_stacked_bar_chart(
        party_pivot("video_count"),
        "Aantal video's per partij, gesplitst naar type zoekterm",
        "Partij",
        "Aantal video's",
        IMAGES_DIR / "aantal_videos_per_partij.png",
    )

    # --- Per partij: mediaan per video, één balk (partij + persoon gebundeld) ---
    for key, label in METRICS:
        save_bar_chart(
            by_party,
            "party",
            f"median_{key}",
            f"Mediaan aantal {label} per video per partij",
            "Partij",
            f"Mediaan aantal {label} per video",
            IMAGES_DIR / f"{key}_per_partij.png",
        )

    # --- Per thema: aantal video's + mediaan per video ---
    save_bar_chart(
        by_theme,
        "theme",
        "video_count",
        "Aantal video's per thema",
        "Thema",
        "Aantal video's",
        IMAGES_DIR / "aantal_videos_per_thema.png",
    )
    for key, label in METRICS:
        save_bar_chart(
            by_theme,
            "theme",
            f"median_{key}",
            f"Mediaan aantal {label} per video per thema",
            "Thema",
            f"Mediaan aantal {label} per video",
            IMAGES_DIR / f"{key}_per_thema.png",
        )

    # --- Cumulatief aantal video's per week, gestapeld per partij / per thema ---
    save_weekly_chart(
        weekly_by_party, "week", "video_count", "party",
        "Cumulatief aantal video's per week, gestapeld per partij",
        "Cumulatief aantal video's",
        "Partij",
        IMAGES_DIR / "cumulatief_videos_per_week_per_partij.png",
    )
    save_weekly_chart(
        weekly_by_theme, "week", "video_count", "theme",
        "Cumulatief aantal video's per week, gestapeld per thema",
        "Cumulatief aantal video's",
        "Thema",
        IMAGES_DIR / "cumulatief_videos_per_week_per_thema.png",
    )

    # --- Cumulatief per week, gestapeld per partij / per thema (totalen) ---
    for key, label in METRICS:
        save_weekly_chart(
            weekly_by_party, "week", f"total_{key}", "party",
            f"Cumulatief aantal {label} per week, gestapeld per partij",
            f"Cumulatief aantal {label}",
            "Partij",
            IMAGES_DIR / f"cumulatief_{key}_per_week_per_partij.png",
        )
        save_weekly_chart(
            weekly_by_theme, "week", f"total_{key}", "theme",
            f"Cumulatief aantal {label} per week, gestapeld per thema",
            f"Cumulatief aantal {label}",
            "Thema",
            IMAGES_DIR / f"cumulatief_{key}_per_week_per_thema.png",
        )

    # --- Samenhang weergaven vs interactie per video ---
    # Per partij: grijze wolk + mediaan-punt per partij; per thema: video's gekleurd per thema
    save_party_median_scatter(
        df, IMAGES_DIR / "samenhang_weergaven_interactie_per_partij.png"
    )
    save_views_engagement_scatter(
        df, "theme", "thema", "Thema",
        IMAGES_DIR / "samenhang_weergaven_interactie_per_thema.png", "tab10",
    )

    # --- Accountleeftijd van reageerders per partij / per thema ---
    if COMMENTS_FILE.exists():
        comments = pd.DataFrame(read_jsonl(COMMENTS_FILE))
    else:
        comments = pd.DataFrame()

    if not comments.empty and "commenter_account_age_days" in comments.columns:
        comments["party"] = comments["search_terms"].apply(classify_party)
        comments["theme"] = comments["search_terms"].apply(classify_theme)
        comments_with_theme = comments.dropna(subset=["theme"])

        for group_col, x_label, label, token, source in [
            ("party", "Partij", "partij", "partij", comments),
            ("theme", "Thema", "thema", "thema", comments_with_theme),
        ]:
            summary, cohort_pct = account_age_breakdown(source, group_col)

            # CSV: aandeel nieuw + cohort-percentages naast elkaar
            cohort_table = cohort_pct.reset_index().rename(columns={group_col: group_col})
            summary.merge(cohort_table, on=group_col).to_csv(
                TABLES_DIR / f"accountleeftijd_per_{token}.csv", index=False
            )

            save_bar_chart(
                summary,
                group_col,
                "pct_nieuw",
                f"Aandeel nieuwe reageerder-accounts (< {NEW_ACCOUNT_MAX_YEARS:g} jaar) per {label}",
                x_label,
                "Aandeel nieuwe accounts (%)",
                IMAGES_DIR / f"nieuwe_accounts_per_{token}.png",
                decimals=1,
            )
            save_cohort_stack_chart(
                cohort_pct,
                f"Leeftijdsverdeling reageerder-accounts per {label}",
                x_label,
                IMAGES_DIR / f"accountcohorten_per_{token}.png",
            )
    else:
        print("Let op: geen accountleeftijd-data gevonden in dataset_processed/comments.jsonl - "
              "draai retrieve.py, anonymize.py en process.py opnieuw. Grafieken overgeslagen.")

    print("Klaar.")
    print(f"Tabellen: {TABLES_DIR}")
    print(f"Afbeeldingen: {IMAGES_DIR}")
    print(f"Aantal video's geanalyseerd: {len(df)}")
    print(f"Totaal views: {df['view_count'].sum()}")
    print(f"Totaal comments: {df['comment_count'].sum()}")
    print(f"Totaal likes: {df['like_count'].sum()}")


if __name__ == "__main__":
    main()
