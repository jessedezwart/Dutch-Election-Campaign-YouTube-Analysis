import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path("sentiment/comments_sentiment.jsonl")

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tabellen"
IMAGES_DIR = RESULTS_DIR / "images"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


PARTY_KEYWORDS = {
    "PVV": ["PVV", "Partij voor de Vrijheid", "Geert Wilders"],
    "GroenLinks-PvdA": ["GroenLinks PvdA", "GroenLinks-PvdA", "Frans Timmermans"],
    "VVD": ["VVD", "Volkspartij voor Vrijheid en Democratie", "Dilan Yeşilgöz"],
    "NSC": ["NSC", "Nieuw Sociaal Contract", "Pieter Omtzigt"],
    "D66": ["D66", "Democraten 66", "Rob Jetten"],
    "BBB": ["BBB", "BoerBurgerBeweging", "Caroline van der Plas"],
    "CDA": ["CDA", "Christen-Democratisch Appèl", "Henri Bontenbal"],
    "SP": ["SP", "Socialistische Partij", "Jimmy Dijk"],
    "PvdD": ["PvdD", "Partij voor de Dieren", "Esther Ouwehand"],
    "FVD": ["FVD", "Forum voor Democratie", "Thierry Baudet", "Lidewij de Vos"],
    "SGP": ["SGP", "Staatkundig Gereformeerde Partij", "Chris Stoffer"],
    "ChristenUnie": ["ChristenUnie", "Mirjam Bikker"],
    "Volt": ["Volt", "Volt Nederland", "Laurens Dassen"],
    "JA21": ["JA21", "Joost Eerdmans"],
    "DENK": ["DENK", "DENK Nederland", "Stephan van Baarle"],
    "50PLUS": ["50PLUS"],
    "BIJ1": ["BIJ1"],
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

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "positive": "#2ca02c",
}

# Jaarcohorten voor accountleeftijd (gelijk aan deelvraag1_2_combined.py)
DAYS_PER_YEAR = 365.25
MAX_COHORT_YEARS = 6
COHORT_ORDER = [f"{i}-{i + 1} jaar" for i in range(MAX_COHORT_YEARS)] + [
    f"{MAX_COHORT_YEARS}+ jaar"
]

# Minimaal aantal reacties per (partij, cohort)-cel om in de heatmap te tonen
HEATMAP_MIN_N = 20


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


# Map search terms
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


def sentiment_breakdown(
    df: pd.DataFrame, group_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        df.groupby([group_col, "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )

    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)

    percentages = counts[SENTIMENT_ORDER].div(counts["total"], axis=0).mul(100).round(2)
    percentages["total"] = counts["total"]

    # Sorting by raw total is meaningless once every bar is normalized to 100%,
    # so order by net sentiment instead to make the distribution comparable.
    percentages["net_sentiment"] = percentages["positive"] - percentages["negative"]
    percentages = percentages.sort_values("net_sentiment", ascending=False)

    return counts.reset_index(), percentages.reset_index()


def save_stacked_bar_chart(
    df: pd.DataFrame, x_col: str, title: str, ylabel: str, output_path: Path
) -> None:
    if df.empty:
        return

    plt.figure(figsize=(12, 6))
    bottom = pd.Series(0, index=df.index, dtype=float)

    for label in SENTIMENT_ORDER:
        plt.bar(
            df[x_col].astype(str),
            df[label],
            bottom=bottom,
            label=label,
            color=SENTIMENT_COLORS[label],
        )
        bottom += df[label]

    max_height = bottom.max()
    plt.ylim(top=max_height * 1.12)

    for x, (top, n) in enumerate(zip(bottom, df["total"])):
        plt.text(
            x, top + max_height * 0.02, f"n={n}", ha="center", va="bottom", fontsize=8
        )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def cohort_label(age_days: float) -> str:
    age_years = max(age_days, 0) / DAYS_PER_YEAR
    bucket = int(age_years)
    if bucket >= MAX_COHORT_YEARS:
        return f"{MAX_COHORT_YEARS}+ jaar"
    return f"{bucket}-{bucket + 1} jaar"


def net_sentiment(group: pd.DataFrame) -> float:
    """Net-sentiment (% positief − % negatief) van een groep reacties."""
    n = len(group)
    if n == 0:
        return float("nan")
    pos = (group["sentiment_label"] == "positive").sum()
    neg = (group["sentiment_label"] == "negative").sum()
    return (pos - neg) / n * 100


def save_net_sentiment_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap partij (rij) x accountleeftijdscohort (kolom), kleur = net-sentiment.
    Cellen met te weinig reacties (< HEATMAP_MIN_N) worden grijs gemaskeerd."""
    if df.empty:
        return

    counts = (
        df.groupby(["party", "cohort", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    counts["net"] = (counts["positive"] - counts["negative"]) / counts["total"] * 100

    net = counts["net"].unstack("cohort").reindex(columns=COHORT_ORDER)
    n = counts["total"].unstack("cohort").reindex(columns=COHORT_ORDER)

    # Partijen ordenen op algeheel net-sentiment (positief bovenaan)
    party_order = (
        df.groupby("party")
        .apply(net_sentiment, include_groups=False)
        .sort_values(ascending=False)
        .index
    )
    net = net.reindex(party_order)
    n = n.reindex(party_order)

    masked = np.ma.masked_invalid(net.where(n >= HEATMAP_MIN_N).to_numpy())
    if masked.count() == 0:
        return
    v = float(np.nanmax(np.abs(masked)))
    v = v if v > 0 else 1.0

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(masked, cmap=cmap, vmin=-v, vmax=v, aspect="auto")

    ax.set_xticks(range(len(COHORT_ORDER)))
    ax.set_xticklabels(COHORT_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(net.index)))
    ax.set_yticklabels(net.index)
    ax.set_xlabel("Accountleeftijd van de reageerder")
    ax.set_ylabel("Partij")
    ax.set_title("Net-sentiment van reacties per partij en accountleeftijdscohort")

    # Annotaties: net-sentiment + aantal reacties
    net_vals = net.to_numpy()
    n_vals = n.to_numpy()
    for r in range(net_vals.shape[0]):
        for c in range(net_vals.shape[1]):
            count = n_vals[r, c]
            if np.isnan(net_vals[r, c]) or np.isnan(count) or count < HEATMAP_MIN_N:
                continue
            ax.text(
                c,
                r,
                f"{net_vals[r, c]:+.0f}\n(n={int(count)})",
                ha="center",
                va="center",
                fontsize=6.5,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Net-sentiment (% positief − % negatief)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    records = read_jsonl(INPUT_FILE)
    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(f"Geen data gevonden in {INPUT_FILE}")

    df["search_text"] = df["search_terms"]
    df["party"] = df["search_text"].apply(lambda value: classify(value, PARTY_KEYWORDS))
    df["theme"] = df["search_text"].apply(lambda value: classify(value, THEME_KEYWORDS))

    df_with_theme = df.dropna(subset=["theme"])

    by_party_counts, by_party_pct = sentiment_breakdown(df, "party")
    by_theme_counts, by_theme_pct = sentiment_breakdown(df_with_theme, "theme")

    by_party_counts.to_csv(TABLES_DIR / "sentiment_per_partij.csv", index=False)
    by_theme_counts.to_csv(TABLES_DIR / "sentiment_per_thema.csv", index=False)
    by_party_pct.to_csv(TABLES_DIR / "sentiment_per_partij_percentage.csv", index=False)
    by_theme_pct.to_csv(TABLES_DIR / "sentiment_per_thema_percentage.csv", index=False)

    save_stacked_bar_chart(
        by_party_counts,
        "party",
        "Sentiment van reacties per partij",
        "Aantal reacties",
        IMAGES_DIR / "sentiment_per_partij.png",
    )

    save_stacked_bar_chart(
        by_theme_counts,
        "theme",
        "Sentiment van reacties per thema",
        "Aantal reacties",
        IMAGES_DIR / "sentiment_per_thema.png",
    )

    save_stacked_bar_chart(
        by_party_pct,
        "party",
        "Sentiment van reacties per partij (%)",
        "Percentage reacties",
        IMAGES_DIR / "sentiment_per_partij_percentage.png",
    )

    save_stacked_bar_chart(
        by_theme_pct,
        "theme",
        "Sentiment van reacties per thema (%)",
        "Percentage reacties",
        IMAGES_DIR / "sentiment_per_thema_percentage.png",
    )

    # --- Sentiment naar accountleeftijd van de reageerder ---
    if (
        "commenter_account_age_days" in df.columns
        and df["commenter_account_age_days"].notna().any()
    ):
        df_age = df.dropna(subset=["commenter_account_age_days"]).copy()
        df_age["cohort"] = df_age["commenter_account_age_days"].apply(cohort_label)

        # Aggregaat over alle partijen: welke leeftijden reageren positief/negatief?
        by_cohort_counts, by_cohort_pct = sentiment_breakdown(df_age, "cohort")
        # Chronologische cohort-volgorde i.p.v. sortering op net-sentiment
        by_cohort_counts = (
            by_cohort_counts.set_index("cohort")
            .reindex(COHORT_ORDER)
            .dropna(subset=["total"])
            .reset_index()
        )
        by_cohort_pct = (
            by_cohort_pct.set_index("cohort")
            .reindex(COHORT_ORDER)
            .dropna(subset=["total"])
            .reset_index()
        )

        by_cohort_counts.to_csv(
            TABLES_DIR / "sentiment_per_accountleeftijd.csv", index=False
        )
        by_cohort_pct.to_csv(
            TABLES_DIR / "sentiment_per_accountleeftijd_percentage.csv", index=False
        )

        save_stacked_bar_chart(
            by_cohort_counts,
            "cohort",
            "Sentiment van reacties naar accountleeftijd",
            "Aantal reacties",
            IMAGES_DIR / "sentiment_per_accountleeftijd.png",
        )
        save_stacked_bar_chart(
            by_cohort_pct,
            "cohort",
            "Sentiment van reacties naar accountleeftijd (%)",
            "Percentage reacties",
            IMAGES_DIR / "sentiment_per_accountleeftijd_percentage.png",
        )

        # Hoofdgrafiek: partij x cohort, gekleurd op net-sentiment
        save_net_sentiment_heatmap(
            df_age, IMAGES_DIR / "sentiment_net_partij_per_cohort.png"
        )
    else:
        print(
            "Let op: geen accountleeftijd in de sentiment-data - draai de pipeline "
            "(retrieve.py, anonymize.py, process.py, sentiment_analysis.py) opnieuw. "
            "Leeftijdsgrafieken overgeslagen."
        )

    print("Klaar.")
    print(f"Tabellen: {TABLES_DIR}")
    print(f"Afbeeldingen: {IMAGES_DIR}")
    print(f"Aantal reacties geanalyseerd: {len(df)}")
    print(f"Aantal reacties met thema: {len(df_with_theme)}")


if __name__ == "__main__":
    main()
