import json
from pathlib import Path
from datetime import datetime, timezone
from typing import TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


# Leest de geanonimiseerde dump (video-/kanaal-ids zijn al gehasht door anonymize.py).
ANON_DIR = Path("dataset_anonymized")
SEARCH_FILE = ANON_DIR / "search.list.jsonl"
COMMENTS_FILE = ANON_DIR / "commentThreads.list.jsonl"
VIDEOS_FILE = ANON_DIR / "videos.list.jsonl"
CHANNELS_FILE = ANON_DIR / "channels.list.jsonl"

# Vaste referentiedatum (Tweede Kamerverkiezingen 2025) voor reproduceerbare accountleeftijd.
REFERENCE_DATE = datetime(2025, 10, 29, tzinfo=timezone.utc)

OUTPUT_DIR = Path("dataset_processed")
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEOS_OUTPUT_FILE = OUTPUT_DIR / "videos.jsonl"
COMMENTS_OUTPUT_FILE = OUTPUT_DIR / "comments.jsonl"

console = Console()
_PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("{task.fields[detail]}"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(file_obj: TextIO, record: dict) -> None:
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_items_by_id(
    path: Path,
    progress: Progress,
    description: str,
    detail_label: str,
) -> dict[str, dict]:
    """Bouw een dict op id uit een .jsonl met batch-responses (videos/channels).
    Een dubbele batch na een herstart is onschadelijk: laatste wint."""
    total_batches = count_jsonl_records(path)
    task_id = progress.add_task(description, total=total_batches, detail=f"0 {detail_label}")
    items_by_id: dict[str, dict] = {}
    for record in read_jsonl(path):
        for item in record.get("response", {}).get("items", []):
            item_id = item.get("id")
            if item_id:
                items_by_id[item_id] = item
        progress.update(task_id, advance=1, detail=f"{len(items_by_id)} {detail_label}")
    return items_by_id


def load_video_details(progress: Progress) -> dict[str, dict]:
    """Geanonimiseerde videos.list items op video_hash."""
    return load_items_by_id(
        VIDEOS_FILE,
        progress,
        "Video-details laden",
        "video's",
    )


def load_channels(progress: Progress) -> dict[str, dict]:
    """Geanonimiseerde channels.list items op channel_hash."""
    return load_items_by_id(
        CHANNELS_FILE,
        progress,
        "Kanaal-details laden",
        "kanalen",
    )


def extract_stats(video_item: dict) -> dict:
    """Haal de cijfers uit een ruw videos.list item (statistics ontbreekt soms)."""
    stats = video_item.get("statistics", {})
    return {
        "view_count": int(stats.get("viewCount", 0) or 0),
        "like_count": int(stats.get("likeCount", 0) or 0),
        "comment_count": int(stats.get("commentCount", 0) or 0),
    }


def account_age_days(channel_id: str | None, channels: dict[str, dict]) -> int | None:
    """Leeftijd van het account in dagen op de referentiedatum, of None als onbekend."""
    if not channel_id:
        return None

    channel_item = channels.get(channel_id)
    if not channel_item:
        return None

    published_at = channel_item.get("snippet", {}).get("publishedAt")
    if not published_at:
        return None

    created = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    return (REFERENCE_DATE - created).days


def is_dutch_or_unknown(video_item: dict) -> bool:
    """Behoud video als er geen taalsignaal is, of als een van de taalvelden Nederlands is.
    Sluit alleen video's uit die expliciet een niet-Nederlandse taal aangeven."""
    snippet = video_item.get("snippet", {})
    langs = [
        snippet.get("defaultAudioLanguage"),
        snippet.get("defaultLanguage"),
    ]
    langs = [lang for lang in langs if lang]

    if not langs:
        return True

    return any(lang.lower().startswith("nl") for lang in langs)


def build_video_lookup(video_details: dict[str, dict], progress: Progress) -> dict[str, dict]:
    video_lookup = {}
    total_pages = count_jsonl_records(SEARCH_FILE)
    task_id = progress.add_task(
        "Video lookup bouwen",
        total=total_pages,
        detail="0 unieke video's",
    )

    for record in read_jsonl(SEARCH_FILE):
        search_term = record.get("request", {}).get("q")

        for item in record.get("response", {}).get("items", []):
            # Het video-id is al gehasht door anonymize.py; het dient meteen als video_hash.
            video_hash = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})

            if not video_hash:
                continue

            # Filter niet-Nederlandse video's eruit (op basis van taalvelden uit videos.list).
            details_entry = video_details.get(video_hash)
            if details_entry is not None and not is_dutch_or_unknown(details_entry):
                continue

            if video_hash not in video_lookup:
                video_lookup[video_hash] = {
                    "video_hash": video_hash,
                    "published_at": snippet.get("publishedAt"),
                    "search_terms": set(),
                }

            video_lookup[video_hash]["search_terms"].add(search_term)
        progress.update(
            task_id,
            advance=1,
            detail=f"{len(video_lookup)} unieke video's",
        )

    return video_lookup


def write_videos(
    video_lookup: dict[str, dict],
    video_details: dict[str, dict],
    progress: Progress,
) -> None:
    task_id = progress.add_task(
        "Video-output schrijven",
        total=len(video_lookup),
        detail="0 video's",
    )
    with VIDEOS_OUTPUT_FILE.open("w", encoding="utf-8") as videos_file:
        written_videos = 0
        for video_hash, video_data in video_lookup.items():
            stats = extract_stats(video_details.get(video_hash, {}))
            record = {
                "video_hash": video_data["video_hash"],
                "published_at": video_data["published_at"],
                "search_terms": sorted(video_data["search_terms"]),
                "view_count": stats["view_count"],
                "like_count": stats["like_count"],
                "comment_count": stats["comment_count"],
            }

            write_jsonl(videos_file, record)
            written_videos += 1
            progress.update(task_id, advance=1, detail=f"{written_videos} video's")


def write_comments(
    video_lookup: dict[str, dict],
    channels: dict[str, dict],
    progress: Progress,
) -> None:
    total_pages = count_jsonl_records(COMMENTS_FILE)
    task_id = progress.add_task(
        "Comments verwerken",
        total=total_pages,
        detail="0 comments",
    )
    with COMMENTS_OUTPUT_FILE.open("w", encoding="utf-8") as comments_file:
        # De .jsonl-stream kan een dubbele pagina bevatten na een herstart; dedup op comment-id.
        seen_comment_ids: set[str] = set()
        emitted_comments = 0

        # Emit een comment naar het output-bestand, mits nog niet gezien.
        def emit(comment: dict, video_hash: str, search_terms: list[str]) -> bool:
            comment_id = comment.get("id")
            if comment_id is not None:
                if comment_id in seen_comment_ids:
                    return False
                seen_comment_ids.add(comment_id)

            snippet = comment.get("snippet", {})
            # authorChannelId is al gehasht door anonymize.py en dient meteen als commenter_hash.
            commenter_hash = snippet.get("authorChannelId")

            write_jsonl(comments_file, {
                "video_hash": video_hash,
                "commenter_hash": commenter_hash,
                "text": snippet.get("textDisplay"),
                "like_count": snippet.get("likeCount"),
                "published_at": snippet.get("publishedAt"),
                "updated_at": snippet.get("updatedAt"),
                "search_terms": search_terms,
                "commenter_account_age_days": account_age_days(commenter_hash, channels),
            })
            return True

        for record in read_jsonl(COMMENTS_FILE):
            video_hash = record.get("request", {}).get("videoId")
            video_data = video_lookup.get(video_hash)

            # Video niet in lookup -> uitgefilterd (niet-Nederlands) of onbekend: comments overslaan.
            if not video_data:
                progress.update(task_id, advance=1, detail=f"{emitted_comments} comments")
                continue

            video_hash = video_data["video_hash"]
            search_terms = sorted(video_data["search_terms"])

            for item in record.get("response", {}).get("items", []):
                top_comment = item.get("snippet", {}).get("topLevelComment", {})
                if emit(top_comment, video_hash, search_terms):
                    emitted_comments += 1

                for reply in item.get("replies", {}).get("comments", []):
                    if emit(reply, video_hash, search_terms):
                        emitted_comments += 1
            progress.update(task_id, advance=1, detail=f"{emitted_comments} comments")


def main():
    with Progress(*_PROGRESS_COLUMNS, console=console) as progress:
        # Laad geanonimiseerde videos.list/channels.list en bouw de video-lookup met taalfilter
        video_details = load_video_details(progress)
        channels = load_channels(progress)
        video_lookup = build_video_lookup(video_details, progress)

        # Schrijf de verwerkte video-metadata en comments naar JSONL-bestanden
        write_videos(video_lookup, video_details, progress)
        write_comments(video_lookup, channels, progress)

    console.print(f"Verwerkte video's: {VIDEOS_OUTPUT_FILE}")
    console.print(f"Verwerkte comments: {COMMENTS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
