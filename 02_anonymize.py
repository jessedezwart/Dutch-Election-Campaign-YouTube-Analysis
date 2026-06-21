import os
import json
import hashlib
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv
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


load_dotenv()

SALT = os.getenv("HASH_SALT")
if not SALT:
    raise RuntimeError("HASH_SALT ontbreekt in .env")

UNPROCESSED_DIR = Path("dataset_unprocessed")
SEARCH_FILE = UNPROCESSED_DIR / "search.list.jsonl"
COMMENTS_FILE = UNPROCESSED_DIR / "commentThreads.list.jsonl"
VIDEOS_FILE = UNPROCESSED_DIR / "videos.list.jsonl"
CHANNELS_FILE = UNPROCESSED_DIR / "channels.list.jsonl"

ANON_DIR = Path("dataset_anonymized")
ANON_DIR.mkdir(exist_ok=True)

ANON_SEARCH_FILE = ANON_DIR / "search.list.jsonl"
ANON_VIDEOS_FILE = ANON_DIR / "videos.list.jsonl"
ANON_COMMENTS_FILE = ANON_DIR / "commentThreads.list.jsonl"
ANON_CHANNELS_FILE = ANON_DIR / "channels.list.jsonl"

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


def hash_value(value: str, prefix: str) -> str:
    raw = f"{SALT}:{prefix}:{value}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_video(video_id: str | None) -> str | None:
    return hash_value(video_id, "video") if video_id else None


def hash_commenter(channel_id: str | None) -> str | None:
    return hash_value(channel_id, "commenter") if channel_id else None


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_jsonl(file_obj: TextIO, record: dict) -> None:
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_author_channel_id(comment_snippet: dict) -> str | None:
    author_channel_id = comment_snippet.get("authorChannelId")

    if isinstance(author_channel_id, dict):
        return author_channel_id.get("value")

    return None


def anonymize_search(progress: Progress) -> None:
    """search.list: video-id hashen, alleen q + publishedAt behouden."""
    total_pages = count_jsonl_records(SEARCH_FILE)
    task_id = progress.add_task(
        "Search anonimiseren", total=total_pages, detail="0 videos"
    )
    total_videos = 0
    with ANON_SEARCH_FILE.open("w", encoding="utf-8") as search_file:
        for record in read_jsonl(SEARCH_FILE):
            items = []
            for item in record.get("response", {}).get("items", []):
                video_id = item.get("id", {}).get("videoId")
                items.append(
                    {
                        "id": {"videoId": hash_video(video_id)},
                        "snippet": {
                            "publishedAt": item.get("snippet", {}).get("publishedAt")
                        },
                    }
                )

            write_jsonl(
                search_file,
                {
                    "request": {"q": record.get("request", {}).get("q")},
                    "response": {"items": items},
                },
            )
            total_videos += len(items)
            progress.update(task_id, advance=1, detail=f"{total_videos} videos")


def anonymize_videos(progress: Progress) -> None:
    """videos.list: id hashen, alleen statistics + taalvelden behouden."""
    total_batches = count_jsonl_records(VIDEOS_FILE)
    task_id = progress.add_task(
        "Video's anonimiseren", total=total_batches, detail="0 videos"
    )
    total_videos = 0
    with ANON_VIDEOS_FILE.open("w", encoding="utf-8") as videos_file:
        for record in read_jsonl(VIDEOS_FILE):
            items = []
            for item in record.get("response", {}).get("items", []):
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                items.append(
                    {
                        "id": hash_video(item.get("id")),
                        "statistics": {
                            "viewCount": stats.get("viewCount"),
                            "likeCount": stats.get("likeCount"),
                            "commentCount": stats.get("commentCount"),
                        },
                        "snippet": {
                            "defaultAudioLanguage": snippet.get("defaultAudioLanguage"),
                            "defaultLanguage": snippet.get("defaultLanguage"),
                        },
                    }
                )

            write_jsonl(videos_file, {"response": {"items": items}})
            total_videos += len(items)
            progress.update(task_id, advance=1, detail=f"{total_videos} videos")


def anonymize_comment(comment: dict) -> dict:
    snippet = comment.get("snippet", {})
    commenter_id = extract_author_channel_id(snippet)
    return {
        "id": comment.get("id"),
        "snippet": {
            "authorChannelId": hash_commenter(commenter_id),
            "textDisplay": snippet.get("textDisplay"),
            "likeCount": snippet.get("likeCount"),
            "publishedAt": snippet.get("publishedAt"),
            "updatedAt": snippet.get("updatedAt"),
        },
    }


def anonymize_comments(progress: Progress) -> None:
    """commentThreads.list: video-id + commenter-id hashen, gebruikte commentvelden behouden."""
    total_pages = count_jsonl_records(COMMENTS_FILE)
    task_id = progress.add_task(
        "Comments anonimiseren", total=total_pages, detail="0 comments"
    )
    total_comments = 0
    with ANON_COMMENTS_FILE.open("w", encoding="utf-8") as comments_file:
        for record in read_jsonl(COMMENTS_FILE):
            video_id = record.get("request", {}).get("videoId")
            items = []
            for item in record.get("response", {}).get("items", []):
                top_comment = item.get("snippet", {}).get("topLevelComment", {})
                replies = item.get("replies", {}).get("comments", [])
                items.append(
                    {
                        "snippet": {"topLevelComment": anonymize_comment(top_comment)},
                        "replies": {
                            "comments": [anonymize_comment(reply) for reply in replies]
                        },
                    }
                )

            write_jsonl(
                comments_file,
                {
                    "request": {"videoId": hash_video(video_id)},
                    "response": {"items": items},
                },
            )
            total_comments += sum(
                1 + len(item.get("replies", {}).get("comments", []))
                for item in record.get("response", {}).get("items", [])
            )
            progress.update(task_id, advance=1, detail=f"{total_comments} comments")


def anonymize_channels(progress: Progress) -> None:
    """channels.list: id hashen (zelfde prefix als commenter), alleen publishedAt behouden."""
    total_batches = count_jsonl_records(CHANNELS_FILE)
    task_id = progress.add_task(
        "Kanalen anonimiseren", total=total_batches, detail="0 kanalen"
    )
    total_channels = 0
    with ANON_CHANNELS_FILE.open("w", encoding="utf-8") as channels_file:
        for record in read_jsonl(CHANNELS_FILE):
            items = []
            for item in record.get("response", {}).get("items", []):
                items.append(
                    {
                        "id": hash_commenter(item.get("id")),
                        "snippet": {
                            "publishedAt": item.get("snippet", {}).get("publishedAt")
                        },
                    }
                )

            write_jsonl(channels_file, {"response": {"items": items}})
            total_channels += len(items)
            progress.update(task_id, advance=1, detail=f"{total_channels} kanalen")


def main():
    with Progress(*_PROGRESS_COLUMNS, console=console) as progress:
        anonymize_search(progress)
        anonymize_videos(progress)
        anonymize_comments(progress)
        anonymize_channels(progress)

    console.print(f"Geanonimiseerde dataset: {ANON_DIR}")


if __name__ == "__main__":
    main()
