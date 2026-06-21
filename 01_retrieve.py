from datetime import datetime
import os
import json
import logging
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Laadt de API-sleutel uit een .env-bestand
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY ontbreekt in .env")

# Initialiseer de YouTube API-client
youtube = build("youtube", "v3", developerKey=API_KEY)

console = Console()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"error_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

_file_handler = logging.FileHandler(log_file, encoding="utf-8")
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
)

logging.basicConfig(
    level=logging.WARNING,
    handlers=[RichHandler(console=console, show_path=False), _file_handler],
)

logger = logging.getLogger(__name__)
_PROGRESS_COLUMNS = (
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)

# Outputs
OUTPUT_DIR = Path("dataset_unprocessed")
OUTPUT_DIR.mkdir(exist_ok=True)

SEARCH_FILE = OUTPUT_DIR / "search.list.jsonl"
VIDEOS_FILE = OUTPUT_DIR / "videos.list.jsonl"
COMMENTS_FILE = OUTPUT_DIR / "commentThreads.list.jsonl"
CHANNELS_FILE = OUTPUT_DIR / "channels.list.jsonl"

# Daterange
PUBLISHED_AFTER = "2025-09-23T00:00:00Z"
PUBLISHED_BEFORE = "2025-10-28T00:00:00Z"

# Termen waarop gezocht wordt
TERMS_FILE = Path("trefwoordenlijst.txt")

# Maximale results per zoekopdracht (API-limiet)
MAX_RESULTS_PER_SEARCH_TERM = 50
# Maximaal aantal pagina's per zoekterm (om te voorkomen dat we te diep gaan bij termen met veel resultaten)
MAX_PAGES_PER_SEARCH = 5

# Max per batch voor videos.list en channels.list, waarbij we zelf de batches maken. De API accepteert maximaal 50 id's per request
BATCH_SIZE = 50  # API max

# Maximaal aantal comment-pagina's per video (om te voorkomen dat we te diep gaan bij video's met veel comments)
MAX_COMMENT_PAGES_PER_VIDEO = 10
# Max results per pagina bij commentThreads.list (API-limiet)
MAX_COMMENTS_PER_PAGE = 100


# Krijg de zoektermen uit het bestand, negeer lege regels en whitespace
def read_search_terms(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# Schrijf de request (de kwargs van de API-call) en bijbehorende response naar een al geopende .jsonl-stream.
def append_jsonl(file_obj: TextIO, request: dict, response: dict) -> None:
    # De ascii moet op False staan zodat tekens zoals emoji's correct worden opgeslagen, anders worden ze vervangen door \uXXXX
    record = {"request": request, "response": response}
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


# Lees een .jsonl-bestand als die bestaat
def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Geef een line terug dmv een generator zodat we niet alles in het geheugen hoeven te laden
                yield json.loads(line)


# Verzamel de ids die al opgehaald zijn uit een .jsonl met batch-responses
# (videos.list / channels.list). Zo weten we wat we kunnen overslaan op een hervatte run.
def fetched_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for record in read_jsonl(path):
        for item in record.get("response", {}).get("items", []):
            item_id = item.get("id")
            if item_id:
                ids.add(item_id)
    return ids


# Verzamel de progressie van pagina's per key (zoekterm of video-id) uit een .jsonl met pagina-responses.
# key_field is een veld uit de opgeslagen request-kwargs (bijv. "q" voor search, "videoId" voor comments).
def page_progress(path: Path, key_field: str) -> dict[str, dict]:
    progress: dict[str, dict] = {}
    for record in read_jsonl(path):
        key = record.get("request", {}).get(key_field)
        if key is None:
            continue
        entry = progress.setdefault(key, {"next_page_token": None, "pages": 0})
        entry["pages"] += 1
        entry["next_page_token"] = record.get("response", {}).get("nextPageToken")
    return progress


# Deze functie haalt authorChannelIds uit een comment.
# Dit wordt gebruikt bij het ophalen van de kanalen van de auteurs van comments.
def extract_author_channel_id(comment_snippet: dict) -> str | None:
    author_channel_id = comment_snippet.get("authorChannelId")

    if isinstance(author_channel_id, dict):
        return author_channel_id.get("value")

    return None


# Retry-settings
def should_retry_youtube_error(exc: BaseException) -> bool:
    # Geen retry voor niet-HTTP-fouten
    if not isinstance(exc, HttpError):
        return False

    status = getattr(exc.resp, "status", None)

    if status in {429, 500, 502, 503, 504}:
        return True

    return False


_retry_api = retry(
    retry=retry_if_exception(should_retry_youtube_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
    reraise=True,
)


@_retry_api
def api_search(kwargs: dict) -> dict:
    return youtube.search().list(**kwargs).execute()


@_retry_api
def api_videos(kwargs: dict) -> dict:
    return youtube.videos().list(**kwargs).execute()


@_retry_api
def api_comment_page(kwargs: dict) -> dict:
    return youtube.commentThreads().list(**kwargs).execute()


@_retry_api
def api_channels(kwargs: dict) -> dict:
    return youtube.channels().list(**kwargs).execute()


# Krijg alle unieke video-IDs uit de zoekresultaten
def get_video_ids() -> set[str]:
    video_ids: set[str] = set()
    for record in read_jsonl(SEARCH_FILE):
        for item in record.get("response", {}).get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                video_ids.add(vid)
    console.print(f"Unieke video-IDs gevonden: {len(video_ids)}")
    return video_ids


# Voert de daadwerkelijke zoekopdrachten uit en slaat de ruwe responses op in een .jsonl-bestand.
def fetch_search_results() -> None:
    search_terms = read_search_terms(TERMS_FILE)
    console.print(f"Aantal zoektermen: {len(search_terms)}")

    progress = page_progress(SEARCH_FILE, "q")

    # Helperfunctie om te bepalen of we klaar zijn met een zoekterm, op basis van het aantal opgehaalde pagina's en of er een volgende pagina is.
    def is_term_search_done(term: str) -> bool:
        entry = progress.get(term)
        if entry is None:
            return False  # Nog geen enkele pagina opgehaald
        if entry["pages"] >= MAX_PAGES_PER_SEARCH:
            logger.warning("search\t%s\tmax_pages_reached", term)
            return True  # Maximaal aantal pagina's bereikt
        return entry["next_page_token"] is None

    # Breadth-first: pagina 1 voor alle termen, dan pagina 2, enz. Zo houdt elke
    # term bij quota-uitval de eerder opgehaalde pagina's.
    with SEARCH_FILE.open("a", encoding="utf-8") as search_file:
        for page in range(MAX_PAGES_PER_SEARCH):
            console.print(f"Zoekpagina {page + 1} van {MAX_PAGES_PER_SEARCH}")
            fetched_any = False

            with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as bar:
                task = bar.add_task("Zoeken", total=len(search_terms))
                for term in search_terms:
                    bar.advance(task)
                    if is_term_search_done(term):
                        continue

                    entry = progress.get(term, {"next_page_token": None, "pages": 0})
                    kwargs = dict(
                        part="snippet",
                        q=term,
                        type="video",
                        order="date",
                        maxResults=MAX_RESULTS_PER_SEARCH_TERM,
                        publishedAfter=PUBLISHED_AFTER,
                        publishedBefore=PUBLISHED_BEFORE,
                        relevanceLanguage="nl",
                        regionCode="NL",
                    )
                    # Als er een next_page_token is, voeg die toe aan de request-parameters zodat we de volgende pagina ophalen.
                    if entry["next_page_token"]:
                        kwargs["pageToken"] = entry["next_page_token"]

                    try:
                        response = api_search(kwargs)
                    except HttpError as e:
                        reason = str(e).replace("\n", " ")
                        logger.error("search\t%s\t%s", term, reason)
                        # Markeer als klaar zodat we niet in een retry-lus blijven hangen
                        progress[term] = {
                            "next_page_token": None,
                            "pages": entry["pages"],
                        }
                        continue

                    # Schrijf de request en response weg
                    append_jsonl(search_file, kwargs, response)

                    # Update de progressie voor deze term
                    progress[term] = {
                        "next_page_token": response.get("nextPageToken"),
                        "pages": entry["pages"] + 1,
                    }
                    fetched_any = True

            if not fetched_any:
                console.print(
                    "Geen verdere pagina's beschikbaar voor enige zoekterm. Zoeken gestopt."
                )
                break


# Krijg details van de video's
def fetch_video_details(video_ids: set[str]) -> None:
    already = fetched_ids(VIDEOS_FILE)
    todo = sorted(vid for vid in video_ids if vid not in already)

    if not todo:
        return

    # Maak batches van video-IDs
    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    stored = 0

    with VIDEOS_FILE.open("a", encoding="utf-8") as videos_file:
        with Progress(*_PROGRESS_COLUMNS, console=console) as bar:
            task = bar.add_task("Video-details ophalen", total=len(batches))
            for batch in batches:
                kwargs = dict(
                    part="statistics,snippet,contentDetails,status,topicDetails,recordingDetails,localizations",
                    id=",".join(batch),
                    maxResults=BATCH_SIZE,
                )
                try:
                    response = api_videos(kwargs)
                except HttpError as e:
                    reason = str(e).replace("\n", " ")
                    for vid in batch:
                        logger.error("video\t%s\t%s", vid, reason)
                    bar.advance(task)
                    continue

                # Bewaar de hele ruwe response per batch (items dragen hun eigen id).
                append_jsonl(videos_file, kwargs, response)

                returned = {
                    item.get("id") for item in response.get("items", []) if item.get("id")
                }
                stored += len(returned)

                # IDs die de API niet teruggeeft (verwijderd/privé) markeren als mislukt
                for vid in set(batch) - returned:
                    logger.error("video\t%s\tniet_gevonden_in_api_response", vid)

                bar.advance(task)


# Krijg comment's van de video's
def fetch_comments(video_ids: set[str]) -> None:
    progress = page_progress(COMMENTS_FILE, "videoId")

    def video_done(vid: str) -> bool:
        entry = progress.get(vid)
        if entry is None:
            return False
        if entry["pages"] >= MAX_COMMENT_PAGES_PER_VIDEO:
            return True
        return entry["next_page_token"] is None

    todo = sorted(vid for vid in video_ids if not video_done(vid))

    with COMMENTS_FILE.open("a", encoding="utf-8") as comments_file:
        with Progress(*_PROGRESS_COLUMNS, console=console) as bar:
            task = bar.add_task("Comments ophalen", total=len(todo))
            for vid in todo:
                entry = progress.get(vid, {"next_page_token": None, "pages": 0})
                page_count = entry["pages"]
                next_page_token = entry["next_page_token"]

                try:
                    while page_count < MAX_COMMENT_PAGES_PER_VIDEO:
                        kwargs = dict(
                            part="snippet,replies",
                            videoId=vid,
                            maxResults=MAX_COMMENTS_PER_PAGE,
                            order="time",
                            textFormat="plainText",
                        )
                        # Voeg de pageToken alleen toe als we een volgende pagina ophalen
                        if next_page_token:
                            kwargs["pageToken"] = next_page_token

                        response = api_comment_page(kwargs)

                        append_jsonl(comments_file, kwargs, response)

                        next_page_token = response.get("nextPageToken")
                        page_count += 1

                        if not next_page_token:
                            break

                except HttpError as e:
                    pass

                bar.advance(task)


# Krijg de kanalen van de auteurs van de comments
def fetch_channels() -> None:
    already = fetched_ids(CHANNELS_FILE)

    channel_ids: set[str] = set()
    for record in read_jsonl(COMMENTS_FILE):
        for item in record.get("response", {}).get("items", []):
            top_snippet = (
                item.get(
                    "snippet,contentDetails,brandingSettings,topicDetails,status,localizations",
                    {},
                )
                .get("topLevelComment", {})
                .get("snippet", {})
            )
            cid = extract_author_channel_id(top_snippet)
            if cid:
                channel_ids.add(cid)
            for reply in item.get("replies", {}).get("comments", []):
                reply_cid = extract_author_channel_id(reply.get("snippet", {}))
                if reply_cid:
                    channel_ids.add(reply_cid)

    todo = sorted(cid for cid in channel_ids if cid not in already)

    if not todo:
        return

    # Maak batches van kanaal-IDs
    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    stored = 0

    with CHANNELS_FILE.open("a", encoding="utf-8") as channels_file:
        with Progress(*_PROGRESS_COLUMNS, console=console) as bar:
            task = bar.add_task("Kanalen ophalen", total=len(batches))
            for batch in batches:
                kwargs = dict(part="snippet", id=",".join(batch), maxResults=BATCH_SIZE)
                try:
                    response = api_channels(kwargs)
                except HttpError as e:
                    reason = str(e).replace("\n", " ")
                    for cid in batch:
                        logger.error("channel\t%s\t%s", cid, reason)
                    bar.advance(task)
                    continue

                # Bewaar de hele ruwe response per batch (items dragen hun eigen id).
                append_jsonl(channels_file, kwargs, response)

                returned = {
                    item.get("id") for item in response.get("items", []) if item.get("id")
                }
                stored += len(returned)

                for cid in set(batch) - returned:
                    logger.error("channel\t%s\tniet_gevonden_in_api_response", cid)

                bar.advance(task)


def main():
    fetch_search_results()
    video_ids = get_video_ids()
    fetch_video_details(video_ids)
    fetch_comments(video_ids)
    fetch_channels()

    console.print(f"Klaar. Responses opgeslagen in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
