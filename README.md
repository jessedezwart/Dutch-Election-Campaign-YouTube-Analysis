## Installatie

Vereist Python 3.13.

Met `uv` (aanbevolen):

```bash
uv sync
```

Of met pip:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Maak een `.env`-bestand aan in de projectmap met:

```
YOUTUBE_API_KEY=jouw-youtube-data-api-v3-sleutel
HASH_SALT=een-willekeurige-geheime-string
```

- `YOUTUBE_API_KEY` is een [YouTube Data API v3](https://console.cloud.google.com/)-sleutel, gebruikt bij het ophalen van data.
- `HASH_SALT` is een geheime waarde waarmee `02_anonymize.py` de SHA-256-hashes van video-ID's en kanaal-ID's salt. Houd deze waarde geheim en gebruik steeds dezelfde waarde als je wil dat hashes stabiel blijven tussen runs.

De zoektermen (partijen, politici, thema's) staan in `trefwoordenlijst.txt`, één term per regel. Regels die beginnen met `#` worden genegeerd.

## 1. Ophalen

```bash
python 01_retrieve.py
```

Voor elke term in `trefwoordenlijst.txt` worden vier YouTube Data API-methoden achtereenvolgens aangeroepen. De **ruwe responses** (zonder veldtrimming) worden opgeslagen in `dataset_unprocessed/`:

1. `search.list` → `dataset_unprocessed/search.list.jsonl` (één paginaresponse per regel, getagd met zoekterm)
2. `videos.list` → `dataset_unprocessed/videos.list.jsonl` (statistieken en taal; één batchresponse per regel)
3. `commentThreads.list` → `dataset_unprocessed/commentThreads.list.jsonl` (één paginaresponse per regel, getagd met video-ID)
4. `channels.list` → `dataset_unprocessed/channels.list.jsonl` (kanaalinformatie van reageerders; één batchresponse per regel)

- Zoekvenster, regio/taal en maximaal aantal resultaten worden bepaald door de constanten bovenaan het script (`PUBLISHED_AFTER`, `PUBLISHED_BEFORE`, `MAX_RESULTS_PER_SEARCH_TERM`, `MAX_PAGES_PER_SEARCH`).
- Het script is **hervat­baar**: bij een herstart gaat het verder waar het gebleven was.
- Fouten worden gelogd naar `logs/error_<tijdstempel>.log`.

De YouTube Data API heeft een daglimiet (`search.list` kost 100 eenheden per aanroep, de overige methoden 1 eenheid). Bij grote trefwoordenlijsten kan het ophalen meerdere dagen duren. Start het script opnieuw na elk quotumreset.

Bij de standaard instellingen met de standaard termen, kan het ophalen van de dataset in één run worden voltooid.

## 2. Anonimiseren

```bash
python 02_anonymize.py
```

- Leest de ruwe dumps in `dataset_unprocessed/` en schrijft gecleande, gede-identificeerde versies naar `dataset_anonymized/` (zelfde vier bestandsnamen, zelfde JSONL-structuur).
- Iedere video-ID en kanaal-ID van reageerders wordt gesalt en SHA-256-gehasht, consistent over alle bestanden zodat joins downstream blijven werken.
- Alleen de velden die het verwerkingsscript nodig heeft worden bewaard; gebruikersnamen en overige velden worden verwijderd.

Mocht je extra velden willen behouden, pas dan `02_anonymize.py` aan.

## 3. Verwerken

```bash
python 03_process.py
```

- Leest `dataset_anonymized/`, koppelt de search/video/comment/channel-dumps, filtert niet-Nederlandstalige video's weg, haalt weergave-/like-/commentaantallen op, berekent de accountleeftijd van reageerders en dedupliceert comments.
- Schrijft `dataset_processed/videos.jsonl` en `dataset_processed/comments.jsonl`. Er wordt hier niet gehasht — het script werkt uitsluitend met al geanonimiseerde ID's.

## 4. Analysen uitvoeren

Voor de sentimentanalyse kan mogelijk gebruik worden gemaakt van de videokaart om het proces te versnellen. Mogelijk moet je dan PyTorch met CUDA installeren. Zie [PyTorch Get Started](https://pytorch.org/get-started/locally/) voor instructies.

```bash
python 04_analyze.py            # publicatiefrequentie + weergaven/likes/reacties per week, partij, thema
python 05_sentiment_analysis.py # sentimentanalyse van reacties via cardiffnlp/twitter-xlm-roberta-base-sentiment
python 06_sentiment_results.py  # tabellen en figuren op basis van de sentimentoutput
```

Outputs:

- `results/tabellen/*.csv` en `results/images/*.png`
- `sentiment/comments_sentiment.jsonl` — elke reactie aangevuld met `sentiment_label` en `sentiment_score`
