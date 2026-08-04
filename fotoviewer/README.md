# Foto Viewer

Lokale semantische Fotosuche mit Objekterkennung. Alle Modelle (SigLIP2 für Text-/Bildsuche, RT-DETR für Objekterkennung) laufen vollständig lokal — es gibt keine Cloud-API-Anbindung. Modellgewichte werden beim ersten Start einmalig von Hugging Face heruntergeladen und danach lokal gecacht.

## Setup

```
uv sync
cp .env.example .env   # und bei Bedarf anpassen
```

`.env` wird automatisch beim Start geladen (siehe [fotoviewer/core/config.py](fotoviewer/core/config.py)) und ist gitignored — `.env.example` dient als Vorlage mit allen verfügbaren Werten.

## Starten

```
uv run uvicorn fotoviewer.main:app --reload
```

App läuft dann unter `http://127.0.0.1:8000`.

Die App bindet standardmäßig nur an `127.0.0.1` und hat **keine Authentifizierung** — das ist für ein lokales Ein-Personen-Tool eine bewusste Entscheidung. Vor dem Binden an `0.0.0.0` oder einer Freigabe im Netzwerk sollte ein Auth-Layer (z.B. ein Reverse Proxy mit Basic Auth) davorgesetzt werden.

## Seiten

| Route | Inhalt |
|---|---|
| `/` | Semantische Text-/Bildsuche |
| `/image?path=...` | Detailseite: Metadaten, Objekterkennung, Individuen-Tagging |
| `/folders` | Datei-Browser über die konfigurierte Fotobibliothek |
| `/sources` | Übersicht über Indizes, Modelle und Schwellwerte |
| `/individuals` | Verwaltung der Individuen-Galerie (umbenennen, löschen) |
| `/duplicates` | Anzeige möglicher Duplikat-Gruppen (nur Review, kein Löschen) |

## Modelle

Alle Modelle laufen lokal über [`transformers`](https://github.com/huggingface/transformers), es gibt keine weiteren Modelle im Projekt.

| Modell | Hugging-Face-ID (Default) | Wofür | Code |
|---|---|---|---|
| SigLIP2 | `google/siglip2-base-patch16-224` | Text-/Bild-Embeddings für die semantische Suche **und** Grundlage der Individuen-Wiedererkennung (Embedding des zugeschnittenen Bildausschnitts) | [fotoviewer/services/image_processor.py](fotoviewer/services/image_processor.py) |
| RT-DETR | `PekingU/rtdetr_r50vd` | Objekterkennung (91 COCO-Klassen) — sowohl für die Detailseiten-Erkennung eines gespeicherten Fotos als auch für den WebSocket-Upload-Flow | [fotoviewer/services/object_detector/object_detector.py](fotoviewer/services/object_detector/object_detector.py) |

Beide Modellnamen sind über `FOTOVIEWER_EMBEDDING_MODEL_NAME` / `FOTOVIEWER_DETECTION_MODEL_NAME` austauschbar (siehe Konfiguration unten). Die Individuen-Wiedererkennung nutzt kein eigenes Modell, sondern ausschließlich SigLIP2 auf zugeschnittenen Bildausschnitten (siehe [fotoviewer/services/identity.py](fotoviewer/services/identity.py) und die Einschränkungen unten).

## Konfiguration

Über Umgebungsvariablen mit Prefix `FOTOVIEWER_` (siehe [fotoviewer/core/config.py](fotoviewer/core/config.py)):

| Variable | Default | Bedeutung |
|---|---|---|
| `FOTOVIEWER_IMAGES_PATH` | – (Pflichtfeld) | Pfad zur Fotobibliothek |
| `FOTOVIEWER_INDEX_DIR` | `../indices` | Ablageort der Embedding-Indizes |
| `FOTOVIEWER_EMBEDDING_MODEL_NAME` | `google/siglip2-base-patch16-224` | Modell für Text-/Bildsuche |
| `FOTOVIEWER_DETECTION_MODEL_NAME` | `PekingU/rtdetr_r50vd` | Modell für Objekterkennung |
| `FOTOVIEWER_IDENTITY_MATCH_THRESHOLD` | `0.75` | Schwellwert für Individuen-Wiedererkennung (siehe Einschränkungen unten) |
| `FOTOVIEWER_DUPLICATE_MATCH_THRESHOLD` | `0.95` | Schwellwert für die Duplikat-Erkennung auf `/duplicates` |
| `FOTOVIEWER_DETECTION_THRESHOLD` | `0.7` | Mindest-Score, ab dem RT-DETR eine Erkennung meldet |

Zusätzlich `HF_HOME`, um den Hugging-Face-Modell-Cache auf ein Laufwerk mit ausreichend Speicherplatz zu legen (Standard ist `~/.cache/huggingface`; die Modelle summieren sich auf mehrere GB).

## Such-Index bauen

```
uv run fotoviewer-create-index
uv run fotoviewer-search <bildpfad>
```

## Tests

```
uv run pytest
```

## Bekannte Einschränkungen

### Individuen-Wiedererkennung ist unzuverlässig

Auf der Bild-Detailseite lässt sich eine erkannte Box mit einem Namen versehen (z.B. "Rex" statt nur "dog"). Künftige Erkennungen werden dann per Cosine-Similarity gegen die so aufgebaute Galerie von Referenz-Embeddings abgeglichen ([fotoviewer/services/identity.py](fotoviewer/services/identity.py)).

**Das funktioniert mechanisch, ist aber in der Praxis nicht zuverlässig genug für echten Einsatz.** Bei einem Test mit einem Referenzbild ("Rex") gegen ein komplett anderes Foto mit anderen Personen und Tieren matchten *alle* elf darauf erkannten Objekte "Rex" mit Scores zwischen 0.79 und 0.85 — nur knapp unter dem Score des tatsächlichen Selbst-Matches (0.9999) und deutlich über dem Standard-Schwellwert von 0.75.

**Ursache:** SigLIP2 (`FOTOVIEWER_EMBEDDING_MODEL_NAME`) ist ein Allzweck-Vision-Language-Embedding, trainiert auf grobe semantische Ähnlichkeit ("das ist eine Person", "das ist ein Pferd"), nicht auf feingranulare Identitäts-Unterscheidung zwischen zwei verschiedenen Personen oder Tieren. Generische Bounding-Box-Crops landen im Embedding-Raum näher beieinander, als für zuverlässige Wiedererkennung nötig wäre — Pose, Beleuchtung und Hintergrund beeinflussen das Embedding stärker als die Identität selbst.

**Umgesetzte Abmilderung:** Statt nur den einen ähnlichsten Referenz-Crop zu vergleichen, mittelt `identify_crop` ([fotoviewer/services/identity.py](fotoviewer/services/identity.py)) den Score über *alle* Referenzfotos eines Namens und vergleicht dann die Durchschnittswerte der Kandidaten-Individuen. Ein einzelner zufällig sehr ähnlicher Fehltreffer wird so von den übrigen, weniger ähnlichen Referenzfotos desselben Namens abgeschwächt — bei nur einer Referenz bleibt das Verhalten unverändert. Das mindert das oben beschriebene Problem, löst es aber nicht grundsätzlich: Bei durchgängig fehlklassifizierten Referenzfotos (z.B. wenn alle Referenzen eines falschen Namens zufällig ähnlich aussehen) bleibt die Fehlerkennung bestehen.

**Weitere mögliche Schritte** (bisher nicht umgesetzt):
- Schwellwert deutlich anheben (z.B. 0.95+) — riskiert dann False Negatives bei echten Wiedererkennungen in anderer Pose/Beleuchtung.
- Mindestanzahl an Referenzfotos pro Individuum erzwingen, bevor ein Name überhaupt als Match infrage kommt.
- Dedizierte Embeddings für Personen (Gesichtserkennung) ergänzen — SigLIP2 ist dafür nicht das richtige Werkzeug, würde aber eine zusätzliche Modell-Dependency bedeuten.

### Objekterkennung ist auf 91 COCO-Klassen begrenzt

RT-DETR erkennt nur die festen COCO-Kategorien (z.B. `person`, `dog`, `horse`). Objekte außerhalb dieser Liste (z.B. Kamele) werden der nächstliegenden bekannten Klasse zugeordnet statt korrekt benannt.
