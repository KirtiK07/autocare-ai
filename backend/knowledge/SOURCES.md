# Knowledge sources

Put 3–5 freely downloadable PDFs in `knowledge/pdfs/`, add one row each below, then run
`python -m rag.ingest`. The **Title** column is what citations show, e.g. `[Title, page 12]`.

Only use documents whose licence or terms allow redistribution/indexing, and record it.

| File | Title | URL | Licence / terms |
| --- | --- | --- | --- |
| `example-owners-manual.pdf` | Example Owner's Manual (warning lights, maintenance) | https://… | … |

Markdown sources in `knowledge/` (indexed automatically, one section per `##` heading):

| File | Title | Author | Licence |
| --- | --- | --- | --- |
| `sensor_glossary.md` | AutoCare sensor glossary | project author | same as repository |
