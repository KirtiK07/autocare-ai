# Knowledge sources

All PDFs are works of the US federal government (US Army / War Department technical
manuals). Under 17 U.S.C. § 105 they are in the public domain, so they can be indexed and
redistributed. The **Title** column is what citations show, e.g. `[TM 9-8000 Principles of Automotive Vehicles, page 120]`.
Page numbers are PDF page numbers (what a PDF viewer shows), not the manual's printed page labels.

To add a document: put the PDF in `knowledge/pdfs/`, add a row, run `python -m rag.ingest`.

| File | Title | URL | Licence / terms |
| --- | --- | --- | --- |
| `tm-9-8000-principles-of-automotive-vehicles.pdf` | TM 9-8000 Principles of Automotive Vehicles | https://everyspec.com/ARMY/TM-Tech-Manual/TM_9-8000_CHG-1_48512/ | US Army, 1985 (Change 1, 1988). US government work, public domain |
| `tm-9-2858-cooling-systems.pdf` | TM 9-2858 Cooling Systems | https://www.govinfo.gov/app/details/GOVPUB-W-2ef8479817998f58cace10cd00a5f70a | War Department, 1945. US government work, public domain (via GPO govinfo) |
| `tm-10-570-internal-combustion-engine.pdf` | TM 10-570 The Internal Combustion Engine | https://www.govinfo.gov/app/details/GOVPUB-W-4251326e95fd1360d74b0778679fed0d | War Department, 1941. US government work, public domain (via GPO govinfo) |

Markdown sources in `knowledge/` (indexed automatically, one section per `##` heading):

| File | Title | Author | Licence |
| --- | --- | --- | --- |
| `sensor_glossary.md` | AutoCare sensor glossary | project author | same as repository |

Note: these manuals describe military and older vehicles. The principles (lubrication, cooling,
fuel systems, gauges, troubleshooting) carry over, but a specific vehicle's owner's manual always
takes precedence.
