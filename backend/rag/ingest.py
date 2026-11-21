"""Build the manuals vector store from knowledge/ (idempotent: rebuilds from scratch).

    python -m rag.ingest

* ``knowledge/pdfs/*.pdf``  -> PyPDFLoader, one document per page (page numbers kept)
* ``knowledge/*.md``        -> split on ``## `` headings (SOURCES.md / README.md skipped)

Display names for PDFs come from the table in knowledge/SOURCES.md (file -> title);
otherwise the file name is used.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from rag.store import make_embeddings, open_store

log = logging.getLogger("rag.ingest")
SKIP_MD = {"sources.md", "readme.md"}


def source_titles(knowledge_dir: Path) -> dict[str, str]:
    """Parse `| file.pdf | Title | ... |` rows from SOURCES.md."""
    path = knowledge_dir / "SOURCES.md"
    titles: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0].lower().endswith(".pdf"):
                titles[cells[0]] = cells[1]
    return titles


def load_pdfs(knowledge_dir: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    titles = source_titles(knowledge_dir)
    docs: list[Document] = []
    for pdf in sorted((knowledge_dir / "pdfs").glob("*.pdf")):
        name = titles.get(pdf.name, pdf.stem.replace("_", " ").replace("-", " "))
        for page in PyPDFLoader(str(pdf)).load():
            text = " ".join(page.page_content.split())
            if len(text) < 40:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_name": name,
                        "file": pdf.name,
                        "page": int(page.metadata.get("page", 0)) + 1,
                    },
                )
            )
        log.info("Loaded %s", pdf.name)
    return docs


def load_markdown(knowledge_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for md in sorted(knowledge_dir.glob("*.md")):
        if md.name.lower() in SKIP_MD:
            continue
        text = md.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else md.stem
        for section in re.split(r"(?m)^##\s+", text)[1:]:
            heading, _, body = section.partition("\n")
            body = body.strip()
            if body:
                docs.append(
                    Document(
                        page_content=f"{heading.strip()}: {body}",
                        metadata={"source_name": f"{title}: {heading.strip()}", "file": md.name, "page": -1},
                    )
                )
        log.info("Loaded %s", md.name)
    return docs


def ingest() -> int:
    settings = get_settings()
    docs = load_pdfs(settings.knowledge_dir) + load_markdown(settings.knowledge_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    if settings.vectorstore_dir.exists():
        shutil.rmtree(settings.vectorstore_dir)
    store = open_store(
        settings.vectorstore_dir, settings.collection_name, make_embeddings(settings.embedding_model)
    )
    if chunks:
        store.add_documents(chunks)
    log.info("Indexed %d chunks from %d documents into %s", len(chunks), len(docs), settings.vectorstore_dir)
    return len(chunks)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    n = ingest()
    print(f"Indexed {n} chunks")


if __name__ == "__main__":
    main()
