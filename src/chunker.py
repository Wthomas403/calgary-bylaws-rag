"""Split Calgary bylaw PDFs into section-level chunks with metadata.

Calgary bylaws are structured as numbered sections and subsections
(e.g. "1.", "1.1", "2.3.4"). We extract text page by page with pypdf,
then split on lines that look like a section header, carrying forward
the bylaw name, section number, and starting page for each chunk.
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdf

# Matches a leading section number like "1.", "1.1", "2.3.4" followed by
# a space and the start of the section heading/body text.
SECTION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(?=\S)(.*)$")


@dataclass
class Chunk:
    bylaw_name: str
    section: str | None
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the extracted text of each page in the PDF, in order."""
    reader = pypdf.PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def chunk_pdf(pdf_path: Path) -> list[Chunk]:
    """Chunk a single bylaw PDF at section boundaries."""
    pages = extract_pages(pdf_path)
    return chunk_pages(pdf_path.stem, pages)


def chunk_pages(bylaw_name: str, pages: list[str]) -> list[Chunk]:
    """Chunk already-extracted page text at section boundaries."""
    chunks: list[Chunk] = []
    current_section: str | None = None
    current_page = 1
    current_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(Chunk(bylaw_name, current_section, current_page, text))

    for page_num, page_text in enumerate(pages, start=1):
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = SECTION_PATTERN.match(line)
            if match:
                flush()
                current_section = match.group(1)
                current_page = page_num
                current_lines = [line]
            else:
                if not current_lines:
                    current_page = page_num
                current_lines.append(line)

    flush()
    return chunks


def chunk_all(bylaws_dir: Path) -> list[Chunk]:
    """Chunk every PDF found in ``bylaws_dir``."""
    all_chunks: list[Chunk] = []
    for pdf_path in sorted(bylaws_dir.glob("*.pdf")):
        all_chunks.extend(chunk_pdf(pdf_path))
    return all_chunks
