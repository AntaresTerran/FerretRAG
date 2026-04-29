from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ferret_rag.index.chunking import chunk_text
from ferret_rag.index.loaders import discover_files, load_document


@dataclass(frozen=True)
class SourceChunk:
    id: str
    file_path: str
    file_name: str
    file_type: str
    chunk_id: int
    text: str
    file_hash: str
    modified_time: float
    page_num: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: SourceChunk
    score: float


@dataclass(frozen=True)
class FileFailure:
    file_path: str
    error: str


class LocalIndex:
    def __init__(self, data_dir: Path, chunk_words: int = 300, overlap: int = 50) -> None:
        self.data_dir = data_dir
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.index_dir = data_dir / "vectors" / "local"
        self.index_file = self.index_dir / "chunks.json"
        self.hash_file = self.index_dir / "hashes.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks = self._load_chunks()
        self._hashes = self._load_hashes()
        self._chroma = self._connect_chroma()

    def index_folder(self, root: Path) -> dict[str, object]:
        files = discover_files(root)
        indexed = 0
        skipped = 0
        failed = 0
        failures: list[FileFailure] = []

        for path in files:
            digest = hash_file(path)
            key = str(path.resolve())
            if self._hashes.get(key) == digest:
                skipped += 1
                continue

            try:
                sections = load_document(path)
            except Exception as exc:
                failed += 1
                failures.append(FileFailure(file_path=key, error=str(exc)))
                continue

            self._chunks = [chunk for chunk in self._chunks if chunk.file_path != key]
            self._delete_chroma_file(key)
            chunk_id = 0
            modified_time = path.stat().st_mtime
            for section in sections:
                chunks = chunk_text(section.text, self.chunk_words, self.overlap)
                for chunk in chunks:
                    source_chunk = SourceChunk(
                        id=hashlib.sha256(f"{key}:{chunk_id}:{digest}".encode()).hexdigest(),
                        file_path=key,
                        file_name=path.name,
                        file_type=path.suffix.lower().lstrip("."),
                        chunk_id=chunk_id,
                        text=chunk,
                        file_hash=digest,
                        modified_time=modified_time,
                        page_num=section.page_num,
                        label=section.label,
                    )
                    self._chunks.append(source_chunk)
                    self._upsert_chroma(source_chunk)
                    chunk_id += 1
            self._hashes[key] = digest
            indexed += 1

        self._save()
        return {
            "files_found": len(files),
            "files_indexed": indexed,
            "files_skipped": skipped,
            "files_failed": failed,
            "chunks_total": len(self._chunks),
            "failures": [asdict(failure) for failure in failures],
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        chroma_results = self._search_chroma(query, top_k)
        if chroma_results is not None:
            return chroma_results

        query_terms = _terms(query)
        if not query_terms:
            return []

        results: list[SearchResult] = []
        for chunk in self._chunks:
            chunk_terms = _terms(chunk.text)
            if not chunk_terms:
                continue
            overlap = query_terms & chunk_terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query_terms), 1)
            results.append(SearchResult(chunk=chunk, score=score))

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def sources(self) -> list[SourceChunk]:
        return list(self._chunks)

    def _load_chunks(self) -> list[SourceChunk]:
        if not self.index_file.exists():
            return []
        raw = json.loads(self.index_file.read_text(encoding="utf-8"))
        return [_source_chunk_from_dict(item) for item in raw]

    def _load_hashes(self) -> dict[str, str]:
        if not self.hash_file.exists():
            return {}
        return json.loads(self.hash_file.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.index_file.write_text(
            json.dumps([asdict(chunk) for chunk in self._chunks], indent=2),
            encoding="utf-8",
        )
        self.hash_file.write_text(json.dumps(self._hashes, indent=2), encoding="utf-8")

    def _connect_chroma(self):
        try:
            import chromadb
        except ImportError:
            return None

        client = chromadb.PersistentClient(path=str(self.data_dir / "vectors" / "chroma"))
        return client.get_or_create_collection(name="ferret_rag")

    def _delete_chroma_file(self, file_path: str) -> None:
        if self._chroma is None:
            return
        try:
            self._chroma.delete(where={"file_path": file_path})
        except Exception:
            return

    def _upsert_chroma(self, chunk: SourceChunk) -> None:
        if self._chroma is None:
            return
        try:
            self._chroma.upsert(
                ids=[chunk.id],
                documents=[chunk.text],
                embeddings=[_embedding(chunk.text)],
                metadatas=[
                    {
                        "file_path": chunk.file_path,
                        "file_name": chunk.file_name,
                        "file_type": chunk.file_type,
                        "chunk_id": chunk.chunk_id,
                        "file_hash": chunk.file_hash,
                        "modified_time": chunk.modified_time,
                        "page_num": chunk.page_num if chunk.page_num is not None else -1,
                        "label": chunk.label or "",
                    }
                ],
            )
        except Exception:
            self._chroma = None

    def _search_chroma(self, query: str, top_k: int) -> list[SearchResult] | None:
        if self._chroma is None:
            return None
        try:
            raw = self._chroma.query(query_embeddings=[_embedding(query)], n_results=top_k)
        except Exception:
            self._chroma = None
            return None

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        results: list[SearchResult] = []
        rows = zip(ids, documents, metadatas, distances, strict=False)
        for item_id, document, metadata, distance in rows:
            chunk = SourceChunk(
                id=item_id,
                file_path=str(metadata["file_path"]),
                file_name=str(metadata.get("file_name") or Path(str(metadata["file_path"])).name),
                file_type=str(metadata.get("file_type") or Path(str(metadata["file_path"])).suffix),
                chunk_id=int(metadata["chunk_id"]),
                text=str(document),
                file_hash=str(metadata["file_hash"]),
                modified_time=float(metadata.get("modified_time") or 0),
                page_num=_metadata_page_num(metadata.get("page_num")),
                label=str(metadata.get("label") or "") or None,
            )
            results.append(SearchResult(chunk=chunk, score=1 / (1 + float(distance))))
        return results


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _terms(text: str) -> set[str]:
    return {term.strip(".,:;!?()[]{}\"'").lower() for term in text.split() if len(term) > 2}


def _embedding(text: str, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    for term in _terms(text):
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0

    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _source_chunk_from_dict(item: dict[str, object]) -> SourceChunk:
    file_path = str(item["file_path"])
    path = Path(file_path)
    return SourceChunk(
        id=str(item["id"]),
        file_path=file_path,
        file_name=str(item.get("file_name") or path.name),
        file_type=str(item.get("file_type") or path.suffix.lower().lstrip(".")),
        chunk_id=int(item["chunk_id"]),
        text=str(item["text"]),
        file_hash=str(item["file_hash"]),
        modified_time=float(item.get("modified_time") or 0),
        page_num=_metadata_page_num(item.get("page_num")),
        label=str(item.get("label") or "") or None,
    )


def _metadata_page_num(value: object) -> int | None:
    if value in (None, "", -1):
        return None
    page_num = int(value)
    return page_num if page_num > 0 else None
