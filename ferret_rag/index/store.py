from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ferret_rag.index.chunking import chunk_text
from ferret_rag.index.loaders import SUPPORTED_EXTENSIONS, discover_files, load_document


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


@dataclass(frozen=True)
class IndexedFile:
    file_path: str
    file_name: str
    file_type: str
    chunk_count: int
    modified_time: float
    root_path: str | None = None


class LocalIndex:
    def __init__(self, data_dir: Path, chunk_words: int = 300, overlap: int = 50) -> None:
        self.data_dir = data_dir
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.index_dir = data_dir / "vectors" / "local"
        self.index_file = self.index_dir / "chunks.json"
        self.hash_file = self.index_dir / "hashes.json"
        self.roots_file = self.index_dir / "roots.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks = self._load_chunks()
        self._hashes = self._load_hashes()
        self._roots = self._load_roots()
        self._chroma = self._connect_chroma()

    def index_folder(self, root: Path) -> dict[str, object]:
        root = root.resolve()
        files = discover_files(root)
        indexed = 0
        skipped = 0
        failed = 0
        failures: list[FileFailure] = []

        for path in files:
            result = self._index_file(path)
            if result == "indexed":
                indexed += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(result)

        self._roots[str(root)] = {
            "path": str(root),
            "file_count": len(files),
            "last_result": {
                "files_found": len(files),
                "files_indexed": indexed,
                "files_skipped": skipped,
                "files_failed": failed,
            },
        }
        self._save()
        return {
            "files_found": len(files),
            "files_indexed": indexed,
            "files_skipped": skipped,
            "files_failed": failed,
            "chunks_total": len(self._chunks),
            "failures": [asdict(failure) for failure in failures],
        }

    def index_path(self, path: Path) -> dict[str, object]:
        target = path.resolve()
        if target.is_dir():
            return self.index_folder(target)
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")
        if not target.is_file():
            raise ValueError(f"Path is not a supported file or folder: {target}")
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {target.suffix or target.name}")

        result = self._index_file(target)
        failures = [asdict(result)] if isinstance(result, FileFailure) else []
        self._save()
        return {
            "files_found": 1,
            "files_indexed": 1 if result == "indexed" else 0,
            "files_skipped": 1 if result == "skipped" else 0,
            "files_failed": 1 if failures else 0,
            "chunks_total": len(self._chunks),
            "failures": failures,
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        chroma_results = self._search_chroma(query, top_k)
        lexical_results = self._search_lexical(query, top_k=top_k * 2)
        if chroma_results is not None:
            return _merge_results(chroma_results, lexical_results, top_k)
        return lexical_results[:top_k]

    def _search_lexical(self, query: str, top_k: int) -> list[SearchResult]:
        query_terms = _terms(query)
        if not query_terms:
            return []

        results: list[SearchResult] = []
        for chunk in self._chunks:
            searchable_text = " ".join(
                [
                    chunk.text,
                    chunk.file_name,
                    chunk.file_type,
                    Path(chunk.file_path).parent.name,
                    chunk.label or "",
                ]
            )
            chunk_terms = _terms(searchable_text)
            if not chunk_terms:
                continue
            overlap = query_terms & chunk_terms
            if not overlap:
                continue
            exact_bonus = 0.25 if query.lower() in searchable_text.lower() else 0
            title_bonus = 0.15 * len(query_terms & _terms(chunk.file_name))
            score = (len(overlap) / max(len(query_terms), 1)) + exact_bonus + title_bonus
            results.append(SearchResult(chunk=chunk, score=score))

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def sources(self) -> list[SourceChunk]:
        return list(self._chunks)

    def indexed_roots(self) -> list[dict[str, object]]:
        roots = []
        for root in self._roots.values():
            path = str(root["path"])
            root_chunks = [
                chunk for chunk in self._chunks if _is_relative_to(chunk.file_path, path)
            ]
            roots.append(
                {
                    **root,
                    "active_file_count": len({chunk.file_path for chunk in root_chunks}),
                    "chunk_count": len(root_chunks),
                }
            )
        return sorted(roots, key=lambda item: str(item["path"]).lower())

    def indexed_files(self) -> list[IndexedFile]:
        files: dict[str, list[SourceChunk]] = {}
        for chunk in self._chunks:
            files.setdefault(chunk.file_path, []).append(chunk)

        indexed: list[IndexedFile] = []
        for file_path, chunks in files.items():
            first = chunks[0]
            indexed.append(
                IndexedFile(
                    file_path=file_path,
                    file_name=first.file_name,
                    file_type=first.file_type,
                    chunk_count=len(chunks),
                    modified_time=first.modified_time,
                    root_path=_root_for_file(file_path, self._roots.keys()),
                )
            )
        return sorted(indexed, key=lambda item: item.file_path.lower())

    def remove_path(self, path: Path) -> dict[str, int]:
        target = str(path.resolve())
        removed_files = {
            chunk.file_path
            for chunk in self._chunks
            if chunk.file_path == target or _is_relative_to(chunk.file_path, target)
        }
        removed_chunks = len([chunk for chunk in self._chunks if chunk.file_path in removed_files])
        self._chunks = [chunk for chunk in self._chunks if chunk.file_path not in removed_files]
        for file_path in removed_files:
            self._hashes.pop(file_path, None)
            self._delete_chroma_file(file_path)

        removed_roots = 0
        if target in self._roots:
            self._roots.pop(target, None)
            removed_roots = 1

        self._save()
        return {
            "files_removed": len(removed_files),
            "chunks_removed": removed_chunks,
            "roots_removed": removed_roots,
            "chunks_total": len(self._chunks),
        }

    def _index_file(self, path: Path) -> str | FileFailure:
        digest = hash_file(path)
        key = str(path.resolve())
        if self._hashes.get(key) == digest:
            return "skipped"

        try:
            sections = load_document(path)
        except Exception as exc:
            return FileFailure(file_path=key, error=str(exc))

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
        return "indexed"

    def _load_chunks(self) -> list[SourceChunk]:
        if not self.index_file.exists():
            return []
        raw = json.loads(self.index_file.read_text(encoding="utf-8"))
        return [_source_chunk_from_dict(item) for item in raw]

    def _load_hashes(self) -> dict[str, str]:
        if not self.hash_file.exists():
            return {}
        return json.loads(self.hash_file.read_text(encoding="utf-8"))

    def _load_roots(self) -> dict[str, dict[str, object]]:
        if not self.roots_file.exists():
            return {}
        return json.loads(self.roots_file.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.index_file.write_text(
            json.dumps([asdict(chunk) for chunk in self._chunks], indent=2),
            encoding="utf-8",
        )
        self.hash_file.write_text(json.dumps(self._hashes, indent=2), encoding="utf-8")
        self.roots_file.write_text(json.dumps(self._roots, indent=2), encoding="utf-8")

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


def _merge_results(
    chroma_results: list[SearchResult],
    lexical_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    merged: dict[str, SearchResult] = {}
    for result in chroma_results:
        merged[result.chunk.id] = result
    for result in lexical_results:
        existing = merged.get(result.chunk.id)
        if existing is None or result.score > existing.score:
            merged[result.chunk.id] = result
    return sorted(merged.values(), key=lambda result: result.score, reverse=True)[:top_k]


def _is_relative_to(file_path: str, root_path: str) -> bool:
    try:
        Path(file_path).resolve().relative_to(Path(root_path).resolve())
    except ValueError:
        return False
    return True


def _root_for_file(file_path: str, roots: object) -> str | None:
    matching_roots = [root for root in roots if _is_relative_to(file_path, str(root))]
    if not matching_roots:
        return None
    return max(matching_roots, key=len)
