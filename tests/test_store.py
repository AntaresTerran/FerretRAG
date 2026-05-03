from __future__ import annotations

from pathlib import Path

from ferret_rag.index.store import LocalIndex, hash_file


def test_hash_file_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("first", encoding="utf-8")
    first_hash = hash_file(path)

    path.write_text("second", encoding="utf-8")

    assert hash_file(path) != first_hash


def test_index_folder_skips_unchanged_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("ferret search local files", encoding="utf-8")
    index = LocalIndex(tmp_path / "data", chunk_words=5, overlap=1)

    first = index.index_folder(docs)
    second = index.index_folder(docs)

    assert first["files_indexed"] == 1
    assert second["files_skipped"] == 1
    assert index.search("local files", top_k=1)


def test_index_folder_records_chunk_metadata(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "notes.md"
    path.write_text("ferret search local files", encoding="utf-8")
    index = LocalIndex(tmp_path / "data", chunk_words=5, overlap=1)

    index.index_folder(docs)
    chunk = index.sources()[0]

    assert chunk.file_name == "notes.md"
    assert chunk.file_type == "md"
    assert chunk.modified_time > 0


def test_index_folder_reports_failed_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    bad_xlsx = docs / "broken.xlsx"
    bad_xlsx.write_text("not a real workbook", encoding="utf-8")
    index = LocalIndex(tmp_path / "data", chunk_words=5, overlap=1)

    result = index.index_folder(docs)

    assert result["files_failed"] == 1
    assert result["failures"]
    assert result["failures"][0]["file_path"].endswith("broken.xlsx")


def test_indexed_roots_files_and_remove_path(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "notes.txt"
    path.write_text("ferret search local files", encoding="utf-8")
    index = LocalIndex(tmp_path / "data", chunk_words=5, overlap=1)

    index.index_folder(docs)
    roots = index.indexed_roots()
    files = index.indexed_files()
    removed = index.remove_path(path)

    assert roots[0]["active_file_count"] == 1
    assert files[0].file_name == "notes.txt"
    assert removed["files_removed"] == 1
    assert index.sources() == []


def test_lexical_search_uses_file_name_metadata(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "taxes.md").write_text("deductible expenses and invoices", encoding="utf-8")
    index = LocalIndex(tmp_path / "data", chunk_words=20, overlap=1)

    index.index_folder(docs)
    results = index.search("taxes", top_k=1)

    assert results
    assert results[0].chunk.file_name == "taxes.md"
