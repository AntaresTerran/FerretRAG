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
