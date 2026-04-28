from __future__ import annotations

from pathlib import Path

from ferret_rag.index.loaders import discover_files, load_document


def test_discover_files_ignores_unsupported_and_hidden(tmp_path: Path) -> None:
    visible = tmp_path / "notes.md"
    visible.write_text("# Ferret\nLocal search.", encoding="utf-8")
    (tmp_path / "image.png").write_text("not supported", encoding="utf-8")
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.md").write_text("hidden", encoding="utf-8")

    assert discover_files(tmp_path) == [visible]


def test_load_document_reads_markdown(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Ferret\nLocal search.", encoding="utf-8")

    assert "Local search" in load_document(path)
