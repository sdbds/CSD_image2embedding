from csd_image2embedding.data.discovery import discover_directory


def test_caption_change_only_changes_caption_digest(tmp_path):
    image = tmp_path / "nested" / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fake-image")
    sidecar = image.with_suffix(".txt")
    sidecar.write_text("a lake", encoding="utf-8")

    first = discover_directory(tmp_path)
    sidecar.write_text("a mountain", encoding="utf-8")
    second = discover_directory(tmp_path)

    assert first.image_digest == second.image_digest
    assert first.caption_digest != second.caption_digest


def test_image_change_only_changes_image_digest(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"first-image")
    image.with_suffix(".txt").write_text("a lake", encoding="utf-8")

    first = discover_directory(tmp_path)
    image.write_bytes(b"second-image")
    second = discover_directory(tmp_path)

    assert first.image_digest != second.image_digest
    assert first.caption_digest == second.caption_digest


def test_discovery_reports_missing_empty_and_invalid_sidecars(tmp_path):
    for name in ("valid.jpg", "missing.jpg", "empty.jpg", "invalid.jpg"):
        (tmp_path / name).write_bytes(name.encode("ascii"))
    (tmp_path / "valid.txt").write_text("a lake", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  ", encoding="utf-8")
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xfe\x00")

    snapshot = discover_directory(tmp_path)

    assert [record.relative_path for record in snapshot.records] == [
        "empty.jpg",
        "invalid.jpg",
        "missing.jpg",
        "valid.jpg",
    ]
    assert snapshot.caption_counts == {
        "valid": 1,
        "missing": 1,
        "empty": 1,
        "unreadable": 1,
    }


def test_discovery_decodes_utf8_bom_and_gb18030_captions(tmp_path):
    (tmp_path / "bom.PNG").write_bytes(b"bom-image")
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbfa prompt\n")
    (tmp_path / "chinese.JPEG").write_bytes(b"chinese-image")
    (tmp_path / "chinese.txt").write_bytes("一条普通描述".encode("gb18030"))

    snapshot = discover_directory(tmp_path)

    assert [record.caption for record in snapshot.records] == [
        "a prompt",
        "一条普通描述",
    ]
    assert snapshot.caption_counts["valid"] == 2
