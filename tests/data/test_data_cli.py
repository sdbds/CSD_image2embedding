import lance
import pytest
from PIL import Image

from csd_image2embedding.data.cli import main


def test_data_cli_writes_a_lance_snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (7, 5), "red").save(source / "sample.png")
    (source / "sample.txt").write_text("plain prompt", encoding="utf-8")
    output = tmp_path / "snapshot.lance"

    assert main([str(source), "--output", str(output)]) == 0

    table = lance.dataset(output).to_table()
    assert table.num_rows == 1
    assert table["relative_path"].to_pylist() == ["sample.png"]
    assert table["captions"].to_pylist() == ["plain prompt"]
    assert (output / "_source_manifest.json").is_file()


def test_data_cli_refuses_to_replace_an_existing_output(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (7, 5), "red").save(source / "sample.png")
    output = tmp_path / "snapshot.lance"
    output.mkdir()

    with pytest.raises(SystemExit):
        main([str(source), "--output", str(output)])

    assert "already exists" in capsys.readouterr().err
