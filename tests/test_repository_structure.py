from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_root_contains_no_python_modules():
    assert list(ROOT.glob("*.py")) == []


def test_step2_uses_the_package_entry_and_public_options():
    script = (ROOT / "Step2_embedding.ps1").read_text(encoding="utf-8")

    assert "-m csd_image2embedding" in script
    assert "--backend" in script
    assert "--text-mode" in script
    assert "--model_type" not in script
    assert "--sd_config" not in script


def test_transform_script_uses_the_package_data_entry():
    script = (ROOT / "transform_to_lance.ps1").read_text(encoding="utf-8")

    assert "-m csd_image2embedding.data" in script
    assert "lancedatasets.py" not in script
