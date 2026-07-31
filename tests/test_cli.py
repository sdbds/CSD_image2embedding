import subprocess
import sys

import pytest

from csd_image2embedding.cli import parse_args


def test_cli_defaults_to_csd_image_only():
    args = parse_args([])

    assert args.backend == "csd"
    assert args.text_mode == "image-only"


def test_legacy_sd_option_maps_to_siglip_backend():
    args = parse_args(["--model_type", "sd"])

    assert args.backend == "siglip-dinov3"


def test_conflicting_new_and_legacy_backend_options_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--backend", "csd", "--model_type", "sd"])


def test_removed_raw_embedding_path_fails_with_a_migration_hint(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--embeddings-path", "old.lance"])

    assert "--artifact-root" in capsys.readouterr().err


def test_removed_worker_pool_fails_instead_of_being_silently_ignored(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--num-workers", "2"])

    assert "deterministic Lance reader" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--batch-size=0", "--k-clusters=-1"])
def test_positive_numeric_options_are_validated(option):
    with pytest.raises(SystemExit):
        parse_args([option])


def test_module_help_does_not_import_dash_or_models():
    result = subprocess.run(
        [sys.executable, "-m", "csd_image2embedding", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--backend" in result.stdout
    assert "--text-mode" in result.stdout

    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import csd_image2embedding.cli; "
                "assert not ({'torch', 'dash', 'lance'} & set(sys.modules)); "
                "assert not any(name.startswith('csd_image2embedding.models') "
                "for name in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_check.returncode == 0, import_check.stderr
