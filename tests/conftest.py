def pytest_addoption(parser):
    parser.addoption(
        "--run-model-smoke",
        action="store_true",
        default=False,
        help="run tests that load the local SigLIP2-DINOv3 weights",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "model_smoke: loads large local model assets and is opt-in",
    )
