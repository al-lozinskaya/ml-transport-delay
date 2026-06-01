from src import config


def test_ensure_runtime_dirs_creates_artifact_directories(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    models_dir = tmp_path / "models"
    mlflow_dir = tmp_path / "mlruns"

    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(config, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "MLFLOW_DIR", mlflow_dir)

    config.ensure_runtime_dirs()

    assert reports_dir.is_dir()
    assert figures_dir.is_dir()
    assert models_dir.is_dir()
    assert mlflow_dir.is_dir()
