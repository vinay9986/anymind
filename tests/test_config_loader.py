import json
from pathlib import Path

import pytest

from anymind.config import loader


def test_find_config_uses_env(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps({"model": "gpt-4.1"}))
    monkeypatch.setenv(loader.MODEL_CONFIG_ENV, str(config_path))
    found = loader._find_config(loader.MODEL_CONFIG_ENV, [])
    assert found == config_path


def test_find_config_env_missing_raises(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setenv(loader.MODEL_CONFIG_ENV, str(missing))
    with pytest.raises(FileNotFoundError):
        loader._find_config(loader.MODEL_CONFIG_ENV, [])


def test_find_config_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(loader.MODEL_CONFIG_ENV, raising=False)
    candidate = tmp_path / "pricing.json"
    candidate.write_text(json.dumps({"currency": "USD"}))
    found = loader._find_config(loader.PRICING_CONFIG_ENV, [candidate])
    assert found == candidate


def test_load_model_config_from_path(tmp_path) -> None:
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps({"model": "gpt-4.1"}))
    model = loader.load_model_config_from_path(config_path)
    assert model.model == "gpt-4.1"


def test_load_mcp_config_non_dict(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps([1, 2, 3]))
    monkeypatch.setenv(loader.MCP_CONFIG_ENV, str(config_path))
    with pytest.raises(ValueError):
        loader.load_mcp_config()
