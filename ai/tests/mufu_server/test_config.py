from pathlib import Path

import pytest

from mufu_server.config import Settings

REQUIRED = {"MUFU_CHECKPOINT_DIR": "checkpoints", "MUFU_API_KEY": "s3cr3t"}


def test_from_env_reads_overrides():
    settings = Settings.from_env(
        {
            **REQUIRED,
            "MUFU_MODEL_NAME": "mufu_test",
            "MUFU_PORT": "9000",
            "MUFU_REQUEST_TIMEOUT_S": "2.5",
            "MUFU_HF_REPO": "org/mufu",
            "MUFU_MODEL_VERSION": "v1.0.0",
        }
    )
    assert settings.checkpoint_dir == Path("checkpoints")
    assert settings.model_name == "mufu_test"
    assert settings.port == 9000
    assert settings.request_timeout_s == 2.5
    assert settings.hf_repo == "org/mufu"
    assert "s3cr3t" not in repr(settings)


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"MUFU_API_KEY": "s3cr3t"}, "MUFU_CHECKPOINT_DIR"),
        ({"MUFU_CHECKPOINT_DIR": "x", "MUFU_API_KEY": " "}, "MUFU_API_KEY"),
        ({**REQUIRED, "MUFU_PORT": "0"}, "MUFU_PORT"),
        ({**REQUIRED, "MUFU_PORT": "70000"}, "MUFU_PORT"),
        ({**REQUIRED, "MUFU_REQUEST_TIMEOUT_S": "0"}, "MUFU_REQUEST_TIMEOUT_S"),
        ({**REQUIRED, "MUFU_REQUEST_TIMEOUT_S": "nan"}, "MUFU_REQUEST_TIMEOUT_S"),
        ({**REQUIRED, "MUFU_HF_REPO": "org/mufu"}, "MUFU_MODEL_VERSION"),
    ],
)
def test_invalid_settings_fail_fast(env, match):
    with pytest.raises(ValueError, match=match):
        Settings.from_env(env)
