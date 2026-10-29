from pathlib import Path

import pytest

from mufu_server.config import Settings

REQUIRED = {"MUFU_CHECKPOINT": "checkpoints/mufu_seed_4.pt", "MUFU_API_KEY": "s3cr3t"}


def test_from_env_reads_overrides():
    settings = Settings.from_env(
        {**REQUIRED, "MUFU_PORT": "9000", "MUFU_REQUEST_TIMEOUT_S": "2.5"}
    )
    assert settings.checkpoint == Path("checkpoints/mufu_seed_4.pt")
    assert settings.port == 9000
    assert settings.request_timeout_s == 2.5
    assert "s3cr3t" not in repr(settings)


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"MUFU_API_KEY": "s3cr3t"}, "MUFU_CHECKPOINT"),
        ({"MUFU_CHECKPOINT": "x.pt", "MUFU_API_KEY": " "}, "MUFU_API_KEY"),
        ({**REQUIRED, "MUFU_PORT": "0"}, "MUFU_PORT"),
        ({**REQUIRED, "MUFU_PORT": "70000"}, "MUFU_PORT"),
        ({**REQUIRED, "MUFU_REQUEST_TIMEOUT_S": "0"}, "MUFU_REQUEST_TIMEOUT_S"),
        ({**REQUIRED, "MUFU_REQUEST_TIMEOUT_S": "nan"}, "MUFU_REQUEST_TIMEOUT_S"),
    ],
)
def test_invalid_settings_fail_fast(env, match):
    with pytest.raises(ValueError, match=match):
        Settings.from_env(env)
