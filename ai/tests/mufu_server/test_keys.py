import stat

import pytest

from mufu_server.keys import hash_key, load_keys, main


def test_add_saves_only_the_hash_in_an_owner_only_file(tmp_path, capsys):
    path = tmp_path / "mufu" / "api_keys"

    main(["add", "alex", "--file", str(path)])
    main(["add", "sam.lee+mufu@ufl.edu", "--file", str(path)])

    alex, sam = capsys.readouterr().out.split()
    assert load_keys(path) == {
        "alex": hash_key(alex),
        "sam.lee+mufu@ufl.edu": hash_key(sam),
    }
    assert alex not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("name", ["alex", "bad name", "a,b@ufl.edu", ""])
def test_add_rejects_repeated_or_invalid_names(tmp_path, name):
    path = tmp_path / "api_keys"
    main(["add", "alex", "--file", str(path)])

    with pytest.raises(SystemExit):
        main(["add", name, "--file", str(path)])


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("alex\n", "expected"),
        (f"a {'A' * 64}\n", "expected"),
        (f"a {'0' * 64}\na {'1' * 64}\n", "twice"),
        (f"a {'0' * 64}\nb {'0' * 64}\n", "reuses"),
    ],
)
def test_load_rejects_malformed_files(tmp_path, text, match):
    path = tmp_path / "api_keys"
    path.write_text(text)

    with pytest.raises(ValueError, match=match):
        load_keys(path)
