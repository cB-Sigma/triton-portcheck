import pytest

from triton_portcheck.targets import TargetError, matches, parse_target, parse_targets


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cuda:90", "cuda:90"),
        ("CUDA:sm_90a", "cuda:90"),
        ("cuda:8.0", "cuda:80"),
        ("hip:gfx942", "hip:gfx942"),
        ("hip:gfx90a", "hip:gfx90a"),
        ("hip:gfx942:64", "hip:gfx942:64"),
        ("xpu:arc", "xpu:arc"),
    ],
)
def test_parse_target_normalises(text, expected):
    assert str(parse_target(text)) == expected


@pytest.mark.parametrize("text", ["cuda", "cuda:", "cuda:hopper", ":90", "cuda 90", ""])
def test_parse_target_rejects(text):
    with pytest.raises(TargetError):
        parse_target(text)


def test_parse_targets_splits_and_dedups():
    assert parse_targets(["cuda:80,cuda:90", "cuda:80", " hip:gfx950 "]) == ["cuda:80", "cuda:90", "hip:gfx950"]


def test_matches_globs_and_versions():
    assert matches("hip:*", "hip:gfx942")
    assert not matches("hip:*", "cuda:80")
    assert matches("cuda:7?", "cuda:75")
    assert matches("cuda:90@3.8.*", "cuda:90", "3.8.0")
    assert not matches("cuda:90@3.8.*", "cuda:90", "3.9.0")
    assert matches("@3.9.0", "cuda:90", "3.9.0")
