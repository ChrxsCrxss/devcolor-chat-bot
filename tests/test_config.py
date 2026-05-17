from devcolor_rag.config import SessionConfig, parse_set_command


def test_set_top_k():
    cfg = SessionConfig()
    result = parse_set_command("/SET top-k 7", cfg)
    assert result is not None
    assert result.kind == "applied"
    assert cfg.top_k == 7


def test_set_strict_bool():
    cfg = SessionConfig()
    result = parse_set_command("/SET strict on", cfg)
    assert result is not None
    assert result.kind == "applied"
    assert cfg.strict is True


def test_set_unknown():
    cfg = SessionConfig()
    result = parse_set_command("/SET foo bar", cfg)
    assert result is not None
    assert result.kind == "error"


def test_set_list():
    cfg = SessionConfig()
    result = parse_set_command("/SET", cfg)
    assert result is not None
    assert result.kind == "list"
