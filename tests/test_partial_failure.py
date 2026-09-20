"""酶切部分失败反推：圆环合并、失败边界子集枚举与唯一诊断的单元 + API 测试。"""
from fastapi.testclient import TestClient

from app.main import app
from app.models import PartialFailureRequest
from app.partial_failure import merge_fragments, resolve_partial_failure

client = TestClient(app)


# ---------------------------------------------------------------- 圆环合并（纯函数）

def test_merge_no_failures_preserves_fragments_and_marks_last_crossing():
    names = ["B0", "B1", "B2", "B3"]
    lengths = [200, 300, 250, 250]
    ms = merge_fragments(names, lengths, frozenset())
    assert [m.length for m in ms] == [200, 300, 250, 250]
    assert [(m.start_boundary, m.end_boundary) for m in ms] == [
        ("B3", "B0"), ("B0", "B1"), ("B1", "B2"), ("B2", "B3"),
    ]
    assert [m.crosses_origin for m in ms] == [False, False, False, True]


def test_merge_single_failure_joins_adjacent_fragments():
    # B1 失败：片段 1 与片段 2 合并为 550
    ms = merge_fragments(["B0", "B1", "B2", "B3"], [200, 300, 250, 250], frozenset({1}))
    assert [(m.length, m.start_boundary, m.end_boundary, m.crosses_origin) for m in ms] == [
        (200, "B3", "B0", False),
        (550, "B0", "B2", False),
        (250, "B2", "B3", True),
    ]
    assert ms[1].component_fragments == [300, 250]


def test_merge_consecutive_failures_merge_longer_block():
    # B0、B1 连续失败：片段 0、1、2 合并为 750，片段 3 单独
    ms = merge_fragments(["B0", "B1", "B2", "B3"], [200, 300, 250, 250], frozenset({0, 1}))
    assert [(m.length, m.start_boundary, m.end_boundary) for m in ms] == [
        (750, "B3", "B2"), (250, "B2", "B3"),
    ]
    assert ms[0].component_fragments == [200, 300, 250]


def test_merge_last_boundary_failure_crosses_origin_with_first_fragment():
    # B3 失败（末段与首段之间）：末段 250 + 首段 200 = 450，标记跨原点
    ms = merge_fragments(["B0", "B1", "B2", "B3"], [200, 300, 250, 250], frozenset({3}))
    assert [(m.length, m.start_boundary, m.end_boundary, m.crosses_origin) for m in ms] == [
        (300, "B0", "B1", False),
        (250, "B1", "B2", False),
        (450, "B2", "B0", True),
    ]
    assert ms[-1].component_fragments == [250, 200]


def test_merge_last_and_leading_boundaries_form_one_wrap_block():
    # B3、B0 失败：B0 失败使首段并入片段 1（B1 切开），B3 失败使末段与
    # [0,1] 段连成跨原点段 => 250 单独 + 250+200+300=750 跨原点
    ms = merge_fragments(["B0", "B1", "B2", "B3"], [200, 300, 250, 250], frozenset({3, 0}))
    assert [(m.length, m.start_boundary, m.end_boundary, m.crosses_origin) for m in ms] == [
        (250, "B1", "B2", False),
        (750, "B2", "B1", True),
    ]
    assert ms[-1].component_fragments == [250, 200, 300]
    # B2、B3 连续失败：300 单独，250+250+200=700 跨原点
    ms2 = merge_fragments(["B0", "B1", "B2", "B3"], [200, 300, 250, 250], frozenset({2, 3}))
    assert [(m.length, m.start_boundary, m.end_boundary, m.crosses_origin) for m in ms2] == [
        (300, "B0", "B1", False),
        (700, "B1", "B0", True),
    ]
    assert ms2[-1].component_fragments == [250, 250, 200]


def test_merge_single_remaining_cut_linearizes_whole_plasmid():
    # 仅一个边界切开时整条质粒线性化为一段（同一边界既是起点又是终点）
    ms = merge_fragments(["B0", "B1", "B2", "B3"], [200, 300, 250, 250], frozenset({0, 1, 2}))
    assert len(ms) == 1
    assert ms[0].length == 1000
    assert (ms[0].start_boundary, ms[0].end_boundary) == ("B3", "B3")
    assert ms[0].component_fragments == [200, 300, 250, 250]


def test_merge_all_boundaries_fail_returns_unnamed_single_block():
    ms = merge_fragments(["B0", "B1", "B2", "B3"], [200, 300, 250, 250], frozenset({0, 1, 2, 3}))
    assert len(ms) == 1
    assert ms[0].length == 1000
    assert ms[0].start_boundary is None and ms[0].end_boundary is None
    assert ms[0].crosses_origin is True


def test_merge_single_fragment_plasmid():
    ms = merge_fragments(["B0"], [1000], frozenset())
    assert len(ms) == 1 and ms[0].length == 1000
    assert ms[0].start_boundary is None and ms[0].end_boundary is None


# ---------------------------------------------------------------- 子集枚举与选定（纯函数）

def _req(fragments, observed, tolerance=10, max_failed=None, length=1000):
    return PartialFailureRequest(
        plasmid_length=length,
        fragments=[{"length": ln, "next_boundary": name} for name, ln in fragments],
        observed_bands=observed,
        band_match_tolerance=tolerance,
        max_failed_boundaries=len(fragments) if max_failed is None else max_failed,
    )


FRAGMENTS = [
    ("EcoRI@300", 200),
    ("BamHI@600", 300),
    ("HindIII@850", 250),
    ("EcoRI@100", 250),
]


def test_resolve_detects_single_failed_boundary():
    # BamHI@600 失败 => 条带 200 / 550 / 250；观测 202 / 548 / 251
    res = resolve_partial_failure(_req(FRAGMENTS, [202, 251, 548], max_failed=2))
    assert res.failed_boundaries == ["BamHI@600"]
    assert res.failed_boundary_count == 1
    assert res.predicted_bands == [200.0, 250.0, 550.0]
    match = res.match
    assert match.matched_count == 3
    assert match.missing_count == 0 and match.extra_count == 0
    assert match.total_error == 5  # 2 + 2 + 1
    # 1 + C(4,1) + C(4,2) = 11
    assert res.evaluated_subset_count == 11
    assert res.selection_reason


def test_resolve_zero_failures_when_complete_digest_matches():
    res = resolve_partial_failure(_req(FRAGMENTS, [200, 300, 250, 250]))
    assert res.failed_boundaries == []
    assert res.failed_boundary_count == 0
    assert res.predicted_bands == [200.0, 250.0, 250.0, 300.0]
    assert res.match.total_error == 0


def test_resolve_wrap_around_failure():
    # EcoRI@100（B3，末段/首段之间）失败 => 300 / 250 / 450
    res = resolve_partial_failure(_req(FRAGMENTS, [301, 250, 449]))
    assert res.failed_boundaries == ["EcoRI@100"]
    assert res.predicted_bands == [250.0, 300.0, 450.0]
    crossing = [m for m in res.merged_fragments if m.crosses_origin]
    assert len(crossing) == 1 and crossing[0].length == 450


def test_resolve_two_consecutive_failures_required():
    # EcoRI@300 与 BamHI@600（B0、B1）连续失败 => 750 / 250
    res = resolve_partial_failure(_req(FRAGMENTS, [748, 251]))
    assert res.failed_boundaries == ["EcoRI@300", "BamHI@600"]
    assert res.predicted_bands == [250.0, 750.0]
    assert res.match.matched_count == 2 and res.match.total_error == 3


def test_resolve_lexicographic_tiebreak_equal_length_fragments():
    # 四个等长片段 4×250，观测 [250,500,250]：恰好一个边界失败 =>
    # [500,250,250] 全部匹配。每个单失败子集都得到同一多重集 {500,250,250}
    # （B0/B1/B2 失败或 B3 跨原点失败均为 1 个失败、0 未匹配、0 误差），
    # 前三级完全相同，按失败边界名（圆环顺序）字典序取最小者
    frags = [("d-cut", 250), ("a-cut", 250), ("b-cut", 250), ("c-cut", 250)]
    res = resolve_partial_failure(_req(frags, [250, 500, 250], length=1000))
    assert res.failed_boundaries == ["a-cut"]
    assert res.failed_boundary_count == 1
    assert res.predicted_bands == [250.0, 250.0, 500.0]
    assert res.match.total_error == 0


def test_resolve_max_failed_boundaries_limits_enumeration():
    # 上限 0：只评估完全切开，即使存在单失败完美解释也不选
    res = resolve_partial_failure(_req(FRAGMENTS, [202, 251, 548], max_failed=0))
    assert res.failed_boundaries == []
    assert res.evaluated_subset_count == 1
    # 上限 1：C(4,0)+C(4,1)=5，且能选中单失败
    res1 = resolve_partial_failure(_req(FRAGMENTS, [202, 251, 548], max_failed=1))
    assert res1.evaluated_subset_count == 5
    assert res1.failed_boundaries == ["BamHI@600"]


def test_resolve_lexicographic_tiebreak_on_failure_name_sequence():
    # 片段 [100,100,100,100]，观测 [100,100,200]：
    # 空集 => 3 匹配 + 1 缺失（未匹配 1）；恰好一个边界失败 =>
    # [200,100,100] 全部匹配（未匹配 0），4 个单失败子集结果完全相同，
    # 按圆环顺序的失败边界名字典序取 a
    frags = [("d", 100), ("a", 100), ("b", 100), ("c", 100)]
    res = resolve_partial_failure(_req(frags, [100, 100, 200], length=400))
    assert res.failed_boundaries == ["a"]
    assert res.match.missing_count == 0 and res.match.extra_count == 0


# ---------------------------------------------------------------- API 端到端

def _payload(**kw):
    payload = {
        "plasmid_length": 1000,
        "fragments": [
            {"length": 200, "next_boundary": "EcoRI@300"},
            {"length": 300, "next_boundary": "BamHI@600"},
            {"length": 250, "next_boundary": "HindIII@850"},
            {"length": 250, "next_boundary": "EcoRI@100"},
        ],
        "observed_bands": [202, 251, 548],
        "band_match_tolerance": 10,
        "max_failed_boundaries": 2,
    }
    payload.update(kw)
    return payload


def test_api_partial_failure_full_evidence():
    res = client.post("/api/v1/digest/partial-failure", json=_payload())
    assert res.status_code == 200
    body = res.json()

    assert body["failed_boundaries"] == ["BamHI@600"]
    assert body["failed_boundary_count"] == 1
    assert body["boundary_count"] == 4
    assert body["predicted_bands"] == [200.0, 250.0, 550.0]
    assert body["evaluated_subset_count"] == 11
    assert body["observed_band_count"] == 3
    assert body["band_match_tolerance"] == 10
    assert body["max_failed_boundaries"] == 2
    assert body["selection_reason"]

    # 合并片段：200 / 550(300+250) / 250(跨原点)，总长仍为质粒长度
    assert [m["length"] for m in body["merged_fragments"]] == [200, 550, 250]
    assert sum(m["length"] for m in body["merged_fragments"]) == 1000
    assert sum(1 for m in body["merged_fragments"] if m["crosses_origin"]) == 1

    match = body["match"]
    assert match["predicted_band_count"] == 3
    assert match["matched_count"] == 3
    assert match["missing_count"] == 0 and match["extra_count"] == 0
    assert match["missing_bands"] == [] and match["extra_bands"] == []
    assert match["total_error"] == 5
    assert [m2["predicted_length"] for m2 in match["matches"]] == [200.0, 250.0, 550.0]
    assert [m2["observed_length"] for m2 in match["matches"]] == [202, 251, 548]
    for m2 in match["matches"]:
        assert m2["error"] == abs(m2["predicted_length"] - m2["observed_length"])
        assert m2["description"]


def test_api_partial_failure_counts_consistent():
    body = client.post("/api/v1/digest/partial-failure", json=_payload(
        observed_bands=[202, 999],
    )).json()
    match = body["match"]
    assert match["matched_count"] + match["missing_count"] == match["predicted_band_count"]
    assert match["matched_count"] + match["extra_count"] == body["observed_band_count"]
    assert match["missing_count"] == len(match["missing_bands"])
    assert match["extra_count"] == len(match["extra_bands"])


# ---------------------------------------------------------------- 422 校验

def _expect_422(payload, code_part=None):
    res = client.post("/api/v1/digest/partial-failure", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "validation_failed"
    assert body["errors"]
    if code_part:
        assert any(code_part in e["type"] for e in body["errors"])
    return body


def test_422_fragment_length_mismatch():
    body = _expect_422(_payload(fragments=[
        {"length": 200, "next_boundary": "EcoRI@300"},
        {"length": 300, "next_boundary": "BamHI@600"},
        {"length": 250, "next_boundary": "HindIII@850"},
        {"length": 251, "next_boundary": "EcoRI@100"},
    ]), "fragment_length_mismatch")
    assert any("fragments" in e["loc"] for e in body["errors"])


def test_422_duplicate_boundary_name():
    _expect_422(_payload(fragments=[
        {"length": 200, "next_boundary": "dup"},
        {"length": 300, "next_boundary": "dup"},
        {"length": 250, "next_boundary": "x"},
        {"length": 250, "next_boundary": "y"},
    ]), "duplicate_boundary_name")


def test_422_blank_boundary_name():
    _expect_422(_payload(fragments=[
        {"length": 200, "next_boundary": "  "},
        {"length": 800, "next_boundary": "y"},
    ]), "invalid_boundary_name")


def test_422_max_failed_boundaries_too_large():
    body = _expect_422(_payload(max_failed_boundaries=5), "max_failed_boundaries_too_large")
    assert any("max_failed_boundaries" in e["loc"] for e in body["errors"])


def test_422_basic_field_constraints():
    # 空片段列表、空观测、负长度片段、负容差、负上限
    _expect_422(_payload(fragments=[]))
    _expect_422(_payload(observed_bands=[]))
    _expect_422(_payload(fragments=[
        {"length": 0, "next_boundary": "x"}, {"length": 1000, "next_boundary": "y"},
    ]))
    _expect_422(_payload(band_match_tolerance=-1))
    _expect_422(_payload(max_failed_boundaries=-1))


def test_422_too_many_boundaries(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_PARTIAL_FAILURE_BOUNDARIES", 3)
    _expect_422(_payload(), "too_many_partial_failure_boundaries")


def test_422_reports_multiple_errors_at_once():
    body = _expect_422(_payload(
        fragments=[
            {"length": 200, "next_boundary": "dup"},
            {"length": 301, "next_boundary": "dup"},
            {"length": 250, "next_boundary": "x"},
            {"length": 250, "next_boundary": "y"},
        ],
        max_failed_boundaries=9,
    ))
    types = {e["type"] for e in body["errors"]}
    assert "fragment_length_mismatch" in types
    assert "duplicate_boundary_name" in types
    assert "max_failed_boundaries_too_large" in types
