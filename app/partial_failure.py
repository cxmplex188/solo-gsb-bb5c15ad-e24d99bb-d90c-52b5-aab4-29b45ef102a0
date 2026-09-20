"""核心算法：从异常凝胶反推最可能未切开的酶切边界（部分酶切失败）。

所有函数均为纯函数，不依赖任何外部服务或全局状态。

部分失败合并规则
----------------
请求给出按**圆环顺序**排列的理论片段：片段 i 长度为 L_i，其后边界名为
B_i（即片段 i 与片段 (i+1) mod n 之间的酶切边界）。某边界 B_i
"失败"（该酶在此处未切开）时，片段 i 与片段 i+1 合并：

* **连续失败**：一串连续的失败边界把相邻的多个理论片段合并为一段，
  合并长度为各段长度之和；一段的起点边界（前一切开边界）与终点边界
  （后一切开边界）均为成功切开的边界，串内边界全部失败；
* **首尾跨原点合并**：圆环首尾没有天然接缝——边界 B_{n-1} 位于末段与
  首段之间，它失败时末段与首段合并，且可与首段之后继续失败的边界连成
  跨原点的一整段，输出中以 ``crosses_origin: true`` 标记；
* **全部失败**：所有边界都失败时整条质粒连成一段（长度即质粒长度），
  该段没有切开的起点 / 终点边界。

对每个失败边界数不超过 ``max_failed_boundaries`` 的边界子集（含空集，
即完全切开），计算合并后的片段长度作为预测条带，再复用
:func:`app.identify.match_sample` 与一次凝胶观测做一对一最小代价匹配
（先最大化匹配对数，再最小化总误差）。

唯一诊断选定
------------
全部评估子集依次按以下次序比较，次序④保证即使指标完全相同也有唯一
结果（片段的提交顺序即圆环顺序，失败边界名序列也按该顺序规范化）：

  1. 未匹配条带总数（缺失 + 额外）升序；
  2. 总误差（各配对长度差之和）升序；
  3. 失败边界数升序；
  4. 失败边界名按圆环顺序排列后的序列字典序升序。
"""
from itertools import combinations

from app.identify import match_sample
from app.models import (
    MergedFragmentOut,
    PartialFailureRequest,
    PartialFailureResponse,
    SampleMatchOut,
)


# --------------------------------------------------------------- 圆环合并

def merge_fragments(names: list[str], lengths: list[int], failed: frozenset[int]
                    ) -> list[MergedFragmentOut]:
    """按下标集合 failed 中的失败边界合并圆环上的理论片段。

    边界 i 失败 => 片段 i 与片段 (i+1) % n 合并。先按切开的边界
    B_0..B_{n-2} 把线性排列 0..n-1 切成连续段；输出沿圆环顺序排列：

    * B_{n-1} 切开时，首段即片段 0，末段（含片段 n-1）为跨原点段，排
      在最后，与 :func:`app.digest.circular_fragment` 的末段跨原点约定
      一致；
    * B_{n-1} 失败时，含片段 n-1 的末段与首段开始的连续失败前缀连成
      跨原点段（``component_fragments`` 为 ``[j, …, n-1, 0, …, k]``），
      该段在首个切开的边界处闭合，其余段从该边界之后开始排列，跨原点
      段仍排在最后；
    * 全部边界失败（或仅 1 段）时返回单段（起止边界为 None）。
    """
    n = len(lengths)
    if n == 1 or len(failed) == n:
        # 唯一边界无论成败物理上都只有完整质粒一段
        return [MergedFragmentOut(
            length=sum(lengths), start_boundary=None, end_boundary=None,
            crosses_origin=True, component_fragments=list(lengths),
        )]

    # 先按"切开"的边界 B_0..B_{n-2} 把线性排列 0..n-1 切成连续段
    runs: list[list[int]] = []
    run = [0]
    for i in range(n - 1):
        if i in failed:  # B_i 失败：片段 i 与 i+1 同段
            run.append(i + 1)
        else:           # B_i 切开：段在此断开
            runs.append(run)
            run = [i + 1]
    runs.append(run)

    def close(block: list[int]) -> MergedFragmentOut:
        first_index = block[0]
        return MergedFragmentOut(
            length=sum(lengths[i] for i in block),
            start_boundary=names[(first_index - 1) % n],
            end_boundary=names[block[-1]],
            crosses_origin=n - 1 in block,
            component_fragments=[lengths[i] for i in block],
        )

    if (n - 1) not in failed:
        # B_{n-1} 切开：首段与末段不合并，按提交顺序输出（末段自然跨原点）
        return [close(run) for run in runs]

    # B_{n-1} 失败：末段与首段开始的连续失败前缀 runs[0] 连成跨原点段。
    # 其余普通段保持输入中的圆环顺序输出，跨原点段排在最后；
    # 能走到这里说明 B_0..B_{n-2} 至少一个切开，故 runs 至少有 2 段。
    crossing = runs[-1] + runs[0]
    return [close(run) for run in runs[1:-1]] + [close(crossing)]


# --------------------------------------------------------------- 子集评估

def resolve_partial_failure(req: PartialFailureRequest) -> PartialFailureResponse:
    """枚举不超过上限的全部失败边界子集，按固定次序选定唯一诊断。"""
    names = [f.next_boundary for f in req.fragments]
    lengths = [f.length for f in req.fragments]
    observed = list(req.observed_bands)
    tolerance = req.band_match_tolerance

    best_key: tuple[int, float, int, tuple[str, ...]] | None = None
    best_match: SampleMatchOut | None = None
    best_bands: list[float] = []
    best_merged: list[MergedFragmentOut] = []
    evaluated = 0

    for size in range(0, req.max_failed_boundaries + 1):
        for subset in combinations(range(len(names)), size):
            evaluated += 1
            failed = frozenset(subset)
            merged = merge_fragments(names, lengths, failed)
            predicted = sorted(float(m.length) for m in merged)
            match = match_sample(predicted, observed, tolerance)
            unmatched = match.missing_count + match.extra_count
            failed_names = tuple(names[i] for i in subset)  # 已是圆环顺序
            key = (unmatched, match.total_error, size, failed_names)
            if best_key is None or key < best_key:
                best_key = key
                best_match = match
                best_bands = predicted
                best_merged = merged

    unmatched, total_error, size, failed_names = best_key
    reason = (
        f"在 {evaluated} 个失败边界数不超过 {req.max_failed_boundaries} 的边界子集中，"
        f"判定未切开边界 {list(failed_names)}（{size} 个）：未匹配条带 {unmatched} 条、"
        f"总误差 {total_error:g} bp、失败边界数 {size}，依次按未匹配条带数升序、"
        f"总误差升序、失败边界数升序、失败边界名序列（圆环顺序）字典序比较后为唯一诊断"
    )
    return PartialFailureResponse(
        plasmid_length=req.plasmid_length,
        band_match_tolerance=tolerance,
        observed_band_count=len(observed),
        boundary_count=len(names),
        max_failed_boundaries=req.max_failed_boundaries,
        failed_boundaries=list(failed_names),
        failed_boundary_count=size,
        predicted_bands=best_bands,
        merged_fragments=best_merged,
        match=best_match,
        evaluated_subset_count=evaluated,
        selection_reason=reason,
    )
