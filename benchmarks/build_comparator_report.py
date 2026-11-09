"""Build a Chinese, offline reading report from completed comparator measurements.

No torch import or GPU access. Synthetic fixtures are clearly marked and cannot
be published accidentally. Main metrics are never taken from the older run.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "dense": "普通 BERT", "deebert": "DeeBERT",
    "router_loop": "RouterBERT · 逐条路由", "router_batched": "RouterBERT · 批量路由",
    "hdc_loop": "HDC · 逐条路由", "hdc_batched": "HDC · 批量路由",
    "hdc_exitonly": "HDC · 仅保留早退",
    "router_legacy": "RouterBERT · 原 KV 裁剪", "router_densegate": "RouterBERT · 全量计算后门控",
}
PRIMARY = ["dense", "deebert", "router_loop", "router_batched", "hdc_loop", "hdc_batched", "hdc_exitonly"]
CHART_LABELS = ["Dense BERT", "DeeBERT", "Router / loop", "Router / batched", "HDC / loop", "HDC / batched", "HDC / exit only"]
P50 = "latency_p50_ms_mean_of_repeats"
P95 = "latency_p95_ms_mean_of_repeats"
FLOPS = "timed_inputs_matmul_flops_per_sample"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value), quote=True)


def pct_change(after, before):
    value = 100 * (after / before - 1)
    return "持平" if abs(value) < .005 else f"{'增加' if value > 0 else '降低'} {abs(value):.1f}%"


def pp_change(after, before):
    value = (after - before) * 100
    return "相同" if abs(value) < .00001 else f"{'提高' if value > 0 else '下降'} {abs(value):.2f} 个百分点"


def value_range(values):
    return "未记录" if not values else f"{min(values):.3f}–{max(values):.3f}"


def validate(summary, manifest, directory, allow_example=False):
    if (summary.get("synthetic") or manifest.get("synthetic") or summary.get("smoke")) and not allow_example:
        raise ValueError("Synthetic/smoke records cannot be published as experiment results")
    if not summary.get("protocol_sha256") or summary["protocol_sha256"] != manifest.get("protocol_sha256"):
        raise ValueError("Protocol identity missing or mismatched")
    if not manifest.get("shared_backbone_identity", {}).get("verified_identical"):
        raise ValueError("Shared-backbone equality has not been verified")
    rows = summary["rows"]
    keys = [(row["variant"], row["batch_size"]) for row in rows]
    expected = {(name, batch) for name in manifest["models"] for batch in manifest["batch_sizes"]}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing/duplicate/unexpected comparison rows")
    repeats = manifest["config"]["benchmark_repeats"]
    for row in rows:
        if row["variant"] not in LABELS:
            raise ValueError("Unknown model path")
        for field in (P50, P95, FLOPS, "accuracy"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"Invalid numeric value: {field}")
        if not 0 <= row["accuracy"] <= 1 or row[P50] <= 0:
            raise ValueError("Invalid accuracy/latency range")
        if len(row["latency_p50_ms_by_repeat"]) != repeats:
            raise ValueError("Incomplete timing repetitions")
        p95 = []
        for repeat in range(repeats):
            path = directory / f"latency_{row['variant']}_b{row['batch_size']}_r{repeat}.json"
            if not path.exists():
                raise ValueError(f"Missing raw timing record: {path.name}")
            raw = read(path)
            if (raw["variant"], raw["batch_size"], raw["repeat"]) != (row["variant"], row["batch_size"], repeat):
                raise ValueError("Timing filename/content identity mismatch")
            if not math.isclose(raw["latency"]["p50_ms"], row["latency_p50_ms_by_repeat"][repeat], rel_tol=1e-9):
                raise ValueError("Raw/summary P50 mismatch")
            p95.append(raw["latency"]["p95_ms"])
        if not math.isclose(sum(p95)/len(p95), row[P95], rel_tol=1e-9):
            raise ValueError("Raw/summary P95 mismatch")
        row["report_p95_by_repeat"] = p95
    return rows


def table(rows, *, ranges=False):
    header = ("<th>模型路径</th><th>准确率</th><th>P50<br>ms / 批</th><th>P95<br>ms / 批</th>"
              "<th>矩阵 GFLOPs<br>/ 文本</th><th>矩阵 FLOPs<br>较普通 BERT 节省</th>")
    if ranges:
        header = "<th>模型路径</th><th>Batch</th><th>各轮 P50 范围<br>ms / 批</th><th>各轮 P95 范围<br>ms / 批</th><th>吞吐量<br>文本 / 秒</th>"
    body = []
    for row in rows:
        label = LABELS[row["variant"]]
        if ranges:
            cells = f"<td>{row['batch_size']}</td><td>{value_range(row['latency_p50_ms_by_repeat'])}</td><td>{value_range(row['report_p95_by_repeat'])}</td><td>{row['throughput_samples_per_second_mean']:.1f}</td>"
        else:
            cells = (f"<td>{row['accuracy']:.2%}</td><td>{row[P50]:.3f}</td><td>{row[P95]:.3f}</td>"
                     f"<td>{row[FLOPS]/1e9:.3f}</td><td>{row['timed_inputs_matmul_flops_saving_percent']:.1f}%</td>")
        body.append(f"<tr><th scope='row'>{esc(label)}</th>{cells}</tr>")
    return "<div class='table-wrap'><table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def chart(rows, path, example=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_key = {(row["variant"], row["batch_size"]): row for row in rows}
    batches = sorted({row["batch_size"] for row in rows})
    figure, axes = plt.subplots(1, len(batches), figsize=(14, 5.5), layout="constrained", squeeze=False)
    colours = ["#64748b", "#3b82f6", "#b45309", "#f59e0b", "#0f766e", "#14b8a6", "#7c3aed"]
    for axis, batch in zip(axes[0], batches):
        values = [by_key[name, batch][P50] for name in PRIMARY]
        repeats = [by_key[name, batch]["latency_p50_ms_by_repeat"] for name in PRIMARY]
        errors = [[v-min(r) for v,r in zip(values,repeats)], [max(r)-v for v,r in zip(values,repeats)]]
        bars = axis.barh(range(len(values)), values, color=colours, xerr=errors, capsize=3)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5, fontsize=9)
        axis.set_yticks(range(len(values)), CHART_LABELS)
        axis.invert_yaxis()
        axis.set_title(f"Batch {batch}", loc="left", weight="bold")
        axis.set_xlabel("P50 latency (ms / batch)")
        axis.set_xlim(0, max(max(r) for r in repeats)*1.25)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#e2e8f0")
        axis.spines[["top", "right", "left"]].set_visible(False)
    figure.suptitle("SYNTHETIC EXAMPLE / NOT RESULTS" if example else "Paired latency: model choice and routing implementation", weight="bold", fontsize=14)
    figure.supxlabel("Error bars: min-max across timing repeats. Each panel has its own scale. Lower is faster.", fontsize=9)
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def prediction_agreement(directory, first, second, batch):
    paths = [directory/f"quality_{name}_b{batch}_predictions.csv" for name in (first, second)]
    if not all(path.exists() for path in paths):
        return "逐条预测对照未随报告提供"
    records = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            records.append({int(row["sample_index"]): int(row["prediction"]) for row in csv.DictReader(stream)})
    if set(records[0]) != set(records[1]):
        raise ValueError("Prediction comparison samples do not match")
    differences = sum(records[0][key] != records[1][key] for key in records[0])
    return f"{len(records[0]):,} 条预测全部一致" if differences == 0 else f"{len(records[0]):,} 条中有 {differences} 条预测不同；不能把两种执行方式视为实测完全一致"


def implementation_section(rows, directory):
    lookup = {(r["variant"], r["batch_size"]): r for r in rows}
    parts = []
    for batch in sorted({r["batch_size"] for r in rows}):
        entries = []
        for prefix, label in (("router", "RouterBERT"), ("hdc", "HDC")):
            before, after = lookup[prefix+"_loop", batch], lookup[prefix+"_batched", batch]
            agreement = prediction_agreement(directory, prefix+"_loop", prefix+"_batched", batch)
            entries.append(f"{label}：中位延迟从 {before[P50]:.3f} 到 {after[P50]:.3f} ms，{pct_change(after[P50], before[P50])}；矩阵 FLOPs {pct_change(after[FLOPS], before[FLOPS])}。{agreement}。")
        parts.append(f"<li><strong>Batch {batch}</strong><p>" + "</p><p>".join(esc(x) for x in entries) + "</p></li>")
    return ("<section><h2>同一组权重，只改执行方式会怎样？</h2><p>逐条路由与批量路由保留完整 K/V，计算语义相同；后者把一批中的查询补齐到该批最大保留长度，减少小算子调用。多出来的补齐计算也计入 FLOPs，不能沿用逐条版的较低 FLOPs 数字。</p><ul>"
            + "".join(parts) + "</ul><p class='note'>这是同权重的实现对照，最适合判断 Python 循环、索引和小矩阵调用的代价。准确率只显示两位小数时相同不代表逐条预测必然相同，以上另行核对预测文件。</p></section>")


def ablation_section(rows):
    lookup = {(r["variant"], r["batch_size"]): r for r in rows}
    text = []
    for batch in sorted({r["batch_size"] for r in rows}):
        routed, exitonly = lookup["hdc_batched", batch], lookup["hdc_exitonly", batch]
        text.append(f"<li>Batch {batch}：加入批量路由后，中位延迟{pct_change(routed[P50], exitonly[P50])}，矩阵 FLOPs {pct_change(routed[FLOPS], exitonly[FLOPS])}，准确率{pp_change(routed['accuracy'], exitonly['accuracy'])}。</li>")
    return ("<section><h2>路由到底有没有额外价值？</h2><p>“HDC · 仅保留早退”使用同一骨干、同一前六层出口和同一阈值；未提前退出的样本直接执行后六层普通 BERT。与完整 HDC 对照，可以隔离后半段路由的边际影响。</p><ul>"
            + "".join(text) + "</ul><p class='note'>它是针对现有 HDC 的消融，不等同于可在全部 12 层早退的 DeeBERT，也不证明所有路由算法都具有同样表现。</p></section>")


def training_section(run):
    rows = []
    for name, label, field in (("router", "RouterBERT 路由器", "accuracy"), ("deebert", "DeeBERT 前 11 个出口", "loss")):
        path = run/f"{name}_history.json"
        if not path.exists():
            continue
        history = read(path)
        best = max(history, key=lambda x:x["validation"]["accuracy"]) if field == "accuracy" else min(history,key=lambda x:x["validation_loss"])
        metric = f"验证准确率 {best['validation']['accuracy']:.2%}" if field == "accuracy" else f"验证平均交叉熵 {best['validation_loss']:.4f}"
        rows.append(f"<tr><th scope='row'>{label}</th><td>{len(history)} 轮；选择第 {best['epoch']} 轮</td><td>{metric}</td></tr>")
    selection = read(run/"deebert_selection.json")
    status = "满足" if selection["constraint_met"] else "未满足"
    return ("<section><h2>训练和选择口径</h2><p>共享之前新训练的 Dense 骨干与最终分类头；RouterBERT 在全部 12 层训练 token 路由器，目标保留率 0.7；DeeBERT 独立训练前 11 个出口，第 12 层直接使用冻结的最终分类头。所有选择只使用 6,000 条验证集，最终质量评估使用 7,600 条官方测试集。</p>"
            "<div class='table-wrap'><table><thead><tr><th>新增部分</th><th>训练轮次</th><th>选择指标</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            f"<p>DeeBERT 选定熵阈值 <strong>{selection['entropy_threshold']}</strong>，验证准确率 {selection['accuracy']:.2%}，{status}预设的“相对 Dense 最多下降 1 个百分点”条件。HDC 沿用此前仅根据验证集选定的阈值。各模型阈值固定用于全部 batch；没有为 batch=8/32 单独寻找最优阈值。</p>"
            "<p class='note'>本次是新的单种子补充实验；重新训练的是新增模块，旧团队训练权重仍未找回。各模型的路由位置、保留率及出口数量不同，因此主比较展示指定配置的取舍，不是穷尽超参数后的算法排名。</p></section>")


def diagnostic_section(run, primary_manifest):
    directory = run/"diagnostic_benchmark"
    if not (directory/"summary.json").exists():
        return ""
    manifest = read(directory/"manifest.json")
    if manifest["dataset_sha256"] != primary_manifest["dataset_sha256"]:
        raise ValueError("Diagnostic data identity differs")
    for name in set(manifest["checkpoints"]) & set(primary_manifest["checkpoints"]):
        if manifest["checkpoints"][name] != primary_manifest["checkpoints"][name]:
            raise ValueError(f"Diagnostic checkpoint differs: {name}")
    rows = validate(read(directory/"summary.json"), manifest, directory)
    parts = []
    for batch in manifest["batch_sizes"]:
        subset = [row for row in rows if row["batch_size"] == batch]
        parts.append(f"<h3>独立诊断 · Batch {batch}</h3>" + table(subset))
    return ("<section><h2>原实现与正确语义的独立诊断</h2><p>以下是单独一轮诊断实验，保留独立的 Dense 参考，不将它的延迟拼进主表。“原 KV 裁剪”在推理时同时删掉未选 token 的键和值；“全量计算后门控”保持训练函数，但不节省注意力计算。它们与 query-loop / query-batched 使用同一新训练的 RouterBERT 权重。</p>" + "".join(parts)
            + "<p class='note'>原 KV 裁剪的质量差异有助于判断语义不一致的后果，但不能反推丢失的历史权重会有相同准确率差。这里的 legacy 路径不是原仓库整个 forward 的逐行计时：新测量适配器已移除无用训练统计，避免把统计同步开销混入语义比较。</p></section>")


def semantic_section(run, primary_manifest):
    directory = run/"semantic_diagnostic"
    if not (directory/"summary.json").exists():
        return "<section><h2>改变注意力函数，实际影响了多少预测？</h2><p>完整测试集的语义对照尚未提供。本报告不把微型正确性测试的误差直接换算成实际准确率损失。</p></section>"
    result, manifest = read(directory/"summary.json"), read(directory/"manifest.json")
    if not result.get("protocol_sha256") or result["protocol_sha256"] != manifest.get("protocol_sha256"):
        raise ValueError("Semantic diagnostic protocol identity mismatch")
    if manifest["checkpoint_sha256"] != primary_manifest["checkpoints"]["router_loop"]:
        raise ValueError("Semantic diagnostic uses a different RouterBERT checkpoint")
    if manifest["data_sha256"] != primary_manifest["dataset_sha256"]:
        raise ValueError("Semantic diagnostic data identity differs")
    rows = {row["variant"]: row for row in result["rows"]}
    if set(rows) != {"router_loop", "router_legacy"} or len(result["rows"]) != 2:
        raise ValueError("Semantic diagnostic requires paired query and legacy rows")
    n = result["samples"]
    query, legacy = rows["router_loop"], rows["router_legacy"]
    if n <= 0 or any(row["samples"] != n or row["batch_size"] != manifest["batch_size"] for row in rows.values()):
        raise ValueError("Semantic diagnostic sample/batch mismatch")
    for row in rows.values():
        if not 0 <= row["correct"] <= n or not math.isclose(row["accuracy"],row["correct"]/n):
            raise ValueError("Semantic diagnostic accuracy/correct count mismatch")
    disagreements = result["prediction_disagreements"]
    recovered, lost = result["query_correct_legacy_wrong"], result["legacy_correct_query_wrong"]
    if not 0 <= recovered + lost <= disagreements <= n or recovered < 0 or lost < 0:
        raise ValueError("Invalid semantic disagreement counts")
    if query["correct"] - legacy["correct"] != recovered - lost:
        raise ValueError("Semantic paired outcome counts disagree with accuracies")
    return ("<section><h2>改变注意力函数，实际影响了多少预测？</h2>"
            f"<p>使用与主实验相同的新训练 RouterBERT 权重，固定 batch={manifest['batch_size']}，在全部 {n:,} 条测试文本上分别运行两条路径。这里只测预测与准确率，<strong>没有为这组语义诊断另报 latency</strong>。</p>"
            "<div class='table-wrap'><table><thead><tr><th>注意力语义</th><th>正确条数</th><th>准确率</th></tr></thead><tbody>"
            f"<tr><th scope='row'>训练一致：选中 query 看完整 K/V</th><td>{query['correct']:,} / {n:,}</td><td>{query['accuracy']:.2%}</td></tr>"
            f"<tr><th scope='row'>原推理：同时裁剪 query 和 K/V</th><td>{legacy['correct']:,} / {n:,}</td><td>{legacy['accuracy']:.2%}</td></tr>"
            "</tbody></table></div>"
            f"<p>保持训练语义后，准确率{pp_change(query['accuracy'],legacy['accuracy'])}。两条路径有 <strong>{disagreements:,} 条预测不同</strong>（{disagreements/n:.2%}）：其中 {recovered:,} 条是修正路径正确、原路径错误；{lost:,} 条方向相反。其余不同预测可能两边都错，不能把全部分歧都称为“修复成功”。</p>"
            "<p class='note'>代码级函数不一致由独立正确性测试确认；上面的实测对照衡量它对这组新权重的实际影响。即使某条路径偶然答对更多，也不能证明训练与推理改变函数是正确实现。历史权重已缺失，不能据此反推历史成绩。</p>"
            "<p><a href='../semantic_diagnostic/summary.json'>查看逐条对照汇总</a> · <a href='../semantic_diagnostic/manifest.json'>查看语义诊断身份记录</a></p></section>")


def build(run_dir, allow_example=False):
    run = Path(run_dir).resolve()
    directory = run/"benchmark"
    summary, manifest = read(directory/"summary.json"), read(directory/"manifest.json")
    rows = validate(summary, manifest, directory, allow_example)
    if set(manifest["models"]) != set(PRIMARY):
        raise ValueError("Main report requires all seven paired paths")
    if read(run/"deebert_selection.json")["entropy_threshold"] != manifest["thresholds"]["deebert"]:
        raise ValueError("DeeBERT selected/measured threshold mismatch")
    example = bool(summary.get("synthetic") or manifest.get("synthetic"))
    lookup = {(r["variant"], r["batch_size"]): r for r in rows}
    config, env = manifest["config"], manifest["environment"]
    chart(rows, directory/"comparison.png", example)
    batches = sorted(manifest["batch_sizes"])
    cards, tables = [], []
    for batch in batches:
        # Rank only the four main model choices, using the measured batched
        # implementation. Still disclose accuracy directly alongside speed.
        choices = [lookup[name,batch] for name in ("dense","deebert","router_batched","hdc_batched")]
        fastest = min(choices, key=lambda x:x[P50])
        cards.append(f"<article><span>Batch {batch} · 本配置中位延迟最低</span><strong>{esc(LABELS[fastest['variant']])}</strong><b>{fastest[P50]:.3f} ms / 批</b><small>准确率 {fastest['accuracy']:.2%}；P95 {fastest[P95]:.3f} ms</small></article>")
        tables.append(f"<h3>一次输入 {batch} 条文本</h3>" + table([lookup[name,batch] for name in PRIMARY]))
    tails = []
    for name in ("deebert", "router_batched", "hdc_batched"):
        first, second = lookup["dense",1], lookup[name,1]
        a,b = first["report_p95_by_repeat"], second["report_p95_by_repeat"]
        overlap = max(min(a),min(b)) <= min(max(a),max(b))
        caveat = "重复范围重叠，不能仅凭平均值宣称尾延迟稳定改善。" if overlap else "重复范围在本轮未重叠；仍属于单机、单种子的观测，不是统计显著性检验。"
        tails.append(f"<li>{esc(LABELS[name])} 的单条 P95 为 {second[P95]:.3f} ms，普通 BERT 为 {first[P95]:.3f} ms。{caveat}</li>")
    audit_path = ROOT/"docs/router_code_audit_20261109.md"
    audit = ""
    if audit_path.exists():
        # Embed a snapshot so the report stays readable when archived standalone.
        audit = "<details><summary>展开完整代码审计记录（英文，含复现证据）</summary><pre>" + esc(audit_path.read_text(encoding="utf-8")) + "</pre></details>"
    marker = "<p class='example'>合成测试样例，不是实验结果</p>" if example else ""
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DeeBERT / RouterBERT 公平补测与代码排查</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f2f5f7;color:#172b3a;font:16px/1.75 'Microsoft YaHei','PingFang SC',sans-serif}}main{{max-width:1120px;margin:36px auto;padding:0 24px 50px}}header,section{{background:white;border:1px solid #dce5eb;border-radius:14px;padding:28px 32px;margin:20px 0}}h1{{font-size:30px;line-height:1.4;margin:8px 0 15px}}h2{{font-size:23px;line-height:1.4;margin:0 0 14px}}h3{{font-size:18px;margin:25px 0 12px}}p{{margin:10px 0}}a{{color:#086b80;text-underline-offset:3px}}.eyebrow{{color:#53717f;font-size:13px;letter-spacing:1px}}.note{{color:#526777;font-size:14px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}}article{{background:#f0f8f8;border:1px solid #cce3e4;border-radius:10px;padding:16px}}article span,article strong,article b,article small{{display:block}}article span{{font-size:12px;color:#536879}}article strong{{font-size:18px;line-height:1.5;margin:8px 0}}article b{{color:#067582;font-size:24px}}article small{{font-size:12px;margin-top:8px}}.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;line-height:1.6;font-variant-numeric:tabular-nums}}th,td{{padding:11px 9px;border-bottom:1px solid #dce5eb;text-align:right;white-space:nowrap}}thead th{{background:#edf3f6;color:#35515e;font-weight:600}}th:first-child{{text-align:left}}tbody th{{font-weight:500}}tbody tr:nth-child(even){{background:#f8fafb}}img{{display:block;width:100%;height:auto;margin:20px 0}}li{{margin:10px 0}}li p{{margin:5px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 Consolas,monospace;background:#f3f5f7;padding:20px}}summary{{cursor:pointer;font-weight:600}}.example{{background:#fff0c2;color:#813f08;padding:14px;font-weight:700}}.links{{display:flex;gap:22px;flex-wrap:wrap;margin-top:18px}}@media(max-width:750px){{main{{padding:0 12px}}header,section{{padding:22px 18px}}h1{{font-size:25px}}.cards{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}header,section{{break-inside:avoid;border-radius:0;padding:18px}}.table-wrap{{overflow:visible}}a{{color:inherit}}}}
    </style></head><body><main><header><div class="eyebrow">2026-11-09 · 新增比较与实现诊断</div>{marker}<h1>DeeBERT、RouterBERT<br>公平补测与代码排查</h1><p>此前只测了普通 BERT 与 HDC，尚不能判断收益来自早退还是路由。这次补齐独立基线，并把“同一组权重、不同执行实现”放在一起比较。</p><p class="note">下方主表全部来自本轮新计时，不拼接昨晚的数据。卡片按中位延迟选择当前配置中的最快项，准确率与尾延迟一并展示，不代表综合质量排名。</p><div class="cards">{''.join(cards)}</div><div class="links"><a href="summary.csv">下载结果 CSV</a><a href="summary.json">查看完整结果 JSON</a><a href="manifest.json">查看实验配置与哈希</a></div></header>
    <section><h2>怎样保证比较公平</h2><p>所有路径共享已校验一致的骨干权重和最终分类头；相同输入分组、GPU、FP32、eager attention、关闭 TF32、固定输入长度 {config['max_length']}。设备：{esc(env.get('gpu','未记录'))}。</p><p>每组先预热 {config['warmup_batches']} 批，再执行 {config['benchmark_repeats']} 轮计时；轮换模型执行顺序。使用固定 {len(manifest['sample_indices']):,} 条不同文本，必要时循环同组输入以达到每轮最少 {config['benchmark_min_calls']} 次批调用。循环次数增加不等于增加独立测试样本。</p><p>准确率在完整 7,600 条测试集上分别测量。时延是模型本身的同步墙钟时间，输入已在显卡上，不含分词、网络、排队或服务框架。P50/P95 是各轮分位数的平均；Batch&gt;1 的数字是<strong>整批耗时</strong>，不能直接当作单请求耗时。</p></section>
    <section><h2>完整结果：同时看速度、准确率和计算量</h2><img src="comparison.png" alt="三种批量大小下七条执行路径的中位延迟，误差线表示重复轮次范围">{''.join(tables)}<p class="note">FLOPs 使用测速的相同输入分组；1 MAC = 2 FLOPs。计数覆盖实际执行形状的矩阵乘法，包含补齐查询、完整 FFN、路由和分类头；不包含索引、搬运、归一化与 softmax 等操作。准确率采用全测试集，FLOPs 的对应输入范围不同，不能混用。</p></section>
    {implementation_section(rows,directory)}{ablation_section(rows)}
    <section><h2>代码到底哪里有问题？</h2><ol><li><strong>已确认的语义不一致：</strong>训练时，选中的 token 仍然能看见所有有效 token；原推理把未选 token 连同 K/V 一起移除，改变了注意力函数。简单说，“这一层不更新某个词”，不等于“不让其他词看到它”。修正后保留完整 K/V，只减少需要更新的 query。</li><li><strong>可以优化的执行开销：</strong>原实现逐样本查找索引、逐样本调用小矩阵；普通 forward 还计算训练统计和预算损失，包含无用的 CPU 同步。此次测量把统计与计时分开，另加批量执行版，实际收益见上面的同权重对照。</li><li><strong>设计本身的计算下限：</strong>路由仅跳过注意力，FFN 仍处理全部 token。在 BERT-base、长度 128 时，FFN 约占每层矩阵 FLOPs 的 64.86%；只要仍有查询需要更新，正确的 query 路由还保留完整 K/V 投影，约占 16.22%。因此，保留一半 token 不等于节省一半模型计算；批量补齐到 Kmax 又会增加计算。这是矩阵 FLOPs 的分析，不是延迟占比。跳过 FFN 会改变模型，需要重新训练和评估。</li></ol><p>此外，原稀疏写回在 BF16 下存在 dtype 不匹配；这不解释 FP32 测速的慢。零初始化配合 ≥0.5 阈值实际全保留，原注释中的“一半保留”不准确，但不代表训练无法获得路由梯度。</p>{audit}</section>
    {semantic_section(run,manifest)}
    <section><h2>重复波动与慢请求</h2><p>中位数改善不意味着最慢的一部分请求同样改善。P95 表示约 95% 的调用不超过的耗时；下表给出各轮分位数的范围，而非所有单次调用的最小/最大值。</p><ul>{''.join(tails)}</ul>{table(sorted(rows,key=lambda r:(r['batch_size'],PRIMARY.index(r['variant']))),ranges=True)}</section>
    {training_section(run)}{diagnostic_section(run,manifest)}
    <section><h2>这次结果能说明到哪一步</h2><p>它回答的是这组模型权重、这份 AG News 数据、这台笔记本和这些指定配置的实际取舍。单种子结果不能代表跨随机种子的稳定优势；整批早退受到组内最难样本影响；也尚未测生产服务、请求并发、量化或其他注意力后端。</p><p>新的结果不能替换成历史团队实验的 latency；原来报告中的理论 FLOPs 节省也不能改称速度提升。测试后的实现优化应以独立新实验记录，不回填旧表。</p><div class="links"><a href="../router_history.json">RouterBERT 训练记录</a><a href="../deebert_history.json">DeeBERT 训练记录</a><a href="../deebert_selection.json">DeeBERT 阈值选择</a></div><p class="note">结果协议：{esc(manifest['protocol_sha256'])}<br>完成记录：{esc(summary.get('completed_at','未记录'))}（本地实验时间）</p></section></main></body></html>"""
    destination = directory/"阅读版.html"
    destination.write_text(document, encoding="utf-8")
    return destination


def self_test():
    with tempfile.TemporaryDirectory(prefix="comparator_report_example_") as temp:
        run = Path(temp)
        directory = run/"benchmark"
        directory.mkdir()
        manifest = {"synthetic": True, "protocol_sha256": "SYNTHETIC-ONLY", "models": PRIMARY,
            "batch_sizes": [1,8,32], "shared_backbone_identity": {"verified_identical": True},
            "sample_indices": list(range(8)), "environment": {"gpu": "SYNTHETIC <fixture>"},
            "thresholds": {"deebert": .2}, "checkpoints": {"router_loop": "example-router"},
            "dataset_sha256": "example-data",
            "config": {"max_length":128,"warmup_batches":2,"benchmark_repeats":3,"benchmark_min_calls":8}}
        rows = []
        for batch in manifest["batch_sizes"]:
            for index,name in enumerate(PRIMARY):
                # Deliberately mixed speedups/slowdowns; text must derive its sign.
                p50 = batch*(1+index*.2)
                if name in {"router_batched","hdc_batched"}:
                    p50 = batch*.8
                row = {"variant":name,"batch_size":batch,"accuracy":.95-index*.001,
                    P50:p50,P95:p50*1.4,FLOPS:1e9*(1-index*.03),
                    "latency_p50_ms_by_repeat":[p50*.9,p50,p50*1.1],
                    "timed_inputs_matmul_flops_saving_percent":index*3,
                    "throughput_samples_per_second_mean":batch/p50*1000}
                rows.append(row)
                for repeat in range(3):
                    raw={"variant":name,"batch_size":batch,"repeat":repeat,
                         "latency":{"p50_ms":row["latency_p50_ms_by_repeat"][repeat],"p95_ms":p50*1.4}}
                    (directory/f"latency_{name}_b{batch}_r{repeat}.json").write_text(json.dumps(raw))
        summary={"synthetic":True,"protocol_sha256":"SYNTHETIC-ONLY","rows":rows}
        for name,data in (("manifest.json",manifest),("summary.json",summary)):
            (directory/name).write_text(json.dumps(data))
        (run/"deebert_selection.json").write_text(json.dumps({"constraint_met":True,"entropy_threshold":.2,"accuracy":.945}))
        semantic_dir=run/"semantic_diagnostic"
        semantic_dir.mkdir()
        semantic_manifest={"protocol_sha256":"semantic-example","checkpoint_sha256":"example-router",
                           "data_sha256":"example-data","batch_size":32}
        semantic_result={"protocol_sha256":"semantic-example","samples":10,
            "prediction_disagreements":3,"query_correct_legacy_wrong":2,"legacy_correct_query_wrong":1,
            "scope":"synthetic semantic example",
            "rows":[{"variant":"router_loop","batch_size":32,"samples":10,"correct":9,"accuracy":.9},
                    {"variant":"router_legacy","batch_size":32,"samples":10,"correct":8,"accuracy":.8}]}
        (semantic_dir/"manifest.json").write_text(json.dumps(semantic_manifest))
        (semantic_dir/"summary.json").write_text(json.dumps(semantic_result))
        try:
            build(run)
        except ValueError:
            pass
        else:
            raise AssertionError("Synthetic report published without opt-in")
        output=build(run,allow_example=True)
        contents=output.read_text(encoding="utf-8")
        assert "合成测试样例，不是实验结果" in contents
        assert "SYNTHETIC &lt;fixture&gt;" in contents
        assert "降低" in contents and "增加" in contents
        assert "3 条预测不同" in contents and "提高 10.00 个百分点" in contents
        assert (directory/"comparison.png").stat().st_size > 1000
        try:
            semantic_section(run,dict(manifest,checkpoints={"router_loop":"mismatched"}))
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong semantic checkpoint accepted")
        broken=dict(summary,protocol_sha256="wrong")
        try:
            validate(broken,manifest,directory,True)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched identity accepted")
    print("Synthetic comparator report tests passed; fixtures removed.")


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"outputs/comparator_audit_20261109")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(build(args.run_dir))


if __name__ == "__main__":
    main()
