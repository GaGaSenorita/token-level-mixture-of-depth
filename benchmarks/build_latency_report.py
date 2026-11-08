"""Build the Chinese reading report and standard static chart from completed runs.

No model imports, GPU access, training, or source-data mutation. Synthetic checks
are confined to a TemporaryDirectory and clearly marked as protocol examples.
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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, decimals=3):
    return "未记录" if value is None else f"{float(value):,.{decimals}f}"


def latency_change(row):
    reduction = float(row["latency_p50_reduction_percent"])
    return f"延迟降低 {reduction:.1f}%" if reduction > 0 else f"延迟增加 {-reduction:.1f}%" if reduction < 0 else "延迟持平"


def accuracy_change(row):
    difference = float(row["accuracy_delta_percentage_points"])
    return f"提高 {difference:.2f} 个百分点" if difference > 0 else f"下降 {-difference:.2f} 个百分点" if difference < 0 else "相同"


def flops_change(row):
    saving = float(row["timed_inputs_matmul_flops_saving_percent"])
    return f"减少 {saving:.1f}%" if saving > 0 else f"增加 {-saving:.1f}%" if saving < 0 else "持平"


def range_text(values, decimals=3):
    if not values:
        return "未记录"
    return f"{min(values):.{decimals}f}–{max(values):.{decimals}f}"


def validate(summary, manifest, allow_smoke):
    if summary.get("smoke") or manifest.get("smoke"):
        if not allow_smoke:
            raise ValueError("Refusing to publish a smoke/synthetic benchmark as a result report")
    if summary.get("protocol_sha256") != manifest.get("protocol_sha256"):
        raise ValueError("Summary and manifest protocol hashes differ")
    if not manifest.get("shared_backbone_identity", {}).get("verified_identical"):
        raise ValueError("Missing evidence of identical dense backbone/final-head weights")
    rows = summary["rows"]
    keys = [(row["variant"], row["batch_size"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate benchmark rows")
    required_batches = manifest["batch_sizes"]
    for batch in required_batches:
        for variant in ("dense", "query"):
            if (variant, batch) not in keys:
                raise ValueError(f"Missing completed row {variant}, batch={batch}")
    for row in rows:
        for key in ("accuracy", "latency_p50_ms_mean_of_repeats", "latency_p95_ms_mean_of_repeats", "timed_inputs_matmul_flops_per_sample"):
            value = row[key]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} for {row['variant']}")
        if not 0 <= row["accuracy"] <= 1:
            raise ValueError("Accuracy outside [0, 1]")
    return rows


def comparison_chart(rows, path, *, example=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_key = {(r["variant"], r["batch_size"]): r for r in rows}
    batches = sorted(b for variant, b in by_key if variant == "dense" and ("query", b) in by_key)
    x = np.arange(len(batches))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), layout="constrained")
    colours = {"dense": "#64748b", "query": "#087f8c"}
    labels = {"dense": "Dense BERT", "query": "HDC (query-sparse)"}
    for axis, metric, title, ylabel in (
        (axes[0], "latency_p50_ms_mean_of_repeats", "Measured model latency", "P50 latency (ms / batch)"),
        (axes[1], "timed_inputs_matmul_flops_per_sample", "Executed-path matrix computation", "Analytical GFLOPs / text"),
    ):
        for variant, offset in (("dense", -.19), ("query", .19)):
            values = [by_key[(variant, b)][metric] / (1e9 if axis is axes[1] else 1) for b in batches]
            error = None
            if axis is axes[0]:
                lows, highs = [], []
                for b, value in zip(batches, values):
                    repeats = by_key[(variant, b)].get("latency_p50_ms_by_repeat", [value])
                    lows.append(max(0, value - min(repeats)))
                    highs.append(max(0, max(repeats) - value))
                error = [lows, highs]
            bars = axis.bar(x + offset, values, .36, color=colours[variant], label=labels[variant],
                            yerr=error, capsize=3, error_kw={"elinewidth": 1.1, "ecolor": "#233243"})
            axis.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=9)
        axis.set_xticks(x, [f"Batch {b}" for b in batches])
        axis.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.set_ylim(bottom=0)
        axis.margins(y=.18)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e2e8f0", linewidth=.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#cbd5e1")
        axis.tick_params(colors="#334155")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("SYNTHETIC PROTOCOL EXAMPLE — NOT EXPERIMENT RESULTS" if example else "HDC-BERT: latency and computation under matched conditions", fontsize=13, weight="bold")
    fig.supxlabel("Same grouped timing inputs; latency error bars = repeat min–max. 1 MAC = 2 FLOPs. Lower bars are better.", fontsize=9, color="#475569")
    fig.savefig(path, dpi=180, facecolor="white", metadata={"Description": "Synthetic protocol example" if example else "Measured latency and executed-path analytical matmul FLOPs"})
    plt.close(fig)


def summary_csv(rows, path):
    columns = ["路径", "batch_size", "完整测试集准确率(%)", "准确率差(百分点)", "P50(ms/batch)", "P95(ms/batch)",
               "P50重复轮次最小(ms)", "P50重复轮次最大(ms)", "P50延迟降低(%)", "P50加速倍数", "测速同组批GFLOPs/text",
               "测速同组批矩阵FLOPs节省(%)", "全测试集GFLOPs/text", "吞吐量(texts/s)", "峰值已分配显存(GiB)", "测试样本数"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for r in rows:
            repeats = r.get("latency_p50_ms_by_repeat", [r["latency_p50_ms_mean_of_repeats"]])
            writer.writerow([
                r["variant"], r["batch_size"], 100*r["accuracy"], r["accuracy_delta_percentage_points"],
                r["latency_p50_ms_mean_of_repeats"], r["latency_p95_ms_mean_of_repeats"], min(repeats), max(repeats),
                r["latency_p50_reduction_percent"], r["latency_p50_speedup"], r["timed_inputs_matmul_flops_per_sample"]/1e9,
                r["timed_inputs_matmul_flops_saving_percent"], r["matmul_flops_per_sample"]/1e9,
                r["throughput_samples_per_second_mean"],
                "" if r.get("memory_peak_allocated_bytes") is None else r["memory_peak_allocated_bytes"]/2**30, r.get("samples", "")])


def main_table(rows):
    body = []
    for row in sorted(rows, key=lambda r: (r["batch_size"], r["variant"] != "dense")):
        label = "普通 BERT" if row["variant"] == "dense" else "HDC-BERT（修正推理）"
        saved = "—" if row["variant"] == "dense" else f"{row['timed_inputs_matmul_flops_saving_percent']:.1f}%"
        body.append(f"<tr><td>{label}</td><td>{row['batch_size']}</td><td>{row['accuracy']:.2%}</td>"
                    f"<td>{row['latency_p50_ms_mean_of_repeats']:.3f}</td><td>{row['latency_p95_ms_mean_of_repeats']:.3f}</td>"
                    f"<td>{row['timed_inputs_matmul_flops_per_sample']/1e9:.3f}</td><td>{saved}</td></tr>")
    return "<div class='table-wrap'><table><thead><tr><th>模型路径</th><th>Batch</th><th>完整测试集<br>准确率</th><th>P50<br>ms / batch</th><th>P95<br>ms / batch</th><th>矩阵 GFLOPs<br>/ text</th><th>矩阵 FLOPs<br>节省</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def repeat_table(rows):
    body = []
    for row in sorted(rows, key=lambda r: (r["batch_size"], r["variant"] != "dense")):
        label = "普通 BERT" if row["variant"] == "dense" else "HDC-BERT"
        memory = row.get("memory_peak_allocated_bytes")
        body.append(f"<tr><td>{label}</td><td>{row['batch_size']}</td><td>{esc(range_text(row.get('latency_p50_ms_by_repeat', [])))}</td>"
                    f"<td>{number(row['throughput_samples_per_second_mean'], 1)}</td><td>{esc(range_text(row.get('throughput_by_repeat', []), 1))}</td>"
                    f"<td>{number(None if memory is None else memory/2**30)}</td></tr>")
    return "<div class='table-wrap'><table><thead><tr><th>模型路径</th><th>Batch</th><th>各轮 P50 范围<br>ms / batch</th><th>平均吞吐量<br>texts / s</th><th>各轮吞吐量范围<br>texts / s</th><th>峰值已分配显存<br>GiB</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def training_section(run_dir, selection, config):
    rows, links = [], []
    for stage, name in ((1, "微调骨干与最终分类头"), (2, "训练早退出口"), (3, "训练 token 路由器")):
        path = run_dir / f"stage{stage}_history.json"
        if not path.exists():
            rows.append(f"<tr><td>{stage} · {name}</td><td colspan='3'>未提供阶段记录</td></tr>")
            continue
        history = read_json(path)
        if not history:
            rows.append(f"<tr><td>{stage} · {name}</td><td colspan='3'>阶段记录为空</td></tr>")
            continue
        if stage == 2:
            best = min(history, key=lambda x: x["validation_loss"])
            metric = f"验证损失 {best['validation_loss']:.4f}"
        else:
            best = max(history, key=lambda x: x["validation"]["accuracy"])
            metric = f"验证准确率 {best['validation']['accuracy']:.2%}"
        seconds = sum(float(x.get("seconds", 0)) for x in history)
        duration = f"{seconds/3600:.2f} 小时" if seconds else "未单独记录"
        rows.append(f"<tr><td>{stage} · {name}</td><td>{len(history)} 轮；选第 {best['epoch']} 轮</td><td>{metric}</td><td>{duration}</td></tr>")
        links.append(f"<a href='../{path.name}'>阶段 {stage} 训练记录</a>")
    limit = float(config.get("maximum_validation_accuracy_drop", .01)) * 100
    met = selection.get("constraint_met")
    constraint = (f"所选阈值满足“batch=1 的验证准确率较普通 BERT 最多下降 {limit:g} 个百分点”的预设要求。"
                  if met is True else
                  f"没有候选阈值满足“batch=1 的验证准确率最多下降 {limit:g} 个百分点”的预设要求，因此按预注册规则选择验证准确率最高的候选，并保留未达标结论。"
                  if met is False else "是否满足预设准确率要求，请核对阈值选择记录。")
    dense_validation = selection.get("dense_validation", {}).get("accuracy")
    selected_validation = selection.get("accuracy")
    metrics = (f"普通 BERT 验证准确率 {dense_validation:.2%}，所选 HDC 验证准确率 {selected_validation:.2%}。"
               if dense_validation is not None and selected_validation is not None else "")
    return ("<section><h2>模型和阈值怎样选出来</h2>"
            f"<p>在验证集选择训练轮次与熵阈值；测试集只用于最终评估。所选熵阈值为 <strong>{esc(selection['entropy_threshold'])}</strong>。{esc(constraint)}{esc(metrics)}</p>"
            "<p>这条准确率约束只用于 batch=1 的验证集选择，不保证最终测试集或 batch=8/32 的降幅也在同一范围内；主表直接报告各情形实测值。</p>"
            "<div class='table-wrap'><table><thead><tr><th>训练阶段</th><th>轮次</th><th>选择依据</th><th>已记录阶段耗时</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>"
            "<p>本次 Stage 2 缓存冻结、处于 eval 模式的骨干 CLS 特征，再训练早退出口；query 稀疏推理修复了原训练与推理的注意力差异。这些均属于本次补充复现的实现选择，不代表原团队历史实验采用同样流程。</p>"
            "<p class='muted'>阶段耗时按记录汇总；不包含未记录的数据下载、环境准备和全部后续评估时间。</p>"
            "<p class='links'><a href='../selection.json'>阈值选择与全部候选</a>" + "".join(links) + "</p></section>")


def diagnosis_section(run_dir, protocol):
    paths = sorted((run_dir / "diagnostics").glob("profile_b32_cuda_*/profile_summary.json"))
    if not paths:
        return ""
    path = paths[-1]
    record = read_json(path)
    if record.get("primary_protocol_sha256") != protocol:
        raise ValueError("Diagnostic belongs to a different benchmark protocol")
    rows = []
    for variant in record["variants"]:
        counts = {x["operator"]: x["count"] / variant["profiled_calls"] for x in variant["key_operators"]}
        label = "普通 BERT" if variant["variant"] == "dense" else "HDC-BERT"
        rows.append(f"<tr><td>{label}</td><td>{counts.get('aten::linear', 0):.0f}</td>"
                    f"<td>{counts.get('aten::bmm', 0):.0f}</td><td>{counts.get('aten::nonzero', 0):.0f}</td></tr>")
    timing_note = ("本机 CUPTI 没有成功采集 CUDA 内核时间；这里仅使用有效的 CPU 侧算子调用次数，不作 GPU 内核耗时占比判断。"
                   if not all(x.get("device_timing_captured") for x in record["variants"]) else
                   "诊断也记录了 CPU / GPU 事件，但不使用带 profiler 开销的时间替换正式时延。")
    relative = "../" + path.relative_to(run_dir).as_posix()
    return ("<section><h2>批量模式为什么少算却更慢</h2>"
            "<p>正式测速全部结束后，选取 batch=32 中最常见的执行路径的一批输入作独立诊断。下表是该批输入每次调用模型时的算子调用次数。</p>"
            "<div class='table-wrap'><table><thead><tr><th>路径</th><th>线性层调用</th><th>批矩阵乘法调用</th><th>保留位置查找</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>"
            "<p>HDC 的后半部分逐样本执行注意力计算，产生更多小矩阵调用和位置查找；前馈网络仍然处理完整长度。整批早退限制又让 batch=32 中的大多数样本走完整模型。这些实现开销有助于解释 FLOPs 下降而实际耗时增加的现象。</p>"
            f"<p class='muted'>{timing_note} 这是一个代表 batch 的诊断，不是全体输入的耗时分解。</p>"
            f"<p><a href='{esc(relative)}'>查看诊断记录与算子 CSV 索引</a></p></section>")


def build_report(run_dir, allow_smoke=False):
    run_dir = Path(run_dir).resolve()
    directory = run_dir / "benchmark"
    summary, manifest = read_json(directory / "summary.json"), read_json(directory / "manifest.json")
    selection = read_json(run_dir / "selection.json")
    if selection.get("entropy_threshold") != manifest.get("entropy_threshold"):
        raise ValueError("Selection and measured threshold differ")
    rows = validate(summary, manifest, allow_smoke)
    primary = [r for r in rows if r["variant"] in ("dense", "query")]
    by_key = {(r["variant"], r["batch_size"]): r for r in primary}
    if ("dense", 1) not in by_key or ("query", 1) not in by_key:
        raise ValueError("The reader report requires completed batch-1 comparison")
    d, h = by_key[("dense", 1)], by_key[("query", 1)]
    config, environment = manifest["config"], manifest["environment"]
    example = bool(summary.get("smoke") or manifest.get("smoke"))
    reduction, saving = h["latency_p50_reduction_percent"], h["timed_inputs_matmul_flops_saving_percent"]
    if reduction > 0:
        title = f"单条预测的中位延迟降低 {reduction:.1f}%"
    elif saving > 0:
        title = f"矩阵计算减少 {saving:.1f}%，单条预测的中位延迟未降低"
    else:
        title = "本次单条预测没有获得计算量与延迟的双重改善"
    if example:
        title = "合成协议检查示例 · 不是真实实验结果"
    conclusion = (f"在 batch=1 的单条输入下，普通 BERT 的 P50 为 <strong>{d['latency_p50_ms_mean_of_repeats']:.3f} ms</strong>，"
                  f"修正推理后的 HDC-BERT 为 <strong>{h['latency_p50_ms_mean_of_repeats']:.3f} ms</strong>，{latency_change(h)}。"
                  f"同一组测速输入的矩阵 FLOPs <strong>{flops_change(h)}</strong>；完整测试集准确率从 "
                  f"<strong>{d['accuracy']:.2%}</strong> 变为 <strong>{h['accuracy']:.2%}</strong>，{accuracy_change(h)}。")
    batch_notes = []
    for batch in sorted(b for variant, b in by_key if variant == "query"):
        if batch == 1:
            continue
        row = by_key[("query", batch)]
        batch_notes.append(f"<li><strong>一次处理 {batch} 条：</strong>{latency_change(row)}，同组批矩阵 FLOPs {flops_change(row)}，完整测试准确率{accuracy_change(row)}。</li>")
    legacy = [r for r in rows if r["variant"] == "legacy"]
    if legacy:
        item = legacy[0]
        legacy_text = (f"原先的紧凑 K×K attention 路径在 batch={item['batch_size']} 下：准确率 {item['accuracy']:.2%}，"
                       f"P50 {item['latency_p50_ms_mean_of_repeats']:.3f} ms/batch，同组批矩阵计算量 {item['timed_inputs_matmul_flops_per_sample']/1e9:.3f} GFLOPs/text。"
                       "这条路径让保留的 token 只关注其他保留的 token，与训练时的注意力语义不同。这里保留它作诊断，不把它混入修正实现的主结果。")
    else:
        legacy_text = "本次结果未提供原 K×K attention 路径的诊断数据。"
    gpu = environment.get("device_name", "未记录 GPU")
    max_length, repeats = manifest["max_length"], manifest["repeats"]
    seed = config.get("seed", "未记录")
    unique_count = len(manifest.get("sample_indices", []))
    calls = manifest.get("min_timed_calls", "未记录")
    tests = d.get("samples", "未记录")
    text_sections = []
    for result in summary.get("text_to_label", []):
        if result.get("variant") not in ("dense", "query") or "latency" not in result:
            continue
        label = "普通 BERT" if result["variant"] == "dense" else "HDC-BERT"
        text_sections.append(f"<li>{label}：P50 {result['latency']['p50_ms']:.3f} ms，P95 {result['latency']['p95_ms']:.3f} ms，{result['latency']['calls']} 条文本。</li>")
    local_pipeline = ("<details><summary>补充：从原始文本到标签的本地耗时</summary><p>包含分词、CPU→GPU 搬运、模型推理及取回标签；单次补充评估，不包含网络与服务排队。</p><ul>"
                      + "".join(text_sections) + "</ul></details>") if text_sections else ""
    shared = manifest["shared_backbone_identity"]
    constraint_notice = ("<p><strong>验证约束未满足：</strong>没有候选阈值达到预设的 batch=1 验证准确率要求；本次按预注册回退规则保留最高验证准确率的候选，结果如下。</p>"
                         if selection.get("constraint_met") is False else "")
    counts = read_json(run_dir / "data_manifest.json").get("counts", {}) if (run_dir / "data_manifest.json").exists() else {}
    data_sentence = (f"AG News：训练 {counts['train']:,} 条，验证 {counts['validation']:,} 条，测试 {counts['test']:,} 条。"
                     if all(k in counts for k in ("train", "validation", "test")) else f"AG News 完整测试集：{tests} 条。")
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HDC-BERT 实测结果｜{esc(summary.get('completed_at', ''))}</title>
<style>
:root{{--ink:#172b3a;--muted:#536778;--line:#dce5eb;--accent:#087f8c;--paper:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf2f5;color:var(--ink);font:16px/1.8 "Microsoft YaHei UI","Microsoft YaHei",system-ui,sans-serif}}
main{{max-width:1120px;margin:32px auto;background:var(--paper);padding:40px 44px;border:1px solid var(--line);border-radius:14px}}
.eyebrow{{font-size:13px;letter-spacing:.08em;color:var(--accent);font-weight:700}}h1{{font-size:30px;line-height:1.45;margin:12px 0 20px}}h2{{font-size:22px;line-height:1.5;margin:0 0 16px}}p{{margin:12px 0}}section{{margin-top:34px;padding-top:28px;border-top:1px solid var(--line)}}
.lead{{font-size:18px;background:#f0f8f8;border-left:4px solid var(--accent);padding:18px 20px;border-radius:4px}}.muted{{color:var(--muted);font-size:14px}}.meta{{display:flex;gap:8px;flex-wrap:wrap}}.meta span{{padding:3px 10px;background:#f3f6f8;border:1px solid var(--line);border-radius:20px;font-size:13px}}
.table-wrap{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6}}th,td{{padding:12px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:#f2f6f8;font-weight:650}}th:first-child,td:first-child{{text-align:left}}tbody tr:nth-child(even){{background:#f8fbfc}}
ul{{padding-left:23px}}li{{margin:10px 0}}figure{{margin:24px 0}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:14px;color:var(--muted);margin-top:10px}}strong{{font-weight:700}}a{{color:#056a82;text-underline-offset:3px}}.links{{display:flex;gap:16px;flex-wrap:wrap}}details{{background:#f8fafc;padding:16px 20px;border:1px solid var(--line);border-radius:8px;margin-top:20px}}summary{{cursor:pointer;font-weight:650}}code{{overflow-wrap:anywhere;font-size:13px}}footer{{margin-top:30px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}
@media(max-width:720px){{main{{margin:0;padding:24px 18px;border:0;border-radius:0}}h1{{font-size:25px}}.lead{{font-size:17px}}}}
@media print{{body{{background:white}}main{{border:0;margin:0;padding:0;max-width:none}}section,figure,tr{{break-inside:avoid}}details{{display:block}}}}
</style></head><body><main>
<div class="eyebrow">HDC-BERT · 本机补充实验 · 结果阅读版</div>
<h1>{esc(title)}</h1>
<div class="meta"><span>{esc(gpu)}</span><span>FP32 · eager attention · TF32 关闭</span><span>固定 padding {max_length}</span><span>训练 seed {esc(seed)}</span></div>
<p class="lead">{conclusion}</p>
{constraint_notice}
<p class="muted">这次从头训练后补测，不是历史权重的复测。{esc(data_sentence)} {esc(summary.get('completed_at', ''))} 完成评估。</p>

<section><h2>准确率、实际等待时间和计算量放在一起看</h2>
{main_table(primary)}
<p><strong>P50</strong> 表示一半调用可在该时间内完成；<strong>P95</strong> 表示 95% 的调用可在该时间内完成。表中为 {repeats} 轮各自分位数的平均值。<strong>ms/batch 是整批等待时间</strong>，batch=32 时不能把除以 32 后的数字当作单个用户的等待时间。</p>
<p><strong>较慢请求需要单独看：</strong>单条输入的 P95 分别为普通 BERT {d['latency_p95_ms_mean_of_repeats']:.3f} ms、HDC-BERT {h['latency_p95_ms_mean_of_repeats']:.3f} ms。中位延迟的收益不能直接推广到 P95。</p>
<p class="muted">准确率来自完整测试集。GFLOPs 来自与测速完全相同的随机输入及组批，统计在另一次不计时的遍历中完成；它和全测试集按原顺序组批的平均 FLOPs 分开保存。</p>
<p class="muted">FLOPs 节省为负数时表示计算量增加；准确率差使用百分点，不是相对百分比。</p>
<figure><img src="comparison.png" alt="普通 BERT 与 HDC-BERT 在各 batch 下的实际 P50 延迟和相同测速组批的矩阵 GFLOPs 对照"><figcaption>左图为实际时间，误差线表示各轮 P50 的最小值到最大值；右图为执行路径的矩阵计算量。两图都从零起算，单位分别为 ms/batch 和 GFLOPs/text。</figcaption></figure>
<ul>{''.join(batch_notes)}</ul>
<p>当前实现只有整批文本都达到置信条件才会早退。一批里出现较难样本时，其他样本也会继续计算。因此 batch 变化可能同时改变准确率、FLOPs 和延迟，不能只把单条结果按比例放大。</p>
</section>

<section><h2>重复测量是否稳定</h2>
{repeat_table(primary)}
<p>从测试集固定抽取 {unique_count} 个唯一样本，在每种 batch 下循环使用，达到每轮至少 {calls} 次计时调用。预热 {manifest['warmup']} 批后测 {repeats} 轮，交替模型顺序。范围是这些重复测量的实际跨度，不是统计置信区间，也不代表重新训练多个随机种子的波动。</p>
<p class="muted">吞吐量另开连续处理窗口测量。显存为 PyTorch 峰值已分配量，包含唯一驻留模型、测速输入和临时张量，不包含驱动和 CUDA 上下文；原始记录也保留基线与增量。</p>
{local_pipeline}
</section>

<section><h2>这次比较具体控制了什么</h2>
<ul>
<li><strong>同一套任务微调权重：</strong>普通 BERT 使用 Stage 1 骨干与最终分类头；HDC 只额外训练早退出口和路由器。已验证 {shared['tensor_count']} 个骨干及最终分类头张量逐字节相同，所以主结果衡量增加动态推理后的取舍。</li>
<li><strong>相同计算条件：</strong>同一 GPU、同一文本输入、固定 padding {max_length}、FP32、eager attention、关闭 TF32。两条路径都不执行未使用的 pooler。这里没有评估 SDPA 等额外优化的普通 BERT。</li>
<li><strong>训练与推理语义一致：</strong>修正实现只省去未保留 query 的 attention；被保留的 query 仍能关注全部有效文本的 key/value，padding key 被屏蔽。前馈网络仍处理全部 padding 后的位置。</li>
<li><strong>等待 GPU 真正完成：</strong>计时前后同步，计时段不收集路由统计、不读取标签、不加载模型。主表只包含模型推理；不包含分词、数据搬运、网络、服务排队或并发请求。</li>
</ul>
<p>矩阵 FLOPs 按实际访问层数和每条输入的保留 token 数计算，约定一次乘加为 2 FLOPs，包含路由器、分类头以及实际执行的 padding 投影和前馈计算。softmax、归一化、激活、索引与搬运没有计入该矩阵指标，但仍会影响实测延迟。因此这里的 FLOPs 不是全部硬件指令数，历史的 65.4% 也不会自动延用到这次结果。</p>
</section>

{diagnosis_section(run_dir, manifest['protocol_sha256'])}

{training_section(run_dir, selection, config)}

<details><summary>原推理路径的独立诊断</summary><p>{esc(legacy_text)}</p></details>

<section><h2>可核查的原始记录</h2>
<p class="links"><a href="summary.csv">下载对照表 CSV</a><a href="summary.json">完整汇总 JSON</a><a href="manifest.json">硬件、配置与权重指纹</a><a href="report.md">技术说明</a></p>
<p>同目录下的 <code>latency_*.csv</code> 保留逐次计时，<code>*_routes.jsonl</code> 保留计数所用路径，<code>quality_*_predictions.csv</code> 保留完整测试预测。主结果以新训练的单个 seed 为范围，不把推理重复测量写成多次独立训练，也不把本地基准称为上线后的服务 latency。</p>
<p class="muted">实验标识：<code>{esc(config.get('experiment_id', '未记录'))}</code><br>协议指纹：<code>{esc(manifest['protocol_sha256'])}</code></p>
</section>
<footer>本报告由已完成的实验记录生成；图表与文字共用同一份 summary.json。{'当前页面为合成检查示例，所有数字仅用于验证排版。' if example else ''}</footer>
</main></body></html>"""
    comparison_chart(primary, directory / "comparison.png", example=example)
    summary_csv(rows, directory / "summary.csv")
    destination = directory / "阅读版.html"
    destination.write_text(body, encoding="utf-8")
    return destination


def self_test():
    """Exercise faster/slower conclusions and artifact schemas without any GPU."""
    with tempfile.TemporaryDirectory(prefix="hdc_report_synthetic_") as temporary:
        run_dir = Path(temporary)
        directory = run_dir / "benchmark"
        directory.mkdir()
        rows = []
        for batch in (1, 8, 32):
            for variant in ("dense", "query"):
                dense_latency = {1: 8, 8: 17, 32: 46}[batch]
                ratio = 1 if variant == "dense" else {1: .6, 8: 1.2, 32: 1.5}[batch]
                latency = dense_latency * ratio
                flop_ratio = 1 if variant == "dense" else {1: .45, 8: .8, 32: .92}[batch]
                rows.append({"variant": variant, "batch_size": batch, "accuracy": .94 if variant == "dense" else .935,
                             "accuracy_delta_percentage_points": 0 if variant == "dense" else -.5,
                             "latency_p50_ms_mean_of_repeats": latency, "latency_p95_ms_mean_of_repeats": latency*1.2,
                             "latency_p50_ms_by_repeat": [latency*.97, latency, latency*1.03],
                             "latency_p50_reduction_percent": 100*(1-ratio), "latency_p50_speedup": 1/ratio,
                             "timed_inputs_matmul_flops_per_sample": 20e9*flop_ratio,
                             "timed_inputs_matmul_flops_saving_percent": 100*(1-flop_ratio),
                             "matmul_flops_per_sample": 20e9*flop_ratio*1.01,
                             "throughput_samples_per_second_mean": 1000*batch/latency,
                             "throughput_by_repeat": [980*batch/latency, 1000*batch/latency, 1020*batch/latency],
                             "memory_peak_allocated_bytes": int(1.5*2**30), "samples": 7600})
        manifest = {"protocol_sha256": "SYNTHETIC_ONLY", "smoke": True, "batch_sizes": [1, 8, 32],
                    "config": {"seed": 42, "experiment_id": "SYNTHETIC_ONLY"}, "entropy_threshold": .2,
                    "environment": {"device_name": "Synthetic RTX 5050 example"}, "shared_backbone_identity": {"verified_identical": ["stage1", "stage3"], "tensor_count": 201},
                    "max_length": 128, "repeats": 3, "warmup": 30, "min_timed_calls": 1024, "sample_indices": list(range(1024))}
        summary = {"protocol_sha256": "SYNTHETIC_ONLY", "smoke": True, "rows": rows, "completed_at": "SYNTHETIC", "text_to_label": []}
        for name, value in (("manifest", manifest), ("summary", summary)):
            (directory / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
        (run_dir / "selection.json").write_text(json.dumps({"entropy_threshold": .2, "constraint_met": True, "accuracy": .935, "dense_validation": {"accuracy": .94}}), encoding="utf-8")
        for stage in (1, 2, 3):
            history = [{"epoch": 1, "validation_loss": .3}] if stage == 2 else [{"epoch": 1, "validation": {"accuracy": .94}, "seconds": 3600}]
            (run_dir / f"stage{stage}_history.json").write_text(json.dumps(history), encoding="utf-8")
        try:
            build_report(run_dir)
        except ValueError as error:
            assert "smoke/synthetic" in str(error)
        else:
            raise AssertionError("Synthetic data was accepted without explicit allowance")
        result = build_report(run_dir, allow_smoke=True)
        content = result.read_text(encoding="utf-8")
        assert "不是真实实验结果" in content and "延迟降低 40.0%" in content and "延迟增加 20.0%" in content
        assert "ms / batch" in content and "summary.json" in content and "../selection.json" in content
        assert (directory / "comparison.png").stat().st_size > 20000
        with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            table = list(csv.reader(stream))
        assert len(table) == 7 and len(table[0]) == 16
        # Exercise the no-speedup wording without creating a formal result.
        row = next(r for r in rows if r["variant"] == "query" and r["batch_size"] == 1)
        row["latency_p50_reduction_percent"] = -20
        assert latency_change(row) == "延迟增加 20.0%"
    print("PASS: temporary synthetic HTML/PNG/CSV, safety gate, mixed speed outcomes, relative evidence links. No model or GPU was used.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/latency_20261108"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true", help="Clearly watermark smoke/example reports; never use to represent real results")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(build_report(args.run_dir, allow_smoke=args.allow_smoke))


if __name__ == "__main__":
    main()
