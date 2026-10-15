"""Standalone Chinese LaTeX supplement; all quantitative tables derive from receipts."""

from __future__ import annotations

from pathlib import Path

from .analysis import pilot, write_table
from .core import METHODS, OUT, PROJECT, read, save, source_hashes, verify_sources

NAMES = dict(lda="LDA", btm="BTM", gsdmm="GSDMM", nmf="NMF", bertopic="BERTopic", cst_mil_r0="CST")


def esc(value):
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(mapping.get(char, char) for char in str(value))


def f(value, digits=4):
    return "--" if value is None else f"{value:.{digits}f}"


def table(headers, rows, widths=None):
    spec = widths or ("l" + "r" * (len(headers) - 1))
    environment = "longtable" if len(rows) > 24 else "tabular"
    return (
        "\\begin{center}\\small\n\\begin{"
        + environment
        + "}{"
        + spec
        + "}\n\\toprule\n"
        + " & ".join(headers)
        + r" \\ \midrule"
        + ("\n\\endhead\n" if environment == "longtable" else "\n")
        + "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
        + "\n\\bottomrule\n\\end{"
        + environment
        + "}\n\\end{center}\n"
    )


def build(out=OUT):
    gemini, local = pilot(out)
    quote = read(out / "gemini/quote.json")
    hardware = read(out / "hardware.json")
    heads = read(out / "robustness/summary.json")
    review_path = out / "gemini/reason_review.json"
    if not review_path.exists():
        raise ValueError("Complete all textual reason reviews before building final report")
    reason_review = read(review_path)
    if len(reason_review["sessions"]) != gemini["n_valid"]:
        raise ValueError("Reason review coverage mismatch")
    report_dir = PROJECT / "reports/专题研究报告/第二轮补充报告_20260929"
    report_dir.mkdir(parents=True, exist_ok=True)
    cst = next(r for r in local if r["method"] == "cst_mil_r0")
    ratio = gemini["latency"]["median_seconds"] / cst["resident_median_seconds"]
    errors = gemini["errors"]
    nonexistent = sum(r.get("nonexistent_id_count") or 0 for r in errors)
    fabricated = sum(r.get("fabricated_quote_count") or 0 for r in errors)
    class_errors = sum(r["classification_error"] is True for r in errors)
    small_count = sum(r["interpretation"] == "small_exploratory" for r in heads["contrasts"])
    selected_cst = next(r for r in heads["selected_configurations"] if r["method"] == "cst_mil_r0")
    selected_best_test = max(heads["selected_configurations"], key=lambda r: r["test_ap_mean"])
    sections = [
        r"""\documentclass[UTF8,a4paper,zihao=-4,fontset=fandol]{ctexart}
\usepackage[margin=23mm,headheight=15pt]{geometry}
\usepackage{lmodern}
\usepackage{amsmath,booktabs,longtable,array,tabularx,enumitem,fancyhdr,xcolor,setspace}
\usepackage[colorlinks=true,linkcolor=blue!40!black,urlcolor=blue!40!black]{hyperref}
\setstretch{1.18}\setlength{\emergencystretch}{3em}
\setlist{nosep,leftmargin=2em}\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{短文本风险筛查：第二轮实验补充}\fancyhead[R]{固定协议结果}
\fancyfoot[C]{\thepage}
\title{CST-MIL 第二轮实验补充报告\\\large 小规模大模型对照与风险层稳健性}
\author{CST-MIL 研究项目}\date{2026年9月29日固定协议}
\begin{document}\maketitle
\begin{abstract}
本报告回答在风险排序效果、响应时间、逐请求费用和部署条件之间，CST适合何种使用场景。
原SCCD的406/135/136划分、冻结CST-r0及第一轮证据保留。
本轮额外执行6个开发会话和30个测试会话的Gemini试点、相同30条的六本地方法计时、
以及5个随机种子的110组风险头对照。
结论依据固定实验清单，不依据测试成绩反向选择样本、随机种子或有利配置。
\end{abstract}
\setcounter{tocdepth}{1}\tableofcontents\clearpage
\section{主要发现与适用场景}
"""
    ]
    sections.append(
        f"在本次QuickRouter接口与本地4个逻辑CPU条件下，相同30条首次推理累计耗时为"
        f"Gemini {f(gemini['latency']['total_seconds'], 2)}秒、CST "
        f"{f(cst['first_pass_total_seconds'], 2)}秒，本地模型载入时间另列。"
        f"Gemini测试会话的端到端中位延迟为"
        f"{f(gemini['latency']['median_seconds'], 3)}秒；CST驻留模型重复推理的中位延迟为"
        f"{f(cst['resident_median_seconds'], 3)}秒，二者比值约{f(ratio, 1)}。"
        f"相同30条测试上的AP分别为{f(gemini['metrics']['ap'])}和{f(cst['ap'])}。"
        "这些数字支持在已有训练标注的条件下，面向本地部署、持续批量筛查和对响应延迟敏感任务的成本收益判断。"
        "30条探索性试点不能替代完整136条测试，也不能证明所有大模型具有相同速度。\n\n"
        f"证据定位呈现不同的取舍：Gemini的Top-3命中为{gemini['metrics']['top3_hits']}/"
        f"{gemini['metrics']['top3_eligible_sessions']}，CST为{cst['top3_hits']}/{cst['top3_eligible']}。"
        "因此，本轮支持CST在排序与本地响应方面的应用价值，同时保留Gemini在该子集证据命中上的较高结果。\n\n"
        f"全部36条API调用按最高公开分组费率计的token费用估算为"
        f"{f(gemini['all_cost_upper_rate_usd'], 4)}美元；该试点金额较小，不能称为高额成本。"
        "本地推理无需逐请求支付外部API费用，但有训练、硬件、内存、存储与维护成本。"
        "选择CST的理由应是这些约束下的实测组合，而非预设其在所有指标上最优。\n\n"
        f"22项风险层指标对照中，有{small_count}项同时满足预设区间与排名保持的小影响条件。"
        f"CST按五种子平均开发AP选出的会话分类器为{selected_cst['session_head'].upper()}。"
        f"其测试AP均值为{f(selected_cst['test_ap_mean'])}；验证规则选出的六套配置中，"
        f"测试AP最高点估计来自{NAMES[selected_best_test['method']]}，为"
        f"{f(selected_best_test['test_ap_mean'])}。"
        "验证集选型不保证测试优势；前述Gemini与CPU对照仍对应原冻结CST-r0，"
        "不能与本节的新配置拼接成同一套最优结果。"
        "未通过等效判断时，"
        "不把统计上不显著解释为影响可忽略。完整配置与全部种子数据位于附录及对应JSON/CSV。\n"
    )
    sections.append(r"""
\section{统一研究设计与指标}
\subsection{数据单位、拟合范围与隔离}
SCCD以完整会话为单位；主题表示仅由训练分区拟合。六个本地模型沿用第一轮冻结参数。
新数据重新训练六方法，种子42、最多4个逻辑CPU；所有正式计算串行。
训练会话分类器使用按完整会话交叉拟合的风险预测，分类阈值只在开发集选择。
Gemini使用单独固定0.5阈值，6个开发样本仅检查接口和格式，不反复调整提示词。
来源代码、原数据、模型和历史结果以哈希保护；检测到影响实验的源码变化时停止受影响阶段。
\subsection{AP、AUC与证据指标}
设按风险分数阈值由高到低得到精确率$P_j$和召回率$R_j$，则
\[\mathrm{AP}=\sum_j(R_j-R_{j-1})P_j.\]
它衡量逐步纳入高分会话时的正类集中程度，召回增加较多的阈值贡献较大；
AP依赖正类比例，本轮平衡子集的AP不能直接解释为真实部署比例下的效果。
ROC-AUC为随机正、负会话成对时，正类分数高于负类的概率，平分记半：
\[\mathrm{AUC}=\Pr(s^+>s^-)+\tfrac12\Pr(s^+=s^-).\]
两者均考察排序，不要求概率已校准。macro-F1对两类F1取均值。
Top-3命中率的分母为含危险评论金标准的会话数，分子为前三候选至少包含一条危险评论的会话数。
危险评论召回率的分母为全部危险评论数量，分子为前三候选中实际覆盖的数量，不能与会话命中率混淆。
\section{Gemini与六方法的相同子集比较}
\subsection{抽样、调用与费用记录}
只用406个训练会话计算总文本字符长度的两个三分位边界。按正负标签与短中长形成六层，
每层开发1条、测试5条，以固定哈希顺序选择。请求输入完整消息、编号和真实回复关系，排除标签。
接口固定为QuickRouter的OpenAI兼容端点，模型显式指定\texttt{gemini-2.5-flash}，
不使用凭证文件里的默认模型，也不自动切换模型。完整请求签名、逐条回执、token和错误保存。
Gemini为未提供SCCD标注示例的零样本直接调用；本地模型使用406个标注训练会话并在135个开发会话上选择阈值。
这里比较两种给定的使用流程，监督信息和准备成本不同；标注及维护劳动未折算为费用。
最多两次尝试，只对超时、429、临时服务端错误重试；预算2美元、连续三次基础设施失败或身份不符即停止。
公开价格来自\href{https://api.quickrouter.ai/api/pricing}{QuickRouter价格接口}。
实际账户分组未知，因此按最高公开费率保守计费；金额属于token费用估算，账户实际扣款未知。
""")
    sections.append(
        "本机处理器为"
        + esc(hardware["cpu"]["Name"].strip())
        + "，系统可见物理内存约"
        + f(hardware["physical_ram_bytes"] / 2**30, 2)
        + "GiB；运行Windows 11。"
        "实验将进程绑定到4个逻辑CPU，并将数值库线程上限设为4。\n"
    )
    sections.append(
        f"调用前报价复核时间为{esc(quote['checked_utc'])}。采用输入"
        f"{f(quote['highest']['input_usd_per_million'], 6)}美元/百万token、输出"
        f"{f(quote['highest']['output_usd_per_million'], 6)}美元/百万token。"
        f"按完整输入UTF-8字节数保守预留输入token、输出上限{quote['output_token_cap']}token，"
        f"36条单轮预留约{f(quote['one_pass_upper_usd'], 4)}美元，最多两次尝试约"
        f"{f(quote['two_attempt_upper_usd'], 4)}美元。预留上界与实际返回token估算分别保留。\n"
    )
    sections.append(
        f"本次具有有效JSON风险分数的输出{gemini['n_valid']}/{gemini['n_test']}；"
        f"返回至少min(3,消息数)条候选的会话为{gemini['complete_top3_sessions']}/{gemini['n_test']}。"
        "提示词使用“最多3条”，所以不足3条作为候选数量统计，未判为JSON格式失败；"
        "这不是强制返回3条协议下的失败率，也不能与伪造摘录混为一谈。全部请求数"
        f"{gemini['all_attempts']}。36条总输入/输出token为"
        f"{gemini['all_prompt_tokens']}/{gemini['all_completion_tokens']}；"
        f"30条测试费用估算为{f(gemini['test_cost_upper_rate_usd'], 4)}美元。\n"
    )
    rows = [
        [
            NAMES[r["method"]],
            f(r["ap"]),
            f(r["auc"]),
            f(r["macro_f1"]),
            f"{r['top3_hits']}/{r['top3_eligible']}",
            f(r["dangerous_comment_recall"]),
        ]
        for r in local
    ]
    m = gemini["metrics"]
    rows.append(
        [
            "Gemini",
            f(m["ap"]),
            f(m["auc"]),
            f(m["macro_f1"]),
            f"{m['top3_hits']}/{m['top3_eligible_sessions']}",
            f(m["dangerous_comment_recall"]),
        ]
    )
    sections.append(table(["方法", "AP", "AUC", "macro-F1", "Top-3命中", "评论召回"], rows))
    unit_fees = [
        (
            gemini["test_prompt_tokens"] * group["input_usd_per_million"]
            + gemini["test_completion_tokens"] * group["output_usd_per_million"]
        )
        / (gemini["n_test"] * 1_000_000)
        for group in quote["groups"]
    ]
    fee_scenarios = [
        dict(sessions=n, public_min_usd=n * min(unit_fees), public_max_usd=n * max(unit_fees))
        for n in [1000, 10000, 100000]
    ]
    save(
        out / "gemini/fee_scenarios.json",
        dict(
            scenarios=fee_scenarios,
            based_on_test_sessions=gemini["n_test"],
            assumptions=(
                "Same mean input/output tokens as this test sample; frozen public rates; "
                "no retries, discounts or other fees; account group and debit unknown"
            ),
            not_additional_calls=True,
        ),
    )
    sections.append(
        "\\subsection{规模化费用情景}\n"
        "以下固定每条输入/输出token均值为本次30条测试的实测均值，"
        "沿用报价快照，不计重试、折扣或其他费用，列出公开分组的最低至最高费率范围。"
        "这是算术情景估算，未执行新增请求，也不是实际账户账单。"
        "本地方案仍须承担硬件、训练、标注及维护成本，现有数据不足以估计完整盈亏平衡点。\n"
    )
    sections.append(
        table(
            ["会话量", "最低公开费率下美元", "最高公开费率下美元"],
            [
                [r["sessions"], f(r["public_min_usd"], 2), f(r["public_max_usd"], 2)]
                for r in fee_scenarios
            ],
        )
    )
    sections.append(r"""\subsection{响应时间与部署成本}
每个本地方法在新进程加载冻结模型，模型载入时间单列；首次整批推理后，再对相同30条做三次驻留重复。
载入计时从joblib反序列化开始，不包含解释器启动和入口已导入的公共Python依赖；首次调用仍可能触发后端懒加载。
表中驻留中位数/P95来自90个逐会话时间。BERTopic首次使用新空嵌入缓存，重复阶段允许命中该缓存；
因此重复时间不等于面对全新文本的冷推理吞吐率。原生后端每会话的子进程开销包含在时间内。
更换后端实现或采用常驻原生服务会改变这部分成本，测得倍数不代表算法的固有效率差距。
API只执行单轮，不重复调用测缓存；其延迟包含中转、网络、服务端排队与生成，不能分离为纯模型计算时间。
本地输出主题与候选消息，Gemini还生成简短自然语言理由；计时包含各自的输出流程。
计时阶段暂停本轮的数据下载与其他训练；操作系统、编辑器及其他项目后台活动未完全隔离，仍可能造成波动。
BERTopic首轮触发了原$10^{-6}$数值一致性检查：818个缓存向量有816个逐位一致，
另外2个float32向量最大差约$5.22\times10^{-8}$，会话分数最大差约$2.81\times10^{-6}$。
核对30条的排序、分类、Top-3和主题编号均未变化后，保留原首轮记录，
在独立修订目录重新完成四轮计时；修订规则要求概率差不大于$10^{-5}$，同时上述离散输出及排序严格一致。
这是数值复现检查修订，没有调整模型、样本或分类阈值。表中BERTopic采用完整修订轮次。
""")
    rows = [
        [
            NAMES[r["method"]],
            f(r["load_seconds"], 3),
            f(r["first_session_seconds"], 3),
            f(r["first_pass_total_seconds"], 2),
            f(r["resident_median_seconds"], 3),
            f(r["resident_p95_seconds"], 3),
        ]
        for r in local
    ]
    rows.append(
        [
            "Gemini",
            "未知",
            "--",
            f(gemini["latency"]["total_seconds"], 2),
            f(gemini["latency"]["median_seconds"], 3),
            f(gemini["latency"]["p95_seconds"], 3),
        ]
    )
    sections.append(table(["方法", "加载秒", "首条秒", "首次30条秒", "中位秒", "P95秒"], rows))
    sections.append(
        "Gemini最后两列为单轮API延迟；本地两列为三轮驻留重复。首次整批总时间与驻留中位时间口径不同。\n"
    )
    sections.append(
        table(
            ["方法", "模型MiB", "额外依赖MiB", "推理峰值MiB", "历史训练秒"],
            [
                [
                    NAMES[r["method"]],
                    f(r["model_mib"], 2),
                    f(r["historical_extra_dependencies_mib"], 2),
                    f(r["peak_rss_mib"], 1),
                    f(r["historical_training_seconds"], 1),
                ]
                for r in local
            ],
        )
    )
    sections.append(r"""上述本地峰值包括当前Python进程及子进程，每50毫秒采样，短暂尖峰可能遗漏。
额外依赖和历史训练来自第一轮账本：BERTopic包含编码器，GSDMM包含Java运行时；
通用Python环境和操作系统未计入“模型MiB”。本地均用CPU，不要求GPU。
Gemini服务端模型大小、内存和GPU使用情况未公开，均记未知。
\subsection{错误与理由支持程度}
""")
    sections.append(
        f"自动检查发现不存在消息编号{nonexistent}处、非原文连续片段摘录{fabricated}处；"
        f"固定阈值分类错误{class_errors}个会话，单独统计。"
        "理由另按完整会话逐条进行文本审核，审核者为Codex，未冒充独立人工金标准。"
        "支持、不支持与不确定三类判断及依据均逐项保存。下表以实际返回的75条理由为单位；"
        "未返回理由的三个会话没有被计为有原文支持。\n"
    )
    sections.append(
        table(
            ["审核类别", "条数"],
            [
                [
                    {
                        "supported": "有原文支持",
                        "unsupported": "缺乏原文支持",
                        "uncertain": "解释不确定",
                    }.get(k, esc(k)),
                    v,
                ]
                for k, v in reason_review["counts"].items()
            ],
        )
    )
    sections.append(r"""
\section{风险层稳健性：110组预设配置}
五个基线各进行实例LR/RF与会话LR/RF的$2\times2$对照；CST实例层保留MIL机制，只替换会话头。
种子42--46同时传入分折与分类器，所有种子均保留。
实例LR与RF使用相同显式会话平衡权重，RF不再额外叠加class\_weight；
会话LR与RF统一使用balanced类别权重。
主题表示固定，因此比较回答给定表示下风险头的敏感性，未重采样主题模型学习过程。
区间条件于给定训练集、模型和配置，只重采样测试会话，不包含重新训练或重新选型的不确定性。

最终配置按五种子平均开发AP选择，其次开发AUC，再优先更简单的头；测试集不参与配置选择。
表中AP给出全部种子的均值与样本标准差，AUC与macro-F1列出均值，不挑最好种子。
前文30条延迟对应原冻结full677配置；本节重新选出的风险头配置未另行测量部署延迟。
""")
    sections.append(
        table(
            ["方法", "实例/会话", "开发AP", "测试AP均值$\\pm$SD", "测试AUC", "macro-F1"],
            [
                [
                    NAMES[r["method"]],
                    r["instance_head"].upper() + "/" + r["session_head"].upper(),
                    f(r["dev_ap_mean"]),
                    f(r["test_ap_mean"]) + r"$\pm$" + f(r["test_ap_sd"]),
                    f(r["test_roc_auc_mean"]),
                    f(r["test_macro_f1_mean"]),
                ]
                for r in heads["selected_configurations"]
            ],
        )
    )
    sections.append(r"""
\subsection{预设配对对照与解释}
每个方法比较会话RF减LR（实例头固定LR）；五个基线比较实例RF减LR（会话头固定LR）。
AP和AUC组成22项预设对照。按会话标签分层、所有方法/配置/种子共用相同重采样索引，
进行10000次配对bootstrap。统计量为每种子指标差的均值，不是先平均概率再算指标。
普通95\%区间与Bonferroni家族95\%区间同时报告；后者尾部分位估计仍有Monte Carlo误差。
探索性小影响界限为绝对值$\pm0.01$，尚未验证为业务容忍标准。
只有调整区间完全位于界限内且比较范围内主要排名保持，才作限定的小影响判断；
不满足时报告存在影响或证据不足，不追加搜索直至出现想要的结论。
""")
    rows = []
    for r in heads["contrasts"]:
        ci = r["bonferroni_family_ci95"]
        rows.append(
            [
                NAMES[r["method"]],
                r["factor"],
                r["metric"].replace("roc_auc", "AUC").upper(),
                f(r["estimate"]),
                f"[{f(ci[0])}, {f(ci[1])}]",
            ]
        )
    sections.append(table(["方法", "变化层", "指标", "差值", "家族95\\%区间"], rows, "lllrr"))
    labels = {}
    for r in heads["contrasts"]:
        labels[r["interpretation"]] = labels.get(r["interpretation"], 0) + 1
    label_names = {
        "small_exploratory": "满足探索性小影响条件",
        "meaningful_positive": "正向变化完整超过界限",
        "meaningful_negative": "负向变化完整超过界限",
        "inconclusive": "不足以判定影响小或完整超过界限",
    }
    sections.append(
        "按预设规则，在22项指标对照中，"
        + "；".join(label_names[k] + str(v) + "项" for k, v in labels.items())
        + "。\n"
    )
    zero_excluded = [
        r
        for r in heads["contrasts"]
        if r["bonferroni_family_ci95"][0] > 0 or r["bonferroni_family_ci95"][1] < 0
    ]
    if zero_excluded:
        sections.append(
            "这里的“证据不足”针对预设小影响或完整超过界限的判别，"
            "不等同于所有差值区间都包含零。例如，"
            + "；".join(
                NAMES[r["method"]]
                + {"instance": "实例层", "session": "会话层"}[r["factor"]]
                + r["metric"].replace("roc_auc", "AUC").upper()
                + "的家族区间["
                + ", ".join(f(x) for x in r["bonferroni_family_ci95"])
                + "]排除零"
                for r in zero_excluded
            )
            + "。是否跨越零与是否完整落入或越过探索性正负0.01界限，是不同判断。\n"
        )
    rank_checks = {}
    for row in heads["contrasts"]:
        rank_checks.setdefault((row["factor"], row["metric"]), row["rankings"])
    sections.append(
        table(
            ["变化层", "指标", "均值排名保持", "发生排名变化的种子数"],
            [
                [
                    {"session": "会话层", "instance": "实例层"}[factor],
                    metric.replace("roc_auc", "AUC").upper(),
                    "是" if ranks[0]["stable"] else "否",
                    str(sum(not r["stable"] for r in ranks[1:])) + "/5",
                ]
                for (factor, metric), ranks in rank_checks.items()
            ],
            "llcr",
        )
    )
    sections.append(
        "会话层排名比较包含六方法，实例层只含五基线；"
        "均值与每个种子均检查完整方法顺序，出现任一逆转便不称排名保持。\n"
    )
    cst_effect = next(
        r for r in heads["contrasts"] if r["method"] == "cst_mil_r0" and r["metric"] == "ap"
    )
    sections.append(
        "CST仅更换会话分类器时，五种子平均AP变化为"
        + f(cst_effect["estimate"])
        + "，家族95\\%区间为["
        + ", ".join(f(x) for x in cst_effect["bonferroni_family_ci95"])
        + "]。"
        + (
            "该对照满足预设探索性小影响条件。"
            if cst_effect["interpretation"] == "small_exploratory"
            else "该结果不能支持将此范围内的风险层影响概括为可忽略。"
        )
        + "\n"
    )
    sections.append(r"""
第一轮\texttt{no\_reply}同时移除了父回复文本与六项结构统计，含邻接和回复统计，
应称为组合消融。其差异不能解释为纯回复边的因果贡献。本轮不改写该历史实验。
\section{可复现性、限制与交付索引}
新增代码：\par\noindent{\small\path{src/cst_mil/research_followup/round2/}}\par
实验结果：\par\noindent{\small\path{benchmark/research_followup/20260929/}}\par
\texttt{protocol.json}为固定协议，\texttt{task\_plan.md}为执行记录，
\texttt{REPRODUCE.md}给出逐阶段命令。
完整预测与成本在\texttt{local\_timing/}和\texttt{gemini/}，
全风险层种子预测、分折、权重及区间位于\texttt{robustness/}。
费用未获账户账单核验；理由支持程度为模型辅助文本审阅，尚无独立人工一致性评估。
单机重复计时不能覆盖生产并发条件，也不能替代真实新平台会话验证。
\appendix\section{全部风险层配置均值}
""")
    sections.append(
        table(
            ["方法", "实例/会话", "开发AP", "测试AP$\\pm$SD", "测试AUC", "macro-F1"],
            [
                [
                    NAMES[r["method"]],
                    r["instance_head"].upper() + "/" + r["session_head"].upper(),
                    f(r["dev_ap_mean"]),
                    f(r["test_ap_mean"]) + r"$\pm$" + f(r["test_ap_sd"]),
                    f(r["test_roc_auc_mean"]),
                    f(r["test_macro_f1_mean"]),
                ]
                for r in heads["configurations"]
            ],
        )
    )
    all_runs = [read(p) for p in sorted((out / "robustness/runs").glob("*/result.json"))]
    write_table(
        out / "robustness/full_runs.csv",
        [
            dict(
                method=r["method"],
                instance_head=r["instance_head"],
                session_head=r["session_head"],
                seed=r["seed"],
                seconds=r["seconds"],
                seconds_scope=r["cost_note"],
                peak_rss_mib=r["peak_rss_mib"],
                **{
                    split + "_" + metric: r[split + "_metrics"][metric]
                    for split in ["dev", "test"]
                    for metric in ["ap", "roc_auc", "macro_f1"]
                },
                crossfit_membership_sha256=r["crossfit_membership_sha256"],
            )
            for r in all_runs
        ],
    )
    write_table(
        out / "robustness/configurations.csv",
        [{k: v for k, v in r.items() if not isinstance(v, list)} for r in heads["configurations"]],
    )
    write_table(
        out / "robustness/contrasts.csv",
        [
            dict(
                method=r["method"],
                factor=r["factor"],
                metric=r["metric"],
                estimate=r["estimate"],
                ci95_low=r["percentile_ci95"][0],
                ci95_high=r["percentile_ci95"][1],
                family_ci95_low=r["bonferroni_family_ci95"][0],
                family_ci95_high=r["bonferroni_family_ci95"][1],
                rankings_stable=r["rankings_stable"],
                interpretation=r["interpretation"],
            )
            for r in heads["contrasts"]
        ],
    )
    sections.append("\\clearpage\\section{全部110组种子结果}\n")
    sections.append(
        table(
            ["方法", "实例/会话", "种子", "开发AP", "测试AP", "测试AUC", "macro-F1"],
            [
                [
                    NAMES[r["method"]],
                    r["instance_head"].upper() + "/" + r["session_head"].upper(),
                    r["seed"],
                    f(r["dev_metrics"]["ap"]),
                    f(r["test_metrics"]["ap"]),
                    f(r["test_metrics"]["roc_auc"]),
                    f(r["test_metrics"]["macro_f1"]),
                ]
                for r in all_runs
            ],
        )
    )
    sections.append("\\clearpage\\section{普通与家族区间并列}\n")
    sections.append(
        table(
            ["方法", "层/指标", "差值", "普通95\\%", "家族95\\%"],
            [
                [
                    NAMES[r["method"]],
                    r["factor"] + "/" + r["metric"].replace("roc_auc", "AUC"),
                    f(r["estimate"]),
                    "[" + ", ".join(f(x) for x in r["percentile_ci95"]) + "]",
                    "[" + ", ".join(f(x) for x in r["bonferroni_family_ci95"]) + "]",
                ]
                for r in heads["contrasts"]
            ],
            "llrrr",
        )
    )
    sections.append("\\end{document}\n")
    path = report_dir / "第二轮补充报告.tex"
    path.write_text("".join(sections), encoding="utf-8")
    sources = [
        out / "pilot_comparison.json",
        out / "gemini/evaluation.json",
        review_path,
        out / "robustness/summary.json",
        out / "hardware.json",
        out / "runtime.json",
        *list((out / "robustness/runs").glob("*/result.json")),
        *list((out / "local_timing").glob("*/summary.json")),
        *list((out / "local_timing_v2").glob("*/summary.json")),
        out / "gemini/summary.json",
        out / "gemini/selection.json",
        out / "gemini/quote.json",
        Path(__file__).with_name("analysis.py"),
        PROJECT / "benchmark/research_followup/20260922/costs.json",
    ]
    binding = source_hashes(sources + [Path(__file__)])
    verify_sources(binding)
    save(
        out / "report_sources.json",
        dict(sources=binding, tex=str(path), tex_sha256=source_hashes([path])),
    )
    return path
