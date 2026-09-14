# D9.4 单人标注与隔时复标协议

> 版本：d9.4-v1  
> 用途：仅用于 MusicAgent 内部判断节奏告警是否具备进入评分的证据。  
> 边界：不是客观准确率、标注者间一致性或普适可唱性结论。

## 1. 评估对象

D9.4 分别评估两个确定性告警：

- `line_length_outlier`：同一段内某行是否因句长明显偏离而需要修改。
- `phrase_count_mismatch`：同一段内某行是否因短语或停顿层级明显不协调而需要修改。

平仄只保留描述性统计，不进入本轮人工标签、阈值搜索或评分。主题关键词直接命中继续保持 `advisory_only`，不进入 D9.4 混淆矩阵或总分；主题语义仍由 D8 匿名评审负责。

## 2. 样本计划

目标约 120 张完整段落卡：

- 60 张未修改的自然段落；
- 30 张一次只改变句长的受控段落；
- 30 张一次只改变停顿层级的受控段落。

样本至少覆盖 16 个歌词家族、6 个主题和 4 类节奏或曲风。Verse、Chorus、Bridge，以及近似等长、ABAB、AABB、自由句长都要覆盖。同一原稿、候选稿、修订稿及其受控变体共享 `family_id`，不得拆到阈值开发集和锁定验证集两边。

当前四条影子歌词已经参与规则设计，只能作为开发集种子，不能进入锁定验证集。阈值开发集按歌词家族划分为 11 个家族、80 张卡；锁定验证集为 5 个新家族、40 张卡，并在阈值冻结前隐藏检测结果。

每种告警的最低有效标签数：

- 开发集：明确正例不少于 30，明确反例不少于 60；
- 锁定验证集：明确正例不少于 20，明确反例不少于 40；
- 验证集每种告警最好有至少 10 个来自未经受控修改的自然缺陷。

如果标签数量不足，结论必须为“证据不足”，不能通过复制重复副歌或同源变体补足分母。

## 3. 首轮标注

标注时只打开 `cards.round1.jsonl` 和 `annotations.round1.jsonl`，不要查看 `manifest.private.jsonl`。每张卡使用不含主题含义的编号，保留完整段落和原始行序，但隐藏来源、开发/验证分组、检测结果和受控变体类型。

为了避免手工编辑 JSONL，可以启动项目 Web 服务后访问 `http://127.0.0.1:8000/d9-4-annotator.html`。页面依次选择：

- `data/output/evaluations/d9-4-annotation-v1/cards.round1.jsonl`
- `data/output/evaluations/d9-4-annotation-v1/annotations.round1.jsonl`

页面完全在本地读取文件，输入时自动保存到浏览器本地存储，并通过“导出当前标注”下载 JSONL。重新载入时会核对盲卡、模板和本地断点的 `sample_id` 与逐行数是否一致；不匹配的旧断点不会覆盖当前模板。未完成时导出文件名会标为 `incomplete-backup`，完成后才标为 `completed`。导出部分进度是允许的，但执行指标计算前必须完成全部必填项。

`[Verse]`、`[Chorus]`、`[Bridge]`、`[Hook]`、`[Outro]` 等结构标记不是歌词行，页面会显示其为“结构标记，不参与标注”；即使旧卡片序列化了这些行，导出后的指标计算也会排除它们。

逐行填写：

- `length_fit`：0=自然；1=边界或可接受的风格选择；2=明显破坏段内节奏且应该修改。
- `pause_fit`：使用相同的 0/1/2 定义。
- `reason_code`：可选的简短理由代码。
- `notes`：必要时记录判断依据。

逐段填写：

- `stanza_pattern`：`near_equal`、`abab`、`aabb`、`free` 或 `uncertain`。
- `rhythm_rating`：文本节奏顺畅度 1～5。
- `confidence`：`high`、`medium` 或 `low`。
- `needs_melody`：当前判断是否必须依赖旋律。

没有旋律、BPM、重音和实际演唱时，统一使用“文本节奏适配度”，不标注或宣称普适可唱性。

## 4. 误报、漏报与边界标签

主指标只使用明确标签：2 是人工正例，0 是人工反例。

- TP：程序告警且人工标 2；
- FP：程序告警但人工标 0；
- FN：程序未告警但人工标 2；
- TN：程序未告警且人工标 0。

程序分别计算精确率、召回率、误报率、漏报率及 Wilson 95% 区间。标签 1 不进入主混淆矩阵，另做两次敏感性分析：一次把 1 当反例，一次把 1 当正例。两种分析导致相反结论时，不准入评分。

## 5. 隔时复标

首轮完成后间隔 7～14 天。使用固定随机种子，按来源类型、段落、首轮句长最高标签和停顿最高标签分层抽取 30%，且不少于 40 张段落卡。第二轮继续隐藏首轮标签、检测结果和样本来源。

当前首轮文件已于 2026-09-13 冻结为 `data/output/evaluations/d9-4-annotation-v1/annotations.round1.completed.jsonl`，SHA-256 为 `13d7811b4989b54855196cf0f7d42a56678ef42c76e3ca3afac0f9826f1b47e9`。复标窗口为 2026-09-20～2026-09-27；在此之前不得用首轮结果生成复标卡或调节业务规则。

最低个人一致性门槛：

- 线性加权 Cohen's kappa 不低于 0.75；
- “明确问题 2 / 非明确问题 0 或 1”的二分类一致率不低于 85%；
- 0 与 2 的严重翻转率不高于 10%。

这些指标只能称为同一标注者的隔时一致性。未来增加第二标注者时，新增 `annotator_id` 记录，原始记录不覆盖；分歧通过独立仲裁记录保存。

## 6. 内部准入条件

锁定验证集中的每一种告警须同时满足：

- 精确率不低于 80%；
- 误报率不高于 10%；
- 漏报率不高于 30%；
- 样本数量与个人一致性达到前述要求；
- 有效样本不少于 20 的任一曲风、段落或句长模式子组，误报率不高于 20%。

检测指标通过后，还需要至少 24 组隐藏前后关系、随机顺序的修改前后比较：至少 17 组文本节奏提高，最多 2 组变差，且主题、自然度或押韵被破坏的比例不高于 10%。没有正向修改证据时，告警只能继续展示。

## 7. 候选评分方式

当前 `rhythm.score` 会被全歌总行数稀释，不能直接作为权重。未来候选节奏分应先按段计算告警行比例，再对独立段落宏平均；完全重复的副歌只计一次。

如果全部准入条件通过，第一版最多采用 5 分的单向软惩罚：

```text
原规则分 B = 0.70 × 押韵分 + 0.30 × 结构分
候选新分 = B - min(5, 0.05 × (100 - 节奏分))
```

节奏不能给低押韵或低结构歌词补分。正式接入 `passed` 和回炉路由前，还必须影子回放跨越 75 分的样本，并确认人工认为不应回炉却被新增回炉的比例不高于 5%。当前不修改公式、门槛或路由。

## 8. 文件与命令

完整首轮盲标包已经冻结在 `data/output/evaluations/d9-4-annotation-v1/`。标注者只打开本地页面 `http://127.0.0.1:8000/d9-4-annotator.html`，并选择其中的 `cards.round1.jsonl` 与 `annotations.round1.jsonl`；不要查看 `manifest.private.jsonl`。

如需从已冻结来源池重新构建同一批标注卡：

```powershell
$env:PYTHONPATH = ''
.\.venv\Scripts\python.exe -m scripts.prepare_d9_4_annotation `
  --shadow-summary data/output/evaluations/quality-v2-shadow-6606956777e6497cbf6c78632c2758d4/summary.json `
  --source-pool data/output/evaluations/d9-4-source-pool `
  --out data/output/evaluations/d9-4-annotation-v1 `
  --replace
```

首轮完成并间隔 7～14 天后生成复标子集：

```powershell
$env:PYTHONPATH = ''
.\.venv\Scripts\python.exe -m scripts.prepare_d9_4_retest `
  --cards data/output/evaluations/d9-4-annotation-v1/cards.round1.jsonl `
  --manifest data/output/evaluations/d9-4-annotation-v1/manifest.private.jsonl `
  --round1 data/output/evaluations/d9-4-annotation-v1/annotations.round1.jsonl `
  --out data/output/evaluations/d9-4-retest
```

完成两轮后计算指标：

```powershell
$env:PYTHONPATH = ''
.\.venv\Scripts\python.exe -m scripts.evaluate_d9_4_annotations `
  --manifest data/output/evaluations/d9-4-annotation-v1/manifest.private.jsonl `
  --round1 data/output/evaluations/d9-4-annotation-v1/annotations.round1.jsonl `
  --round2 data/output/evaluations/d9-4-retest/annotations.round2.jsonl `
  --out data/output/evaluations/d9-4-metrics.json
```

以上标注包构建、复标抽样和指标脚本均离线运行，不调用 DeepSeek、HeartMuLa 或 GPU，也不修改业务评分。

本轮完整样本包已经生成在 `data/output/evaluations/d9-4-annotation-v1/`。其来源池通过 `scripts/generate_d9_4_source_pool.py` 一次性调用 DeepSeek 生成 12 个新主题家族的结构型和意象型歌词；后续标注、复标和指标脚本仍全部离线，来源生成未调用评审、制作或音频模型。
