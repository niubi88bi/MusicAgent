# D10-D11 RAG 素材、字段与验收方案

> 状态：方案已形成；D10.0-D10.3 已落地来源目录、17 张风格草稿卡与 390 个韵脚候选；17 张风格卡已完成字段级来源复核，均只可进入固定查询，仍为 `draft`。近邻分组复核和语义影子策略已完成，逐卡准入审核包已生成；韵脚词 RAG 必要性审计不支持首稿默认注入。
> 日期：2026-09-13
> 当前边界：只保存公开指南的来源事实、项目化短卡和待人工审核的韵脚候选；尚未保存指南全文、安装依赖、构建索引或接入 Agent。

## 1. 目标与非目标

当前 RAG 只解决两个窄问题：

1. 按“目标辙 + 主题/情绪”给作词 Agent 提供少量可追溯候选韵脚。
2. 按“曲风 + 情绪 + 目标音频后端”给制作 Agent 提供结构化风格卡和可直接使用的 tags。

它不承担歌词质检，不替代 D8 主观评审，也不证明生成歌词或音频质量提升。D9.4 的首轮冻结、复标协议、评分权重和回炉路由均不在本任务范围内。

## 2. 当前事实

- `tools/rag/` 与正式的 `data/rhyme_lexicon.json` 当前不存在；`data/miaoxiang/` 已包含来源说明、7 条来源目录、4 张 v1 粗粒度卡和 17 张 v2 风格草稿卡。
- v1 的 4 张风格卡共塞入 16 个子风格标签，`top_k=3` 会返回全库 75%，无法形成有意义的检索区分。v2 已拆成动感流行 4、EDM 6、民族国韵 4、新世纪 Rap 3，共 17 张；其中 House 进一步拆为 Deep House 与 Tropical House，“苦情民谣电音”明确为项目组合。全部仍为 `draft`，HeartMuLa 后端验证仍为 `unverified`。
- 已生成 `data/rhyme_candidate_seeds.v1.json` 和 `data/rhyme_lexicon.candidates.v1.jsonl`：十三辙各 30 个、共 390 个项目候选词。它们只通过了 Jieba 固定快照收录、pypinyin 读音和现有十三辙映射校验，主题、情绪、常用度与歌词句尾适用性仍待人工审核。
- 项目还没有 ChromaDB、sentence-transformers 或 BGE 的已确认安装与锁定记录。
- 妙响“更多 → 创作指南”会打开飞书 Wiki；2026-09-13 已确认该知识库标注为“互联网公开”，无需登录即可阅读正文。已核验动感流行、潮流电子（EDM）、民族国韵、新世纪 Rap 四篇官方指南，并把来源事实和项目化摘要先整理为 4 张 v1 粗卡，再形成 17 张 v2 风格草稿卡；未保存或再分发指南全文、示例完整歌词、图片或音频。
- HeartMuLa 官方仓库确认模型以逗号分隔、无空格的 tags 作为条件输入；官方示例包含乐器、情绪与场景类标签。因此风格库必须保留面向 HeartMuLa 的 `backend_tags`，但不能承诺模型严格遵循每个标签。
- BAAI 的 `bge-small-zh-v1.5` 模型卡标注 MIT License，可作为本地中文向量模型候选；真正安装前仍需锁定版本、下载来源与文件哈希。
- Chroma 官方文档支持持久化客户端和 metadata filter。第一版采用“元数据硬过滤 + 向量语义排序”，而不是只靠向量相似度判断辙部。

## 3. 来源策略

### 3.1 允许进入正式素材库

| 级别 | 来源 | 可保存内容 | 必须记录 |
|---|---|---|---|
| A | 官方模型/产品文档 | 支持的输入格式、短标签示例、可验证的功能事实 | 页面标题、URL、访问日期、许可/使用说明、内容哈希 |
| A | 妙响互联网公开官方知识库 | 原文仅作来源证据；进入检索库的是项目重写后的短风格卡 | 指南标题、公开 URL、访问日期、页面公开状态、内容哈希 |
| B | 项目原创整理 | 自写风格卡、人工筛选的常用押韵词与标签 | 作者类型、审核状态、版本、变更说明 |
| B | 许可明确的开源词音资源 | 单字/词语、拼音、韵母映射等事实性字段 | 仓库、版本/提交、License、派生规则 |

### 3.2 不进入正式素材库

- 未确认著作权与再利用许可的整首歌词、歌词网站爬取结果和第三方打包歌词语料。
- 把“模仿某首歌/某位在世歌手”直接作为风格模板的材料。
- 无原始 URL、访问时间、许可说明或人工审核状态的匿名文本。
- 仅来自短视频作者、论坛帖或搜索摘要且无法回到官方原文的“妙响指南”。

网上可见的 `dengxiuqi/ChineseLyrics` 自述为从网络收集的 10 万余首歌词，且只说明“仅供学习交流使用”，仓库未提供明确 LICENSE。由于正式索引首先需要可复用许可，现阶段已排除该来源，不再为入库目的继续下载或抽样；只有未来取得明确授权或许可发生变化时才重新评估。

### 3.3 追加来源调研结论（2026-09-13）

风格模板的来源需要按职责拆分，不能要求一个网页同时提供曲风知识、规范名称和 HeartMuLa 可执行标签：

- **妙响四篇曲风指南**：继续作为中文场景和提示词内容的主要事实锚点。知识库目录当前只列动感流行、潮流电子、民族国韵、新世纪 Rap 四篇“常见热门曲风”指南；另有一篇 Rap 创作范式指南，可补充提示词与分段方法，但不是第五种曲风。页面公开可读不等于允许全文再分发，正式库仍只保存来源事实和项目原创短卡。
- **HeartMuLa-Benchmark**：作为后端标签白名单种子。官方 ModelScope 数据集标注 Apache-2.0；固定提交 `ab18c82248b7bc0ba5bda8acc76357516b221060` 中有 80 个 `tag.txt`，覆盖中文 20、英文 30、日/韩/西语各 10。实测每条 4～7 个标签，共 51 个去重标签；人工按用途归类为 8 个乐器、8 个大类曲风、18 个主题/场景和 17 个情绪/质感标签。它不含 BPM，也不足以证明标签闭集或稳定控制能力。
- **HeartMuLa 官方仓库**：只确认逗号分隔、无空格的 tags 输入格式和极少量示例。官方 issue #17 对“支持哪些标签、是否支持 BPM”仍无维护者答复，因此不能据此宣称固定标签词典或精确 BPM 控制。
- **MusicBrainz Core Data**：曲风名称、别名和乐器名称属于 CC0 核心数据，可作为规范名/别名层；用户提交的标签关联属于补充数据，不纳入第一版。MusicBrainz 只提供分类骨架，不直接生成风格描述。
- **Suno 官方帮助中心**：公开术语页覆盖节奏、结构、曲风、乐器、人声和制作效果，可用于校验通用 `prompt_template` 字段是否完整；未观察到可再分发许可，因此只保留来源和项目原创总结，也不能拿它证明 HeartMuLa 能力。

韵脚词库不应从整首歌词语料直接抽取，推荐采用“许可明确的候选词表 + 项目确定性校音/定辙 + 人工语义审核”：

- **Jieba `dict.txt`**：仓库标注 MIT。固定 `master` 提交 `67fa2e36e72f69d9134b8a1037b83fbb070b9775`，本次下载文件 SHA-256 为 `7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8`。实测 349,046 条，其中 1～2 个汉字词 125,714 条、频次不低于 100 的 28,660 条、频次不低于 1000 的 6,720 条。高频项以功能词和通用新闻词为主，词频只能用于候选裁剪，不能直接标成“常用歌词词”或质量分。
- **pypinyin 及其数据仓库**：项目当前版本为 `0.55.0`；`pinyin-data`、`phrase-pinyin-data` 仓库标注 MIT，但数据来源链还包含 Unihan、CC-CEDICT 和其他词典。第一版只复用既有 pypinyin 对完整词语校音，并对多音词人工确认；不把其大词表整体复制为项目词库。
- **CC-CEDICT**：提供词语、拼音和英文释义，但官方为 CC BY-SA 3.0。第一版只作为人工交叉核验候选，不混入项目原创核心词库，避免不必要的署名与相同方式共享边界。
- **THUOCL**：官方说明允许研究与商业使用，仓库标注 MIT，但主体是 IT、财经、地名、诗词名句等领域词表，并非现代歌词韵脚库。可作极少量补充抽查，不作为主种子。
- **大连理工情感词汇本体**：只允许学术研究，商业使用需另行联系；不进入第一版正式库。
- **MEmoLon**：生成式多语言情感词典数据标注 CC BY 4.0，覆盖广但归档约 2.2 GB，中文词的情感值为跨语言自动扩展结果，既不面向歌词也不值得为 390 个首批词引入。仅保留为未来独立校验候选。
- **ChineseLyrics**：仓库没有明确 LICENSE，正文说明数据来自网络收集、覆盖 2019 年以前 102,197 首歌且“仅供学习交流使用”。这已经构成正式索引的许可阻断；现阶段不下载、不做技术抽样、不进入正式 RAG。只有未来获得明确授权或出现许可变更时才重新评估。

基于以上证据，当前推荐来源组合为：Jieba 只产生候选词，pypinyin 负责完整词语读音，现有十三辙代码负责确定性分组，`themes`、`moods`、常用度和是否适合作韵脚由项目人工审核。外部歌词、情感词典和词频都不直接替代人工标签。

十三辙在本项目中的定位是“普通话韵脚粗粒度分组”，不是现代流行歌曲的唯一规范。候选词库只辅助作词 Agent 找到可替换的句尾词，不要求每句一韵到底，不把跨辙自动等同于歌词质量差，也不适用于粤语、方言或大量中英混写歌词。现有评分与路由本轮不变；是否放宽规则必须等待既有人工标注证据并单独审批。

## 4. 两类数据契约

原始 JSONL 中保留数组；写入 Chroma metadata 时，为兼容尚未锁定的 Chroma 版本，将多值字段规范化为按字典序排列的 `|` 分隔字符串，同时把完整数组保留在 `document` 或原始记录中。

### 4.1 风格卡 `style_template`

必填字段：

```json
{
  "id": "style-cn-pop-warm-v1",
  "schema_version": "1.0",
  "kind": "style_template",
  "genre": "流行",
  "subgenres": ["华语流行"],
  "moods": ["温暖", "治愈"],
  "tempo_bpm_min": 78,
  "tempo_bpm_max": 96,
  "instruments": ["piano", "acoustic guitar"],
  "vocal": "clean vocal",
  "dynamics": "主歌克制，副歌抬升",
  "structure_notes": "Verse-Chorus-Verse-Chorus-Bridge-Chorus",
  "prompt_template": "温暖的华语流行，钢琴与木吉他，主歌克制、副歌抬升",
  "backend_tags": {
    "heartmula": ["pop", "warm", "piano", "acoustic guitar"]
  },
  "supported_backends": ["heartmula-local"],
  "source_type": "project_authored",
  "source_title": "MusicAgent 自编风格卡",
  "source_url": null,
  "accessed_at": "2026-09-13",
  "license_or_usage_note": "项目原创整理",
  "content_sha256": "<生成时填写>",
  "review_status": "draft",
  "version": 1
}
```

约束：

- 每张卡只描述一个可复用风格组合，不复制指南长段落或歌曲歌词。
- BPM、结构、乐器只表示建议，不标注为 HeartMuLa 的严格控制能力。
- `review_status=approved` 才能进入正式索引；来源事实与项目推断必须在卡片说明中区分。

2026-09-16 影子契约补充字段：`card_role`/`retrieval_scope` 区分组合查询模板与直接候选；`vocal_type`、`rap_default`、`instrumental_default`、`default_language` 表达项目派生的人声路由属性；`query_expansion_terms` 和 `scene_tags` 不属于核心风格词。当前仅由 `scripts/validate_style_template_contract.py` 检查，正式 `tools/rag/schemas.py` 尚未实现。

### 4.2 韵脚词条 `rhyme_word`

必填字段：

```json
{
  "id": "rhyme-jiangyang-guang-v1",
  "schema_version": "1.0",
  "kind": "rhyme_word",
  "word": "光",
  "pinyin": "guang1",
  "final": "uang",
  "rhyme_group": "江阳",
  "tone_pattern": "平",
  "themes": ["希望", "夜晚"],
  "moods": ["明亮", "温暖"],
  "part_of_speech": "名词",
  "frequency_tier": "common",
  "source_type": "project_curated",
  "source_title": "MusicAgent 人工整理",
  "source_url": null,
  "license_or_usage_note": "词语及拼音事实字段；主题标签为项目人工判断",
  "review_status": "draft",
  "version": 1
}
```

约束：

- `rhyme_group` 必须由现有十三辙映射代码校验，不能由 embedding 猜测。
- `frequency_tier` 只使用人工档位 `common/less_common`；没有许可明确的语料统计前，不填写伪精确词频。
- 一词多音时拆成不同读音记录，或明确 `context_required=true`；不能用不确定读音强行入辙。
- 不要求候选词一定被模型采用，避免为了“命中 RAG”牺牲自然度。

## 5. 文件与组件范围

当前与后续涉及：

- `data/miaoxiang/README.md`：来源保存、事实/派生字段与审核边界说明；已实现。
- `data/miaoxiang/source_catalog.jsonl`：所有来源的证据目录；已实现 4 条。
- `data/miaoxiang/style_templates.v1.jsonl`：4 张粗粒度历史草稿，保留用于对比。
- `data/miaoxiang/style_templates.v2.jsonl`：17 张风格候选卡，全部为 `draft`。
- `data/rhyme_candidate_seeds.v1.json`：按歌词句尾可用性整理的 390 个项目候选种子。
- `data/rhyme_lexicon.candidates.v1.jsonl`：由脚本补齐拼音、辙部和来源信息的待审核记录。
- `scripts/build_rhyme_candidates.py`：固定来源快照、校验种子并生成候选记录；不生成正式词库。
- `data/rhyme_lexicon.json`：人工审核后的正式十三辙韵脚词条，当前尚未生成。
- `data/rag/chroma/`：本地持久化索引；属于可重建产物，不提交大体积缓存。
- `tools/rag/schemas.py`：字段校验与 metadata 扁平化。
- `tools/rag/build_index.py`：建立两个 collection。
- `tools/rag/retriever.py`：检索及来源返回。
- `tests/test_rag.py`：契约、过滤、空召回与固定查询测试。

第一版使用两个 collection：

1. `rhyme_words_v1`：先按 `rhyme_group` 精确过滤，再按主题/情绪向量排序。
2. `style_templates_v1`：按后端和已知曲风过滤，再按用户描述向量排序。

## 6. 检索接口草案

```python
retrieve_rhyme_words(
    target_rhyme_group: str,
    theme: str,
    mood: str | None = None,
    top_k: int = 10,
) -> list[RhymeHit]

retrieve_style_templates(
    query: str,
    backend: str,
    genre: str | None = None,
    top_k: int = 3,
) -> list[StyleHit]
```

每个结果必须返回 `id`、`score`、业务字段、`source_type`、`source_title` 和 `source_url`。未知辙名、索引缺失或零召回必须返回明确状态；禁止悄悄跨辙兜底。

作词 Agent 每次最多注入 8～12 个韵脚候选；制作 Agent 每次最多注入 3 张风格卡。限制上下文数量是为了减少堆词、套模板和提示词漂移。

2026-09-13 必要性审计后，以上韵脚词注入路径暂停实施：现有无 RAG 数据更像暴露“模型不理解十三辙名称对应韵母”，而不是缺少词汇。先比较现有反馈、确定性辙部说明、辙部说明加少量候选词三种回炉提示；只有候选词相对确定性说明产生额外收益，才讨论在失败回炉阶段条件注入。详见 `docs/rhyme-rag-necessity-audit.md`。

## 7. 分阶段验收

### A. 来源与数据契约

- 所有记录通过 schema 校验，`id` 唯一，来源字段完整率 100%。
- 正式索引只接收 `review_status=approved`。
- 妙响风格卡必须能回溯到已核验的公开官方指南页面，且区分原文事实与项目整理内容。
- 外部歌词库只有在许可先通过后才进入技术抽样；`ChineseLyrics` 当前因许可不明确直接排除，不进入正式索引。

### B. 最小素材规模

- 风格卡共 17 张草稿，覆盖动感流行 4、EDM 6、民族国韵 4、新世纪 Rap 3。House 已拆为 Deep House 与 Tropical House，“苦情民谣电音”已明确为项目组合。
- 2026-09-14 字段级来源复核后，原 15 张候选可进入固定查询；Trance 与 Drum & Bass 已分别用 Wikipedia 固定版本补充宽泛风格事实，也只可进入固定查询。腾讯多媒体实验室中文官方文档补充了中文歌曲的风格/情绪/语种/场景/强度元数据维度，但不承担具体制作模板来源。Wikipedia 两张补证卡采用 CC BY-SA 4.0，必须保留署名、修改说明和相同许可提示。17 张仍全部为 `draft`，0 张 `approved`。
- 韵脚候选已按十三辙每辙 30 个生成，共 390 条。这个均衡数量只服务首轮覆盖审核，不是行业标准，也不是正式词库必须保持的配额；人工审核可以删除不自然词，不为凑数补低质量词。
- 390 条候选的全面人工精审暂缓。先从中东辙选择小批候选完成必要性对照；只有候选词相对固定辙部说明表现出额外收益，才继续审核歌词句尾适用性、主题、情绪和常用度。
- 17 张风格卡完成固定查询后，再判断是否增加流行抒情、民谣/原声、摇滚、R&B/Soul、Lo-fi、City Pop 等缺口；未出现真实召回问题前不继续扩库。

2026-09-14 固定查询词面基线：8 组预登记查询已覆盖 17 张草稿卡的相关候选集合。top-1 主卡命中 8/8、top-3 相关候选命中 8/8、top-1 来源追溯率 100%；q04“苦情民谣电音”和 q08“国风说唱”仅有 3 分词面证据，标记为低信号。结果文件为 `data/output/evaluations/style-fixed-query-v1/result.json`，只证明固定登记语义的词面排序，不证明 embedding 检索或生成质量。下一步先补强两个低信号查询并保留新版本基线，再决定是否安装 ChromaDB/BGE。

补强后登记 `style_fixed_queries.v1.1.json`，只改 q04/q08 的中文用户表达和关键词，不改风格卡或排序算法。v1.1 仍为 top-1 8/8、top-3 8/8，低信号查询降为 0；结果文件为 `data/output/evaluations/style-fixed-query-v1.1/result.json`。v1 保留为不可变历史基线，v1.1 不代表向量检索或生成效果已通过。

2026-09-14 embedding 对照：安装 `chromadb==1.5.9`、`sentence-transformers==6.0.1`，将 `BAAI/bge-small-zh-v1.5`（512 维）缓存于 D 盘，并在隔离 collection `style_draft_v11` 上运行 v1.1 查询。embedding top-1/top-3 均为 8/8、来源追溯率 100%，与词面基线没有指标提升。结果文件为 `data/output/evaluations/style-embedding-v1.1/result.json`；这只是 draft 卡离线对照，不是生产索引或生成质量证据。

随后根据 BGE Top-3 近邻生成 4 组人工复核包，覆盖 17 张卡：中文动感舞曲、Deep/Tropical House、高速或旋律型 EDM、国风与中文 Rap。复核包只展示摘要、检索词、来源和观察邻居，人工决定字段保持 `null`；产物为 `data/output/evaluations/style-near-neighbor-review-v1/review_packet.json`，不改变 `review_status`。

2026-09-16 人工复核后执行语义修订：`苦情民谣电音`降级为 `composite_query_template/query_expansion_only`，不作为独立风格候选；国风/Rap 七张卡增加 `vocal_type`、`rap_default`、`instrumental_default`、`default_language`。v1.2 影子查询验证组合模板先扩展查询、明确说唱优先 Rap、明确纯音乐优先 Ambient/器乐、模糊“国风”保留多候选；结果为组合模板识别 1/1、直接主卡 3/3、相关 Top-3 5/5，且直接候选 Top-3 不包含组合模板。该策略仍未接入生产。

2026-09-16 扩大固定查询 holdout：新增 `style_fixed_queries.v1.3.json` 的 20 条 q14～q33，覆盖中文隐藏别名、查询扩展、人声/器乐属性及国风/Rap、动感舞曲近邻表达。词面 top-1 主卡命中 19/20、Top-3 预期召回 20/20；BGE-small-zh-v1.5 隔离对照同为 19/20、20/20，来源追溯均为 100%，没有相对词面的指标提升。q31“国风 + 说唱”在两种方法中均表现为组合歧义，登记为多候选观察而非单一主卡失败。该 holdout 仍只读取 draft 卡，不改变生产索引、approved 状态或 Agent 路由。

### C. 离线检索

- 13 个固定辙部查询中，top-k 返回项的辙部正确率必须为 100%。
- 未知辙部必须明确失败，零条跨辙结果。
- 8 个固定风格查询中，top-3 至少包含 1 张预先登记的相关风格卡。
- 所有命中项来源可追溯率 100%。
- 记录冷启动与热查询耗时，但第一轮不把本机时延设为产品性能承诺。

### D. Agent 接入前的小型归因测试

先固定 6 个主题和 DeepSeek 参数，做配对的 RAG-off / RAG-on 文本生成；保留请求、召回项、歌词和规则报告。只在以下条件同时满足时，才讨论默认开启：

1. on 组没有出现跨辙候选或来源缺失。
2. 目标韵脚使用情况有改善，且不是机械重复候选词造成。
3. 人工盲评的自然度没有下降。
4. 不改变现有规则分、75 分门槛、D8 评审权重和回炉上限。

6 个主题只能作为可归因的小试点，不能宣称普遍提升、客观准确率或生产效果。具体配对样本、盲评字段与放行阈值应在执行前另行确认。

## 8. 主要风险与控制

- **版权/许可风险**：候选歌词库先验证许可和实际用途；所有外部素材保留来源与用途说明，未经确认不进入正式索引。
- **来源冒充风险**：公开作品页不等于官方曲风指南；只有已核验的妙响官方知识库页面可作为该来源。
- **检索逻辑风险**：辙部使用确定性过滤，embedding 只负责主题排序。
- **生成自然度风险**：限制 top-k，把候选描述为参考而非强制命中，并通过 RAG-off/on 盲评检查。
- **后端控制夸大**：HeartMuLa tags 是条件输入，不把 BPM、乐器或结构写成已验证的精确控制。
- **依赖与磁盘风险**：依赖、模型、缓存均放 D 盘；实施前锁定版本并记录模型文件哈希。

## 9. 当前已确认与待验证事项

1. **已实现候选数据**：妙响官方知识库的四篇指南、两篇 Wikipedia 固定版本补充来源和一篇腾讯多媒体实验室中文官方标签文档已形成 7 条来源目录、4 张 v1 粗粒度历史卡和 17 张 v2 风格草稿卡。House 已拆为 Deep House 与 Tropical House，“苦情民谣电音”已明确为项目组合。v2 的内容哈希、唯一 ID、主/补充来源引用与 HeartMuLa-Benchmark 标签子集检查已通过；17 张均只达到固定查询候选状态。所有卡仍为 `draft`，后端标签跟随效果为 `unverified`。
2. **已实现候选清单**：已生成十三辙各 30 个、共 390 个韵脚候选。候选由项目按歌词句尾可用性整理，经固定 Jieba 快照、pypinyin 和现有十三辙映射校验；尚未进行人工主题、情绪、常用度和最终适用性审核，因此不是正式词库。
3. **已完成来源排查**：`ChineseLyrics` 当前无明确再利用许可，正式索引不采用；Jieba 只承担候选收录核验，不把其源频次冒充歌词频率。
4. **已完成必要性审计**：24 份无 RAG 新主题草稿中，16 份押韵分达到 70，双候选机制让 10/12 个主题至少有一份押韵达标稿；但目标辙行命中率只有 64.8%，并出现把言前、乜斜、人辰误称为中东辙的明确错误。当前证据支持先补确定性辙部说明，不支持首稿默认注入韵脚词。
5. **待完成**：固定查询、BGE/Chroma 隔离对照、近邻人工复核、本轮影子语义修订、逐卡审核包生成、17 张卡人工结论、6 张 revise 卡最小修订、影子字段契约检查和 v1.3 扩展 holdout 已完成；11 张 keep 卡内容未改。v1.3 的词面与 BGE 指标均为 19/20 top-1、20/20 top-3，q31 作为组合歧义保留。影子契约 17 张卡错误数为 0，但正式 `tools/rag/schemas.py` 尚未实现，仍需先汇总失败类型再决定生产 schema、approved-only 索引和多候选策略；组合模板和人声属性策略仍未接入生产，`approved` 尚为 0；韵脚路线已完成必要性与 section-consistency 检查，390 词审核及候选词 RAG 继续暂停。

下一步不先精审全部 390 词。风格卡先完成用户内容确认和固定查询设计，再决定是否安装检索依赖；候选数据通过结构检查、AI 审计或来源补证都不等于 RAG 效果已验证。

## 10. 核验来源

- 妙响公开工作室：https://music.douyin.com/studio
- 妙响动感流行官方指南：https://bytedance.larkoffice.com/wiki/MVeBw3xvQi8MpCkAVjxcDlSQn0f
- 妙响潮流电子（EDM）官方指南：https://bytedance.larkoffice.com/wiki/XiePwOqqtinAfzkag5xcXd3In9g
- 妙响民族国韵官方指南：https://bytedance.larkoffice.com/wiki/JauMwFPdyiOY6OkxX5LcqwZhnvc
- 妙响新世纪 Rap 官方指南：https://bytedance.larkoffice.com/wiki/EWznw3NAPilTZJkgDkhc22fYnnb
- Wikipedia Trance 固定版本（CC BY-SA 4.0）：https://en.wikipedia.org/w/index.php?title=Trance_music&oldid=1371451095
- Wikipedia Drum and bass 固定版本（CC BY-SA 4.0）：https://en.wikipedia.org/w/index.php?title=Drum_and_bass&oldid=1369972946
- Wikipedia 内容复用与署名规则：https://en.wikipedia.org/wiki/Wikipedia:Copyrights
- Native Instruments 法律信息（因 AI/数据挖掘用途限制而不纳入 RAG 来源）：https://www.native-instruments.com/pages/legal-information
- HeartMuLa 官方仓库：https://github.com/HeartMuLa/heartlib
- HeartMuLa-Benchmark：https://modelscope.cn/datasets/HeartMuLa/HeartMuLa-Benchmark
- HeartMuLa 标签范围未答复 issue：https://github.com/HeartMuLa/heartlib/issues/17
- HeartMuLa-oss-3B 模型卡：https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B
- MusicBrainz 数据许可：https://musicbrainz.org/doc/About/Data_License
- MusicBrainz 核心数据范围：https://musicbrainz.org/doc/MusicBrainz_Database
- Suno 音乐术语表：https://help.suno.com/en/articles/9010177
- Jieba 官方仓库：https://github.com/fxsjy/jieba
- BGE-small-zh-v1.5 模型卡：https://huggingface.co/BAAI/bge-small-zh-v1.5
- Chroma metadata filter：https://docs.trychroma.com/reference/where-filter
- Chroma PersistentClient：https://docs.trychroma.com/docs/run-chroma/cloud-client
- pypinyin 官方仓库：https://github.com/mozillazg/python-pinyin
- pinyin-data：https://github.com/mozillazg/pinyin-data
- phrase-pinyin-data：https://github.com/mozillazg/phrase-pinyin-data
- CC-CEDICT 官方许可：https://cc-cedict.org/wiki/
- THUOCL 官方仓库：https://github.com/thunlp/THUOCL
- 大连理工情感词汇本体：https://ir.dlut.edu.cn/zyxz/qgbtk.htm
- MEmoLon 官方仓库：https://github.com/JULIELab/MEmoLon
- 排除的歌词候选库说明：https://github.com/dengxiuqi/ChineseLyrics/blob/master/README.md
