# MusicAgent

中文歌曲创作工作台。当前 Web 版支持主题输入、双作词人并行创作、匿名主观评审与规则组合选优、最多三稿的规则回炉记录、制作方案，以及本地 HeartMuLa 的 30 秒试听或完整歌曲按需生成。

> 这是用于求职展示的公开审查副本。旧实验数据、内部规划资料、模型权重、密钥、历史生成歌词和音频未随仓库发布。下述完整 Web 启动方式记录了原开发环境；全新克隆仅保证下方自拟小样本有独立依赖清单，不代表可立即运行 DeepSeek→HeartMuLa 链路。2026-10-16 借用原项目 Python 环境和本机临时 DeepSeek 配置，公开副本代码完成过 2 条真实文本 API 任务；尚未验证全新克隆的完整链路、浏览器交互或该副本音频生成。

## 可复现的公开小样本

`data/public_demo/` 含 1 条从零自拟的风格卡、来源记录和失败关闭准入策略，不含第三方网页文本。样本状态为 `draft`，仅用于展示字段契约、哈希校验与准入判断；不会进入本地演示或公开产品索引，也不证明风格 RAG 或 HeartMuLa 生成效果。

在有 Python 3.10 和 pip 的新环境中，可只安装该小样本需要的依赖并运行专项检查：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-public-demo.txt
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m unittest tests.test_public_style_demo -v
```

仓库仍保留历史 RAG/Producer 实验代码和测试，以便展示设计与取舍；依赖未发布风格卡或实验数据的测试会明确 `skip`，不能计为通过。完整 Web/音频链路需要另行准备 DeepSeek 密钥、HeartMuLa 权重及项目开发依赖；本小样本不提供这些依赖的安装或真实生成验收。

## 启动（Windows PowerShell）

以下命令只适用于已自行配齐原项目完整开发依赖、密钥和模型权重的本地环境；在仓库根目录执行：

```powershell
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8123 --workers 1
```

浏览器打开 http://127.0.0.1:8123/ 。使用项目 Python 3.10 虚拟环境；依赖统一由 uv 管理。`.env` 中需配置 `DEEPSEEK_API_KEY`，其他配置见 `.env.example`。健康检查只返回密钥是否存在，不返回密钥。

若只想查看人工自拟的静态页面示例，可在仓库根目录执行 `python -m http.server 8123 --bind 127.0.0.1 --directory frontend`；它不连接真实任务接口。真实创作需启动上述 FastAPI 服务，同一端口不能同时运行两种服务。

## 使用

1. 输入 2～200 字主题，可以包含曲风和情绪；选择“30 秒试听”（默认）或“完整歌曲”，再点击“生成歌词”。完整歌曲使用约两分钟的长歌词创作约束；它不是把 30 秒短歌词直接拉长。
2. 作词人 A 偏重押韵结构，作词人 B 偏重意象表达；匿名评审 Agent 只看需求和 A/B 正文，从主题、自然度、情绪、叙事意象和 Hook 五维评分。页面显示并行节点、候选分数、入选依据，以及后续各稿质检与修改反馈。
3. 文本完成后按所选模式生成音频。完整歌曲需同时通过原有规则质检和额外的歌词结构/长度准入，才可启动 GPU；未通过时页面显示原因，不自动消耗 GPU。音频任务不会重新调用 DeepSeek；失败后保留文本并允许手动重试。
4. 同一服务进程内，刷新带 `?task=任务编号` 的地址可继续查询；终态音频记录可在服务重启后恢复查看。

完整歌曲的 240 秒是 HeartMuLa 生成安全上限，不是固定成品时长，也不保证歌词唱全。生成后必须人工试听核对漏唱、发音和结尾；自动文本准入、MP3 存在或播放器可加载都不等于歌曲质量验收。长歌生成可能占用 GPU 数十分钟，本地仍只允许一个音频任务。

主观评审不读取 persona 名称和规则分；单方重大主题问题先淘汰，主观总分相差不足 3 分时才用规则结果决胜。双方都有重大主题问题时，两位作词人最多共同回炉 1 次。入选后仍执行确定性规则质检，最多生成 3 稿（初稿 + 最多 2 次改写）；达到上限时采用规则分最高的一稿，`capped` 仍表示未通过。任务 `done` 只表示文本流程完成。

质检报告还会展示句长、停顿、平仄统计和主题关键词直接命中。D9 第一版处于影子模式：总分与通过门槛仍只由押韵和结构决定，节奏参考与主题词面信号只作建议，不会单独触发回炉，也不能代表完整语义或可唱性。详细口径见 `docs/quality-design.md`。

## 接口和保存

- `GET /health`：服务和文本配置状态。
- `POST /api/tasks`：请求体 `{"theme":"夏夜回家","song_mode":"preview"}` 或 `{"theme":"夏夜回家","song_mode":"full"}`；省略模式仍为 30 秒试听。接收成功立即返回 202 和 `task_id`。
- `GET /api/tasks/{task_id}`：任务状态、真实节点事件、部分文本及最终结果。
- `GET /api/tasks`：本次服务进程内的任务列表。
- `POST /api/tasks/{task_id}/audio`：按文本任务已保存的模式创建音频任务；试听上限 30 秒，完整歌曲上限 240 秒。模式不可在音频请求中临时改写。
- `GET /api/tasks/{task_id}/audio/status`：查询关联音频任务状态。
- `GET /api/tasks/{task_id}/audio`：仅在生成成功后返回对应 MP3。

一个进程同时执行 1 个文本任务，最多有 2 个等待文本任务；音频侧同时只接受 1 个任务，满载返回 429。应保持 `--workers 1`，不要在任务运行期间启用自动重载。正常关闭会等待已接收任务完成。

产物保存于 `data/output/web/<task_id>/`：最新 `task.json`、两份作词候选 `lyric_candidates.json`、`subjective_review.json`、评审历史、综合选择依据 `candidate_selection.json`、最终歌词、结构化需求、制作方案、逐轮质检报告，以及 `audio_task.json`、`audio_process.log` 和成功后的 `audio/<audio_task_id>/song.mp3`。音频通过独立子进程运行，原生崩溃不会带走 Web 服务。服务重启后会恢复终态记录；中断的运行态会标记失败，不会自动重复消耗 GPU。

HeartMuLa 在本机 16 GB 内存环境会产生较高系统提交压力。若日志停在 checkpoint 加载且 Windows 事件 2004 报虚拟内存不足，应先释放内存或调整页面文件后再重试，不能把 `No module named 'triton'` 提示直接判作根因。

当前同一虚拟环境兼容 HeartMuLa 与 BGE 检索的已验证组合为 `transformers==4.57.0`、`tokenizers==0.22.1`、`sentence-transformers==5.7.0`。HeartMuLa 源码固定要求前两项；升级到 Transformers 5.x 会导致本机推理报 `RoPE cache is not built`。调整依赖后应先运行 `uv pip check --python .venv\Scripts\python.exe`，再分别验证风格检索与试听生成。

静态服务仅公开 `frontend/`，不公开 `.env`、模型权重或 `data/`。`frontend/demo-data.json` 仅含人工自拟的页面展示样本，没有真实任务、质检分数或音频；启动本地服务后产生的实际任务记录另行展示，不随公开仓库发布。原开发项目保留的历史长歌验证音频与生成歌词不作为公开版预置示例。

## 离线验证

```powershell
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

可运行的离线测试通过替身 Agent 检查双作词并行、匿名评审、组合选优、有限回炉、队列、失败和接口行为，不调用 DeepSeek 或 GPU。依赖未公开数据的历史实验测试会跳过。完整歌曲 Web 新路径目前只有离线接口/规则验证，尚无公开可复现的真实 DeepSeek→HeartMuLa 端到端复验；技术背景和旧阶段边界见 `docs/03-review.md`、`docs/decisions.md`。

## 第三方与许可边界

`third_party/heartlib/` 仅保留上游 HeartMuLa 的 Python 包源码、安装元数据、来源说明和 Apache-2.0 许可证；上游示例脚本与素材未纳入公开副本。它不属于 MusicAgent 的原创实现。仓库根目录暂未添加许可证，因此公开可见不等于向他人授予本项目原创代码的开源复用许可。公开样本的自拟说明不对第三方材料作再授权。
