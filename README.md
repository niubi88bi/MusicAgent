# MusicAgent

中文歌曲创作工作台。当前 Web 版支持主题输入、双作词人并行创作、匿名主观评审与规则组合选优、最多三稿的规则回炉记录、制作方案，以及基于最终采用歌词的本地 HeartMuLa 30 秒按需试听。

## 启动（Windows PowerShell）

在 `D:\MusicAgent` 项目根目录执行：

```powershell
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8123 --workers 1
```

浏览器打开 http://127.0.0.1:8123/ 。使用项目 Python 3.10 虚拟环境；依赖统一由 uv 管理。`.env` 中需配置 `DEEPSEEK_API_KEY`，其他配置见 `.env.example`。健康检查只返回密钥是否存在，不返回密钥。

旧的 `python -m http.server` 命令仅支持历史样例预览；真实创作需启动上述 FastAPI 服务，同一端口不能同时运行两种服务。

## 使用

1. 输入 2～200 字主题，可以包含曲风和情绪，点击“生成歌词”。当前文本任务以 30 秒短歌为约束。
2. 作词人 A 偏重押韵结构，作词人 B 偏重意象表达；匿名评审 Agent 只看需求和 A/B 正文，从主题、自然度、情绪、叙事意象和 Hook 五维评分。页面显示并行节点、候选分数、入选依据，以及后续各稿质检与修改反馈。
3. 文本完成后点击“生成 30 秒试听”。音频任务不会重新调用 DeepSeek；生成期间仍可查看歌词，失败后保留文本并允许手动重试。
4. 同一服务进程内，刷新带 `?task=任务编号` 的地址可继续查询；终态音频记录可在服务重启后恢复查看。

主观评审不读取 persona 名称和规则分；单方重大主题问题先淘汰，主观总分相差不足 3 分时才用规则结果决胜。双方都有重大主题问题时，两位作词人最多共同回炉 1 次。入选后仍执行确定性规则质检，最多生成 3 稿（初稿 + 最多 2 次改写）；达到上限时采用规则分最高的一稿，`capped` 仍表示未通过。任务 `done` 只表示文本流程完成。

质检报告还会展示句长、停顿、平仄统计和主题关键词直接命中。D9 第一版处于影子模式：总分与通过门槛仍只由押韵和结构决定，节奏参考与主题词面信号只作建议，不会单独触发回炉，也不能代表完整语义或可唱性。详细口径见 `docs/quality-design.md`。

## 接口和保存

- `GET /health`：服务和文本配置状态。
- `POST /api/tasks`：请求体 `{"theme":"夏夜回家"}`，接收成功立即返回 202 和 `task_id`。
- `GET /api/tasks/{task_id}`：任务状态、真实节点事件、部分文本及最终结果。
- `GET /api/tasks`：本次服务进程内的任务列表。
- `POST /api/tasks/{task_id}/audio`：为已完成文本任务创建 30 秒本地试听任务，立即返回 202。
- `GET /api/tasks/{task_id}/audio/status`：查询关联音频任务状态。
- `GET /api/tasks/{task_id}/audio`：仅在生成成功后返回对应 MP3。

一个进程同时执行 1 个文本任务，最多有 2 个等待文本任务；音频侧同时只接受 1 个任务，满载返回 429。应保持 `--workers 1`，不要在任务运行期间启用自动重载。正常关闭会等待已接收任务完成。

产物保存于 `data/output/web/<task_id>/`：最新 `task.json`、两份作词候选 `lyric_candidates.json`、`subjective_review.json`、评审历史、综合选择依据 `candidate_selection.json`、最终歌词、结构化需求、制作方案、逐轮质检报告，以及 `audio_task.json`、`audio_process.log` 和成功后的 `audio/<audio_task_id>/song.mp3`。音频通过独立子进程运行，原生崩溃不会带走 Web 服务。服务重启后会恢复终态记录；中断的运行态会标记失败，不会自动重复消耗 GPU。

HeartMuLa 在本机 16 GB 内存环境会产生较高系统提交压力。若日志停在 checkpoint 加载且 Windows 事件 2004 报虚拟内存不足，应先释放内存或调整页面文件后再重试，不能把 `No module named 'triton'` 提示直接判作根因。

静态服务仅公开 `frontend/`，不公开 `.env`、模型权重或 `data/`。历史样例来源记录在 `frontend/demo-data.json` 中；`frontend/assets/d5-summer-night.mp3` 受现有 Git 忽略规则影响，换机器时需准备对应本地产物。

## 离线验证

```powershell
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试通过替身 Agent 验证双作词并行、匿名评审、组合选优、有限回炉、D9 影子报告与 D9.4 标注指标、队列、失败和接口行为，不调用 DeepSeek 或 GPU。当前完整回归为 90/90；真实运行证据和测试边界见 `docs/work-log.md`，详细计划见 `docs/02-implementation-plan.md`。

D9.4 首轮人工标注使用 `http://127.0.0.1:8000/d9-4-annotator.html`。只选择 `data/output/evaluations/d9-4-annotation-v1/cards.round1.jsonl` 和同目录的 `annotations.round1.jsonl`，不要打开包含程序预测与来源信息的 `manifest.private.jsonl`。
