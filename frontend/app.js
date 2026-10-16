/* 歌曲模式随文本任务保存；音频上限由服务端根据任务模式决定。 */
'use strict';
const $ = (id) => document.getElementById(id);
let records = [];
let current = null;
let toastTimer;
let apiReady = false;
let submitting = false;
let pollTimer;
let pollFailures = 0;
let syncBusy = false;
let pendingClarification = null;
const stageNames = {validate_input:'校验输入',requirement:'理解创作意图',lyricist_structured:'作词人 A 创作',lyricist_imagery:'作词人 B 创作',subjective_judge:'匿名主观评审',lyricist_selection:'候选综合选优',lyricist:'入选歌词回炉',quality:'规则质检',producer:'整理制作方案'};
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function notify(message) {
  clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 5000);
}
function clearClarification() {
  pendingClarification = null;
  $('clarification-card').hidden = true;
  $('clarification-question').textContent = '';
  $('clarification-options').replaceChildren();
  setSubmitState();
}
function renderClarification(decision) {
  const question = decision.questions?.[0];
  if (!question) { clearClarification(); return; }
  pendingClarification = decision;
  $('clarification-question').textContent = question.question;
  const options = $('clarification-options'); options.replaceChildren();
  for (const option of question.options) {
    const label = node('label', 'clarification-option');
    const input = document.createElement('input');
    input.type = 'radio'; input.name = 'clarification-option'; input.value = option.code;
    const copy = node('span'); copy.append(node('strong', '', option.label), node('small', '', option.description));
    label.append(input, copy); options.append(label);
  }
  $('clarification-card').hidden = false;
  setSubmitState();
  $('clarification-card').scrollIntoView({block:'nearest', behavior:'smooth'});
}
function selectedClarificationAnswers() {
  if (!pendingClarification) return [];
  const selected = document.querySelector('input[name="clarification-option"]:checked');
  return selected ? [{code: pendingClarification.questions[0].code, option_code: selected.value}] : [];
}
function showView(name) {
  if (name !== 'workspace') {
    const url = new URL(location.href); url.searchParams.delete('task');
    history.replaceState(null, '', url);
  }
  ['home', 'library', 'workspace'].forEach((view) => { $(view + '-view').hidden = view !== name; });
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.action === name));
  $('breadcrumb').textContent = {home: '创作工作台', library: '创作记录', workspace: '创作记录 / 作品详情'}[name];
  window.scrollTo({top: 0});
}
function sampleCard(record) {
  const button = node('button', 'example-card');
  const cover = node('span', 'cover ' + (record.audio_url ? 'record' : 'paper'), record.audio_url ? undefined : '词');
  cover.setAttribute('aria-hidden', 'true');
  const copy = node('span', 'example-copy');
  copy.append(node('strong', '', record.title), node('p', '', record.subtitle), node('small', '', record.task ? '文本任务 · 查看进度与结果' : record.demo_kind === 'self_authored' ? '自拟示例 · 非模型产物' : record.audio_url ? '真实音频 · 点击试听' : '真实文本 · 查看质检'));
  button.append(cover, copy, node('span', 'example-arrow', '↗'));
  button.addEventListener('click', () => openRecord(record.id));
  return button;
}
function tab(name) {
  ['lyrics', 'process'].forEach((key) => {
    $(key + '-panel').hidden = key !== name;
    $(key + '-tab').classList.toggle('active', key === name);
    $(key + '-tab').setAttribute('aria-selected', String(key === name));
    $(key + '-tab').tabIndex = key === name ? 0 : -1;
  });
}
function addProcess(title, detail) {
  const item = node('div', 'process-item');
  item.append(node('h3', '', title), node('p', '', detail));
  $('process-panel').append(item);
}
function renderProcess(record) {
  $('process-panel').replaceChildren();
  if (record.demo_kind === 'self_authored') {
    addProcess('自拟展示样本', '歌词与制作标签为公开版人工编写，仅用于展示页面结构；没有调用模型、规则质检或音频生成。');
    return;
  }
  if (record.task) {
    const task = record.task;
    for (const event of task.progress.events || []) {
      const status = {running:'开始',done:'完成',failed:'失败'}[event.status];
      addProcess(`${stageNames[event.stage] || event.stage} · ${status}`, `${new Date(event.at).toLocaleTimeString()}${event.attempt ? ` · 第 ${event.attempt} 稿` : ''}`);
    }
    if (!task.progress.events?.length) addProcess('等待执行', '任务已进入队列，尚未开始调用文本模型。');
    const candidateRounds = (record.candidates || []).map((item) => item.generation_round || 1);
    const latestCandidateRound = candidateRounds.length ? Math.max(...candidateRounds) : 1;
    for (const candidate of (record.candidates || []).filter((item) => (item.generation_round || 1) === latestCandidateRound)) {
      const selected = record.selection?.selected_candidate_id === candidate.candidate_id;
      addProcess(
        `${candidate.display_name}${selected ? ' · 综合评审入选' : ''}`,
        `规则总分 ${candidate.scores.total} / 100 · 押韵 ${candidate.scores.rhyme} · 结构 ${candidate.scores.structure}`
      );
    }
    if (record.subjective_review) {
      const reviews = record.subjective_review.candidate_reviews || {};
      const scores = Object.entries(reviews).map(([id, review]) => `${id} ${review.weighted_total}`).join(' · ');
      addProcess('匿名主观评审已保存', `${scores || '评分已完成'} · ${record.subjective_review.comparison_summary || ''}`);
    }
    record.history.forEach((round) => addProcess(`第 ${round.attempt} 稿${round.scores ? ` · 规则总分 ${round.scores.total}` : ' · 待质检'}`, round.revision_feedback || (round.scores ? '本轮规则检查已完成。' : '草稿已生成，尚未完成规则检查。')));
    if (record.report?.selected_attempt) addProcess(`采用第 ${record.report.selected_attempt} 稿`, `生成 ${record.report.attempts} 次 · 回炉 ${Math.max(0, record.report.attempts - 1)} 次`);
    if (task.status === 'failed') addProcess('任务失败', task.error);
    if (task.status === 'done') addProcess('文本创作完成', '文本产物已保存，音频会作为独立任务按需生成。');
    if (task.input?.song_mode === 'full' && task.result?.full_song_audit) addProcess(task.result.full_song_audit.passed ? '完整歌曲文本准入通过' : '完整歌曲文本准入未通过', task.result.full_song_audit.passed ? '仅确认歌词结构与长度；是否唱完及发音仍需人工试听。' : task.result.full_song_audit.issues.join('；'));
    if (record.audio_task) {
      const audioStatus = {pending:'等待生成',running:'HeartMuLa 生成中',done:record.task.input?.song_mode === 'full' ? '完整歌曲音频已生成' : '30 秒试听已生成',failed:'音频生成失败'}[record.audio_task.status];
      addProcess(audioStatus, record.audio_task.error || `使用最终采用的第 ${record.audio_task.selected_attempt} 稿，不重新生成歌词。`);
    }
    return;
  }
  addProcess('理解创作意图', record.requirement.theme);
  if (record.history.length) {
    record.history.forEach((round) => {
      const status = {passed:'规则检查通过',retrying:'需要改写',capped:'达到生成上限'}[round.quality_status] || '检查完成';
      addProcess(`第 ${round.attempt} 稿 · ${status}`, `规则总分 ${round.scores.total} / 100` + (round.revision_feedback ? '\n' + round.revision_feedback : ''));
    });
    addProcess(`采用第 ${record.report.selected_attempt} 稿`, `生成 ${record.report.attempts} 次 · 回炉 ${Math.max(0, record.report.attempts - 1)} 次`);
  } else if (record.report?.quality_report) {
    addProcess('最终歌词质检已保存', `规则总分 ${record.report.scores.total} / 100；独立验证只保存最终版本，不展示不存在的逐轮记录。`);
  } else { addProcess('歌词已保存', '这份历史作品生成时尚未记录逐轮质检，无法展示改写过程。'); }
  if (record.provenance?.kind === 'isolated_validation') {
    addProcess('独立长歌验证 · 人工试听通过', `实际时长 ${record.provenance.actual_duration_seconds} 秒；240 秒是生成安全上限。这是导入的历史成品，不代表新的 Web 生成路径已验收。`);
  }
  addProcess('制作方案已保存', record.production.tags.join(' · '));
  addProcess(record.audio_url ? '音频已生成' : '本任务未生成音频', record.audio_url ? '播放器中的音频与本作品保存的歌词对应。' : '本次验证跳过了音频模型，没有可播放音频。');
}
function renderLyrics(round) {
  const content = $('lyrics-content'); content.replaceChildren();
  const rhymeLines = round.quality_report?.rhyme?.lines || [];
  let stanza;
  const labels = {Verse:'主歌',Chorus:'副歌',Bridge:'桥段',Intro:'前奏',Outro:'尾声'};
  round.lyrics.split(/\r?\n/).forEach((line, index) => {
    const section = line.match(/^\[([^\]]+)\]$/);
    if (section) {
      stanza = node('section', 'stanza'); stanza.append(node('h3', '', `${labels[section[1]] || section[1]} / ${section[1].toUpperCase()}`)); content.append(stanza);
    } else if (line.trim()) {
      if (!stanza) { stanza = node('section', 'stanza'); content.append(stanza); }
      const row = node('p', 'lyric-line'); row.id = `line-${index + 1}`; row.append(node('span', '', line));
      const analysis = rhymeLines.find((entry) => entry.line_no === index + 1);
      if (analysis?.rhyme_group) row.append(node('span', 'rhyme-label', `${analysis.ending_char} · ${analysis.rhyme_group}`));
      stanza.append(row);
    }
  });
}
function renderQuality(round) {
  const area = $('quality-content'); area.replaceChildren();
  const report = round.quality_report;
  $('quality-status').className = 'status-label';
  if (!report) {
    $('quality-status').textContent = current?.task ? '尚无结果' : current?.demo_kind === 'self_authored' ? '未执行' : '未记录';
    const detail = node('div', 'quality-detail');
    detail.append(node('h3', '', current?.task ? '等待规则检查结果' : current?.demo_kind === 'self_authored' ? '自拟展示样本' : '这是一份早期音频样例'), node('p', '', current?.task ? '只有本稿完成规则检查后才显示评分。' : current?.demo_kind === 'self_authored' ? '此样本未运行规则质检，不提供分数或通过结论。' : '原任务没有保存规则质检报告，本页不补造分数，也不把音频生成成功标为歌词质检通过。'));
    area.append(detail); return;
  }
  const status = round.quality_status;
  $('quality-status').textContent = status === 'capped' ? '达到上限' : report.passed ? (report.issues.length ? '达标 · 有建议' : '通过') : '需改写';
  $('quality-status').classList.toggle('warning', !report.passed);
  const score = node('div', 'score-block'); const number = node('div', 'score-number', String(report.total_score));
  number.append(node('small', '', '/ 100')); score.append(number, node('p', 'score-caption', '规则检查总分')); area.append(score);
  const metrics = [['押韵', report.rhyme.score], ['结构', report.structure.score]];
  if (report.rhythm) metrics.push(['节奏参考（不计分）', report.rhythm.score]);
  metrics.forEach(([label, value]) => {
    const metric = node('div', 'metric'); const heading = node('div', 'metric-label'); heading.append(node('span', '', label), node('span', '', String(value)));
    const bar = node('div', 'metric-bar'); const fill = node('span'); fill.style.width = `${Math.max(0, Math.min(100, value))}%`; bar.append(fill); metric.append(heading, bar); area.append(metric);
  });
  const detail = node('div', 'quality-detail'); detail.append(node('h3', '', '检查依据'));
  detail.append(node('p', '', `已分析 ${report.rhyme.analyzable_lines} 个韵脚`), node('p', '', report.structure.hook_present ? 'Hook 已出现在副歌中' : 'Hook 未满足副歌位置要求'));
  for (const [section, count] of Object.entries(report.structure.section_line_counts)) detail.append(node('p', '', `${({Verse:'主歌',Chorus:'副歌',Bridge:'桥段'})[section] || section} · ${count} 行`));
  if (report.rhythm) {
    const patterns = Object.entries(report.rhythm.stanza_patterns || {}).map(([section, pattern]) => `${section}：${pattern}`).join(' · ');
    detail.append(node('p', '', `节奏参考：${patterns || '暂无可比较段落'}`));
  }
  if (report.theme_signal) {
    const signal = report.theme_signal;
    const coverage = signal.keywords.length ? `${signal.matched_keywords.length}/${signal.keywords.length}` : '未提供关键词';
    const hookMatches = signal.hook_matches.length ? signal.hook_matches.join('、') : '无直接命中';
    detail.append(node('p', '', `主题词面信号：${coverage}；Hook：${hookMatches}`));
  }
  for (const issue of report.issues) {
    const issueButton = node('button', 'issue-button', issue.message.replace(/。+$/, '') + '。' + issue.suggestion);
    issueButton.addEventListener('click', () => {
      tab('lyrics'); const line = $(`line-${issue.line_no}`);
      if (line) { line.scrollIntoView({block:'center'}); line.classList.add('highlight'); setTimeout(() => line.classList.remove('highlight'), 2500); }
      else notify(issue.suggestion);
    }); detail.append(issueButton);
  }
  if (status === 'capped') detail.append(node('p', '', '已达到修改上限，采用规则分最高的版本。此结果未通过质检。'));
  detail.append(node('p', 'quality-note', '总分和通过门槛仍只由押韵与结构决定；节奏和主题词面信号仅作建议，不代表完整语义、可唱性或音乐听感。')); area.append(detail);
}
function selectVersion(index) {
  let round = current.history[index] || {lyrics:current.lyrics,attempt:current.report?.selected_attempt || 1,
    quality_report:current.report?.quality_report,quality_status:current.report?.quality_status};
  // capped 的最高分可能来自早期稿；最终采用稿的结论不能沿用其旧 retrying 状态。
  if (current.report?.selected_attempt === round.attempt && current.report.quality_status === 'capped') round = {...round, quality_status:'capped'};
  renderLyrics(round); renderQuality(round);
  $('version-caption').textContent = current.history.length ? `第 ${round.attempt} 稿 · 共 ${current.history.length} 稿` : current.task ? '尚无歌词草稿' : current.demo_kind === 'self_authored' ? '自拟示例歌词' : current.report?.quality_report ? '最终保存歌词' : '原任务歌词';
  $('selected-version').textContent = current.report?.selected_attempt === round.attempt ? '最终采用' : '';
}
function openRecord(id, refresh = false) {
  const record = records.find((item) => item.id === id); if (!record) return;
  const oldIndex = Number($('version-select').value);
  const wasDone = current?.task?.status === 'done';
  current = record;
  if (!refresh) {
    showView('workspace'); tab('lyrics');
    const url = new URL(location.href);
    if (record.task) url.searchParams.set('task', record.id); else url.searchParams.delete('task');
    history.replaceState(null, '', url);
  }
  $('task-status').hidden = !record.task;
  if (record.task) {
    const task = record.task;
    $('task-status').textContent = task.status === 'failed' ? task.error : task.status === 'done' ? (record.report.quality_status === 'capped' ? '文本已完成 · 达到修改上限，采用最高规则分版本，仍未通过质检。' : '文本创作完成 · 规则检查通过') : task.status === 'pending' ? '任务已提交，等待执行…' : `正在${stageNames[task.progress.stage] || '处理'}…`;
  }
  $('work-title').textContent = record.title; $('work-source').textContent = record.source_label;
  $('work-tags').replaceChildren(...record.production.tags.slice(0, 5).map((tag) => node('span', '', tag)));
  $('version-select').replaceChildren();
  (record.history.length ? record.history : [{attempt:1}]).forEach((round, index) => { const option = node('option', '', !record.history.length ? (record.task ? '尚无草稿' : '最终歌词') : `第 ${round.attempt} 稿`); option.value = index; $('version-select').append(option); });
  const selectedIndex = Math.max(0, record.history.findIndex((round) => round.attempt === record.report?.selected_attempt));
  const index = refresh && (record.task?.status !== 'done' || wasDone) ? Math.min(oldIndex, Math.max(0,record.history.length - 1)) : selectedIndex;
  $('version-select').value = index; $('version-select').disabled = record.history.length < 2;
  selectVersion(index); renderProcess(record);
  $('copy-button').disabled = !record.lyrics;
  if (!record.lyrics) $('lyrics-content').append(node('p','muted', record.task?.status === 'failed' ? '任务尚未产出歌词。' : '歌词完成后会在这里出现。'));
  const audio = $('audio-player');
  if (record.audio_url) {
    if (audio.getAttribute('src') !== record.audio_url) {
      audio.pause();
      audio.src = record.audio_url;
    }
    $('audio-title').textContent = record.title; $('download-audio').href = record.audio_url;
    $('audio-meta').textContent = record.audio_task ? (record.task?.input?.song_mode === 'full' ? 'HeartMuLa · 完整歌曲（请人工试听）' : 'HeartMuLa · 30 秒试听') : record.audio_label || '历史生成音频';
  } else if (audio.hasAttribute('src')) {
    audio.pause();
    audio.removeAttribute('src');
    audio.load();
  }
  $('audio-dock').hidden = !record.audio_url; document.body.classList.toggle('has-audio', Boolean(record.audio_url));
  $('text-audio-note').hidden = Boolean(record.audio_url);
  updateAudioAction(record);
}

function updateAudioAction(record) {
  const button = $('audio-preview-button');
  const note = $('text-audio-note').querySelector('p');
  const status = record.audio_task?.status;
  const full = record.task?.input?.song_mode === 'full';
  const audit = record.task?.result?.full_song_audit;
  const blocked = full && (record.report?.quality_status !== 'passed' || audit?.passed !== true);
  button.disabled = !record.task || record.task.status !== 'done' || blocked || status === 'pending' || status === 'running';
  button.textContent = status === 'pending' ? '等待生成…' : status === 'running' ? 'HeartMuLa 生成中…' : blocked ? '完整歌曲歌词未达准入' : status === 'failed' ? (full ? '重新生成完整歌曲 ↗' : '重新生成 30 秒试听 ↗') : (full ? '生成完整歌曲 ↗' : '生成 30 秒试听 ↗');
  note.textContent = status === 'pending' ? '任务已提交，正在等待唯一的 GPU 执行位。' :
    status === 'running' ? '正在使用最终采用的歌词生成音频，页面可以继续浏览。' :
    status === 'failed' ? record.audio_task.error :
    blocked ? `本次长歌歌词未满足准入：${audit?.issues?.join('；') || '规则质检未通过'}。请重新创作后再生成音频。` :
    record.task ? (full ? `将使用最终采用的第 ${record.report?.selected_attempt || 1} 稿生成长歌；240 秒是生成上限，不保证具体时长或歌词全部唱完，完成后请试听核对。` : `将使用最终采用的第 ${record.report?.selected_attempt || 1} 稿调用本地 HeartMuLa，不会重新调用 DeepSeek。`) :
    '这份历史文本示例没有可关联的按需音频任务。';
}
document.querySelectorAll('[data-action]').forEach((button) => button.addEventListener('click', () => {
  const action = button.dataset.action;
  if (action === 'quality-demo') { const demo = records.find(item => item.id === 'public-self-authored-demo'); if (demo) openRecord(demo.id); else notify('示例数据还未加载，请稍后再试。'); }
  else showView(action);
}));
document.querySelector('.brand').addEventListener('click', (event) => { event.preventDefault(); showView('home'); });
document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => { $('theme').value = button.dataset.prompt; $('theme').focus(); }));
$('composer').addEventListener('submit', submitText);
$('lyrics-tab').addEventListener('click', () => tab('lyrics')); $('process-tab').addEventListener('click', () => tab('process'));
document.querySelector('.tab-list').addEventListener('keydown', (event) => { if (['ArrowLeft','ArrowRight'].includes(event.key)) { event.preventDefault(); const name = $('lyrics-panel').hidden ? 'lyrics' : 'process'; tab(name); $(name + '-tab').focus(); } });
$('theme').addEventListener('input', () => { if (pendingClarification) clearClarification(); });
$('clarification-cancel').addEventListener('click', clearClarification);
$('version-select').addEventListener('change', () => selectVersion(Number($('version-select').value)));
$('audio-preview-button').addEventListener('click', submitAudio);
$('audio-player').addEventListener('error', () => notify('音频文件未能加载，请检查本地示例文件是否完整。'));
$('copy-button').addEventListener('click', async () => {
  const index = Number($('version-select').value); const lyrics = current?.history[index]?.lyrics || current?.lyrics;
  if (!lyrics) return;
  try { await navigator.clipboard.writeText(lyrics); notify('当前版本歌词已复制。'); } catch { notify('浏览器未允许复制，请直接选中歌词复制。'); }
});
function renderLists() {
  $('record-count').textContent = records.length;
  $('examples-list').replaceChildren(); $('library-list').replaceChildren(); $('history-list').replaceChildren();
  for (const record of records) {
    if (!record.task || record.task.status === 'done') $('examples-list').append(sampleCard(record));
    $('library-list').append(sampleCard(record));
    const item = node('button', 'history-item', record.title); item.addEventListener('click', () => openRecord(record.id)); $('history-list').append(item);
  }
}
function taskRecord(task) {
  const text = task.result || task.progress?.text || {};
  const rounds = [...(text.quality_history || [])];
  if (text.lyrics && text.attempts > rounds.length) rounds.push({attempt:text.attempts,lyrics:text.lyrics});
  const report = text.quality_report ? {quality_report:text.quality_report,quality_status:text.quality_status,attempts:text.attempts,selected_attempt:text.selected_attempt} : null;
  return {id:task.task_id, task, title:text.lyric_draft?.title || (task.input.theme.length > 24 ? task.input.theme.slice(0,24) + '…' : task.input.theme),
    subtitle:{pending:'排队中',running:'创作中',done:'文本已完成',failed:'任务失败'}[task.status],
    source_label:'本次服务 / 真实文本任务', requirement:text.requirement || {theme:task.input.theme},
    production:text.production_plan || {tags:[]}, lyrics:text.lyrics || '',
    candidates:text.lyric_candidates || [], selection:text.candidate_selection || null,
    subjective_review:text.subjective_review || null,
    history:rounds, report, audio_task:task.audio || null,
    audio_url:task.audio?.status === 'done' ? task.audio.audio_url : null};
}
function mergeTask(task) {
  const record = taskRecord(task), index = records.findIndex(item => item.id === record.id);
  const changed = index < 0 || records[index].task?.updated_at !== task.updated_at || records[index].audio_task?.updated_at !== task.audio?.updated_at;
  if (index >= 0) records[index] = record; else records.unshift(record);
  if (changed && current?.id === record.id && !$('workspace-view').hidden) openRecord(record.id, true);
}
function setSubmitState() {
  const running = records.some(item => ['pending','running'].includes(item.task?.status));
  const button = $('composer').querySelector('[type="submit"]');
  button.disabled = submitting || running || !apiReady || pollFailures > 0;
  button.textContent = submitting ? '正在提交…' : running ? '创作进行中…' : pendingClarification ? '确认选择并生成 ↗' : '生成歌词 ↗';
}
async function requestJson(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(path, {...options, signal:controller.signal, cache:'no-store'});
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : data.detail?.message;
      throw new Error(detail || '输入不符合要求，请输入 2～200 字的主题。');
    }
    return data;
  } finally { clearTimeout(timeout); }
}
async function syncTasks() {
  if (syncBusy) return;
  syncBusy = true; clearTimeout(pollTimer);
  try {
    const data = await requestJson('/api/tasks');
    const ids = new Set(data.items.map(task => task.task_id));
    if (current?.task && !ids.has(current.id)) {
      $('task-status').hidden = false;
      $('task-status').textContent = '任务已失效，服务可能已重启。页面中的旧内容不能代表任务仍在执行。';
    }
    records = records.filter(record => !record.task || ids.has(record.id));
    for (const task of data.items) mergeTask(task);
    pollFailures = 0; renderLists(); setSubmitState();
    if (records.some(item => ['pending','running'].includes(item.task?.status) || ['pending','running'].includes(item.audio_task?.status))) pollTimer = setTimeout(syncTasks, 1200);
  } catch {
    pollFailures += 1;
    if (pollFailures === 1) notify('任务状态暂时无法获取，正在重连。请勿重复提交。');
    pollTimer = setTimeout(syncTasks, Math.min(1000 * 2 ** Math.min(pollFailures, 4), 15000));
  } finally { syncBusy = false; }
}
async function submitText(event) {
  event.preventDefault();
  if (!apiReady || submitting || records.some(item => ['pending','running'].includes(item.task?.status))) return;
  const theme = $('theme').value.trim();
  if (theme.length < 2 || theme.length > 200) { notify('请输入 2～200 字的创作主题。'); $('theme').focus(); return; }
  const clarificationAnswers = selectedClarificationAnswers();
  if (pendingClarification && !clarificationAnswers.length) { notify('请先选择一种替代配器，或选择“系统按曲风选择”。'); return; }
  const body = {theme, song_mode:document.querySelector('input[name="song-mode"]:checked')?.value || 'preview'};
  if (clarificationAnswers.length) body.clarification_answers = clarificationAnswers;
  submitting = true; setSubmitState();
  try {
    const preflight = await requestJson('/api/tasks/preflight', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if (preflight.status === 'clarification_required') {
      renderClarification(preflight);
      notify('这条需求还缺少一项制作选择。');
      return;
    }
    if (preflight.status === 'advisory_only' && preflight.advisories?.length) notify(preflight.advisories[0]);
    const task = await requestJson('/api/tasks', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    clearClarification();
    mergeTask(task); renderLists(); openRecord(task.task_id); tab('process');
  } catch (error) { notify(error.name === 'AbortError' || error instanceof TypeError ? '提交结果暂时无法确认，正在查询任务列表，请勿重复提交。' : error.message); }
  finally { await syncTasks(); submitting = false; setSubmitState(); }
}
async function submitAudio() {
  if (!current?.task || current.task.status !== 'done' || ['pending','running'].includes(current.audio_task?.status)) return;
  $('audio-preview-button').disabled = true;
  try {
    current.audio_task = await requestJson(`/api/tasks/${encodeURIComponent(current.id)}/audio`, {method:'POST'});
    updateAudioAction(current);
    await syncTasks();
  } catch (error) {
    notify(error.name === 'AbortError' || error instanceof TypeError ? '试听提交结果暂时无法确认，正在查询状态，请勿重复提交。' : error.message);
    await syncTasks();
  }
}
async function initialize() {
  try { const data = await requestJson('demo-data.json'); records = data.records; renderLists(); }
  catch { notify('历史示例暂时无法加载。'); }
  try {
    const health = await requestJson('/health');
    apiReady = health.text_tasks_enabled && health.llm_configured;
    document.querySelector('.preview-label').textContent = '文本创作';
    document.querySelector('.preview-note').textContent = apiReady ? '支持 30 秒试听与完整歌曲两种模式；完整歌曲需先通过歌词准入，生成后仍需人工试听。' : '请先配置文本模型密钥并重启服务。历史示例仍可查看。';
    await syncTasks();
    const id = new URL(location.href).searchParams.get('task');
    if (id) {
      try { const task = await requestJson(`/api/tasks/${encodeURIComponent(id)}`); mergeTask(task); openRecord(id); }
      catch { notify('任务不存在或服务已重启，当前不能恢复该任务。'); }
    }
  } catch { document.querySelector('.preview-note').textContent = '当前只有静态预览。请使用 FastAPI 启动服务后生成歌词。'; }
  setSubmitState();
}
initialize();
