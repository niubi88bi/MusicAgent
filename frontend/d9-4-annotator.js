"use strict";

const state = { cards: [], annotations: [], index: 0, storageKey: "" };

const elements = {
  loader: document.querySelector("#loader"),
  workspace: document.querySelector("#workspace"),
  cardsFile: document.querySelector("#cardsFile"),
  annotationsFile: document.querySelector("#annotationsFile"),
  loadButton: document.querySelector("#loadButton"),
  loadError: document.querySelector("#loadError"),
  progressText: document.querySelector("#progressText"),
  progressBar: document.querySelector("#progressBar"),
  saveStatus: document.querySelector("#saveStatus"),
  completionHint: document.querySelector("#completionHint"),
  samplePosition: document.querySelector("#samplePosition"),
  sectionBadge: document.querySelector("#sectionBadge"),
  lyricsLines: document.querySelector("#lyricsLines"),
  stanzaPattern: document.querySelector("#stanzaPattern"),
  rhythmRating: document.querySelector("#rhythmRating"),
  confidence: document.querySelector("#confidence"),
  needsMelody: document.querySelector("#needsMelody"),
  stanzaNotes: document.querySelector("#stanzaNotes"),
  previousButton: document.querySelector("#previousButton"),
  nextButton: document.querySelector("#nextButton"),
  nextIncompleteButton: document.querySelector("#nextIncompleteButton"),
  exportButton: document.querySelector("#exportButton"),
  exportStatus: document.querySelector("#exportStatus"),
};

function parseJsonl(text) {
  return text.split(/\r?\n/).filter((line) => line.trim()).map((line, index) => {
    try { return JSON.parse(line); }
    catch (error) { throw new Error(`第 ${index + 1} 行不是有效 JSON：${error.message}`); }
  });
}

function isLineComplete(line) {
  return [0, 1, 2].includes(line.length_fit) && [0, 1, 2].includes(line.pause_fit);
}

function isStructuralMarkerLine(line) {
  return /^\[(?:Intro|Verse|Pre-?Chorus|Chorus|Bridge|Hook|Outro|Interlude|Instrumental)(?:\s+\d+)?\]$/i.test(line?.text?.trim() || "");
}

function isCardComplete(annotation, card) {
  return Boolean(
    annotation.stanza_pattern &&
    annotation.rhythm_rating &&
    annotation.confidence &&
    typeof annotation.needs_melody === "boolean" &&
    card?.lines?.every((line, index) => (
      isStructuralMarkerLine(line) || isLineComplete(annotation.lines[index])
    ))
  );
}

function assertMatchingFiles(cards, annotations) {
  if (!Array.isArray(cards) || !Array.isArray(annotations) || !cards.length) {
    throw new Error("盲卡或标注模板为空。请选择本轮配套的两个 JSONL 文件。");
  }
  const cardIds = new Set();
  cards.forEach((card, index) => {
    if (!card?.sample_id || cardIds.has(card.sample_id) || !Array.isArray(card.lines) || !card.lines.length) {
      throw new Error(`第 ${index + 1} 张盲卡格式不正确或 sample_id 重复。`);
    }
    cardIds.add(card.sample_id);
  });
  if (annotations.length !== cards.length) throw new Error("盲卡与标注模板数量不一致。");
  const annotationIds = new Set();
  annotations.forEach((annotation, index) => {
    if (!annotation?.sample_id || annotationIds.has(annotation.sample_id)) {
      throw new Error(`第 ${index + 1} 条标注模板格式不正确或 sample_id 重复。`);
    }
    annotationIds.add(annotation.sample_id);
  });
  const annotationById = new Map(annotations.map((row) => [row.sample_id, row]));
  return cards.map((card) => {
    const annotation = annotationById.get(card.sample_id);
    if (!annotation || !Array.isArray(annotation.lines) || annotation.lines.length !== card.lines.length) {
      throw new Error(`盲卡 ${card.sample_id} 与标注模板的逐行项不匹配。`);
    }
    return annotation;
  });
}

function setSaveStatus(message, isError = false) {
  elements.saveStatus.textContent = message;
  elements.saveStatus.classList.toggle("error-text", isError);
}

function persist() {
  const annotation = state.annotations[state.index];
  if (annotation) annotation.annotated_at = new Date().toISOString();
  try {
    localStorage.setItem(state.storageKey, JSON.stringify(state.annotations));
    setSaveStatus(`已保存到本机浏览器 · ${new Date().toLocaleTimeString()}`);
  } catch (error) {
    setSaveStatus(`本地保存失败：${error.message}；请立即导出备份。`, true);
  }
  updateProgress();
}

function updateProgress() {
  const complete = state.cards.filter((card, index) => isCardComplete(state.annotations[index], card)).length;
  elements.progressText.textContent = `${complete} / ${state.cards.length} 张完成`;
  elements.progressBar.max = Math.max(1, state.cards.length);
  elements.progressBar.value = complete;
  const remaining = state.cards.length - complete;
  elements.completionHint.textContent = remaining
    ? `还差 ${remaining} 张完整标注；未完成时也可导出备份。`
    : "全部完成。请导出 JSONL 后发送给 Codex 进行格式校验与复标抽样。";
}

function choiceGroup(title, field, line, lineIndex) {
  const wrapper = document.createElement("div");
  wrapper.className = "label-group";
  const label = document.createElement("span");
  label.className = "label-title";
  label.textContent = title;
  const choices = document.createElement("div");
  choices.className = "choice-row";
  for (const value of [0, 1, 2]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = String(value);
    if (line[field] === value) button.classList.add("selected");
    button.addEventListener("click", () => {
      state.annotations[state.index].lines[lineIndex][field] = value;
      persist();
      render();
    });
    choices.append(button);
  }
  wrapper.append(label, choices);
  return wrapper;
}

function render() {
  const card = state.cards[state.index];
  const annotation = state.annotations[state.index];
  elements.samplePosition.textContent = `第 ${state.index + 1} / ${state.cards.length} 张 · ${card.sample_id}`;
  elements.sectionBadge.textContent = card.section;
  elements.lyricsLines.replaceChildren();
  card.lines.forEach((line, lineIndex) => {
    const row = document.createElement("div");
    row.className = isStructuralMarkerLine(line) ? "structural-marker" : "lyric-line";
    const text = document.createElement("div");
    text.className = "line-text";
    text.textContent = isStructuralMarkerLine(line)
      ? `${line.text}（结构标记，不参与标注）`
      : line.text;
    const lineAnnotation = annotation.lines[lineIndex];
    row.append(text);
    if (!isStructuralMarkerLine(line)) {
      row.append(
        choiceGroup("句长是否需要修改", "length_fit", lineAnnotation, lineIndex),
        choiceGroup("停顿是否需要修改", "pause_fit", lineAnnotation, lineIndex),
      );
    }
    elements.lyricsLines.append(row);
  });
  elements.stanzaPattern.value = annotation.stanza_pattern || "";
  elements.rhythmRating.value = annotation.rhythm_rating || "";
  elements.confidence.value = annotation.confidence || "";
  elements.needsMelody.checked = annotation.needs_melody === true;
  elements.stanzaNotes.value = annotation.notes || "";
  elements.previousButton.disabled = state.index === 0;
  elements.nextButton.disabled = state.index === state.cards.length - 1;
  updateProgress();
}

function updateStanzaFields() {
  const annotation = state.annotations[state.index];
  if (!annotation) return;
  annotation.stanza_pattern = elements.stanzaPattern.value || null;
  annotation.rhythm_rating = elements.rhythmRating.value ? Number(elements.rhythmRating.value) : null;
  annotation.confidence = elements.confidence.value || null;
  annotation.needs_melody = elements.needsMelody.checked;
  annotation.notes = elements.stanzaNotes.value;
  persist();
}

function goTo(index) {
  updateStanzaFields();
  state.index = Math.max(0, Math.min(index, state.cards.length - 1));
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function loadFiles() {
  elements.loadError.textContent = "";
  try {
    if (!elements.cardsFile.files[0] || !elements.annotationsFile.files[0]) {
      throw new Error("请同时选择盲卡和标注模板。" );
    }
    const [cardsText, annotationsText] = await Promise.all([
      elements.cardsFile.files[0].text(),
      elements.annotationsFile.files[0].text(),
    ]);
    const cards = parseJsonl(cardsText);
    let annotations = parseJsonl(annotationsText);
    annotations = assertMatchingFiles(cards, annotations);
    state.cards = cards;
    const cardsFile = elements.cardsFile.files[0];
    state.storageKey = `d94:${cardsFile.name}:${cardsFile.size}:${cardsFile.lastModified}`;
    const saved = localStorage.getItem(state.storageKey);
    if (saved) {
      const savedAnnotations = JSON.parse(saved);
      try {
        state.annotations = assertMatchingFiles(cards, savedAnnotations);
        setSaveStatus("已恢复本机浏览器中的未导出进度。");
      } catch (error) {
        state.annotations = annotations;
        setSaveStatus("发现不匹配的旧本地进度，已改用你选择的标注模板；旧进度未删除。", true);
      }
    } else {
      state.annotations = annotations;
      setSaveStatus("已载入空白模板；填写时会自动保存到本机浏览器。");
    }
    state.index = 0;
    elements.loader.hidden = true;
    elements.workspace.hidden = false;
    render();
  } catch (error) {
    elements.loadError.textContent = error.message;
  }
}

function exportAnnotations() {
  updateStanzaFields();
  const body = state.annotations.map((row) => JSON.stringify(row)).join("\n") + "\n";
  const remaining = state.cards.filter((card, index) => !isCardComplete(state.annotations[index], card)).length;
  const blob = new Blob([body], { type: "application/x-ndjson;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = remaining
    ? "annotations.round1.incomplete-backup.jsonl"
    : "annotations.round1.completed.jsonl";
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
  elements.exportStatus.textContent = remaining
    ? `已导出备份，仍有 ${remaining} 张未完成；请继续标注，完成后再次导出。`
    : "已导出完整标注。请把该 JSONL 文件发送给 Codex。";
}

elements.loadButton.addEventListener("click", loadFiles);
elements.previousButton.addEventListener("click", () => goTo(state.index - 1));
elements.nextButton.addEventListener("click", () => goTo(state.index + 1));
elements.nextIncompleteButton.addEventListener("click", () => {
  updateStanzaFields();
  const offset = state.cards.slice(state.index + 1).findIndex(
    (card, offsetIndex) => !isCardComplete(state.annotations[state.index + 1 + offsetIndex], card),
  );
  const fallback = state.cards.findIndex((card, index) => !isCardComplete(state.annotations[index], card));
  goTo(offset >= 0 ? state.index + 1 + offset : fallback >= 0 ? fallback : state.index);
});
elements.exportButton.addEventListener("click", exportAnnotations);
for (const input of [elements.stanzaPattern, elements.rhythmRating, elements.confidence, elements.needsMelody]) {
  input.addEventListener("change", updateStanzaFields);
}
elements.stanzaNotes.addEventListener("input", updateStanzaFields);
window.addEventListener("beforeunload", updateStanzaFields);
