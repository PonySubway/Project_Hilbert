const state = {
  query: "苹果",
  selectedWord: null
};

const elements = {
  status: document.querySelector("#status"),
  searchForm: document.querySelector("#search-form"),
  queryInput: document.querySelector("#query-input"),
  layersInput: document.querySelector("#layers-input"),
  perLayerInput: document.querySelector("#per-layer-input"),
  ringStage: document.querySelector("#ring-stage"),
  detailWord: document.querySelector("#detail-word"),
  detailBody: document.querySelector("#detail-body"),
  relationForm: document.querySelector("#relation-form"),
  leftInput: document.querySelector("#left-input"),
  rightInput: document.querySelector("#right-input"),
  scoreFill: document.querySelector("#score-fill"),
  scoreOutput: document.querySelector("#score-output"),
  relationDetail: document.querySelector("#relation-detail")
};

elements.searchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch();
});

elements.relationForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runRelation();
});

window.addEventListener("resize", () => {
  const cached = elements.ringStage.dataset.lastPayload;
  if (cached) {
    renderRing(JSON.parse(cached));
  }
});

boot();

async function boot() {
  await refreshHealth();
  await runSearch();
  await runRelation();
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const payload = await response.json();
    elements.status.textContent = `${payload.model} / ${payload.concept_count}`;
  } catch (error) {
    elements.status.textContent = "API 未连接";
    elements.status.title = error.message;
  }
}

async function runSearch() {
  const query = elements.queryInput.value.trim();
  const layers = Number.parseInt(elements.layersInput.value || "5", 10);
  const perLayer = Number.parseInt(elements.perLayerInput.value || "8", 10);
  if (!query) {
    return;
  }
  state.query = query;
  setRingLoading(query);
  setButtonState(elements.searchForm, true);
  try {
    const payload = await postJson("/api/ring", {
      query,
      layers,
      per_layer: perLayer
    });
    elements.ringStage.dataset.lastPayload = JSON.stringify(payload);
    renderRing(payload);
  } catch (error) {
    elements.ringStage.innerHTML = `<div class="empty-state error">${escapeHtml(error.message)}</div>`;
  } finally {
    setButtonState(elements.searchForm, false);
  }
}

async function runRelation() {
  const left = elements.leftInput.value.trim();
  const right = elements.rightInput.value.trim();
  if (!left || !right) {
    return;
  }
  setButtonState(elements.relationForm, true);
  try {
    const payload = await postJson("/api/relate", { left, right });
    const score = Number(payload.score || 0);
    elements.scoreFill.style.width = `${Math.max(0, Math.min(1, score)) * 100}%`;
    elements.scoreOutput.textContent = score.toFixed(3);
    elements.relationDetail.textContent = `${payload.label}，内积 ${Number(payload.inner_product).toFixed(3)}，模型 ${payload.model}`;
    elements.relationDetail.classList.remove("error");
  } catch (error) {
    elements.relationDetail.textContent = error.message;
    elements.relationDetail.classList.add("error");
  } finally {
    setButtonState(elements.relationForm, false);
  }
}

function renderRing(payload) {
  const layers = payload.layers || [];
  elements.ringStage.innerHTML = "";
  drawCircles(layers.length);

  const center = document.createElement("div");
  center.className = "ring-center";
  center.textContent = payload.query;
  elements.ringStage.appendChild(center);

  layers.forEach((layer, layerIndex) => {
    const items = layer.items || [];
    const radius = radiusForLayer(layerIndex, layers.length);
    const angleOffset = layerIndex * 0.43;
    items.forEach((item, itemIndex) => {
      const angle = angleOffset + (Math.PI * 2 * itemIndex) / Math.max(items.length, 1);
      const x = 50 + Math.cos(angle) * radius;
      const y = 50 + Math.sin(angle) * radius;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ring-word";
      button.dataset.layer = String(layer.layer);
      button.style.left = `${x}%`;
      button.style.top = `${y}%`;
      button.title = `相似度 ${Number(item.similarity).toFixed(3)} / 距离 ${Number(item.distance).toFixed(3)}`;
      button.textContent = item.concept;
      button.addEventListener("click", () => explainWord(item.concept));
      elements.ringStage.appendChild(button);
    });
  });
}

function drawCircles(count) {
  for (let index = count - 1; index >= 0; index -= 1) {
    const diameter = radiusForLayer(index, count) * 2;
    const circle = document.createElement("div");
    circle.className = "ring-circle";
    circle.style.width = `${diameter}%`;
    circle.style.height = `${diameter}%`;
    elements.ringStage.appendChild(circle);
  }
}

function radiusForLayer(index, count) {
  if (count <= 1) {
    return 34;
  }
  return 15 + (index * 28) / (count - 1);
}

async function explainWord(word) {
  state.selectedWord = word;
  elements.detailWord.textContent = word;
  elements.detailBody.innerHTML = '<p class="loading">生成中...</p>';
  try {
    const payload = await postJson("/api/explain", {
      word,
      query: state.query
    });
    elements.detailBody.textContent = payload.explanation;
    elements.detailBody.classList.remove("error");
  } catch (error) {
    elements.detailBody.textContent = error.message;
    elements.detailBody.classList.add("error");
  }
}

function setRingLoading(query) {
  elements.ringStage.innerHTML = `<div class="loading-state">正在计算 ${escapeHtml(query)}</div>`;
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

async function readError(response) {
  const clone = response.clone();
  try {
    const payload = await clone.json();
    if (payload && payload.detail) {
      return String(payload.detail);
    }
  } catch {
    // Fall through to text body.
  }
  const text = await response.text();
  return text || `HTTP ${response.status}`;
}

function setButtonState(form, disabled) {
  form.querySelectorAll("button").forEach((button) => {
    button.disabled = disabled;
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    };
    return entities[character];
  });
}
