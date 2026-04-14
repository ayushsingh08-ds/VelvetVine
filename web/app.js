const form = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const probEl = document.getElementById("prob");
const verdictEl = document.getElementById("verdict");
const predEl = document.getElementById("pred");
const predTextEl = document.getElementById("pred-text");

const sweetness = document.getElementById("sweetness");
const alcohol = document.getElementById("alcohol");
const acidity = document.getElementById("acidity");

const sweetnessVal = document.getElementById("sweetness-val");
const alcoholPct = document.getElementById("alcohol-pct");
const acidityVal = document.getElementById("acidity-val");

const legendSul = document.getElementById("l-sul");
const legendAlc = document.getElementById("l-alc");
const legendSugar = document.getElementById("l-sugar");
const legendChl = document.getElementById("l-chl");
const legendSulBar = document.getElementById("l-sul-bar");
const legendAlcBar = document.getElementById("l-alc-bar");
const legendSugarBar = document.getElementById("l-sugar-bar");
const legendChlBar = document.getElementById("l-chl-bar");
const metricsEl = document.getElementById("model-metrics");

const FEATURES = [
  "fixed acidity",
  "volatile acidity",
  "citric acid",
  "residual sugar",
  "chlorides",
  "free sulfur dioxide",
  "total sulfur dioxide",
  "density",
  "pH",
  "sulphates",
  "alcohol",
];

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function sliderT(el) {
  return Number(el.value) / 100;
}

function toFixedMaybe(x) {
  if (Number.isFinite(x) && Math.abs(x) < 10) return Number(x.toFixed(4));
  if (Number.isFinite(x) && Math.abs(x) < 100) return Number(x.toFixed(3));
  return Number(x.toFixed(2));
}

function getInputByName(name) {
  return form.querySelector(`input[name="${CSS.escape(name)}"]`);
}

function setInput(name, value) {
  const input = getInputByName(name);
  if (!input) return;
  input.value = String(value);
}

function readAdvancedPayload() {
  const payload = {};
  for (const name of FEATURES) {
    const input = getInputByName(name);
    payload[name] = Number(input.value);
  }
  return payload;
}

// Baseline centered around realistic values (red-ish)
const BASE = {
  "fixed acidity": 7.8,
  "volatile acidity": 0.35,
  "citric acid": 0.35,
  "residual sugar": 2.1,
  chlorides: 0.05,
  "free sulfur dioxide": 15,
  "total sulfur dioxide": 50,
  density: 0.994,
  pH: 3.32,
  sulphates: 0.85,
  alcohol: 12.8,
};

// Loose realistic ranges (mostly from red dataset extremes)
const R = {
  "fixed acidity": [4.6, 15.9],
  "volatile acidity": [0.12, 1.58],
  "citric acid": [0.0, 1.0],
  "residual sugar": [0.9, 15.5],
  chlorides: [0.012, 0.611],
  "free sulfur dioxide": [1.0, 72.0],
  "total sulfur dioxide": [6.0, 289.0],
  density: [0.99007, 1.00369],
  pH: [2.74, 4.01],
  sulphates: [0.33, 2.0],
  alcohol: [8.0, 14.9],
};

function applySliders() {
  // Sweetness: residual sugar + small density coupling
  const tSweet = sliderT(sweetness);
  const sugar = lerp(1.4, 6.5, tSweet);
  const density = clamp(BASE.density + (tSweet - 0.5) * 0.0028, R.density[0], R.density[1]);

  // Alcohol: main alcohol + density inverse coupling
  const tAlc = sliderT(alcohol);
  const alc = lerp(10.5, 14.9, tAlc);
  const density2 = clamp(density - (tAlc - 0.5) * 0.0022, R.density[0], R.density[1]);

  // Acidity: blend fixed/volatile/citric and pH inverse
  const tAcid = sliderT(acidity);
  const fixed = lerp(6.6, 10.8, tAcid);
  // Volatile acidity is usually *lower* for better wines, so we invert it vs this slider.
  const volatile = lerp(0.75, 0.18, tAcid);
  const citric = lerp(0.08, 0.55, tAcid);
  const ph = lerp(3.75, 3.0, tAcid);

  // Keep the rest near baseline with small couplings
  const sulphates = clamp(BASE.sulphates + (tAlc - 0.5) * 0.55, R.sulphates[0], R.sulphates[1]);
  const chlorides = clamp(BASE.chlorides - (tAlc - 0.5) * 0.02, R.chlorides[0], R.chlorides[1]);
  const freeSO2 = clamp(BASE["free sulfur dioxide"] + (tSweet - 0.5) * 10, R["free sulfur dioxide"][0], R["free sulfur dioxide"][1]);
  const totalSO2 = clamp(BASE["total sulfur dioxide"] + (tSweet - 0.5) * 40, R["total sulfur dioxide"][0], R["total sulfur dioxide"][1]);

  const payload = {
    ...BASE,
    "residual sugar": sugar,
    density: density2,
    alcohol: alc,
    "fixed acidity": fixed,
    "volatile acidity": volatile,
    "citric acid": citric,
    pH: ph,
    sulphates,
    chlorides,
    "free sulfur dioxide": freeSO2,
    "total sulfur dioxide": totalSO2,
  };

  // Clamp all features into ranges
  for (const k of FEATURES) {
    const [lo, hi] = R[k] || [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY];
    payload[k] = clamp(Number(payload[k]), lo, hi);
    setInput(k, toFixedMaybe(payload[k]));
  }

  // Update labels
  sweetnessVal.textContent = tSweet < 0.33 ? "Dry" : tSweet < 0.66 ? "Medium" : "Sweet";
  alcoholPct.textContent = toFixedMaybe(payload.alcohol);
  acidityVal.textContent = toFixedMaybe(payload.pH);

  // Right-side legend labels (simple thresholds for display)
  const sul = payload.sulphates;
  const alcNow = payload.alcohol;
  const sugarNow = payload["residual sugar"];
  const chlNow = payload.chlorides;

  function state3(v, lo, hi) {
    if (v <= lo) return "Low";
    if (v >= hi) return "Elevated";
    return "Moderate";
  }

  legendSul.textContent = state3(sul, 0.55, 0.95);
  legendAlc.textContent = state3(alcNow, 10.5, 13.2);
  // Sugar: higher is worse -> invert naming
  legendSugar.textContent = sugarNow <= 2.2 ? "Low" : sugarNow >= 5.0 ? "Elevated" : "Moderate";
  // Chlorides: lower is better
  legendChl.textContent = chlNow <= 0.05 ? "Low" : chlNow >= 0.09 ? "Elevated" : "Moderate";

  function setBar(el, v, lo, hi) {
    const t = clamp((v - lo) / (hi - lo), 0, 1);
    el.style.width = `${Math.round(10 + t * 90)}%`;
  }

  setBar(legendSulBar, sul, R.sulphates[0], R.sulphates[1]);
  setBar(legendAlcBar, alcNow, R.alcohol[0], R.alcohol[1]);
  setBar(legendSugarBar, sugarNow, R["residual sugar"][0], R["residual sugar"][1]);
  setBar(legendChlBar, chlNow, R.chlorides[0], R.chlorides[1]);
}

function setResultFromResponse(data) {
  const pred = data?.prediction;
  if (pred === 0 || pred === 1) {
    predEl.textContent = String(pred);
    predTextEl.textContent = pred === 1 ? "Good" : "Bad";
  } else {
    predEl.textContent = "—";
    predTextEl.textContent = "—";
  }

  const p = data?.probability_good;
  if (typeof p === "number") {
    probEl.textContent = `${(p * 100).toFixed(1)}%`;
    verdictEl.textContent =
      p >= 0.5
        ? "This indicates a higher chance of being classified as good."
        : "This indicates a lower chance of being classified as good.";
    return;
  }
  probEl.textContent = "—";
  verdictEl.textContent = "Prediction received.";
}

function formatConfusionMatrix(cm) {
  if (!Array.isArray(cm) || cm.length !== 2) return null;
  const tn = cm?.[0]?.[0];
  const fp = cm?.[0]?.[1];
  const fn = cm?.[1]?.[0];
  const tp = cm?.[1]?.[1];
  if (![tn, fp, fn, tp].every((x) => typeof x === "number")) return null;
  return `CM [[TN ${tn}, FP ${fp}], [FN ${fn}, TP ${tp}]]`;
}

async function loadModelMetrics() {
  try {
    const res = await fetch("/model-info");
    if (!res.ok) {
      metricsEl.textContent = "";
      return;
    }
    const info = await res.json();
    const v = info?.version ?? "";
    const acc = info?.metrics?.accuracy;
    const cm = info?.metrics?.confusion_matrix;
    const cmText = formatConfusionMatrix(cm);
    const accText = typeof acc === "number" ? `Test accuracy ${(acc * 100).toFixed(1)}%` : null;
    const parts = [v ? `Model ${v}` : null, accText, cmText].filter(Boolean);
    metricsEl.textContent = parts.length ? parts.join(" • ") : "";
  } catch {
    metricsEl.textContent = "";
  }
}

async function doPredict(payload) {
  predictBtn.disabled = true;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }
    if (!res.ok) {
      probEl.textContent = "—";
      predEl.textContent = "—";
      predTextEl.textContent = "—";
      verdictEl.textContent = `Error ${res.status}: ${data?.detail ?? "Request failed"}`;
      return;
    }
    setResultFromResponse(data);
    void loadModelMetrics();
  } catch (e) {
    probEl.textContent = "—";
    predEl.textContent = "—";
    predTextEl.textContent = "—";
    verdictEl.textContent = String(e);
  } finally {
    predictBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  applySliders();
  void doPredict(readAdvancedPayload());
});

for (const el of [sweetness, alcohol, acidity]) {
  el.addEventListener("input", applySliders);
}

applySliders();
void loadModelMetrics();

// Bring back sample buttons to match the mock UI.
document.getElementById("sample-balanced")?.addEventListener("click", () => {
  sweetness.value = "35";
  alcohol.value = "70";
  acidity.value = "60";
  applySliders();
});

document.getElementById("sample-low")?.addEventListener("click", () => {
  sweetness.value = "78";
  alcohol.value = "30";
  acidity.value = "30";
  applySliders();
});
