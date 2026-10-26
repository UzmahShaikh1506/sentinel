/* Headless smoke test: drives the real console against the live API via jsdom,
   so any runtime error in a render path surfaces as a failure. */
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const ROOT = "/home/claude/sentinel-ts/frontend";
const BASE = "http://127.0.0.1:8800";

const errors = [];
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");

const dom = new JSDOM(html, {
  runScripts: "outside-only",
  pretendToBeVisual: true,
  url: BASE
});
const { window } = dom;

window.addEventListener("error", (e) => errors.push("window.error: " + e.message));

// Canvas is not implemented in jsdom; stub a context that records nothing.
window.HTMLCanvasElement.prototype.getContext = function () {
  const noop = () => {};
  return new Proxy(
    {
      canvas: this,
      measureText: () => ({ width: 10 }),
      createLinearGradient: () => ({ addColorStop: noop }),
      getImageData: () => ({ data: [] })
    },
    { get: (t, k) => (k in t ? t[k] : noop), set: () => true }
  );
};
window.devicePixelRatio = 2;
Object.defineProperty(window.HTMLElement.prototype, "clientWidth", { get: () => 900 });
Object.defineProperty(window.HTMLElement.prototype, "offsetWidth", { get: () => 160 });
Object.defineProperty(window.HTMLElement.prototype, "offsetHeight", { get: () => 60 });
window.HTMLElement.prototype.getBoundingClientRect = () => ({
  left: 0, top: 0, right: 900, bottom: 300, width: 900, height: 300
});

// Real network calls against the running server.
window.fetch = async (url, opts) => {
  const res = await fetch(url.startsWith("http") ? url : BASE + url, opts);
  return {
    ok: res.ok,
    status: res.status,
    statusText: res.statusText,
    json: () => res.json()
  };
};
window.FormData = globalThis.FormData;

const run = (file) => {
  try {
    window.eval(fs.readFileSync(path.join(ROOT, file), "utf8"));
  } catch (e) {
    errors.push(`${file}: ${e.message}\n${e.stack}`);
  }
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  run("charts.js");
  run("app.js");
  await sleep(2500);

  const doc = window.document;
  const check = (label, cond, extra) => {
    if (!cond) errors.push("ASSERT " + label + (extra ? " :: " + extra : ""));
    else console.log("  ok  " + label);
  };

  check("datasets listed", doc.querySelectorAll(".ds-item").length >= 6,
        doc.querySelectorAll(".ds-item").length + " items");
  check("sparklines drawn", doc.querySelectorAll(".ds-spark polyline").length >= 6);
  check("model toggles built", doc.querySelectorAll("#modelToggles .switch-row").length >= 6);
  check("pipeline stages built", doc.querySelectorAll(".stage").length === 8);
  check("provenance populated", doc.querySelectorAll("#dsInfo .kv-row").length >= 8);
  check("preview chart canvas", !!doc.querySelector("#previewChart canvas"));
  check("rail stats filled", doc.querySelector("#railModels").textContent !== "—");

  // Navigate every view to be sure none throws.
  ["overview", "eda", "models", "detection", "alerts", "benchmark", "about", "control"]
    .forEach((v) => {
      const el = doc.querySelector(`.nav-item[data-view="${v}"]`);
      el.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
  check("nav cycles all views", doc.querySelector("#view-control").classList.contains("active"));

  // Launch a real run through the UI button and wait for it to finish.
  console.log("\n  launching run via UI…");
  doc.querySelector("#runBtn").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  for (let i = 0; i < 200; i++) {
    await sleep(1000);
    const chip = doc.querySelector("#runChip").textContent;
    if (chip === "complete" || chip === "failed") break;
  }
  const chip = doc.querySelector("#runChip").textContent;
  check("run completed", chip === "complete", "chip=" + chip + " stage=" + doc.querySelector("#stageText").textContent);

  if (chip === "complete") {
    check("overview verdict rendered", !!doc.querySelector("#overviewBody .verdict"));
    check("overview tiles rendered", doc.querySelectorAll("#ovTiles .metric").length === 4);
    check("overview main chart", !!doc.querySelector("#ovMain canvas"));
    check("legend built", doc.querySelectorAll("#ovLegend .legend-item").length >= 2);
    check("eda audit rendered", doc.querySelectorAll("#edaBody .kv-row").length >= 10);
    check("eda decomposition charts", doc.querySelectorAll("#edaBody canvas").length >= 6);
    check("stationarity cards", doc.querySelectorAll("#edaBody .chip").length >= 4);
    check("leaderboard rows", doc.querySelectorAll("#modelsBody tbody tr").length >= 6);
    check("winner row highlighted", !!doc.querySelector("#modelsBody tr.winner"));
    check("per-model cards", doc.querySelectorAll("#mdlCards .card").length >= 6);
    check("detection tiles", doc.querySelectorAll("#detectionBody .metric").length === 4);
    check("detection raster rows", doc.querySelectorAll("#detRaster > div").length >= 6);
    check("score model selector", !!doc.querySelector("#scoreModelSel"));
    check("export enabled", !doc.querySelector("#exportBtn").disabled);

    const incidents = doc.querySelectorAll("#incList .incident").length;
    const emptyAlerts = !!doc.querySelector("#alertsBody .empty");
    check("alerts rendered", incidents > 0 || emptyAlerts, incidents + " incidents");

    if (incidents) {
      const first = doc.querySelector("#incList .incident");
      first.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
      check("incident playbook expands", first.classList.contains("open"));
    }

    // Switch the detection chart model — a common source of runtime errors.
    const sel = doc.querySelector("#scoreModelSel");
    sel.value = "seasonal_naive";
    sel.dispatchEvent(new window.Event("change", { bubbles: true }));
    await sleep(300);
    check("score chart switches model", !!doc.querySelector("#detScore canvas"));

    // Toggle a legend entry.
    const li = doc.querySelector("#ovLegend .legend-item[data-name]");
    li.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    check("legend toggles", li.classList.contains("off"));
  }

  console.log("\n" + (errors.length ? "FAILURES:\n" + errors.join("\n---\n") : "ALL UI CHECKS PASSED"));
  process.exit(errors.length ? 1 : 0);
})();
