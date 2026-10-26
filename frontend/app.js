/* ==========================================================================
   SENTINEL-TS :: Console application

   Thin client. Every number it renders was computed by the Python pipeline;
   nothing is derived or re-estimated here. The only client-side logic is
   presentation, interaction and animation.
   ========================================================================== */

(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const C = window.SChart;

  /* ------------------------------------------------------------- state */

  const state = {
    datasets: [],
    models: [],
    selected: null,
    result: null,
    benchmark: null,
    jobId: null,
    benchJobId: null,
    running: false,
    charts: [],
    previewChart: null
  };

  const PIPELINE_STAGES = [
    "Loading telemetry",
    "Cleaning and resampling",
    "Exploratory analysis",
    "Splitting train and test",
    "Fitting models",
    "Scoring and ranking",
    "Building incident report",
    "Finalising"
  ];

  const FAMILY_COLOR = {
    baseline: C.PALETTE.faint,
    statistical: C.PALETTE.sky,
    classical: C.PALETTE.violet,
    "machine learning": C.PALETTE.amber,
    hybrid: C.PALETTE.teal
  };

  /* --------------------------------------------------------------- api */

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch (e) {
        /* non-JSON error body */
      }
      throw new Error(detail);
    }
    return res.json();
  }

  /* ------------------------------------------------------------ toasts */

  function toast(message, isError) {
    const el = document.createElement("div");
    el.className = "toast" + (isError ? " err" : "");
    el.textContent = message;
    $("#toasts").appendChild(el);
    setTimeout(() => {
      el.classList.add("leaving");
      setTimeout(() => el.remove(), 320);
    }, isError ? 6000 : 3600);
  }

  /* ---------------------------------------------------------- helpers */

  const fmt = (v, digits) =>
    v === null || v === undefined || Number.isNaN(v)
      ? "—"
      : typeof v === "number"
      ? v.toFixed(digits === undefined ? 3 : digits)
      : String(v);

  const pct = (v, digits) =>
    v === null || v === undefined ? "—" : (v * 100).toFixed(digits === undefined ? 1 : digits) + "%";

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  function disposeCharts() {
    state.charts.forEach((c) => {
      try {
        c.destroy();
      } catch (e) {
        /* already gone */
      }
    });
    state.charts = [];
  }

  function track(chart) {
    state.charts.push(chart);
    return chart;
  }

  /** Convert a boolean flag array into contiguous index spans. */
  function spans(mask) {
    const out = [];
    let start = null;
    (mask || []).forEach((v, i) => {
      if (v && start === null) start = i;
      else if (!v && start !== null) {
        out.push([start, i - 1]);
        start = null;
      }
    });
    if (start !== null) out.push([start, mask.length - 1]);
    return out;
  }

  /* -------------------------------------------------------------- boot */

  function runBoot() {
    const lines = [
      ["init", "analysis engine"],
      ["load", "NAB realAWSCloudwatch corpus"],
      ["arm", "six-model comparison harness"],
      ["ready", "operations console"]
    ];
    const host = $("#bootLines");
    lines.forEach((pair, i) => {
      const el = document.createElement("div");
      el.className = "boot-line";
      el.style.animationDelay = 180 + i * 300 + "ms";
      el.innerHTML = `<span class="ok">▸ ${pair[0]}</span><span>${pair[1]}</span>`;
      host.appendChild(el);
    });
    setTimeout(() => $("#boot").classList.add("done"), 2150);
  }

  /* --------------------------------------------------------------- nav */

  const TITLES = {
    control: ["Mission Control", "Configure and launch an analysis run"],
    overview: ["Overview", "Winning model, incidents and capacity outlook"],
    eda: ["Data & EDA", "Preprocessing audit and exploratory analysis"],
    models: ["Model Comparison", "Six approaches scored on identical splits"],
    detection: ["Detection", "Anomaly scores, thresholds and confusion"],
    alerts: ["Alerts", "Incidents with recommended operator actions"],
    benchmark: ["Fleet Benchmark", "Aggregated results across every labelled service"],
    about: ["Method", "Problem framing, approach and evaluation stance"]
  };

  function go(view) {
    $$(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.view === view));
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + view));
    const t = TITLES[view] || ["", ""];
    $("#pageTitle").textContent = t[0];
    $("#pageSub").textContent = t[1];
    $("#main").scrollTop = 0;
    $("#rail").classList.remove("open");
  }

  /* ---------------------------------------------------------- datasets */

  async function loadCatalogue() {
    const [ds, md, health] = await Promise.all([
      api("/api/datasets"),
      api("/api/models"),
      api("/api/health")
    ]);
    state.datasets = ds.datasets;
    state.models = md.models;

    $("#railVersion").textContent = "v" + health.version;
    $("#railModels").textContent = health.n_models;
    $("#railDatasets").textContent = state.datasets.length;

    renderDatasetList();
    renderModelToggles();

    if (!state.selected && state.datasets.length) select(state.datasets[0].id);
  }

  function renderDatasetList() {
    const host = $("#dsList");
    host.innerHTML = "";
    state.datasets.forEach((d) => {
      const el = document.createElement("div");
      el.className = "ds-item" + (state.selected === d.id ? " selected" : "");
      el.dataset.id = d.id;
      el.innerHTML = `
        <svg class="ds-spark" data-spark="${escapeHtml(d.id)}"></svg>
        <div class="ds-body">
          <div class="ds-name">${escapeHtml(d.title)}</div>
          <div class="ds-meta">${escapeHtml(d.service)} · ${d.n_obs.toLocaleString()} pts · ${
        d.n_label_windows
      } labelled window${d.n_label_windows === 1 ? "" : "s"}</div>
        </div>`;
      el.addEventListener("click", () => select(d.id));
      host.appendChild(el);
      C.sparkline(
        $('svg[data-spark]', el),
        d.spark,
        d.n_label_windows ? C.PALETTE.teal : C.PALETTE.sky
      );
    });
  }

  function renderModelToggles() {
    const host = $("#modelToggles");
    host.innerHTML = "";
    state.models.forEach((m) => {
      const row = document.createElement("div");
      row.className = "switch-row on";
      row.dataset.model = m.name;
      row.title = m.description;
      row.innerHTML = `<div class="switch"></div><span>${escapeHtml(m.display_name)}</span>`;
      row.addEventListener("click", () => {
        const on = $$('.switch-row.on', host).length;
        if (row.classList.contains("on") && on <= 2) {
          toast("Keep at least two models so there is something to compare.", true);
          return;
        }
        row.classList.toggle("on");
      });
      host.appendChild(row);
    });
  }

  function selectedModels() {
    return $$('#modelToggles .switch-row.on').map((r) => r.dataset.model);
  }

  async function select(id) {
    state.selected = id;
    $$(".ds-item").forEach((el) => el.classList.toggle("selected", el.dataset.id === id));
    const d = state.datasets.find((x) => x.id === id);
    if (!d) return;

    $("#previewMeta").textContent = `${d.native_freq} · ${d.n_obs.toLocaleString()} observations`;

    $("#dsInfo").innerHTML = [
      ["Service", d.service],
      ["Metric", d.title],
      ["Source", d.source],
      ["Unit", d.unit],
      ["Native frequency", d.native_freq],
      ["Observations", d.n_obs.toLocaleString()],
      ["Window start", d.start.replace("T", " ").slice(0, 16)],
      ["Window end", d.end.replace("T", " ").slice(0, 16)],
      ["Labelled anomalies", d.n_label_windows]
    ]
      .map(
        (r) =>
          `<div class="kv-row"><span class="kv-key">${r[0]}</span><span class="kv-value">${escapeHtml(
            String(r[1])
          )}</span></div>`
      )
      .join("") +
      `<div class="kv-row" style="display:block"><span class="kv-key">About</span>
       <div class="prose" style="margin-top:6px">${escapeHtml(d.description)}</div></div>`;

    drawPreview(id);
  }

  /** Raw series preview, with ground-truth windows shaded, before any run. */
  async function drawPreview(id) {
    const host = $("#previewChart");
    host.innerHTML = '<div class="skel" style="height:100%"></div>';
    try {
      const p = await api(`/api/datasets/${encodeURIComponent(id)}/preview`);
      if (state.selected !== id) return;

      const times = p.timestamps.map((t) => Date.parse(t));
      const regions = (p.label_windows || []).map((w, i) => {
        const a = Date.parse(w.start);
        const b = Date.parse(w.end);
        const from = times.findIndex((t) => t >= a);
        let to = times.findIndex((t) => t > b);
        if (to === -1) to = times.length - 1;
        return {
          from: from === -1 ? 0 : from,
          to: Math.max(to - 1, from === -1 ? 0 : from),
          color: "rgba(251,113,133,0.12)",
          label: i === 0 ? "labelled incident" : ""
        };
      });

      const s = [
        {
          name: p.meta.title || "value",
          values: p.values,
          color: C.PALETTE.teal,
          width: 1.4,
          area: true
        }
      ];
      if (state.previewChart) {
        try {
          state.previewChart.destroy();
        } catch (e) {
          /* already gone */
        }
      }
      state.previewChart = new C.TimeChart(host, {
        x: p.timestamps,
        series: s,
        regions,
        height: 230,
        unit: p.meta.unit || "",
        yLabel: p.meta.unit || ""
      });
      buildLegend($("#previewLegend"), state.previewChart, s, [], regions.length > 0, 0);
    } catch (err) {
      host.innerHTML = `<div class="empty" style="padding:40px 10px">${escapeHtml(
        err.message
      )}</div>`;
    }
  }

  /* ----------------------------------------------------------- upload */

  async function handleUpload(file) {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      toast("Uploading " + file.name + "…");
      const res = await api("/api/datasets/upload", { method: "POST", body: form });
      state.datasets = res.datasets;
      $("#railDatasets").textContent = state.datasets.length;
      renderDatasetList();
      select(res.dataset_id);
      toast("Uploaded and registered. It is selected and ready to run.");
    } catch (err) {
      toast(err.message, true);
    }
  }

  /* -------------------------------------------------------- pipeline UI */

  function renderPipeline() {
    const host = $("#pipeline");
    host.innerHTML = PIPELINE_STAGES.map(
      (s) => `<div class="stage" data-stage="${escapeHtml(s)}"><div class="ico"></div>${escapeHtml(s)}</div>`
    ).join("");
  }

  function updatePipeline(stage, progress) {
    const normalised = /^Fitting /.test(stage) ? "Fitting models" : stage;
    let seen = false;
    $$(".stage").forEach((el) => {
      const name = el.dataset.stage;
      if (name === normalised) {
        el.className = "stage active";
        seen = true;
      } else if (!seen) {
        el.className = "stage done";
      } else {
        el.className = "stage";
      }
    });
    $("#progressFill").style.width = progress + "%";
    $("#progressText").textContent = progress + "%";
    $("#stageText").textContent = stage;
  }

  function completePipeline() {
    $$(".stage").forEach((el) => (el.className = "stage done"));
    $("#progressFill").style.width = "100%";
    $("#progressText").textContent = "100%";
  }

  /* --------------------------------------------------------------- run */

  function config() {
    return {
      dataset_id: state.selected,
      freq: $("#freqSel").value,
      test_size: Number($("#testSize").value) / 100,
      threshold_sigma: Number($("#sigma").value),
      contamination: Number($("#contam").value) / 100,
      fourier_k: Number($("#fourierK").value),
      horizon: Number($("#horizon").value),
      models: selectedModels()
    };
  }

  async function run() {
    if (state.running) return;
    if (!state.selected) return toast("Choose a telemetry source first.", true);

    state.running = true;
    $("#runBtn").classList.add("running");
    $("#runBtn").disabled = true;
    $("#runBtnLabel").textContent = "Running…";
    $("#statusDot").className = "status-dot busy";
    $("#runChip").textContent = "running";
    $("#runChip").className = "chip amber dot";
    renderPipeline();
    updatePipeline("Loading telemetry", 2);
    go("control");

    try {
      const job = await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config())
      });
      state.jobId = job.id;
      await poll(job.id, updatePipeline);
      const done = await api(`/api/jobs/${job.id}/result`);
      state.result = done.result;
      completePipeline();
      renderAll();
      $("#runChip").textContent = "complete";
      $("#runChip").className = "chip teal dot";
      $("#railRuntime").textContent = state.result.runtime_seconds + "s";
      $("#exportBtn").disabled = false;
      toast(
        `Analysis complete in ${state.result.runtime_seconds}s — ${
          state.result.models[0].display_name
        } selected.`
      );
      go("overview");
    } catch (err) {
      $("#runChip").textContent = "failed";
      $("#runChip").className = "chip rose dot";
      $("#stageText").textContent = err.message;
      $("#statusDot").className = "status-dot err";
      toast("Run failed: " + err.message, true);
    } finally {
      state.running = false;
      $("#runBtn").classList.remove("running");
      $("#runBtn").disabled = false;
      $("#runBtnLabel").textContent = "Run analysis";
      if ($("#statusDot").className !== "status-dot err") {
        $("#statusDot").className = "status-dot";
      }
    }
  }

  /** Poll a job until it settles, reporting progress as it goes. */
  async function poll(jobId, onProgress) {
    for (;;) {
      await new Promise((r) => setTimeout(r, 650));
      const st = await api(`/api/jobs/${jobId}`);
      onProgress(st.stage, st.progress);
      if (st.status === "complete") return st;
      if (st.status === "failed") throw new Error(st.error || "Run failed.");
    }
  }

  /* =========================================================== renderers */

  function renderAll() {
    disposeCharts();
    renderOverview();
    renderEda();
    renderModels();
    renderDetection();
    renderAlerts();
  }

  /* ---------------------------------------------------------- overview */

  function renderOverview() {
    const r = state.result;
    if (!r) return;
    const winner = r.models.find((m) => m.name === r.winner) || r.models[0];
    const det = winner.detection;
    const fc = winner.forecast || {};
    const series = r.model_series[r.winner];

    const host = $("#overviewBody");
    host.innerHTML = `
      <div class="verdict mb-10">
        <div class="verdict-head">
          <span class="chip teal dot">selected model</span>
          <div class="verdict-title">${escapeHtml(r.narrative.headline)}</div>
          <div class="spacer" style="flex:1"></div>
          <span class="chip">${escapeHtml(r.dataset.service || "")}</span>
          <span class="chip">${escapeHtml(r.config.freq)} buckets</span>
          <span class="chip">${r.runtime_seconds}s</span>
        </div>
        <ul class="bullet-list">${r.narrative.points
          .map((p) => `<li>${escapeHtml(p)}</li>`)
          .join("")}</ul>
      </div>

      <div class="grid g-4 stagger mb-10" id="ovTiles"></div>

      <div class="card mb-10">
        <div class="card-head">
          <h2 class="card-title">Observed versus expected — hold-out window</h2>
          <div class="card-sub">shaded spans are ground-truth incident windows; dots are raised alerts</div>
          <div class="spacer"></div>
          <span class="card-sub mono">${escapeHtml(r.split.test_start.slice(0, 16).replace("T", " "))} → ${escapeHtml(
      r.split.test_end.slice(0, 16).replace("T", " ")
    )}</span>
        </div>
        <div id="ovMain" style="height:330px"></div>
        <div class="legend" id="ovLegend"></div>
      </div>

      <div class="grid g-21">
        <div class="card">
          <div class="card-head">
            <h2 class="card-title">Forward capacity outlook</h2>
            <div class="card-sub">${escapeHtml(r.capacity_outlook.method || "")}</div>
          </div>
          <div id="ovForecast" style="height:250px"></div>
          <p class="prose mt-10">${escapeHtml(r.capacity_outlook.message || "")}</p>
        </div>
        <div class="card">
          <div class="card-head"><h2 class="card-title">Alert profile</h2></div>
          <div class="kv" id="ovDigest"></div>
        </div>
      </div>`;

    /* -- tiles ------------------------------------------------------- */
    const tiles = [
      {
        label: "Incident windows caught",
        value: det.labelled
          ? `${det.n_segments_detected}/${det.n_true_segments}`
          : "unlabelled",
        note: det.labelled
          ? `segment recall ${pct(det.segment_recall)}`
          : "no ground truth on this series",
        accent: C.PALETTE.teal
      },
      {
        label: "Point-adjusted F1",
        value: det.labelled ? fmt(det.pa_f1) : "—",
        note: det.labelled ? `point-wise F1 ${fmt(det.f1)}` : "requires labels",
        accent: C.PALETTE.violet
      },
      {
        label: "Time to first alert",
        value:
          det.mean_detection_latency_minutes !== null &&
          det.mean_detection_latency_minutes !== undefined
            ? det.mean_detection_latency_minutes + " min"
            : "—",
        note: `${det.alerts_per_day} alert episode(s)/day`,
        accent: C.PALETTE.amber
      },
      {
        label: "Forecast error (RMSE)",
        value: fc.rmse !== undefined && fc.rmse !== null ? fmt(fc.rmse, 3) : "—",
        note:
          fc.mase !== null && fc.mase !== undefined
            ? `MASE ${fmt(fc.mase)} — ${
                fc.mase < 1 ? "beats" : "loses to"
              } seasonal naive`
            : "no forecast from this model",
        accent: C.PALETTE.sky
      }
    ];
    $("#ovTiles").innerHTML = tiles
      .map(
        (t) => `<div class="metric" style="--accent:${t.accent}">
          <div class="metric-label">${escapeHtml(t.label)}</div>
          <div class="metric-value">${escapeHtml(String(t.value))}</div>
          <div class="metric-note">${escapeHtml(t.note)}</div>
        </div>`
      )
      .join("");

    /* -- main chart --------------------------------------------------- */
    const x = r.series.test.timestamps;
    const regions = (r.series.truth_mask ? spans(r.series.truth_mask) : []).map(
      (s, i) => ({ from: s[0], to: s[1], color: "rgba(251,113,133,0.10)", label: i === 0 ? "labelled incident" : "" })
    );
    const markers = [];
    (series.flags || []).forEach((f, i) => {
      if (f) markers.push({ index: i, value: r.series.test.values[i], color: C.PALETTE.rose });
    });

    const chartSeries = [
      {
        name: "Observed",
        values: r.series.test.values,
        color: C.PALETTE.text,
        width: 1.5
      }
    ];
    if (series.predictions) {
      chartSeries.push({
        name: "Expected (model)",
        values: series.predictions,
        color: C.PALETTE.teal,
        width: 1.6,
        dash: [5, 3]
      });
    }

    const bands = series.lower
      ? [
          {
            name: "Normal band",
            lower: series.lower,
            upper: series.upper,
            color: C.hexToRgba(C.PALETTE.teal, 0.08)
          }
        ]
      : [];

    const main = track(
      new C.TimeChart($("#ovMain"), {
        x,
        series: chartSeries,
        bands,
        regions,
        markers,
        height: 330,
        unit: r.dataset.unit,
        yLabel: r.dataset.unit
      })
    );
    buildLegend($("#ovLegend"), main, chartSeries, bands, regions.length > 0, markers.length);

    /* -- forecast ----------------------------------------------------- */
    const outlook = r.capacity_outlook;
    if (outlook.available && outlook.values && outlook.values.length) {
      const tail = 96;
      const histVals = r.series.test.values.slice(-tail);
      const histX = r.series.test.timestamps.slice(-tail);
      const combinedX = histX.concat(outlook.timestamps);
      const pad = new Array(histVals.length).fill(null);
      const fSeries = [
        {
          name: "Recent observed",
          values: histVals.concat(new Array(outlook.values.length).fill(null)),
          color: C.PALETTE.text,
          width: 1.5
        },
        {
          name: "Forecast",
          values: pad.concat(outlook.values),
          color: C.PALETTE.violet,
          width: 1.8,
          area: true
        }
      ];
      const fBands = outlook.upper
        ? [
            {
              name: "95% interval",
              lower: pad.concat(outlook.lower),
              upper: pad.concat(outlook.upper),
              color: C.hexToRgba(C.PALETTE.violet, 0.1)
            }
          ]
        : [];
      track(
        new C.TimeChart($("#ovForecast"), {
          x: combinedX,
          series: fSeries,
          bands: fBands,
          height: 250,
          unit: r.dataset.unit,
          threshold: outlook.ceiling,
          thresholdLabel: outlook.ceiling ? `SLO ceiling ${outlook.ceiling}` : null,
          regions: [
            { from: histVals.length, to: combinedX.length - 1, color: "rgba(167,139,250,0.05)" }
          ]
        })
      );
    } else {
      $("#ovForecast").innerHTML =
        '<div class="empty" style="padding:40px 10px">The selected model does not produce a forward forecast.</div>';
    }

    /* -- digest ------------------------------------------------------- */
    const dg = r.digest;
    const sevRows = Object.entries(dg.by_severity)
      .map(
        ([k, v]) =>
          `<div class="kv-row"><span class="kv-key" style="text-transform:capitalize">${k}</span><span class="kv-value">${v}</span></div>`
      )
      .join("");
    const kindRows = Object.entries(dg.by_kind)
      .map(
        ([k, v]) =>
          `<div class="kv-row"><span class="kv-key">${escapeHtml(
            k.replace(/_/g, " ")
          )}</span><span class="kv-value">${v}</span></div>`
      )
      .join("");
    $("#ovDigest").innerHTML =
      `<div class="kv-row"><span class="kv-key">Incidents detected</span><span class="kv-value">${dg.total}</span></div>` +
      sevRows +
      (kindRows ? `<div class="divider" style="margin:10px 0"></div>` + kindRows : "") +
      `<div style="margin-top:12px" class="prose">${escapeHtml(dg.verdict)}</div>`;

    const badge = $("#alertBadge");
    if (dg.actionable > 0) {
      badge.style.display = "";
      badge.textContent = dg.actionable;
    } else badge.style.display = "none";
  }

  function buildLegend(host, chart, series, bands, hasRegions, markerCount) {
    if (!host) return;
    const items = series.map(
      (s) =>
        `<div class="legend-item" data-name="${escapeHtml(s.name)}">
           <i class="legend-sw" style="background:${s.color}"></i>${escapeHtml(s.name)}
         </div>`
    );
    (bands || []).forEach((b) =>
      items.push(
        `<div class="legend-item" data-name="${escapeHtml(b.name)}">
           <i class="legend-sw box" style="background:${b.color}"></i>${escapeHtml(b.name)}
         </div>`
      )
    );
    if (hasRegions)
      items.push(
        `<div class="legend-item" style="cursor:default"><i class="legend-sw box" style="background:rgba(251,113,133,0.3)"></i>labelled incident window</div>`
      );
    if (markerCount)
      items.push(
        `<div class="legend-item" style="cursor:default"><i class="legend-sw box" style="background:${C.PALETTE.rose};border-radius:50%"></i>alert raised (${markerCount})</div>`
      );
    host.innerHTML = items.join("");
    $$(".legend-item[data-name]", host).forEach((el) => {
      el.addEventListener("click", () => {
        el.classList.toggle("off");
        chart.toggle(el.dataset.name);
      });
    });
  }

  /* --------------------------------------------------------------- eda */

  function renderEda() {
    const r = state.result;
    if (!r) return;
    const e = r.eda;
    const pre = r.preprocessing;
    const dec = e.decomposition;
    const st = e.stationarity;
    const dist = e.distribution;

    $("#edaBody").innerHTML = `
      <div class="grid g-4 stagger mb-10">
        ${tile("Observations", pre.n_output.toLocaleString(), `${pre.duration_days} days at ${pre.freq}`, C.PALETTE.teal)}
        ${tile("Coverage", pre.coverage_pct + "%", `${pre.n_missing_after_resample} empty bucket(s) reconstructed`, C.PALETTE.sky)}
        ${tile("Seasonal strength", fmt(dec.seasonal_strength, 3), `trend strength ${fmt(dec.trend_strength, 3)}`, C.PALETTE.violet)}
        ${tile("Stationarity", st.verdict.label, `recommended d = ${st.verdict.recommended_d}`, C.PALETTE.amber)}
      </div>

      <div class="card mb-10">
        <div class="card-head">
          <h2 class="card-title">Preprocessing audit trail</h2>
          <div class="card-sub">every modification made to the raw feed, on the record</div>
        </div>
        <div class="grid g-2">
          <div class="kv">
            ${kv("Raw observations", pre.n_input.toLocaleString())}
            ${kv("After resampling", pre.n_output.toLocaleString())}
            ${kv("Duplicate timestamps", pre.n_duplicate_timestamps)}
            ${kv("Non-finite readings", pre.n_non_finite)}
            ${kv("Empty buckets", pre.n_missing_after_resample)}
          </div>
          <div class="kv">
            ${kv("Filled by interpolation", pre.n_filled_interpolated)}
            ${kv("Filled seasonally", pre.n_filled_seasonal)}
            ${kv("Extreme values flagged", pre.n_outliers_flagged)}
            ${kv("Flag rate", pct(pre.outlier_rate, 2))}
            ${kv("Seasonal period", pre.seasonal_period + " buckets/day")}
          </div>
        </div>
        <ul class="bullet-list mt-16">${pre.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
      </div>

      <div class="card mb-10">
        <div class="card-head">
          <h2 class="card-title">STL decomposition</h2>
          <div class="card-sub">robust to the outliers this project is looking for, unlike a centred moving average</div>
          <div class="spacer"></div>
          <span class="chip teal">trend ${pct(dec.variance_share.trend, 0)}</span>
          <span class="chip violet">seasonal ${pct(dec.variance_share.seasonal, 0)}</span>
          <span class="chip">remainder ${pct(dec.variance_share.remainder, 0)}</span>
        </div>
        <div id="edaObserved" style="height:150px"></div>
        <div id="edaTrend" style="height:120px"></div>
        <div id="edaSeasonal" style="height:120px"></div>
        <div id="edaResid" style="height:130px"></div>
        <p class="prose mt-10">${escapeHtml(dec.interpretation || "")}</p>
      </div>

      <div class="grid g-2 mb-10">
        <div class="card">
          <div class="card-head">
            <h2 class="card-title">Autocorrelation (ACF)</h2>
            <div class="card-sub">bars outside the band are significant</div>
          </div>
          <div id="edaAcf" style="height:180px"></div>
        </div>
        <div class="card">
          <div class="card-head">
            <h2 class="card-title">Partial autocorrelation (PACF)</h2>
            <div class="card-sub">drives the candidate AR order</div>
          </div>
          <div id="edaPacf" style="height:180px"></div>
        </div>
      </div>

      <div class="card mb-10">
        <div class="card-head"><h2 class="card-title">Stationarity evidence</h2></div>
        <div class="grid g-2">
          ${testCard("Augmented Dickey-Fuller", st.adf)}
          ${testCard("KPSS", st.kpss)}
        </div>
        <div class="verdict mt-16">
          <div class="verdict-head">
            <span class="chip amber dot">${escapeHtml(st.verdict.label)}</span>
            <div class="verdict-title">Reconciled verdict: d = ${st.verdict.recommended_d}</div>
          </div>
          <p class="prose" style="margin:0">${escapeHtml(st.verdict.explanation)}</p>
        </div>
      </div>

      <div class="grid g-2 mb-10">
        <div class="card">
          <div class="card-head">
            <h2 class="card-title">Load by hour and weekday</h2>
            <div class="card-sub">mean value per cell</div>
          </div>
          <div id="edaHeat" style="height:190px"></div>
          <p class="prose mt-10">${escapeHtml(e.seasonality.interpretation || "")}</p>
        </div>
        <div class="card">
          <div class="card-head">
            <h2 class="card-title">Value distribution</h2>
            <div class="card-sub">skew ${fmt(dist.skewness, 2)} · excess kurtosis ${fmt(dist.kurtosis, 2)}</div>
          </div>
          <div id="edaHist" style="height:190px"></div>
          <div class="kv mt-10">
            ${kv("Median", fmt(dist.median, 2))}
            ${kv("IQR", `${fmt(dist.q25, 2)} – ${fmt(dist.q75, 2)}`)}
            ${kv("1st–99th percentile", `${fmt(dist.q01, 2)} – ${fmt(dist.q99, 2)}`)}
            ${kv("Volatility non-constant", dist.volatility.heteroskedastic ? "yes" : "no")}
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><h2 class="card-title">What the data is telling us</h2></div>
        <ul class="bullet-list">${e.insights.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>
      </div>`;

    /* decomposition panels */
    if (dec.available) {
      const dx = dec.timestamps;
      const panels = [
        ["#edaObserved", "Observed", dec.observed, C.PALETTE.text, 150, true],
        ["#edaTrend", "Trend", dec.trend, C.PALETTE.teal, 120, true],
        ["#edaSeasonal", "Seasonal", dec.seasonal, C.PALETTE.violet, 120, false],
        ["#edaResid", "Remainder", dec.remainder, C.PALETTE.amber, 130, false]
      ];
      panels.forEach(([sel, name, values, color, h, area]) => {
        track(
          new C.TimeChart($(sel), {
            x: dx,
            series: [{ name, values, color, width: 1.3, area }],
            height: h,
            yLabel: name,
            padding: { top: 10, right: 16, bottom: sel === "#edaResid" ? 26 : 12, left: 52 }
          })
        );
      });
    }

    track(
      new C.StemChart($("#edaAcf"), {
        values: e.correlation.acf,
        band: e.correlation.confidence_band,
        color: C.PALETTE.teal,
        height: 180
      })
    );
    track(
      new C.StemChart($("#edaPacf"), {
        values: e.correlation.pacf,
        band: e.correlation.confidence_band,
        color: C.PALETTE.violet,
        height: 180
      })
    );
    track(
      new C.Heatmap($("#edaHeat"), {
        matrix: e.seasonality.heatmap,
        rows: e.seasonality.heatmap_rows,
        height: 190
      })
    );
    track(
      new C.Histogram($("#edaHist"), {
        counts: dist.histogram.counts,
        edges: dist.histogram.edges,
        height: 190
      })
    );
  }

  function tile(label, value, note, accent) {
    return `<div class="metric" style="--accent:${accent}">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(String(value))}</div>
      <div class="metric-note">${escapeHtml(String(note))}</div>
    </div>`;
  }

  function kv(key, value) {
    return `<div class="kv-row"><span class="kv-key">${escapeHtml(key)}</span><span class="kv-value">${escapeHtml(
      String(value)
    )}</span></div>`;
  }

  function testCard(title, t) {
    if (!t || t.error)
      return `<div class="card" style="background:var(--bg-deep)"><div class="card-title">${escapeHtml(
        title
      )}</div><p class="prose">Test did not converge.</p></div>`;
    const reject = t.reject_null;
    return `<div class="card" style="background:var(--bg-deep)">
      <div class="card-head">
        <h3 class="card-title">${escapeHtml(title)}</h3>
        <div class="spacer"></div>
        <span class="chip ${reject ? "rose" : "teal"} dot">${reject ? "reject H₀" : "fail to reject H₀"}</span>
      </div>
      <div class="kv">
        ${kv("Statistic", fmt(t.statistic, 4))}
        ${kv("p-value", fmt(t.p_value, 5))}
        ${kv("Lags used", t.lags_used)}
        ${kv("Critical 5%", fmt(t.critical_values["5%"], 4))}
      </div>
      <p class="prose mt-10" style="font-size:11.5px">H₀: ${escapeHtml(t.null_hypothesis)}</p>
    </div>`;
  }

  /* ------------------------------------------------------------ models */

  function renderModels() {
    const r = state.result;
    if (!r) return;

    const labelled = r.models.some((m) => m.composite_score !== null);
    const best = {};
    ["pa_f1", "f1", "segment_recall"].forEach((k) => {
      best[k] = Math.max(...r.models.map((m) => m.detection[k] || 0));
    });
    const bestRmse = Math.min(
      ...r.models.filter((m) => m.forecast).map((m) => m.forecast.rmse)
    );

    $("#modelsBody").innerHTML = `
      <div class="card mb-10 pad-0">
        <div class="card-head" style="padding:18px 20px 12px;margin:0">
          <h2 class="card-title">Leaderboard</h2>
          <div class="card-sub">${
            labelled
              ? "ranked by composite operational score"
              : "no labels on this series — ranked by forecast error"
          }</div>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th>#</th><th>Model</th><th>Family</th>
              <th class="right">Composite</th>
              <th class="right">PA&nbsp;F1</th>
              <th class="right">F1</th>
              <th class="right">Seg&nbsp;recall</th>
              <th class="right">Latency</th>
              <th class="right">FA/day</th>
              <th class="right">RMSE</th>
              <th class="right">MASE</th>
              <th class="right">Fit&nbsp;s</th>
            </tr></thead>
            <tbody>${r.models
              .map((m) => {
                const d = m.detection;
                const f = m.forecast || {};
                return `<tr class="${m.name === r.winner ? "winner" : ""}" data-model="${m.name}">
                  <td><span class="rank-pill ${m.rank === 1 ? "gold" : ""}">${m.rank}</span></td>
                  <td><b>${escapeHtml(m.display_name)}</b>${
                  m.error ? ' <span class="chip rose">error</span>' : ""
                }</td>
                  <td><span class="chip" style="color:${FAMILY_COLOR[m.family] || C.PALETTE.dim}">${escapeHtml(
                  m.family
                )}</span></td>
                  <td class="num ${m.rank === 1 ? "best" : ""}">${fmt(m.composite_score)}</td>
                  <td class="num ${d.pa_f1 === best.pa_f1 && d.pa_f1 ? "best" : ""}">${fmt(d.pa_f1)}</td>
                  <td class="num ${d.f1 === best.f1 && d.f1 ? "best" : ""}">${fmt(d.f1)}</td>
                  <td class="num">${d.segment_recall === null ? "—" : pct(d.segment_recall, 0)}</td>
                  <td class="num">${
                    d.mean_detection_latency_minutes === null ||
                    d.mean_detection_latency_minutes === undefined
                      ? "—"
                      : d.mean_detection_latency_minutes + "m"
                  }</td>
                  <td class="num">${fmt(d.false_alarms_per_day, 2)}</td>
                  <td class="num ${f.rmse === bestRmse ? "best" : ""}">${fmt(f.rmse, 3)}</td>
                  <td class="num">${fmt(f.mase, 3)}</td>
                  <td class="num dim">${fmt(m.fit_seconds, 2)}</td>
                </tr>`;
              })
              .join("")}</tbody>
          </table>
        </div>
      </div>

      <div class="grid g-2 mb-10">
        <div class="card">
          <div class="card-head">
            <h2 class="card-title">Composite operational score</h2>
            <div class="card-sub">PA-F1 0.45 · seg recall 0.25 · F1 0.15 · latency 0.10 · alert burden 0.05</div>
          </div>
          <div id="mdlScoreBar" style="height:${Math.max(160, r.models.length * 34)}px"></div>
        </div>
        <div class="card">
          <div class="card-head">
            <h2 class="card-title">Forecast error (RMSE, lower is better)</h2>
            <div class="card-sub">detector-only models are omitted — they produce no forecast</div>
          </div>
          <div id="mdlRmseBar" style="height:${Math.max(160, r.models.length * 34)}px"></div>
        </div>
      </div>

      <div class="card mb-10">
        <div class="card-head">
          <h2 class="card-title">Forecasts side by side</h2>
          <div class="card-sub">click a legend entry to isolate a model</div>
        </div>
        <div id="mdlCompare" style="height:300px"></div>
        <div class="legend" id="mdlCompareLegend"></div>
      </div>

      <div class="grid g-2 stagger" id="mdlCards"></div>`;

    /* bars */
    track(
      new C.BarChart($("#mdlScoreBar"), {
        items: r.models.map((m) => ({
          label: m.display_name,
          value: m.composite_score || 0,
          display: fmt(m.composite_score),
          color: FAMILY_COLOR[m.family],
          highlight: m.name === r.winner
        })),
        max: 1,
        height: Math.max(160, r.models.length * 34)
      })
    );

    const forecasters = r.models.filter((m) => m.forecast);
    if (forecasters.length) {
      track(
        new C.BarChart($("#mdlRmseBar"), {
          items: forecasters.map((m) => ({
            label: m.display_name,
            value: m.forecast.rmse,
            display: fmt(m.forecast.rmse, 3),
            color: FAMILY_COLOR[m.family],
            highlight: m.forecast.rmse === bestRmse
          })),
          height: Math.max(160, r.models.length * 34)
        })
      );
    }

    /* comparison chart */
    const cmpSeries = [
      {
        name: "Observed",
        values: r.series.test.values,
        color: C.PALETTE.text,
        width: 1.6
      }
    ];
    const colors = [C.PALETTE.teal, C.PALETTE.violet, C.PALETTE.amber, C.PALETTE.sky, C.PALETTE.lime];
    let ci = 0;
    forecasters.forEach((m) => {
      const s = r.model_series[m.name];
      if (!s || !s.predictions) return;
      cmpSeries.push({
        name: m.display_name,
        values: s.predictions,
        color: colors[ci++ % colors.length],
        width: 1.2,
        dash: [4, 3]
      });
    });
    const cmp = track(
      new C.TimeChart($("#mdlCompare"), {
        x: r.series.test.timestamps,
        series: cmpSeries,
        regions: (r.series.truth_mask ? spans(r.series.truth_mask) : []).map((s) => ({
          from: s[0],
          to: s[1],
          color: "rgba(251,113,133,0.08)"
        })),
        height: 300,
        unit: r.dataset.unit
      })
    );
    buildLegend($("#mdlCompareLegend"), cmp, cmpSeries, [], true, 0);

    /* per-model cards */
    $("#mdlCards").innerHTML = r.models
      .map((m) => {
        const d = m.diagnostics || {};
        const paramRows = Object.entries(m.params || {})
          .filter(([k]) => k !== "feature_names")
          .map(([k, v]) =>
            kv(
              k.replace(/_/g, " "),
              typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)
            )
          )
          .join("");

        let extra = "";
        if (d.residual_variance_reduction_pct !== undefined) {
          extra += `<div class="verdict mt-10" style="padding:12px 14px">
            <p class="prose" style="margin:0">${escapeHtml(d.interpretation || "")}</p></div>`;
        }
        if (d.order_search || d.stage_1_order_search) {
          const search = d.order_search || d.stage_1_order_search;
          extra += `<div class="mt-10"><div class="card-sub mb-10">AIC order search</div>
            <div class="table-wrap"><table><thead><tr><th>Order</th><th class="right">AIC</th><th class="right">BIC</th></tr></thead>
            <tbody>${search
              .slice(0, 6)
              .map(
                (s, i) =>
                  `<tr><td class="mono">(${s.order.join(", ")})</td><td class="num ${
                    i === 0 ? "best" : ""
                  }">${fmt(s.aic, 1)}</td><td class="num">${fmt(s.bic, 1)}</td></tr>`
              )
              .join("")}</tbody></table></div></div>`;
        }
        if (d.ljung_box_lag24_pvalue !== undefined) {
          extra += `<p class="prose mt-10"><strong>Ljung-Box (lag 24): p = ${fmt(
            d.ljung_box_lag24_pvalue,
            4
          )}.</strong> ${escapeHtml(d.residual_autocorrelation_note || "")}</p>`;
        }
        if (d.residual_learner && d.residual_learner.fallback_reason) {
          extra += `<p class="prose mt-10"><span class="chip amber">note</span> ${escapeHtml(
            d.residual_learner.fallback_reason
          )}</p>`;
        }

        return `<div class="card">
          <div class="card-head">
            <h2 class="card-title">${escapeHtml(m.display_name)}</h2>
            <div class="spacer"></div>
            ${m.name === r.winner ? '<span class="chip teal dot">selected</span>' : ""}
            <span class="chip" style="color:${FAMILY_COLOR[m.family]}">${escapeHtml(m.family)}</span>
          </div>
          ${m.error ? `<p class="prose"><span class="chip rose">failed</span> ${escapeHtml(m.error)}</p>` : ""}
          <p class="prose">${escapeHtml(m.description)}</p>
          <p class="prose" style="color:var(--text-faint)"><strong style="color:var(--teal)">Why it is here.</strong> ${escapeHtml(
            m.rationale
          )}</p>
          <div class="divider"></div>
          <div class="kv">${paramRows}</div>
          ${extra}
        </div>`;
      })
      .join("");
  }

  /* --------------------------------------------------------- detection */

  function renderDetection() {
    const r = state.result;
    if (!r) return;
    const winner = r.models.find((m) => m.name === r.winner) || r.models[0];
    const det = winner.detection;

    const options = r.models
      .map(
        (m) =>
          `<option value="${m.name}" ${m.name === r.winner ? "selected" : ""}>${escapeHtml(
            m.display_name
          )}</option>`
      )
      .join("");

    $("#detectionBody").innerHTML = `
      <div class="grid g-4 stagger mb-10">
        ${tile("True positives", det.tp === null ? "—" : det.tp, "points inside a labelled window", C.PALETTE.teal)}
        ${tile("False positives", det.fp === null ? "—" : det.fp, `${det.false_alarms_per_day} false episode(s)/day`, C.PALETTE.rose)}
        ${tile("False negatives", det.fn === null ? "—" : det.fn, "missed points inside a window", C.PALETTE.amber)}
        ${tile("Precision / Recall", det.labelled ? `${fmt(det.precision, 2)} / ${fmt(det.recall, 2)}` : "—", "point-wise, unadjusted", C.PALETTE.violet)}
      </div>

      <div class="card mb-10">
        <div class="card-head">
          <h2 class="card-title">Anomaly score against threshold</h2>
          <div class="card-sub">score above the dashed rule raises an alert</div>
          <div class="spacer"></div>
          <select id="scoreModelSel" style="width:auto;min-width:200px">${options}</select>
        </div>
        <div id="detScore" style="height:260px"></div>
      </div>

      <div class="card mb-10">
        <div class="card-head">
          <h2 class="card-title">Where each model fires</h2>
          <div class="card-sub">one row per model; blocks are alert episodes, shaded spans are ground truth</div>
        </div>
        <div id="detRaster"></div>
      </div>

      <div class="card">
        <div class="card-head">
          <h2 class="card-title">Reading these numbers</h2>
        </div>
        <div class="prose">
          <p><strong>Point-adjusted F1 is the headline, not point-wise F1.</strong>
          If any bucket inside a real incident window is flagged, the incident
          was caught — one page is enough to put a human on it. Point-wise F1
          additionally demands that the detector agree with the label on
          <em>every</em> bucket, which no operator cares about.</p>
          <p><strong>Latency is inflated by the labelling convention.</strong>
          NAB windows open before the visible excursion so that early detection
          scores well. A reported latency of several hours against the window
          start can still be prompt against the point a human would have
          noticed. The figure is left unadjusted rather than quietly corrected.</p>
          <p><strong>False alarms per day is the number that decides adoption.</strong>
          A detector with perfect recall and ten false pages a day gets muted
          within a week, at which point its real-world recall is zero. That is
          why alert burden carries negative weight in the composite score.</p>
        </div>
      </div>`;

    drawScoreChart(r.winner);
    $("#scoreModelSel").addEventListener("change", (e) => drawScoreChart(e.target.value));
    drawRaster();
  }

  function drawScoreChart(modelName) {
    const r = state.result;
    const s = r.model_series[modelName];
    const model = r.models.find((m) => m.name === modelName);
    const host = $("#detScore");
    if (!s) return;

    const markers = [];
    s.flags.forEach((f, i) => {
      if (f) markers.push({ index: i, value: s.scores[i], color: C.PALETTE.rose });
    });

    track(
      new C.TimeChart(host, {
        x: r.series.test.timestamps,
        series: [
          {
            name: model.display_name + " score",
            values: s.scores,
            color: C.PALETTE.teal,
            width: 1.4,
            area: true
          }
        ],
        regions: (r.series.truth_mask ? spans(r.series.truth_mask) : []).map((sp) => ({
          from: sp[0],
          to: sp[1],
          color: "rgba(251,113,133,0.10)"
        })),
        markers,
        threshold: s.threshold,
        thresholdLabel: `threshold ${s.threshold}`,
        height: 260,
        yLabel: "score"
      })
    );
  }

  function drawRaster() {
    const r = state.result;
    const host = $("#detRaster");
    const n = r.series.test.values.length;
    const truthSpans = r.series.truth_mask ? spans(r.series.truth_mask) : [];

    host.innerHTML = r.models
      .map((m) => {
        const s = r.model_series[m.name];
        const blocks = spans(s.flags)
          .map(
            (sp) =>
              `<i style="position:absolute;left:${(sp[0] / n) * 100}%;width:${Math.max(
                0.35,
                ((sp[1] - sp[0] + 1) / n) * 100
              )}%;top:0;bottom:0;background:${
                m.name === r.winner ? C.PALETTE.teal : C.PALETTE.sky
              };border-radius:2px;opacity:0.9"></i>`
          )
          .join("");
        const truth = truthSpans
          .map(
            (sp) =>
              `<i style="position:absolute;left:${(sp[0] / n) * 100}%;width:${
                ((sp[1] - sp[0] + 1) / n) * 100
              }%;top:0;bottom:0;background:rgba(251,113,133,0.16);border-radius:2px"></i>`
          )
          .join("");
        return `<div class="flex" style="gap:12px;padding:5px 0">
          <div style="width:170px;flex:none;font-size:11.5px;color:${
            m.name === r.winner ? "var(--text)" : "var(--text-dim)"
          }">${escapeHtml(m.display_name)}</div>
          <div style="position:relative;flex:1;height:18px;background:var(--surface-1);border-radius:3px;overflow:hidden">
            ${truth}${blocks}
          </div>
          <div style="width:64px;flex:none;text-align:right;font-size:11px" class="mono dim">${
            s.flags.filter(Boolean).length
          } pts</div>
        </div>`;
      })
      .join("");
  }

  /* ------------------------------------------------------------ alerts */

  function renderAlerts() {
    const r = state.result;
    if (!r) return;
    const inc = r.incidents;

    if (!inc.length) {
      $("#alertsBody").innerHTML = `<div class="card"><div class="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/></svg>
        <div>No incidents were raised in the hold-out window.</div></div></div>`;
      return;
    }

    const SEV = {
      critical: "var(--sev-critical)",
      high: "var(--sev-high)",
      medium: "var(--sev-medium)",
      low: "var(--sev-low)"
    };

    $("#alertsBody").innerHTML = `
      <div class="verdict mb-10">
        <div class="verdict-head">
          <span class="chip rose dot">${r.digest.actionable} actionable</span>
          <div class="verdict-title">${escapeHtml(r.digest.verdict)}</div>
        </div>
        <p class="prose" style="margin:0">Raised by <strong>${escapeHtml(
          (r.models.find((m) => m.name === r.winner) || {}).display_name || ""
        )}</strong> over ${escapeHtml(
      r.split.test_start.slice(0, 16).replace("T", " ")
    )} → ${escapeHtml(r.split.test_end.slice(0, 16).replace("T", " "))}. Click an
        incident to open its playbook.</p>
      </div>

      <div class="grid stagger" style="gap:10px" id="incList">
        ${inc
          .map(
            (i) => `<div class="incident ${i.severity}" style="--sev:${SEV[i.severity]}">
            <div class="incident-top">
              <span class="sev-badge">${i.severity}</span>
              <span class="incident-headline">${escapeHtml(i.headline)}</span>
              <div class="spacer" style="flex:1"></div>
              ${
                i.confirmed_by_ground_truth === true
                  ? '<span class="chip teal dot">confirmed by label</span>'
                  : i.confirmed_by_ground_truth === false
                  ? '<span class="chip amber dot">outside labelled window</span>'
                  : ""
              }
              ${i.slo_breached ? '<span class="chip rose dot">SLO breached</span>' : ""}
              <span class="incident-time">${escapeHtml(
                i.start.slice(5, 16).replace("T", " ")
              )} → ${escapeHtml(i.end.slice(11, 16))}</span>
            </div>
            <div class="incident-body">
              ${escapeHtml(i.summary)}
              <span class="dim"> Peak ${fmt(i.peak_value, 2)} ${escapeHtml(i.unit)} against an
              expected ${fmt(i.expected_value, 2)}${
              i.deviation_pct !== null ? ` (${i.deviation_pct > 0 ? "+" : ""}${i.deviation_pct}%)` : ""
            }, score ${fmt(i.peak_score, 2)}σ-equivalent over ${i.duration_minutes} minutes.</span>
            </div>
            <div class="incident-action"><b>Recommended next step.</b> ${escapeHtml(
              i.recommended_action
            )}</div>
          </div>`
          )
          .join("")}
      </div>`;

    $$("#incList .incident").forEach((el) => {
      el.addEventListener("click", () => el.classList.toggle("open"));
    });
  }

  /* --------------------------------------------------------- benchmark */

  async function runBenchmark() {
    const btn = $("#benchBtn");
    if (btn.disabled) return;
    btn.disabled = true;
    btn.classList.add("running");
    $("#benchTrack").style.display = "";
    try {
      const job = await api("/api/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          freq: $("#freqSel").value,
          test_size: Number($("#testSize").value) / 100,
          threshold_sigma: Number($("#sigma").value),
          models: selectedModels()
        })
      });
      state.benchJobId = job.id;
      await poll(job.id, (stage, p) => {
        $("#benchFill").style.width = p + "%";
        $("#benchPct").textContent = p + "%";
        $("#benchStage").textContent = stage;
      });
      const done = await api(`/api/jobs/${job.id}/result`);
      state.benchmark = done.result;
      renderBenchmark();
      toast("Fleet benchmark complete across " + state.benchmark.n_datasets + " services.");
    } catch (err) {
      toast("Benchmark failed: " + err.message, true);
    } finally {
      btn.disabled = false;
      btn.classList.remove("running");
    }
  }

  function renderBenchmark() {
    const b = state.benchmark;
    if (!b) return;
    const s = b.summary;

    $("#benchBody").innerHTML = `
      <div class="verdict mb-10">
        <div class="verdict-head">
          <span class="chip teal dot">${b.n_datasets} services</span>
          <div class="verdict-title">Aggregated across the labelled corpus</div>
        </div>
        <ul class="bullet-list">${b.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
      </div>

      <div class="card mb-10 pad-0">
        <div class="card-head" style="padding:18px 20px 12px;margin:0">
          <h2 class="card-title">Aggregate leaderboard</h2>
          <div class="card-sub">macro = each service weighted equally · pooled = each observation weighted equally · RMSE is unit-dependent and is not averaged across services</div>
        </div>
        <div class="table-wrap"><table>
          <thead><tr>
            <th>#</th><th>Model</th>
            <th class="right">Macro composite</th>
            <th class="right">Macro PA&nbsp;F1</th>
            <th class="right">Pooled P</th>
            <th class="right">Pooled R</th>
            <th class="right">Pooled F1</th>
            <th class="right">Windows caught</th>
            <th class="right">FA/day</th>
            <th class="right">Macro MASE</th>
            <th class="right">Mean fit s</th>
          </tr></thead>
          <tbody>${s
            .map(
              (m) => `<tr class="${m.rank === 1 ? "winner" : ""}">
              <td><span class="rank-pill ${m.rank === 1 ? "gold" : ""}">${m.rank}</span></td>
              <td><b>${escapeHtml(m.display_name)}</b></td>
              <td class="num ${m.rank === 1 ? "best" : ""}">${fmt(m.macro_composite)}</td>
              <td class="num">${fmt(m.macro_pa_f1)}</td>
              <td class="num">${fmt(m.pooled_precision)}</td>
              <td class="num">${fmt(m.pooled_recall)}</td>
              <td class="num">${fmt(m.pooled_f1)}</td>
              <td class="num">${m.segments_detected}/${m.segments_total}</td>
              <td class="num">${fmt(m.macro_false_alarms_per_day, 2)}</td>
              <td class="num">${fmt(m.macro_mase, 3)}</td>
              <td class="num dim">${fmt(m.mean_fit_seconds, 2)}</td>
            </tr>`
            )
            .join("")}</tbody>
        </table></div>
      </div>

      <div class="grid g-2 mb-10">
        <div class="card">
          <div class="card-head"><h2 class="card-title">Macro composite by model</h2></div>
          <div id="benchBar" style="height:${Math.max(160, s.length * 34)}px"></div>
        </div>
        <div class="card">
          <div class="card-head"><h2 class="card-title">Incident windows recovered</h2></div>
          <div id="benchSeg" style="height:${Math.max(160, s.length * 34)}px"></div>
        </div>
      </div>

      <div class="card pad-0">
        <div class="card-head" style="padding:18px 20px 12px;margin:0">
          <h2 class="card-title">Per-service winners</h2>
          <div class="card-sub">which model each service selected on its own hold-out</div>
        </div>
        <div class="table-wrap"><table>
          <thead><tr><th>Service</th><th>Metric</th><th class="right">Test points</th><th class="right">Labelled windows</th><th>Selected model</th></tr></thead>
          <tbody>${b.datasets
            .map(
              (d) => `<tr>
              <td>${escapeHtml(d.service || "—")}</td>
              <td>${escapeHtml(d.title || d.dataset)}</td>
              <td class="num">${d.n_test}</td>
              <td class="num">${d.n_label_windows}</td>
              <td><span class="chip teal">${escapeHtml(
                (s.find((x) => x.name === d.winner) || {}).display_name || d.winner || "—"
              )}</span></td>
            </tr>`
            )
            .join("")}</tbody>
        </table></div>
      </div>`;

    track(
      new C.BarChart($("#benchBar"), {
        items: s.map((m) => ({
          label: m.display_name,
          value: m.macro_composite || 0,
          display: fmt(m.macro_composite),
          color: FAMILY_COLOR[m.family],
          highlight: m.rank === 1
        })),
        max: 1,
        height: Math.max(160, s.length * 34)
      })
    );
    track(
      new C.BarChart($("#benchSeg"), {
        items: s.map((m) => ({
          label: m.display_name,
          value: m.segment_recall || 0,
          display: `${m.segments_detected}/${m.segments_total}`,
          color: FAMILY_COLOR[m.family],
          highlight: (m.segment_recall || 0) >= 0.999
        })),
        max: 1,
        height: Math.max(160, s.length * 34)
      })
    );
  }

  /* ---------------------------------------------------------------- init */

  function bindControls() {
    $$(".nav-item").forEach((n) =>
      n.addEventListener("click", () => go(n.dataset.view))
    );

    $("#runBtn").addEventListener("click", run);
    $("#benchBtn").addEventListener("click", runBenchmark);

    $("#exportBtn").addEventListener("click", () => {
      if (state.jobId) window.open(`/api/jobs/${state.jobId}/export`, "_blank");
    });

    $("#uploadInput").addEventListener("change", (e) => {
      handleUpload(e.target.files[0]);
      e.target.value = "";
    });

    const sliders = [
      ["#testSize", "#testSizeVal", (v) => v + "%"],
      ["#sigma", "#sigmaVal", (v) => Number(v).toFixed(1) + "σ"],
      ["#contam", "#contamVal", (v) => Number(v).toFixed(1) + "%"],
      ["#fourierK", "#fourierVal", (v) => v],
      ["#horizon", "#horizonVal", (v) => v]
    ];
    sliders.forEach(([input, label, fn]) => {
      const el = $(input);
      const out = $(label);
      el.addEventListener("input", () => (out.textContent = fn(el.value)));
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
    });
  }

  async function init() {
    runBoot();
    bindControls();
    renderPipeline();
    try {
      await loadCatalogue();
    } catch (err) {
      toast("Could not reach the analysis engine: " + err.message, true);
      $("#statusDot").className = "status-dot err";
    }
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
