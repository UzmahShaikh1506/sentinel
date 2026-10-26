/* ==========================================================================
   SENTINEL-TS :: Canvas chart engine

   Hand-written rather than pulled from a CDN for three reasons that matter
   for this project: the console must run on a lab machine with no network
   access; the plots need anomaly bands, ground-truth shading and residual
   stems that general-purpose libraries do not express cleanly; and the
   progressive-reveal animation needs frame-level control.

   Everything is device-pixel-ratio aware and redraws on container resize.
   ========================================================================== */

(function (global) {
  "use strict";

  const CSS = getComputedStyle(document.documentElement);
  const token = (name, fallback) =>
    (CSS.getPropertyValue(name) || fallback || "").trim() || fallback;

  const PALETTE = {
    text: token("--text", "#e8edf5"),
    dim: token("--text-dim", "#9aa5b8"),
    faint: token("--text-faint", "#5f6b80"),
    grid: "rgba(255,255,255,0.055)",
    axis: "rgba(255,255,255,0.14)",
    teal: token("--teal", "#5eead4"),
    violet: token("--violet", "#a78bfa"),
    amber: token("--amber", "#fbbf24"),
    rose: token("--rose", "#fb7185"),
    sky: token("--sky", "#60a5fa"),
    lime: token("--lime", "#a3e635")
  };

  const MONO =
    '11px ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace';

  /* ---------------------------------------------------------------- utils */

  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const isNum = (v) => typeof v === "number" && isFinite(v);

  function niceTicks(min, max, count) {
    if (!isFinite(min) || !isFinite(max)) return [0, 1];
    if (min === max) {
      const pad = Math.abs(min) * 0.1 || 1;
      min -= pad;
      max += pad;
    }
    const raw = (max - min) / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const start = Math.ceil(min / step) * step;
    const out = [];
    for (let v = start; v <= max + step * 1e-6; v += step) out.push(v);
    return out.length ? out : [min, max];
  }

  function fmtNum(v) {
    if (!isNum(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(2) + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return "0";
    return v.toPrecision(3);
  }

  function fmtTime(iso, withDate) {
    const d = new Date(iso);
    if (isNaN(d)) return String(iso);
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mm = String(d.getUTCMinutes()).padStart(2, "0");
    if (!withDate) return hh + ":" + mm;
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return months[d.getUTCMonth()] + " " + d.getUTCDate() + " " + hh + ":" + mm;
  }

  function hexToRgba(hex, alpha) {
    if (hex.startsWith("rgb")) return hex;
    let h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  }

  /* -------------------------------------------------------------- Chart base */

  class Chart {
    constructor(container, opts) {
      this.container = container;
      this.opts = Object.assign({ height: 260, padding: {} }, opts || {});
      this.pad = Object.assign(
        { top: 14, right: 16, bottom: 28, left: 52 },
        this.opts.padding
      );

      container.innerHTML = "";
      container.classList.add("chart-wrap");

      this.canvas = document.createElement("canvas");
      this.ctx = this.canvas.getContext("2d");
      container.appendChild(this.canvas);

      this.tip = document.createElement("div");
      this.tip.className = "chart-tip";
      container.appendChild(this.tip);

      this.anim = 0;
      this.hover = null;
      this.destroyed = false;
      // Declared on the base class because the constructor calls _resize(),
      // which calls draw(), before any subclass constructor body has run. A
      // subclass that initialised this itself would be too late and every
      // first paint would throw.
      this.hidden = new Set();

      this._resize = this._resize.bind(this);
      this._onMove = this._onMove.bind(this);
      this._onLeave = this._onLeave.bind(this);

      this.canvas.addEventListener("mousemove", this._onMove);
      this.canvas.addEventListener("mouseleave", this._onLeave);

      if (global.ResizeObserver) {
        this.ro = new ResizeObserver(() => this._resize());
        this.ro.observe(container);
      } else {
        global.addEventListener("resize", this._resize);
      }
      this._resize();
    }

    destroy() {
      this.destroyed = true;
      if (this.ro) this.ro.disconnect();
      this.canvas.removeEventListener("mousemove", this._onMove);
      this.canvas.removeEventListener("mouseleave", this._onLeave);
    }

    _resize() {
      if (this.destroyed) return;
      const dpr = global.devicePixelRatio || 1;
      const w = this.container.clientWidth || 600;
      const h = this.opts.height;
      this.w = w;
      this.h = h;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.height = h + "px";
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.draw();
    }

    get plot() {
      return {
        x: this.pad.left,
        y: this.pad.top,
        w: Math.max(10, this.w - this.pad.left - this.pad.right),
        h: Math.max(10, this.h - this.pad.top - this.pad.bottom)
      };
    }

    animateIn(duration) {
      const d = duration || 850;
      const start = performance.now();
      const step = (now) => {
        if (this.destroyed) return;
        this.anim = clamp((now - start) / d, 0, 1);
        this.draw();
        if (this.anim < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    _onMove() {}
    _onLeave() {
      this.hover = null;
      this.tip.classList.remove("show");
      this.draw();
    }
    draw() {}
  }

  /* ----------------------------------------------------------- TimeChart */

  class TimeChart extends Chart {
    /**
     * opts:
     *   x        : array of ISO strings (shared axis)
     *   series   : [{ name, values, color, width, dash, area, hidden }]
     *   bands    : [{ lower, upper, color, name }]
     *   regions  : [{ from, to, color, label }]    index-based shaded spans
     *   markers  : [{ index, color, label, value }]
     *   yLabel   : string
     *   yMin/yMax: optional fixed bounds
     *   unit     : appended in tooltips
     */
    constructor(container, opts) {
      super(container, opts);
      this.animateIn(this.opts.animate === false ? 0 : 900);
    }

    setData(opts) {
      Object.assign(this.opts, opts);
      this.anim = 0;
      this.animateIn(700);
    }

    toggle(name) {
      if (this.hidden.has(name)) this.hidden.delete(name);
      else this.hidden.add(name);
      this.draw();
    }

    _bounds() {
      const o = this.opts;
      let min = Infinity;
      let max = -Infinity;
      const consider = (arr) => {
        if (!arr) return;
        for (let i = 0; i < arr.length; i++) {
          const v = arr[i];
          if (isNum(v)) {
            if (v < min) min = v;
            if (v > max) max = v;
          }
        }
      };
      (o.series || []).forEach((s) => {
        if (!this.hidden.has(s.name)) consider(s.values);
      });
      (o.bands || []).forEach((b) => {
        if (this.hidden.has(b.name)) return;
        consider(b.lower);
        consider(b.upper);
      });
      if (!isFinite(min) || !isFinite(max)) {
        min = 0;
        max = 1;
      }
      if (isNum(o.yMin)) min = o.yMin;
      if (isNum(o.yMax)) max = o.yMax;
      const span = max - min || 1;
      return { min: min - span * 0.08, max: max + span * 0.08 };
    }

    _scales() {
      const p = this.plot;
      const n = (this.opts.x || []).length;
      const { min, max } = this._bounds();
      const sx = (i) => p.x + (n <= 1 ? 0 : (i / (n - 1)) * p.w);
      const sy = (v) => p.y + p.h - ((v - min) / (max - min)) * p.h;
      return { sx, sy, min, max, n, p };
    }

    draw() {
      if (this.destroyed || !this.ctx) return;
      const ctx = this.ctx;
      const o = this.opts;
      const { sx, sy, min, max, n, p } = this._scales();
      ctx.clearRect(0, 0, this.w, this.h);
      if (!n) return;

      const reveal = p.x + p.w * easeOut(this.anim);

      /* -- shaded ground-truth / highlight regions ----------------------- */
      (o.regions || []).forEach((r) => {
        const x0 = sx(clamp(r.from, 0, n - 1));
        const x1 = sx(clamp(r.to, 0, n - 1));
        ctx.fillStyle = r.color || "rgba(251,113,133,0.10)";
        ctx.fillRect(x0, p.y, Math.max(1.5, x1 - x0), p.h);
        if (r.label) {
          ctx.fillStyle = hexToRgba(PALETTE.rose, 0.75);
          ctx.font = "10px " + MONO.split("px ")[1];
          ctx.textAlign = "left";
          ctx.fillText(r.label, x0 + 4, p.y + 11);
        }
      });

      /* -- grid + y axis ------------------------------------------------- */
      const ticks = niceTicks(min, max, 5);
      ctx.strokeStyle = PALETTE.grid;
      ctx.lineWidth = 1;
      ctx.font = MONO;
      ctx.fillStyle = PALETTE.faint;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ticks.forEach((t) => {
        const y = Math.round(sy(t)) + 0.5;
        if (y < p.y - 1 || y > p.y + p.h + 1) return;
        ctx.beginPath();
        ctx.moveTo(p.x, y);
        ctx.lineTo(p.x + p.w, y);
        ctx.stroke();
        ctx.fillText(fmtNum(t), p.x - 8, y);
      });

      /* -- x axis labels -------------------------------------------------- */
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const labelCount = clamp(Math.floor(p.w / 108), 2, 8);
      for (let i = 0; i <= labelCount; i++) {
        const idx = Math.round((i / labelCount) * (n - 1));
        const x = sx(idx);
        ctx.fillStyle = PALETTE.faint;
        ctx.fillText(fmtTime(o.x[idx], true), x, p.y + p.h + 8);
      }

      /* -- confidence bands ----------------------------------------------- */
      (o.bands || []).forEach((b) => {
        if (this.hidden.has(b.name)) return;
        ctx.save();
        ctx.beginPath();
        ctx.rect(p.x, p.y, reveal - p.x, p.h);
        ctx.clip();
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          const v = b.upper[i];
          if (!isNum(v)) continue;
          const X = sx(i);
          const Y = clamp(sy(v), p.y - 40, p.y + p.h + 40);
          started ? ctx.lineTo(X, Y) : (ctx.moveTo(X, Y), (started = true));
        }
        for (let i = n - 1; i >= 0; i--) {
          const v = b.lower[i];
          if (!isNum(v)) continue;
          ctx.lineTo(sx(i), clamp(sy(v), p.y - 40, p.y + p.h + 40));
        }
        ctx.closePath();
        ctx.fillStyle = b.color || hexToRgba(PALETTE.teal, 0.08);
        ctx.fill();
        ctx.restore();
      });

      /* -- series ---------------------------------------------------------- */
      (o.series || []).forEach((s) => {
        if (this.hidden.has(s.name)) return;
        const color = s.color || PALETTE.teal;

        ctx.save();
        ctx.beginPath();
        ctx.rect(p.x, p.y, reveal - p.x, p.h);
        ctx.clip();

        if (s.area) {
          ctx.beginPath();
          let begun = false;
          let firstX = p.x;
          for (let i = 0; i < n; i++) {
            const v = s.values[i];
            if (!isNum(v)) continue;
            const X = sx(i);
            if (!begun) {
              firstX = X;
              ctx.moveTo(X, sy(v));
              begun = true;
            } else ctx.lineTo(X, sy(v));
          }
          if (begun) {
            ctx.lineTo(sx(n - 1), p.y + p.h);
            ctx.lineTo(firstX, p.y + p.h);
            ctx.closePath();
            const grad = ctx.createLinearGradient(0, p.y, 0, p.y + p.h);
            grad.addColorStop(0, hexToRgba(color, 0.26));
            grad.addColorStop(1, hexToRgba(color, 0));
            ctx.fillStyle = grad;
            ctx.fill();
          }
        }

        ctx.beginPath();
        ctx.lineWidth = s.width || 1.6;
        ctx.strokeStyle = color;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        if (s.dash) ctx.setLineDash(s.dash);
        let begun = false;
        for (let i = 0; i < n; i++) {
          const v = s.values[i];
          if (!isNum(v)) {
            begun = false;
            continue;
          }
          const X = sx(i);
          const Y = sy(v);
          begun ? ctx.lineTo(X, Y) : (ctx.moveTo(X, Y), (begun = true));
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      });

      /* -- anomaly markers --------------------------------------------------- */
      (o.markers || []).forEach((m) => {
        const X = sx(m.index);
        if (X > reveal) return;
        const Y = isNum(m.value) ? sy(m.value) : p.y + p.h - 6;
        const color = m.color || PALETTE.rose;
        ctx.beginPath();
        ctx.arc(X, Y, 3.6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(X, Y, 7, 0, Math.PI * 2);
        ctx.strokeStyle = hexToRgba(color, 0.4);
        ctx.lineWidth = 1.2;
        ctx.stroke();
      });

      /* -- threshold rule ----------------------------------------------------- */
      if (isNum(o.threshold)) {
        const y = Math.round(sy(o.threshold)) + 0.5;
        if (y > p.y && y < p.y + p.h) {
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = hexToRgba(PALETTE.amber, 0.65);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(p.x, y);
          ctx.lineTo(p.x + p.w, y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = hexToRgba(PALETTE.amber, 0.9);
          ctx.font = "10px " + MONO.split("px ")[1];
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(o.thresholdLabel || "threshold", p.x + 5, y - 3);
          ctx.restore();
        }
      }

      /* -- crosshair ----------------------------------------------------------- */
      if (this.hover !== null && this.anim >= 1) {
        const X = sx(this.hover);
        ctx.save();
        ctx.strokeStyle = "rgba(255,255,255,0.22)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(X, p.y);
        ctx.lineTo(X, p.y + p.h);
        ctx.stroke();
        ctx.setLineDash([]);
        (o.series || []).forEach((s) => {
          if (this.hidden.has(s.name)) return;
          const v = s.values[this.hover];
          if (!isNum(v)) return;
          ctx.beginPath();
          ctx.arc(X, sy(v), 3.4, 0, Math.PI * 2);
          ctx.fillStyle = s.color || PALETTE.teal;
          ctx.fill();
          ctx.lineWidth = 1.6;
          ctx.strokeStyle = token("--bg", "#06080d");
          ctx.stroke();
        });
        ctx.restore();
      }

      /* -- axis frame ---------------------------------------------------------- */
      ctx.strokeStyle = PALETTE.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p.x + 0.5, p.y);
      ctx.lineTo(p.x + 0.5, p.y + p.h + 0.5);
      ctx.lineTo(p.x + p.w, p.y + p.h + 0.5);
      ctx.stroke();

      if (o.yLabel) {
        ctx.save();
        ctx.translate(12, p.y + p.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = PALETTE.faint;
        ctx.font = "10px " + MONO.split("px ")[1];
        ctx.fillText(o.yLabel, 0, 0);
        ctx.restore();
      }
    }

    _onMove(ev) {
      const o = this.opts;
      const n = (o.x || []).length;
      if (!n) return;
      const rect = this.canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const p = this.plot;
      const frac = clamp((mx - p.x) / p.w, 0, 1);
      const idx = Math.round(frac * (n - 1));
      if (idx === this.hover) {
        this._placeTip(ev, idx);
        return;
      }
      this.hover = idx;
      this.draw();
      this._placeTip(ev, idx);
    }

    _placeTip(ev, idx) {
      const o = this.opts;
      const rows = (o.series || [])
        .filter((s) => !this.hidden.has(s.name) && isNum(s.values[idx]))
        .map(
          (s) =>
            `<div class="tt-row"><i class="tt-sw" style="background:${
              s.color || PALETTE.teal
            }"></i><span class="tt-name">${s.name}</span><span class="tt-val">${fmtNum(
              s.values[idx]
            )}${o.unit ? " " + o.unit : ""}</span></div>`
        )
        .join("");

      let extra = "";
      const flagged = (o.markers || []).some((m) => m.index === idx);
      if (flagged) extra += `<div class="tt-row" style="color:${PALETTE.rose}">▲ alert raised</div>`;
      const inRegion = (o.regions || []).some((r) => idx >= r.from && idx <= r.to);
      if (inRegion) extra += `<div class="tt-row" style="color:${PALETTE.amber}">● inside labelled window</div>`;

      this.tip.innerHTML =
        `<div class="tt-time">${fmtTime(o.x[idx], true)}</div>` + rows + extra;
      this.tip.classList.add("show");

      const rect = this.canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;
      const tw = this.tip.offsetWidth;
      const th = this.tip.offsetHeight;
      this.tip.style.left = clamp(mx + 14, 4, this.w - tw - 4) + "px";
      this.tip.style.top = clamp(my - th - 10, 4, this.h - th - 4) + "px";
    }
  }

  /* ------------------------------------------------------------- BarChart */

  class BarChart extends Chart {
    /** opts: { items: [{label, value, color, note}], max, unit } */
    constructor(container, opts) {
      super(
        container,
        Object.assign({ padding: { top: 8, right: 60, bottom: 8, left: 150 } }, opts)
      );
      this.animateIn(900);
    }

    setData(opts) {
      Object.assign(this.opts, opts);
      this.anim = 0;
      this.animateIn(800);
    }

    draw() {
      if (this.destroyed) return;
      const ctx = this.ctx;
      const items = this.opts.items || [];
      ctx.clearRect(0, 0, this.w, this.h);
      if (!items.length) return;

      const p = this.plot;
      const max =
        this.opts.max || Math.max(...items.map((i) => i.value || 0), 1e-9);
      const rowH = p.h / items.length;
      const barH = Math.min(16, rowH * 0.5);
      const t = easeOut(this.anim);

      items.forEach((it, i) => {
        const cy = p.y + rowH * i + rowH / 2;

        ctx.font = "11.5px " + MONO.split("px ")[1];
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillStyle = it.highlight ? PALETTE.text : PALETTE.dim;
        ctx.fillText(it.label, p.x - 10, cy);

        ctx.fillStyle = "rgba(255,255,255,0.05)";
        ctx.beginPath();
        ctx.roundRect(p.x, cy - barH / 2, p.w, barH, barH / 2);
        ctx.fill();

        const width = Math.max(2, (it.value / max) * p.w * t);
        const color = it.color || PALETTE.teal;
        const grad = ctx.createLinearGradient(p.x, 0, p.x + width, 0);
        grad.addColorStop(0, hexToRgba(color, 0.55));
        grad.addColorStop(1, color);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(p.x, cy - barH / 2, width, barH, barH / 2);
        ctx.fill();

        ctx.textAlign = "left";
        ctx.fillStyle = it.highlight ? color : PALETTE.dim;
        ctx.fillText(
          (it.display !== undefined ? it.display : fmtNum(it.value)) +
            (this.opts.unit || ""),
          p.x + p.w + 8,
          cy
        );
      });
    }
  }

  /* --------------------------------------------------------------- Stem */

  class StemChart extends Chart {
    /** opts: { values: [], band: number, color, startLag } */
    constructor(container, opts) {
      super(
        container,
        Object.assign({ padding: { top: 12, right: 12, bottom: 24, left: 44 } }, opts)
      );
      this.animateIn(700);
    }

    draw() {
      if (this.destroyed) return;
      const ctx = this.ctx;
      const vals = this.opts.values || [];
      ctx.clearRect(0, 0, this.w, this.h);
      if (!vals.length) return;

      const p = this.plot;
      const band = this.opts.band || 0;
      const maxAbs = Math.max(
        ...vals.filter(isNum).map(Math.abs),
        band * 1.6,
        0.12
      );
      const sy = (v) => p.y + p.h / 2 - (v / maxAbs) * (p.h / 2 - 6);
      const sx = (i) => p.x + (i / Math.max(1, vals.length - 1)) * p.w;
      const t = easeOut(this.anim);
      const color = this.opts.color || PALETTE.teal;

      if (band) {
        ctx.fillStyle = hexToRgba(PALETTE.sky, 0.09);
        ctx.fillRect(p.x, sy(band), p.w, sy(-band) - sy(band));
        ctx.strokeStyle = hexToRgba(PALETTE.sky, 0.42);
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        [band, -band].forEach((b) => {
          ctx.beginPath();
          ctx.moveTo(p.x, sy(b));
          ctx.lineTo(p.x + p.w, sy(b));
          ctx.stroke();
        });
        ctx.setLineDash([]);
      }

      ctx.strokeStyle = PALETTE.axis;
      ctx.beginPath();
      ctx.moveTo(p.x, sy(0) + 0.5);
      ctx.lineTo(p.x + p.w, sy(0) + 0.5);
      ctx.stroke();

      vals.forEach((v, i) => {
        if (!isNum(v)) return;
        const X = sx(i);
        const V = v * t;
        const significant = Math.abs(v) > band;
        ctx.strokeStyle = significant ? color : hexToRgba(color, 0.32);
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(X, sy(0));
        ctx.lineTo(X, sy(V));
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(X, sy(V), significant ? 2.6 : 1.8, 0, Math.PI * 2);
        ctx.fillStyle = significant ? color : hexToRgba(color, 0.45);
        ctx.fill();
      });

      ctx.font = MONO;
      ctx.fillStyle = PALETTE.faint;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(fmtNum(maxAbs), p.x - 7, sy(maxAbs));
      ctx.fillText("0", p.x - 7, sy(0));
      ctx.fillText(fmtNum(-maxAbs), p.x - 7, sy(-maxAbs));

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const step = Math.max(1, Math.round(vals.length / 8));
      for (let i = 0; i < vals.length; i += step) {
        ctx.fillText(String(i), sx(i), p.y + p.h + 6);
      }
    }
  }

  /* ------------------------------------------------------------ Heatmap */

  class Heatmap extends Chart {
    /** opts: { matrix: [[..]], rows: [], cols: [], unit } */
    constructor(container, opts) {
      super(
        container,
        Object.assign({ padding: { top: 10, right: 10, bottom: 24, left: 42 } }, opts)
      );
      this.animateIn(700);
    }

    draw() {
      if (this.destroyed) return;
      const ctx = this.ctx;
      const m = this.opts.matrix || [];
      ctx.clearRect(0, 0, this.w, this.h);
      if (!m.length) return;

      const p = this.plot;
      const rows = m.length;
      const cols = m[0].length;
      const flat = m.flat().filter(isNum);
      const lo = Math.min(...flat);
      const hi = Math.max(...flat);
      const cw = p.w / cols;
      const ch = p.h / rows;
      const t = easeOut(this.anim);

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = m[r][c];
          const X = p.x + c * cw;
          const Y = p.y + r * ch;
          if (!isNum(v)) {
            ctx.fillStyle = "rgba(255,255,255,0.03)";
          } else {
            const norm = hi > lo ? (v - lo) / (hi - lo) : 0.5;
            // Teal -> violet -> rose ramp: cool is quiet, warm is loaded.
            const stops = [
              [16, 44, 56],
              [45, 212, 191],
              [167, 139, 250],
              [251, 113, 133]
            ];
            const pos = norm * (stops.length - 1);
            const i0 = Math.floor(pos);
            const i1 = Math.min(stops.length - 1, i0 + 1);
            const f = pos - i0;
            const col = stops[i0].map((a, k) => Math.round(a + (stops[i1][k] - a) * f));
            ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${0.3 + 0.6 * norm})`;
          }
          const inset = 0.6;
          ctx.globalAlpha = t;
          ctx.fillRect(X + inset, Y + inset, cw - inset * 2, ch - inset * 2);
          ctx.globalAlpha = 1;
        }
      }

      ctx.font = "10px " + MONO.split("px ")[1];
      ctx.fillStyle = PALETTE.faint;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      (this.opts.rows || []).forEach((label, r) => {
        ctx.fillText(label, p.x - 7, p.y + ch * r + ch / 2);
      });
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let c = 0; c < cols; c += 3) {
        ctx.fillText(String(c).padStart(2, "0"), p.x + cw * c + cw / 2, p.y + p.h + 6);
      }
    }
  }

  /* ---------------------------------------------------------- Histogram */

  class Histogram extends Chart {
    /** opts: { counts: [], edges: [], color } */
    constructor(container, opts) {
      super(
        container,
        Object.assign({ padding: { top: 10, right: 12, bottom: 24, left: 42 } }, opts)
      );
      this.animateIn(700);
    }

    draw() {
      if (this.destroyed) return;
      const ctx = this.ctx;
      const counts = this.opts.counts || [];
      const edges = this.opts.edges || [];
      ctx.clearRect(0, 0, this.w, this.h);
      if (!counts.length) return;

      const p = this.plot;
      const max = Math.max(...counts, 1);
      const bw = p.w / counts.length;
      const t = easeOut(this.anim);
      const color = this.opts.color || PALETTE.violet;

      counts.forEach((c, i) => {
        const h = (c / max) * p.h * t;
        const X = p.x + i * bw;
        const grad = ctx.createLinearGradient(0, p.y + p.h - h, 0, p.y + p.h);
        grad.addColorStop(0, hexToRgba(color, 0.9));
        grad.addColorStop(1, hexToRgba(color, 0.22));
        ctx.fillStyle = grad;
        ctx.fillRect(X + 0.5, p.y + p.h - h, Math.max(1, bw - 1), h);
      });

      ctx.font = "10px " + MONO.split("px ")[1];
      ctx.fillStyle = PALETTE.faint;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(String(max), p.x - 7, p.y + 6);
      ctx.fillText("0", p.x - 7, p.y + p.h);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      [0, Math.floor(edges.length / 2), edges.length - 1].forEach((i) => {
        if (edges[i] === undefined) return;
        const X = p.x + (i / Math.max(1, edges.length - 1)) * p.w;
        ctx.fillText(fmtNum(edges[i]), X, p.y + p.h + 6);
      });
    }
  }

  /* ---------------------------------------------------------- Sparkline */

  function sparkline(svgEl, values, color) {
    const vals = (values || []).filter(isNum);
    if (vals.length < 2) return;
    const w = 68;
    const h = 26;
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const span = hi - lo || 1;
    const pts = vals
      .map((v, i) => {
        const x = (i / (vals.length - 1)) * w;
        const y = h - 2 - ((v - lo) / span) * (h - 4);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    svgEl.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svgEl.innerHTML =
      `<polyline points="${pts}" fill="none" stroke="${color || PALETTE.teal}" ` +
      `stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>`;
  }

  /* -------------------------------------------------------------- export */

  global.SChart = {
    TimeChart,
    BarChart,
    StemChart,
    Heatmap,
    Histogram,
    sparkline,
    PALETTE,
    fmtNum,
    fmtTime,
    hexToRgba
  };
})(window);
