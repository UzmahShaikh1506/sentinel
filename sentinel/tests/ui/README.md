# Console smoke test

Drives the real console (`frontend/index.html` + `charts.js` + `app.js`) inside
jsdom against a running server, launching an actual analysis through the Run
button and asserting that every view renders. It is kept separate from the
pytest suite because it needs Node and a live server.

This test earns its keep. It caught a crash that affected every chart on the
page: `Chart`'s constructor calls `_resize()` → `draw()` before any subclass
constructor body has run, so `TimeChart`'s own field initialisation was too
late and the first paint of every time series threw. Nothing in the Python
suite could have seen that.

## Running it

```bash
# terminal 1
python run.py --port 8800

# terminal 2
npm install jsdom
node tests/ui/ui_smoke.js
```

Exits non-zero on any failure and prints a per-assertion log. Canvas is stubbed
(jsdom has no 2D context), so this verifies that render paths execute and
produce the right DOM, not that pixels are correct.
