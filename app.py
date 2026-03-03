from flask import Flask, render_template, request, jsonify
from screener import run_screener, get_segments
import threading

app = Flask(__name__)

scan_state = {
    "running": False, "progress": 0, "total": 0,
    "current_symbol": "", "results": [], "error": None
}
scan_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html", segments=get_segments())


@app.route("/api/segments")
def api_segments():
    return jsonify(get_segments())


@app.route("/api/scan", methods=["POST"])
def api_scan():
    global scan_state
    with scan_lock:
        if scan_state["running"]:
            return jsonify({"error": "Scan already running"}), 400
        scan_state = {"running": True, "progress": 0, "total": 0,
                      "current_symbol": "", "results": [], "error": None}

    d = request.json or {}

    def cb(done, total, sym):
        with scan_lock:
            scan_state["progress"] = done
            scan_state["total"] = total
            scan_state["current_symbol"] = sym

    def run_bg():
        global scan_state
        try:
            results = run_screener(
                segment=d.get("segment", "All"),
                timeframe=d.get("timeframe", "1W"),
                signal_filter=d.get("signal_filter", "both"),
                lookback=int(d.get("lookback", 3)),
                st_period=int(d.get("st_period", 7)),
                st_multiplier=float(d.get("st_multiplier", 3)),
                use_rsi=bool(d.get("use_rsi", True)),
                use_adx=bool(d.get("use_adx", True)),
                use_ema=bool(d.get("use_ema", True)),
                use_volume=bool(d.get("use_volume", True)),
                rsi_period=int(d.get("rsi_period", 14)),
                rsi_min=float(d.get("rsi_min", 45)),
                rsi_max=float(d.get("rsi_max", 70)),
                rsi_bear_min=float(d.get("rsi_bear_min", 30)),
                rsi_bear_max=float(d.get("rsi_bear_max", 55)),
                adx_period=int(d.get("adx_period", 14)),
                adx_min=float(d.get("adx_min", 20)),
                ema_fast=int(d.get("ema_fast", 20)),
                ema_slow=int(d.get("ema_slow", 50)),
                vol_lookback=int(d.get("vol_lookback", 20)),
                vol_threshold=float(d.get("vol_threshold", 1.5)),
                progress_callback=cb
            )
            with scan_lock:
                scan_state["results"] = results
                scan_state["running"] = False
        except Exception as e:
            with scan_lock:
                scan_state["error"] = str(e)
                scan_state["running"] = False

    threading.Thread(target=run_bg, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/scan/progress")
def api_progress():
    with scan_lock:
        return jsonify({
            "running":        scan_state["running"],
            "progress":       scan_state["progress"],
            "total":          scan_state["total"],
            "current_symbol": scan_state["current_symbol"],
            "results_count":  len(scan_state["results"]),
            "error":          scan_state["error"]
        })


@app.route("/api/scan/results")
def api_results():
    with scan_lock:
        return jsonify({"running": scan_state["running"],
                        "results": scan_state["results"],
                        "error":   scan_state["error"]})


if __name__ == "__main__":
    print("=" * 55)
    print("  Nifty 500 SuperTrend + Multi-Filter Screener")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, threaded=True)
