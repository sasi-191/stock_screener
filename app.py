from flask import Flask, render_template, request, jsonify
from screener import run_screener, get_segments
import threading

app = Flask(__name__)

# Store scan progress in memory
scan_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_symbol": "",
    "results": [],
    "error": None
}
scan_lock = threading.Lock()


@app.route("/")
def index():
    segments = get_segments()
    return render_template("index.html", segments=segments)


@app.route("/api/segments")
def api_segments():
    return jsonify(get_segments())


@app.route("/api/scan", methods=["POST"])
def api_scan():
    global scan_state

    with scan_lock:
        if scan_state["running"]:
            return jsonify({"error": "Scan already running"}), 400

        scan_state = {
            "running": True,
            "progress": 0,
            "total": 0,
            "current_symbol": "",
            "results": [],
            "error": None
        }

    data = request.json or {}
    segment = data.get("segment", "All")
    timeframe = data.get("timeframe", "1W")
    signal_filter = data.get("signal_filter", "both")
    st_period = int(data.get("st_period", 7))
    st_multiplier = float(data.get("st_multiplier", 3))

    def progress_callback(done, total, symbol):
        with scan_lock:
            scan_state["progress"] = done
            scan_state["total"] = total
            scan_state["current_symbol"] = symbol

    def run_in_background():
        global scan_state
        try:
            results = run_screener(
                segment=segment,
                timeframe=timeframe,
                signal_filter=signal_filter,
                st_period=st_period,
                st_multiplier=st_multiplier,
                lookback=int(data.get("lookback", 3)),
                progress_callback=progress_callback
            )
            with scan_lock:
                scan_state["results"] = results
                scan_state["running"] = False
        except Exception as e:
            with scan_lock:
                scan_state["error"] = str(e)
                scan_state["running"] = False

    thread = threading.Thread(target=run_in_background, daemon=True)
    thread.start()

    return jsonify({"status": "started"})


@app.route("/api/scan/progress")
def api_scan_progress():
    with scan_lock:
        return jsonify({
            "running": scan_state["running"],
            "progress": scan_state["progress"],
            "total": scan_state["total"],
            "current_symbol": scan_state["current_symbol"],
            "results_count": len(scan_state["results"]),
            "error": scan_state["error"]
        })


@app.route("/api/scan/results")
def api_scan_results():
    with scan_lock:
        return jsonify({
            "running": scan_state["running"],
            "results": scan_state["results"],
            "error": scan_state["error"]
        })


if __name__ == "__main__":
    print("=" * 50)
    print("  Nifty 500 SuperTrend Screener")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, threaded=True)
