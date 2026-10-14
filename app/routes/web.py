from flask import render_template, request, jsonify

from app.extensions import csrf
from app.services import prediction_service


def register_web_routes(app):

    @app.route("/forecast")
    def forecast_page():
        skus = prediction_service.get_available_skus()
        return render_template("forecast.html", skus=skus)

    @app.route("/api/forecast")
    def forecast_api():
        sku_id = request.args.get("sku_id")
        horizon = int(request.args.get("horizon", 30))
        if not sku_id:
            return jsonify({"error": "sku_id is required"}), 400
        result = prediction_service.forecast_next_n(sku_id, horizon)
        return jsonify(result)

    @app.route("/simulate-disruption")
    def simulate_disruption_page():
        skus = prediction_service.get_available_skus()
        return render_template("forecast.html", skus=skus, simulate_mode=True)

    @app.route("/api/simulate-disruption", methods=["POST"])
    def simulate_disruption_api():
        payload = request.get_json(force=True)
        sku_id = payload.get("sku_id")
        disruption_type = payload.get("disruption_type", "demand_spike")
        start_offset = int(payload.get("start_offset", 5))
        duration = int(payload.get("duration", 14))
        magnitude = float(payload.get("magnitude", 0.5))
        horizon = int(payload.get("horizon", 30))

        if not sku_id:
            return jsonify({"error": "sku_id is required"}), 400

        result = prediction_service.simulate_disruption(
            sku_id, disruption_type, start_offset, duration, magnitude, horizon
        )
        return jsonify(result)

    csrf.exempt(simulate_disruption_api)

    @app.route("/performance")
    def performance_page():
        metrics = prediction_service.load_metrics()
        return render_template("performance.html", metrics=metrics)

    @app.route("/api/performance")
    def performance_api():
        metrics = prediction_service.load_metrics()
        return jsonify(metrics)

    @app.route("/causal-graph")
    def causal_graph_page():
        graph_data = prediction_service.get_causal_graph_json()
        return render_template("causal_graph.html", graph_data=graph_data)

    @app.route("/api/causal-graph")
    def causal_graph_api():
        graph_data = prediction_service.get_causal_graph_json()
        return jsonify(graph_data)