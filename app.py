"""
Flask server for the movie recommendation app.

Nothing fancy here: one route serves the HTML page, another handles
recommendation requests. The heavy lifting happens in recommender.py.
"""

import os
from flask import Flask, request, jsonify, render_template
from recommender import RecommendationEngine

app = Flask(__name__)

# figure out where the CSV is sitting
# in Docker it'll be in the same directory, locally it might be elsewhere
CSV_PATH = os.environ.get("CSV_PATH", "netflix_data.csv")

# spin up the recommendation engine once when the server starts
# yeah this blocks startup for ~30s while it builds embeddings, but
# its a one-time cost and way simpler than async loading for this use case
engine = RecommendationEngine(CSV_PATH)
engine.initialize()


@app.route("/")
def home():
    """Serve the main (and only) page."""
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Handle recommendation requests. Expects JSON with a 'query' field
    and an optional 'top_k' for how many results to return.
    """
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "need a 'query' field in the request body"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "query cant be empty"}), 400

    top_k = data.get("top_k", 10)

    # clamp top_k to something reasonable
    top_k = max(1, min(top_k, 50))

    try:
        results = engine.recommend(query, top_k=top_k)
        return jsonify({
            "query": query,
            "count": len(results),
            "results": results,
        })
    except Exception as e:
        print(f"something went wrong with query '{query}': {e}")
        return jsonify({"error": "recommendation failed, check server logs"}), 500


@app.route("/health")
def health():
    """Quick health check endpoint, useful for Docker."""
    return jsonify({"status": "ok", "titles_loaded": len(engine.df) if engine.df is not None else 0})


if __name__ == "__main__":
    # for local dev, run on port 5000
    # in Docker we use gunicorn on port 80 instead
    app.run(host="0.0.0.0", port=5001, debug=True)
