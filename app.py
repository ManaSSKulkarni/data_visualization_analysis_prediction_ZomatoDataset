from flask import Flask, render_template, send_file
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dataset")
def dataset():
    return send_file(
        "zomato.csv",
        mimetype="text/csv",
        as_attachment=False
    )


@app.route("/analysis")
def analysis():

    import os
    import importlib

    # Delete old graphs
    [os.remove(f"static/graphs/{file}") for file in os.listdir("static/graphs")]

    import main
    import ratingsprediction

    importlib.reload(main)
    importlib.reload(ratingsprediction)

    graphs = sorted(
        os.listdir("static/graphs"),
        key=lambda x: int(x.replace("graph", "").replace(".png", ""))
    )

    return render_template(
        "displayanalysis.html",
        graphs=graphs,
        results=ratingsprediction.results,
        prediction=ratingsprediction.predicted_rating_value,
        logs=ratingsprediction.process_logs
    )


if __name__ == "__main__":
    app.run(debug=True)