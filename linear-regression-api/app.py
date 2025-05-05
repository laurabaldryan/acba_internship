from flask import Flask, request, jsonify
from io import StringIO
import pandas as pd
from linear_regression import LinearRegressionScratch

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "csv" not in data:
        return jsonify({"error": "No csv"}), 400

    try:
        df = pd.read_csv(StringIO(data["csv"]))
    except Exception as e:
        return jsonify({"error": f"CSV parsing error: {str(e)}"}), 400

    try:
        y = df['price'].values
        X = df.drop(columns=['price'])
        X = X.select_dtypes(include='number').values
    except Exception as e:
        return jsonify({"error": f"Missing 'price' column or invalid format: {str(e)}"}), 400

    model = LinearRegressionScratch()
    model.fit(X, y)
    predictions = model.predict(X)

    return jsonify({
        "mse": model.mean_squared_error(y, predictions),
        "r2": model.r2_score(y, predictions),
        # printing first 10 predictions
        "predictions": predictions.tolist()[:10]
    })

if __name__ == '__main__':
    app.run(debug=True)
