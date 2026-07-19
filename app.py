from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and column names
model = pickle.load(open("models/car_price_model.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    present_price = float(request.form["present_price"])
    kms_driven = int(request.form["kms_driven"])
    car_age = int(request.form["car_age"])
    owner = int(request.form["owner"])

    fuel_type = request.form["fuel_type"]
    seller_type = request.form["seller_type"]
    transmission = request.form["transmission"]

    # Create input dataframe
    input_data = pd.DataFrame([[0] * len(columns)], columns=columns)

    input_data["Present_Price"] = present_price
    input_data["Kms_Driven"] = kms_driven
    input_data["Owner"] = owner
    input_data["Car_Age"] = car_age

    # Set categorical values
    if "Fuel_Type_Petrol" in input_data.columns:
        input_data["Fuel_Type_Petrol"] = 1 if fuel_type == "Petrol" else 0

    if "Seller_Type_Individual" in input_data.columns:
        input_data["Seller_Type_Individual"] = 1 if seller_type == "Individual" else 0

    if "Transmission_Manual" in input_data.columns:
        input_data["Transmission_Manual"] = 1 if transmission == "Manual" else 0

    # Predict
    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=f"Predicted Selling Price: ₹ {prediction:.2f} Lakhs"
    )


if __name__ == "__main__":
    app.run(debug=True)