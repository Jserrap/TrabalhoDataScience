from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Global variables to store the model and scaler
model = None
scaler = None

# Load and preprocess the data
def prepare_data():
    global model, scaler

    # Load the dataset
    df = pd.read_csv('mysite/imoveis_joao_pessoa.csv')

    # Separate features and target
    X = df.drop('valor', axis=1)
    y = df['valor']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

# Initialize the model and preprocessors
prepare_data()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        input_data = pd.DataFrame({
            'quartos': [int(request.form['quartos'])],
            'banheiros': [int(request.form['banheiros'])],
            'metros_quadrados': [float(request.form['metros_quadrados'])]
        })

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        predicted_price = model.predict(input_scaled)[0]

        # Prepare context for template
        context = {
            "predicted_price": f"R$ {predicted_price:,.2f}",
            "input_data": {
                "Quartos": request.form['quartos'],
                "Banheiros": request.form['banheiros'],
                "Metros Quadrados": f"{float(request.form['metros_quadrados']):.1f}"
            },
            "error": None
        }

        return render_template("result.html", **context)

    except Exception as e:
        context = {
            "error": f"Erro na previsão: {str(e)}"
        }
        return render_template("result.html", **context)

if __name__ == '__main__':
    app.run(debug=True)