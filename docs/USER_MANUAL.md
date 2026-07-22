# User Manual

## Dashboard Navigation
Access the Streamlit Dashboard at `http://localhost:8501`.
- **Telemetry Stream**: View live incoming motor data.
- **Prediction**: View current health status (Healthy vs Anomalous).
- **Explainability**: SHAP waterfall plots describe *why* the model made its decision.

## API Usage
Access Swagger UI at `http://localhost:8000/docs`.
Use the `/predict` endpoint to manually test telemetry values.