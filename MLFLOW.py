import mlflow

# Set the tracking URI (just the server, not the UI path)
mlflow.set_tracking_uri("http://localhost:5000")

# Specify the model name and version
model_name = "random_forest_model"
model_version = 1

# Download the model
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")