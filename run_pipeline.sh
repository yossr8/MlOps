#!/bin/bash

# Function to check if a port is in use
is_port_in_use() {
  if command -v nc &> /dev/null; then
    nc -z localhost $1 &> /dev/null
    return $?
  elif command -v netstat &> /dev/null; then
    netstat -an | grep LISTEN | grep -q ":$1 "
    return $?
  else
    # Default to assuming port is not in use if we can't check
    return 1
  fi
}

# Create necessary directories
mkdir -p outputs/mlruns

# Check if port 5000 is already in use
if is_port_in_use 5000; then
  echo "⚠️ Port 5000 is already in use. MLFlow server might already be running."
  echo "If the existing server isn't a MLFlow server, please stop it and try again."
  echo "Continuing with the assumption that MLFlow server is running..."
else
  # Start MLFlow server in the background
  echo "Starting MLFlow server..."
  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./outputs/mlruns --host 0.0.0.0 --port 5000 &
  MLFLOW_PID=$!
  
  echo "MLFlow server started with PID: $MLFLOW_PID"
  
  # Wait for MLFlow server to start
  echo "Waiting for MLFlow server to start..."
  for i in {1..10}; do
    if is_port_in_use 5000; then
      echo "MLFlow server is running."
      break
    fi
    if [ $i -eq 10 ]; then
      echo "⚠️ Timeout waiting for MLFlow server to start."
      echo "Continuing anyway, but the pipeline might fail if it can't connect to MLFlow."
    fi
    sleep 1
  done
fi

echo "MLFlow UI is available at http://localhost:5000"
echo ""

# Run DVC pipeline
echo "Running DVC pipeline..."
dvc repro

# Check if DVC pipeline was successful
if [ $? -eq 0 ]; then
  # Print results from the latest output directory
  echo ""
  echo "Pipeline completed successfully. Results:"
  LATEST_OUTPUT=$(ls -td outputs/$(date +%Y-%m-%d)/* | head -1)
  if [ -f "$LATEST_OUTPUT/metrics/metrics.json" ]; then
    cat "$LATEST_OUTPUT/metrics/metrics.json"
  else
    echo "Metrics file not found in $LATEST_OUTPUT"
    dvc metrics show
  fi
else
  echo ""
  echo "⚠️ Pipeline encountered errors. Check the logs above for details."
fi

echo ""
echo "To view experiments in MLFlow UI, visit http://localhost:5000"

# If we started the MLFlow server, keep it running and provide instructions
if [ -n "$MLFLOW_PID" ]; then
  echo "Press Ctrl+C to stop the MLFlow server when done."
  
  # Keep the script running to maintain the MLFlow server
  wait $MLFLOW_PID
else
  echo "MLFlow server was already running. You may need to stop it manually when done."
fi