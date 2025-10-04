#!/bin/bash
# Script to run a batch of PPO training experiments on a remote server.

echo "--- Starting Experiment Batch ---"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="gcp_batch_run_${TIMESTAMP}.log"

echo "Logging output to ${LOG_FILE}"

# Start MLflow in the background if it's not already running
if ! pgrep -f "mlflow server" > /dev/null
then
    echo "Starting MLflow Server in the background..."
    # THIS IS THE CRITICAL FIX: Use 'mlflow server' with the correct artifact root
    nohup mlflow server --host 0.0.0.0 --default-artifact-root ./mlflow_artifacts &> mlflow_server.log &
    sleep 5 # Give it a moment to start
fi

# Run experiments and redirect output to a single log file
{
    echo "--- Experiment 1: Baseline on Jan 2023 Data ---"
    python src/main_train.py environment.data_file="BTCUSDT-5m_2023-01-01_to_2023-01-31_features.csv"

    echo "--- Experiment 2: Faster Learning Rate ---"
    python src/main_train.py environment.data_file="BTCUSDT-5m_2023-01-01_to_2023-01-31_features.csv" agent.lr=0.001

    echo "--- Experiment 3: Lower Drawdown Penalty ---"
    python src/main_train.py environment.data_file="BTCUSDT-5m_2023-01-01_to_2023-01-31_features.csv" environment.drawdown_penalty=0.05

    echo "--- Experiment 4: More PPO Epochs ---"
    python src/main_train.py environment.data_file="BTCUSDT-5m_2023-01-01_to_2023-01-31_features.csv" agent.K_epochs=8

} &>> "${LOG_FILE}"

echo "--- All experiments launched in the background. ---"
echo "Monitor progress with: tail -f ${LOG_FILE}"
echo "Access MLflow UI at your VM's IP address, port 5000"