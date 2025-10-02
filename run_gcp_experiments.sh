#!/bin/bash
# Script to run a batch of PPO training experiments on a remote server.

echo "--- Starting Experiment Batch ---"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="gcp_batch_run_${TIMESTAMP}.log"

echo "Logging output to ${LOG_FILE}"

# Start MLflow in the background if it's not already running
if ! pgrep -f "mlflow ui" > /dev/null
then
    echo "Starting MLflow UI in the background..."
    nohup mlflow ui --host 0.0.0.0 &> mlflow_ui.log &
    sleep 5 # Give it a moment to start
fi

# Run experiments and redirect output to a single log file
{
    echo "--- Experiment 1: Baseline ---"
    python src/main_train.py

    echo "--- Experiment 2: Faster Learning Rate ---"
    python src/main_train.py agent.lr=0.001

    echo "--- Experiment 3: Lower Drawdown Penalty ---"
    python src/main_train.py environment.drawdown_penalty=0.05

    echo "--- Experiment 4: More PPO Epochs ---"
    python src/main_train.py agent.K_epochs=8

} &>> "${LOG_FILE}"

echo "--- All experiments launched in the background. ---"
echo "Monitor progress with: tail -f ${LOG_FILE}"
echo "Access MLflow UI at: http://34.131.15.81:5000"