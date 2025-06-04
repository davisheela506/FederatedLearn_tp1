#!/bin/bash

# CONFIG
CLIENTS=10
ROUNDS=10
BATCH_SIZE=64
EPOCHS=3
ALPHAS=(0.1 1.0 10.0)

# Create output directories
mkdir -p results_logs plots results_json

for ALPHA in "${ALPHAS[@]}"; do
    echo "📦 Generating data for alpha=$ALPHA..."
    python run_simulation.py --clients $CLIENTS --alpha $ALPHA --rounds $ROUNDS --batch_size $BATCH_SIZE --epochs $EPOCHS --only-data

    echo "🚀 Starting server for alpha=$ALPHA..."
    python start_server.py --alpha $ALPHA --rounds $ROUNDS \
        > results_logs/server_alpha_${ALPHA}.log &

    SERVER_PID=$!
    sleep 5  # Wait for the server to initialize

    echo "👥 Launching $CLIENTS clients..."
    CLIENT_PIDS=()
    for ((i=0; i<CLIENTS; i++)); do
        python run_client.py $i > results_logs/client_${i}_alpha_${ALPHA}.log &
        CLIENT_PIDS+=($!)
    done

    echo "🕒 Waiting for server (PID $SERVER_PID) to finish..."
    wait $SERVER_PID

    echo "✅ Training done for alpha=$ALPHA!"

    echo "🧼 Killing any remaining clients..."
    for PID in "${CLIENT_PIDS[@]}"; do
        kill $PID 2>/dev/null
    done

    sleep 3  # Cool down

    echo "💾 Saving and plotting metrics for alpha=$ALPHA..."
       python save_metrics.py --alpha $ALPHA --output "results_json/alpha_${ALPHA}.json"
    python plot_metrics.py "results_json/alpha_${ALPHA}.json" "plots/alpha_${ALPHA}.png"


done

echo "🎉 All experiments completed!"
