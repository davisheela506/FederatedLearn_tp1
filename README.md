#  TP1: Introduction to Flower Framework – Federated Learning Project


In this TP, I explored the complete pipeline of building a federated learning system from scratch using Flower. Below are the key steps and what I explored client-server FL simulation with the Fashion MNIST dataset, model training on distributed data, and custom FL strategy implementation.

### Project steps

## Step 1: Data Generation & Loading

data_utils.py: Splits FashionMNIST into client datasets using the Dirichlet distribution (alpha) and loads client-specific data for training and testing.<br>

## Step 2: Model Implementation

model.py: Defines CustomFashionModel, a neural network for FashionMNIST classification and includes methods to get and set model parameters for federated learning.

## Step 3: Federated Client

run_client.py: Creates FlowerClient to train and evaluate the model on local data and handles parameter exchange with the server for federated updates.

## Step 4: Running a Client

run_client.py: Launches a client with a given ID to connect to the server and starts the client to participate in federated learning rounds.

## Step 5: Server Components

server.py: Sets up the server with FedAvgStrategy and CustomClientManager and provides the server with tools to manage clients and aggregation.<br>
strategy.py: Implements CustomClientManager for client registration and sampling and defines FedAvgStrategy to aggregate client updates using federated averaging.<br>
start_server.py: Starts the server with specified rounds and alpha value and saves training metrics (accuracy, loss) to a JSON file.

## Step 7: Result Analysis

analyze_results.py
        Loads results and displays a table of accuracy/loss per round. <br>
        Creates plots to visualize training progress over rounds.
plot_metrics.py
        Plots accuracy and loss from JSON results as line graphs.<br>
        Saves the graphs as images (e.g., alpha_1.0.png).
save_metrics.py
        Saves training history (accuracy, loss) to a JSON file.<br>
        Ensures results are stored for later analysis.

## Step 8: Running the Simulation

run_simulation.py
        Generates client data and starts the server for a full simulation.<br>
        Coordinates federated learning with specified settings (clients, rounds).
launch_all.sh
        Runs multiple experiments for different alpha values automatically.<br>
        Starts server, clients, and saves results with plots.
launch_experiments.sh
        Runs experiments for different alpha values with fewer logs.
        Focuses on saving results and plots for quick comparison.
### Hyperparameters

Initial: 10 clients, 30 rounds, 1 epoch, α=1.0, batch size=32, learning rate=0.01. 

## Additional Files

logger.py
        Tracks training progress (accuracy, loss) per round.<br>
        Saves data in JSON or CSV format for analysis. <br>
fl_results.json, alpha_0.1.json, alpha_1.0.json, alpha_10.0.json, fedavg_alpha_*.json, results_a0.1_r10_c5.json<br>
        Store training results like accuracy and loss per round.<br>
        Allow comparison of performance across different settings.
