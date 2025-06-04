#  TP1: Introduction to Flower Framework – Federated Learning Project


In this TP, I explored the complete pipeline of building a federated learning system from scratch using Flower. Below are the key steps and what I explored client-server FL simulation with the Fashion MNIST dataset, model training on distributed data, and custom FL strategy implementation.

### Project steps

Step 1: Data Generation & Loading

Data generation is handled by data_utils.py. This script splits the FashionMNIST dataset into client-specific subsets using the Dirichlet distribution, controlled by the alpha parameter, to simulate varying levels of data heterogeneity. It saves the generated datasets to the distributed_data folder. The data_utils.py file also includes a function to load client data, which normalizes, reshapes, and splits the data into training and testing sets, transforming them into tensors and preparing DataLoader objects for model training.
## Step 2: Model Implementation

The model is implemented in model.py. This file defines CustomFashionModel, a neural network with a flatten layer followed by two linear layers (784 → 128 → 10) and a ReLU activation, designed for FashionMNIST classification. It includes a train_epoch function that trains the model for one epoch, initializing metrics to zero, computing loss, and aggregating metrics across the dataset. The test_epoch function evaluates the model on unseen data, initializing metrics, predicting on the test set, and returning the aggregated metrics.
## Step 3: Federated Client

This step is managed by run_client.py. The fit function in this file calls the model’s training method for the number of epochs specified in the configuration, retrieves metrics from train_epoch, and returns them after training. The evaluate function calls test_epoch to assess the model on the client’s test data and returns the evaluation metrics.
## Step 4: Running a Client

Client execution is handled by run_client.py. The command python3 run_client.py --cid INTEGER runs a client, which requires the server to be active; otherwise, it will fail. The script supports a --cid argument to set the client ID and connects to the server at localhost:8080. It calls the load_client_data function from data_utils.py to load the client’s data, creates the model and client instance, and starts the client to participate in federated learning.
## Step 5: Server Components

The client manager is implemented in strategy.py as CustomClientManager, and the strategy is defined as FedAvgStrategy in the same file. The client manager maintains a dictionary of registered clients by their IDs, with the register function adding clients and the unregister function removing them. The wait_for function ensures enough clients are available, and the sample function selects a subset of clients for each round. The configure_fit and configure_evaluate functions in FedAvgStrategy determine the number of clients to sample based on the minimum required or a fraction of connected clients. The aggregate_fit function combines training weights from clients using a weighted average, while aggregate_evaluate does the same for evaluation metrics.
## Step 6: Running the Server

The server is managed by start_server.py. The command python3 start_server.py launches the server. This script sets the server address, imports the number of rounds and alpha value from the configuration, and uses CustomClientManager and FedAvgStrategy from strategy.py. It then configures and runs the server, capturing the training history and saving it to a JSON file (e.g., alpha_1.0.json).
## Step 7: Result Analysis

Result visualization is handled by analyze_results.py and plot_metrics.py. The command python3 analyze_results.py generates visualizations and tables. Results are saved to the plots folder as images (e.g., alpha_1.0.png), with plot titles reflecting the experiment’s parameters. The save_metrics.py file ensures the training history (accuracy, loss) is saved to JSON files for further analysis.

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
        Saves data in JSON or CSV format for analysis.
fl_results.json, alpha_0.1.json, alpha_1.0.json, alpha_10.0.json, fedavg_alpha_*.json, results_a0.1_r10_c5.json<br>
        Store training results like accuracy and loss per round.<br>
        Allow comparison of performance across different settings.
