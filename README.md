#  TP1: Introduction to Flower Framework – Federated Learning Project

This repository contains my implementation of a Federated Learning (FL) system using the framework as part of the TP1 assignment. The project explores client-server FL simulation with the Fashion MNIST dataset, model training on distributed data, and custom FL strategy implementation.

##  Project Overview

In this TP, I explored the complete pipeline of building a federated learning system from scratch using Flower. Below are the key steps and what I accomplished in each:

### Project steps

Step 1: Data Generation & Loading
data_utils.py: Generates distributed FashionMNIST datasets using Dirichlet distribution (saved in client_data) and loads client-specific DataLoader objects.
    data_utils.py
        Splits FashionMNIST into client datasets using the Dirichlet distribution (alpha).
        Loads client-specific data for training and testing.

Step 2: Model Implementation

    model.py
        Defines CustomFashionModel, a neural network for FashionMNIST classification.
        Includes methods to get and set model parameters for federated learning.

Step 3: Federated Client

    run_client.py
        Creates FlowerClient to train and evaluate the model on local data.
        Handles parameter exchange with the server for federated updates.

Step 4: Running a Client

    run_client.py
        Launches a client with a given ID to connect to the server.
        Starts the client to participate in federated learning rounds.

Step 5: Server Components

    server.py
        Sets up the server with FedAvgStrategy and CustomClientManager.
        Provides the server with tools to manage clients and aggregation.
    strategy.py
        Implements CustomClientManager for client registration and sampling.
        Defines FedAvgStrategy to aggregate client updates using federated averaging.
    start_server.py
        Starts the server with specified rounds and alpha value.
        Saves training metrics (accuracy, loss) to a JSON file.

Step 7: Result Analysis

    analyze_results.py
        Loads results and displays a table of accuracy/loss per round.
        Creates plots to visualize training progress over rounds.
    plot_metrics.py
        Plots accuracy and loss from JSON results as line graphs.
        Saves the graphs as images (e.g., alpha_1.0.png).
    save_metrics.py
        Saves training history (accuracy, loss) to a JSON file.
        Ensures results are stored for later analysis.

Step 8: Running the Simulation

    run_simulation.py
        Generates client data and starts the server for a full simulation.
        Coordinates federated learning with specified settings (clients, rounds).
    launch_all.sh
        Runs multiple experiments for different alpha values automatically.
        Starts server, clients, and saves results with plots.
    launch_experiments.sh
        Runs experiments for different alpha values with fewer logs.
        Focuses on saving results and plots for quick comparison.

Additional Files

    logger.py
        Tracks training progress (accuracy, loss) per round.
        Saves data in JSON or CSV format for analysis.
    fl_results.json, alpha_0.1.json, alpha_1.0.json, alpha_10.0.json, fedavg_alpha_*.json, results_a0.1_r10_c5.json
        Store training results like accuracy and loss per round.
        Allow comparison of performance across different settings.
    README.md
        Documents the project and provides an overview of its structure.
        Guides users on the purpose and functionality of each component.
