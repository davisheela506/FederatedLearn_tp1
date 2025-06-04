#  TP1: Introduction to Flower Framework – Federated Learning Project


In this TP, I explored the complete pipeline of building a federated learning system from scratch using Flower. Below are the key steps and what I explored client-server FL simulation with the Fashion MNIST dataset, model training on distributed data, and custom FL strategy implementation.

### Project steps

## Step 1: Data Generation & Loading

This part is handled by the data_utils.py file. It takes the FashionMNIST dataset—a collection of clothing images—and splits it into smaller chunks for each client, like dividing a big photo album among friends. It uses a math rule called the Dirichlet distribution, with a setting called alpha, to decide how evenly or unevenly the types of clothes are shared among clients. The split data is saved in a folder called distributed_data. The same file also helps load a client’s specific chunk of data, making sure it’s ready for training and testing by organizing the images properly.

## Step 2: Model Implementation

The model.py file creates the model we use to identify clothes in the FashionMNIST images. It builds a simple brain-like system called CustomFashionModel that learns to recognize things like shirts or shoes. The model flattens the images into a long list of numbers, processes them through two layers, and makes a guess about the clothing type. It has a train_epoch function that teaches the model by showing it images, tracking how well it’s learning, and adjusting its guesses. The test_epoch function checks how good the model is by showing it new images it hasn’t seen before, then reports how many it got right.

## Step 3: Federated Client

The run_client.py file takes care of this step. It sets up each client—like a person with their own photo collection—to teach the model using their images. The fit function does the teaching for a set number of rounds, called epochs, and keeps track of how well the model learns. After finishing, it shares the learning progress. The evaluate function tests the model on the client’s unseen images and shares how well it did.

## Step 4: Running a Client

To get a client working, we use run_client.py. You start a client by typing python3 run_client.py --cid NUMBER in the terminal, where NUMBER is the client’s ID, like 1 or 2. The server needs to be running first, or the client won’t work. This file lets you pick a client ID with --cid and connects to the server at localhost:8080. It grabs the client’s data using data_utils.py, sets up the model and client, and gets the client started so it can join the team effort.

## Step 5: Server Components

The server has two main parts, handled by strategy.py. First, CustomClientManager keeps track of all the clients, like a team leader with a list of team members. It adds new clients to the list with a register function, removes them with unregister, waits for enough clients to join using wait_for, and picks a group of clients to work each round with sample. Second, FedAvgStrategy is the plan for combining everyone’s work. It decides how many clients to use with configure_fit and configure_evaluate, mixes their training updates with aggregate_fit by averaging them, and does the same for test results with aggregate_evaluate.

## Step 6: Running the Server

The server runs using start_server.py. You start it by typing python3 start_server.py in the terminal. This script sets up the server address, pulls in settings like the number of rounds and alpha value, and uses the team leader and plan from strategy.py. It then gets the server going, collects the results of the team’s work, and saves them in a file like alpha_1.0.json to look at later.

## Step 7: Result Analysis

To see how well the model learned, we use analyze_results.py and plot_metrics.py. You run it with python3 analyze_results.py, and it makes pictures and tables of the results. These pictures, like alpha_1.0.png, are saved in a plots folder and show how the model improved over time, with titles that include the experiment’s details. The save_metrics.py file makes sure all the learning progress, like how many images the model got right, is saved in JSON files for us to check later.

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
