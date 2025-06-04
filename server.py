from strategy import FedAvgStrategy, CustomClientManager

def get_strategy():
    return FedAvgStrategy()

def get_client_manager():
    return CustomClientManager()
