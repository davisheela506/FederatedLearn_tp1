import flwr as fl
import torch
from model import CustomFashionModel
from data_utils import load_client_data

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_dir, batch_size):
        self.cid = cid
        self.model = CustomFashionModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loader, self.val_loader = load_client_data(cid, data_dir, batch_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return self.model.get_model_parameters()

    def set_parameters(self, parameters):
        self.model.set_model_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(config.get("epochs", 1)):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.model.get_model_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        return float(loss / total), total, {"accuracy": accuracy}

def main():
    import sys
    cid = int(sys.argv[1])
    client = FlowerClient(cid, data_dir="./distributed_data", batch_size=64)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
