import flwr as fl
from flwr.common import (
    Code, Status, GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays
)
import torch
from model import CustomFashionModel


class CustomClient(fl.client.Client):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={}
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )

    def fit(self, ins: FitIns) -> FitRes:
        # Update local model with global parameters
        self.model.set_model_parameters(parameters_to_ndarrays(ins.parameters))

        # Train for one epoch and get loss/accuracy
        self.model.train()
        correct = 0
        total = 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total if total > 0 else 0.0

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(self.model.get_model_parameters()),
            num_examples=len(self.train_loader.dataset),
            metrics={"accuracy": acc}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self.model.set_model_parameters(parameters_to_ndarrays(ins.parameters))
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_sum += loss.item() * labels.size(0)

        acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / total if total > 0 else 0.0

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=avg_loss,
            num_examples=total,
            metrics={"accuracy": acc}
        )

    def to_client(self):
        # Optional but useful for compatibility with new Flower versions
        return self
