import json
import csv
import os

class TrainingLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data = {
            "rounds": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "test_loss": []
        }

    def log_round(self, rnd, train_acc, test_acc, test_loss):
        self.data["rounds"].append(rnd)
        self.data["train_accuracy"].append(train_acc)
        self.data["test_accuracy"].append(test_acc)
        self.data["test_loss"].append(test_loss)

    def save_to_json(self):
        with open(self.filename + ".json", "w") as f:
            json.dump(self.data, f, indent=2)

    def save_to_csv(self):
        with open(self.filename + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "train_accuracy", "test_accuracy", "test_loss"])
            for i in range(len(self.data["rounds"])):
                writer.writerow([
                    self.data["rounds"][i],
                    self.data["train_accuracy"][i],
                    self.data["test_accuracy"][i],
                    self.data["test_loss"][i]
                ])
