import torch.nn as nn
import torch

class CustomFashionModel(nn.Module):
    def __init__(self):
        super(CustomFashionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

    def get_model_parameters(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_model_parameters(self, parameters):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)
