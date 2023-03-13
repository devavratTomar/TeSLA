import torch.nn as nn
import torch

class PolicyPredictor(nn.Module):
    def __init__(self, n_sub_policy, sub_policy_dim, name="default") -> None:
        super().__init__()
        self.name = name
        self.n_sub_policy = n_sub_policy
        self.sub_policy_dim = sub_policy_dim
        self.register_parameter("policy_selection_weights", torch.nn.parameter.Parameter(torch.zeros(n_sub_policy)))
        self.register_parameter("policy_mag_weights", torch.nn.parameter.Parameter(torch.zeros(n_sub_policy, sub_policy_dim)))

    def reset_weights(self):
        self.policy_selection_weights.data.zero_()
        self.policy_mag_weights.data = torch.zeros((self.n_sub_policy, self.sub_policy_dim), device=self.policy_mag_weights.data.device)

    def forward(self):
        probs = torch.softmax(self.policy_selection_weights, dim=0)
        if self.name == "Hard":
            mags  = torch.sigmoid(self.policy_mag_weights)
        else:
            mags = 0.5 + self.policy_mag_weights
            mags = torch.clamp(mags, 0, 1)
        
        return probs, mags
