import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.utils import calc_dp_sgd_privacy


class AdaptivePrivacyEngine:
    def __init__(self, model, optimizer, initial_noise_multiplier, final_noise_multiplier, epochs):
        self.model = model
        self.optimizer = optimizer
        self.initial_noise_multiplier = initial_noise_multiplier
        self.final_noise_multiplier = final_noise_multiplier
        self.epochs = epochs
        self.current_epoch = 0
        self.privacy_engine = PrivacyEngine(
            model,
            batch_size=64,
            sample_size=len(train_dataset),
            alphas=[10, 100],
            noise_multiplier=self.initial_noise_multiplier,
            max_grad_norm=1.0,
        )
        self.privacy_engine.attach(optimizer)

    def update_noise_multiplier(self):
        ratio = self.current_epoch / self.epochs
        new_noise_multiplier = (
            self.initial_noise_multiplier * (1 - ratio) +
            self.final_noise_multiplier * ratio
        )
        self.privacy_engine.detach()  # Detach the current PrivacyEngine
        self.privacy_engine = PrivacyEngine(
            self.model,
            batch_size=64,
            sample_size=len(train_dataset),
            alphas=[10, 100],
            noise_multiplier=new_noise_multiplier,
            max_grad_norm=1.0,
        )
        self.privacy_engine.attach(self.optimizer)

    def step(self):
        self.current_epoch += 1
        self.update_noise_multiplier()




adaptive_privacy_engine = AdaptivePrivacyEngine(
    model,
    optimizer,
    initial_noise_multiplier=1.0,
    final_noise_multiplier=0.1,
    epochs=num_epochs
)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data, target in train_loader:
        adaptive_privacy_engine.optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        adaptive_privacy_engine.optimizer.step()
        running_loss += loss.item()

    adaptive_privacy_engine.step()  # Update the noise multiplier

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

    epsilon, delta = calc_dp_sgd_privacy(
        noise_multiplier=adaptive_privacy_engine.privacy_engine.noise_multiplier,
        batch_size=64,
        sample_size=len(train_dataset),
        num_epochs=epoch + 1,
        alphas=[10, 100]
    )
    print(f'Privacy parameters: ε = {epsilon}, δ = {delta}')
