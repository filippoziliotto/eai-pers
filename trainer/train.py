import torch
from torch.utils.data import DataLoader
import wandb  # Import W&B

# Other imports
from utils.losses import compute_loss

# Define the training function
def train(model, data_loader, optimizer, scheduler, num_epochs, loss_choice='L2', device='cpu'):
    """
    Trains a model with the provided parameters.

    Args:
        model: The PyTorch model to train.
        data_loader: DataLoader providing (ground_truth_coords, input_data) batches.
        optimizer: Optimizer for updating model weights.
        num_epochs: Number of epochs to train the model.
        loss_choice: Loss function choice ('L1' or 'L2').
        device: Device to run the training on ('cpu' or 'cuda').
    """
    
    # If model not in device yet
    if device == 'cuda' and not next(model.parameters()).is_cuda:
        model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (ground_truth_coords, input_data) in enumerate(data_loader):
            ground_truth_coords = ground_truth_coords.to(device)
            input_data = input_data.to(device)

            # Forward pass: Get predictions
            predicted_coords = model(input_data)

            # Compute loss
            loss = compute_loss(ground_truth_coords, predicted_coords, loss_choice)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for the epoch
            epoch_loss += loss.item()

        # Log epoch loss to W&B
        epoch_avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_avg_loss}")

    print("Training complete.")

# Example usage (you'll need to adapt this with your dataset and model)
if __name__ == '__main__':

    batch_size, num_epochs = 2, 10
    optimizer = torch.optim.Adam

    # Dataset and DataLoader
    dataset = ExampleDataset()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = ExampleModel()

    # Train the model
    train(model, data_loader, optimizer, num_epochs, loss_choice='L2', device='cpu')
