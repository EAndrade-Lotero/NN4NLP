import torch
from tqdm.auto import tqdm


class NN4NLPTrainer:
    '''
    Dummy class to aggregate helper functions
    '''

    @staticmethod
    def get_device(safe=False):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available() and not safe:
            device = 'mps'
        else:
            device = 'cpu'
        return device

    @staticmethod
    def basic_trainer(model, dataloader, criterion, optimizer, num_epochs=1000):
        """
        Train the model for the specified number of epochs.
        
        Args:
            model: The PyTorch model to be trained.
            dataloader: DataLoader providing data for training.
            criterion: Loss function.
            optimizer: Optimizer for updating model's weights.
            num_epochs: Number of epochs to train the model for.

        Returns:
            model: The trained model.
            epoch_losses: List of average losses for each epoch.
        """
        
        # List to store running loss for each epoch
        epoch_losses = []

        for epoch in tqdm(range(num_epochs)):
            # Storing running loss values for the current epoch
            running_loss = 0.0

            # Using tqdm for a progress bar
            for input, target in dataloader:
                optimizer.zero_grad()               
                predicted = model(input)                               
                loss = criterion(predicted, target)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                running_loss += loss.item()

            # Append average loss for the epoch
            epoch_losses.append(running_loss / len(dataloader))
        
        return model, epoch_losses
       

    @staticmethod
    def train_CBOW_model(model, dataloader, criterion, optimizer, num_epochs=1000):
        """
        Train the model for the specified number of epochs.
        
        Args:
            model: The PyTorch model to be trained.
            dataloader: DataLoader providing data for training.
            criterion: Loss function.
            optimizer: Optimizer for updating model's weights.
            num_epochs: Number of epochs to train the model for.

        Returns:
            model: The trained model.
            epoch_losses: List of average losses for each epoch.
        """
        
        # List to store running loss for each epoch
        epoch_losses = []

        for epoch in tqdm(range(num_epochs)):
            # Storing running loss values for the current epoch
            running_loss = 0.0

            # Using tqdm for a progress bar
            for idx, samples in enumerate(dataloader):

                optimizer.zero_grad()
                
                target, context, offsets = samples
                predicted = model(context, offsets)
                                
                loss = criterion(predicted, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                running_loss += loss.item()

            # Append average loss for the epoch
            epoch_losses.append(running_loss / len(dataloader))
        
        return model, epoch_losses

    @staticmethod
    def train_lm_model(model, dataloader, criterion, optimizer, num_epochs=1000):
        """
        Train the model for the specified number of epochs.
        
        Args:
            model: The PyTorch model to be trained.
            dataloader: DataLoader providing data for training.
            criterion: Loss function.
            optimizer: Optimizer for updating model's weights.
            num_epochs: Number of epochs to train the model for.

        Returns:
            model: The trained model.
            epoch_losses: List of average losses for each epoch.
        """
        
        # List to store running loss for each epoch
        epoch_losses = []

        for epoch in tqdm(range(num_epochs), desc="Training progress"):
            # Storing running loss values for the current epoch
            running_loss = 0.0

            # Using tqdm for a progress bar
            for context, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                optimizer.zero_grad()              
                predicted = model(context)
                loss = criterion(predicted, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                running_loss += loss.item()

            # Append average loss for the epoch
            epoch_losses.append(running_loss / len(dataloader))
        
        return model, epoch_losses


    @staticmethod
    def train_trsf(model, dataloader, optimizer, criterion, num_epochs, device):
        epoch_losses = []
        model.train()
        # Training Loop
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for target_tokens, input_tokens in progress_bar:
                input_tokens, target_tokens = input_tokens.to(device), target_tokens.to(device)

                optimizer.zero_grad()

                # Forward pass
                logits = model(input_tokens)  # Output shape: (batch_size, seq_len, vocab_size)
                
                # Reshape logits and targets for cross-entropy loss
                logits = logits[:, -1, :]

                loss = criterion(logits, target_tokens)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_losses.append(epoch_loss)
                progress_bar.set_postfix(loss=loss.item())

        return model, epoch_losses