import torch
from GTZAN.standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# Create MLP model
class ScoreEstimatorModel(torch.nn.Module):
    # old: [128,64,32,16]
    def __init__(self, input_size=32, output_size=1, hidden_sizes=[32, 16]):
        super(ScoreEstimatorModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.linear1 = torch.nn.Linear(input_size, hidden_sizes[0] if hidden_sizes else output_size)
        for i in range(1, len(hidden_sizes) + 1 if len(hidden_sizes) > 0 else 1):
            setattr(self, f'relu{i}', torch.nn.GELU())
            setattr(self, f'batchnorm{i}', torch.nn.BatchNorm1d(hidden_sizes[i-1]) if i < len(hidden_sizes) else None)
            setattr(self, f'dropout{i}', torch.nn.Dropout(0.2) if i < len(hidden_sizes) else None)
            setattr(self, f'linear{i+1}', torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i] if i < len(hidden_sizes) else output_size))

    def forward(self, x):
        x = self.linear1(x)
        for i in range(1, len(self.hidden_sizes) + 1 if len(self.hidden_sizes) > 0 else 1):
            relu = getattr(self, f'relu{i}', None)
            batchnorm = getattr(self, f'batchnorm{i}', None)
            dropout = getattr(self, f'dropout{i}', None)
            linear = getattr(self, f'linear{i+1}', None)
            # if relu:
            #     x = relu(x)
            # if batchnorm:
            #     x = batchnorm(x)
            if dropout:
                x = dropout(x)
            if linear:
                x = linear(x)
        return x

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)  # For reproducibility
    # Create an instance of the dataset
    dataset = StandardizedFeaturesToScoreDataset(
        features_csv="GTZAN/bands_dev_global_features_nyquist.csv",
        scores_csv="GTZAN/original_song_scores.csv",
        outliers_txt="GTZAN/outliers.txt",
        ignore_columns=[0,1],  # Ignore filename, label
        normalize=True,
        feature_indeces=None,
        shuffle=True,
        shuffle_seed=4,
        pca_components=None,
    )
    # dataset.analytical_normalization(feature_index=dataset.feature_columns.index("TotalRMS"))
    
    dataset_mean_score = dataset.get_mean_score()
    ten_highest = sorted([float(dataset.score_dict[f]) for f in dataset.filenames], reverse=True)[:10]

    split_idx = int(0.8 * len(dataset))
    train_indices = list(range(0, split_idx))
    # remove ten highest from training set
    # train_indices = [i for i in train_indices if float(dataset.score_dict[dataset.filenames[i]]) not in ten_highest]
    val_indices = list(range(split_idx, len(dataset)))
    # union ten highest with validation set
    # val_inscore_dices += [i for i in range(0, split_idx) if float(dataset.score_dict[dataset.filenames[i]]) in ten_highest]

    train_dataset = Subset(dataset, train_indices)  # First half for training
    val_dataset = Subset(dataset, val_indices)      # Second half for validation
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train the model
    input_size = len(dataset.feature_columns)
    print(f"Input feature size: {input_size}")
    hidden_sizes = [32,16]  # Example hidden layer sizes
    output_size = 1  # Single score output
    model = ScoreEstimatorModel(input_size, output_size, hidden_sizes).to(device)
    model.train()
    print(f"Model: {model}")
    criterion = torch.nn.SmoothL1Loss(reduction="none")  # Smooth L1 loss for regression
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.04)  # AdamW optimizer with weight decay
    num_epochs = 1000
    loss = None

    best_loss = float('inf')
    best_pearson = 0.0
    patience = 50
    epochs_no_improve = 0

    # Initialize lists to store losses for plotting
    train_losses = []
    val_losses = []
    fig, ax = plt.subplots(2, 1, figsize=(7, 15))
    loss_line_train, = ax[0].plot([], [], label='Train Loss')
    loss_line_val, = ax[0].plot([], [], label='Val Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    best_state = None

    def evaluate_model(model_chkpt):
        model_chkpt.eval()
        val_loss = 0.0
        r_value = 0.0
        with torch.no_grad():

            train_features = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).to(device) # type: ignore
            train_scores = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))]).to(device) # type: ignore
            train_outputs = model_chkpt(train_features)
            train_outputs = torch.sigmoid(train_outputs)
            
            val_features = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))]).to(device) # type: ignore
            val_scores = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))]).to(device) # type: ignore
            val_outputs = model_chkpt(val_features)
            val_outputs = torch.sigmoid(val_outputs)
            val_loss = criterion(val_outputs.squeeze(), val_scores.squeeze()).mean().item()

            r_value = torch.corrcoef(torch.stack([val_outputs.squeeze(), val_scores.squeeze()]))[0, 1].item()
            from scipy.stats import spearmanr
            spearman_rho, _ = spearmanr(val_outputs.squeeze().cpu().numpy(), val_scores.squeeze().cpu().numpy())
            ax[1].cla()
            ax[1].set_title(f"Validation Set: Pearson r={r_value:.4f} Spearman rho={spearman_rho:.4f}", fontsize=14)
            ax[1].scatter(train_scores.cpu().numpy(), train_outputs.cpu().squeeze().numpy(), alpha=0.6, color='orange', label='Train Set')
            ax[1].scatter(val_scores.cpu().numpy(), val_outputs.cpu().squeeze().numpy(), alpha=0.6, color='blue', label='Validation Set')
            ax[1].set_xlabel("Actual Scores")
            ax[1].set_ylabel("Predicted Scores")
            ax[1].plot([0, 1], [0, 1], 'r--', label='y=x')
            ax[1].legend()
            ax[1].grid(True)
            
        avg_val_loss = 1400 * val_loss / len(val_dataset)

        print(f"\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}", end='')
            
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        loss_line_train.set_data(range(1, len(train_losses)+1), train_losses)
        loss_line_val.set_data(range(1, len(val_losses)+1), val_losses)
        ax[0].set_xlim(1, len(train_losses))
        ax[0].set_ylim(0, max(max(train_losses), max(val_losses)))
        ax[0].relim()
        ax[0].autoscale_view()
        plt.draw()
        plt.pause(0.01)
        return avg_val_loss, r_value

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for features, scores, filenames in train_dataloader:
            features, scores = features.to(device), scores.to(device)
            z = torch.logit(scores, eps=1e-4)  # Convert scores to logits
            outputs = model(features)
            loss = criterion(outputs.squeeze(), z.squeeze())
            # Compute Pearson correlation loss
            # outputs_flat = outputs.squeeze()
            # scores_flat = scores.squeeze()
            # mean_outputs = outputs_flat.mean()
            # mean_scores = scores_flat.mean()
            # cov = ((outputs_flat - mean_outputs) * (scores_flat - mean_scores)).mean()
            # std_outputs = outputs_flat.std()
            # std_scores = scores_flat.std()
            # pearson_corr = cov / (std_outputs * std_scores + 1e-8)
            # pearson_loss = 1 - pearson_corr  # maximize correlation

            w = 1 # Weighting based on distance from mean score (prioritize outliers) 
            # w += 1 * torch.abs(scores - dataset_mean_score)**3
            total_loss = (loss * w).mean()  # Weighted loss
            r_value = torch.corrcoef(torch.stack([outputs.squeeze(), z.squeeze()]))[0, 1].item()
            # total_loss += 10 * (1 - r_value)  # Add Pearson correlation loss
            ax[0].set_title(f"Epoch {epoch+1}/{num_epochs} - Train Set: Pearson r={r_value:.4f}", fontsize=14)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item() * features.size(0)
        
        avg_train_loss = epoch_loss / len(train_dataset)

        # Validation loss calculation
        _, r_value = evaluate_model(model)

        # Early stopping based on validation loss
        # if avg_val_loss < best_loss:
        #     best_loss = avg_val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1

        # Early stopping based on Pearson correlation
        if r_value > best_pearson:
            best_pearson = r_value
            epochs_no_improve = 0
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            # print(f"New best model found with Pearson r: {best_pearson:.4f}")
            # best = 0.6348282098770142
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
    print("Training complete. Best Pearson correlation:", best_pearson)
    # load best model for final evaluation
    if best_state is not None:
        d = {key: value.to(device) for key, value in best_state.items()}
        # torch.save(d, "GTZAN/models/score_estimator_bands_dev_nyquist.pth")
        model.load_state_dict(d)
    else:
        print("No best model checkpoint found. Using the current model state.")
    evaluate_model(model)
    plt.show()
    