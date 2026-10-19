import json
import time
import numpy as np

import torch
import torchvision

LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)




def confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=y_true.device)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm

#stolen from assignment 1, modified for readability and extra features.
def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs, scheduler=None, print_every=1,
                early_stopping_patience=None, monitor="val_accuracy",
                checkpoint_path=None):
    """
    Train a model, optionally with a learning-rate schedule, early stopping and checkpointing.
    This is the PyTorch equivalent of Keras' model.fit() with its callbacks -- written out
    so that nothing is hidden.

    Arguments:
    model                   -- the nn.Module to train
    train_loader/val_loader -- DataLoaders
    criterion               -- loss function
    optimizer               -- optimizer
    epochs                  -- maximum number of passes over the training set
    scheduler               -- optional torch.optim.lr_scheduler, stepped once per epoch
    print_every             -- print a line every N epochs
    early_stopping_patience -- stop if `monitor` has not improved for this many epochs (None = never)
    monitor                 -- "val_accuracy" (higher is better) or "val_loss" (lower is better)
    checkpoint_path         -- if set, the best weights so far are saved here and reloaded at the end

    Returns:
    history -- dict of per-epoch lists: loss, accuracy, val_loss, val_accuracy, lr
    """
    model = model.to(device)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}

    def better(new, best):
        if monitor == "val_accuracy":
            return new > best
        else:
            return new < best

        
    if monitor == "val_accuracy":
        best_score = -np.inf
    else:
        best_score = np.inf


    best_epoch, epochs_without_improvement = 0, 0
    t0 = time.time()

    for epoch in range(epochs):

        model.train()
        running_loss, running_correct, n_seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()               # 1. reset the gradients
            logits = model(x)                   # 2. forward propagation
            loss = criterion(logits, y)         # 3. compute the cost
            loss.backward()                     # 4. backward propagation
            optimizer.step()                    # 5. update the parameters

            bs = y.size(0)
            running_loss += loss.item() * bs
            with torch.no_grad():
                running_correct += (logits.argmax(dim=1) == y).sum().item()
            n_seen += bs

        # ---- validation pass: no gradients, and model.eval() so dropout/batchnorm behave
        model.eval()
        val_loss, val_correct, n_val = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                val_loss += criterion(logits, y).item() * y.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                n_val += y.size(0)

        history["loss"].append(running_loss / n_seen)
        history["accuracy"].append(running_correct / n_seen)
        history["val_loss"].append(val_loss / n_val)
        history["val_accuracy"].append(val_correct / n_val)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        score = history[monitor][-1]
        if better(score, best_score):
            best_score, best_epoch, epochs_without_improvement = score, epoch + 1, 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)   # "model checkpoint" callback
        else:
            epochs_without_improvement += 1

        if print_every and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(f"Epoch {epoch + 1:3d}/{epochs}"
                  f" - lr: {history['lr'][-1]:.4f}"
                  f" - loss: {history['loss'][-1]:.4f} - acc: {history['accuracy'][-1]:.4f}"
                  f" - val_loss: {history['val_loss'][-1]:.4f} - val_acc: {history['val_accuracy'][-1]:.4f}")

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}: no {monitor} improvement for "
                  f"{early_stopping_patience} epochs.")
            break

    run_time = time.time() - t0

    print(f"\nTrained in {run_time:.0f}s. Best {monitor} = {best_score:.4f} at epoch {best_epoch}. Time per epoch = {run_time / (epoch + 1):.2f}s.")
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))         # restore the best epoch
        print(f"Restored the weights from epoch {best_epoch}.")
    history["best_epoch"] = best_epoch
    history["seconds"] = time.time() - t0
    return history




"""
Datasets labels from  https://github.com/UCSC-REAL/cifar-10-100n
"""
noise_file = torch.load('./CIFAR-10_human.pt')
clean_label = noise_file['clean_label']
worst_label = noise_file['worse_label']
aggre_label = noise_file['aggre_label']
random_label1 = noise_file['random_label1']
random_label2 = noise_file['random_label2']
random_label3 = noise_file['random_label3']





# CNN block that convolves with 3x3 kernel and uses relu and then pools into 2x2 blocks, aka half size downsampling, for easier reconfiguration of the model
def CNN_layer(input_channels, output_channels):
    return [torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)] 

modelA = torch.nn.Sequential(
    *CNN_layer(3, 32),   # 32x32 -> 16x16
    *CNN_layer(32, 64),  # 16x16   -> 8x8
    torch.nn.Flatten(),
    torch.nn.Linear(8*8*64, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)

modelB = torch.nn.Sequential(
    *CNN_layer(3, 32),   # 32x32 -> 16x16
    *CNN_layer(32, 64),  # 16x16   -> 8x8
    *CNN_layer(64, 128), # 8x8   -> 4x4
    torch.nn.Flatten(),
    torch.nn.Linear(4*4*128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)

modelC = torch.nn.Sequential(
    *CNN_layer(3, 32),   # 32x32 -> 16x16
    *CNN_layer(32, 64),  # 16x16   -> 8x8
    *CNN_layer(64, 128), # 8x8   -> 4x4
    *CNN_layer(128, 256), # 4x4   -> 2x2
    torch.nn.Flatten(),
    torch.nn.Linear(2*2*256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)

# prints the number of parameters in the model, for sizing purposes
for model in [modelA, modelB, modelC]:
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params:_}")


#seems like model B has the fewest parameters, so we will use that for the rest of the code
model = modelB.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)


train_loader = torch.utils.data.DataLoader(train_set,   batch_size=BATCH_SIZE, shuffle=True) # shuffle for bedre trening i stedenfor rutine visining av data
val_loader   = torch.utils.data.DataLoader(val_set,     batch_size=BATCH_SIZE, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_set,    batch_size=BATCH_SIZE, shuffle=False)


history = train_model(modelB, 
                      train_loader, 
                      val_loader, 
                      torch.nn.CrossEntropyLoss(), 
                      optimizer, 
                      epochs=EPOCHS, 
                      print_every=1,
)

with open("history.json", "w") as f:
    json.dump(history, f)
