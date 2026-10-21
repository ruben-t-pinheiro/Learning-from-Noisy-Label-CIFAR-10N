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

#stolen from assignment 1, modified for readability, extra features and pruned.
def train_model(model, train_loader, val_noisy_loader, val_clean_loader, criterion, optimizer, epochs):
    model = model.to(device)
    history = {"loss": [], "accuracy": [], "val_noisy_loss": [], "val_noisy_accuracy": [], "val_clean_loss": [], "val_clean_accuracy": [], "lr": []}

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
        val_noisy_loss, val_noisy_correct, n_val_noisy = 0.0, 0, 0
        val_clean_loss, val_clean_correct, n_val_clean = 0.0, 0, 0
        with torch.no_grad():

            for x, y in val_noisy_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                val_noisy_loss += criterion(logits, y).item() * y.size(0)
                val_noisy_correct += (logits.argmax(dim=1) == y).sum().item()
                n_val_noisy += y.size(0)

            for x, y in val_clean_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                val_clean_loss += criterion(logits, y).item() * y.size(0)
                val_clean_correct += (logits.argmax(dim=1) == y).sum().item()
                n_val_clean += y.size(0)

        history["loss"].append(running_loss / n_seen)                           # training
        history["accuracy"].append(running_correct / n_seen)                    # training
        history["val_noisy_loss"].append(val_noisy_loss / n_val_noisy)          # noisy validation
        history["val_noisy_accuracy"].append(val_noisy_correct / n_val_noisy)   # noisy validation
        history["val_clean_loss"].append(val_clean_loss / n_val_clean)          # clean validation
        history["val_clean_accuracy"].append(val_clean_correct / n_val_clean)   # clean validation
        history["lr"].append(optimizer.param_groups[0]["lr"])                   # learning rate


        print(f"Epoch {epoch + 1:3d}/{epochs}"
                f" - lr: {history['lr'][-1]:.4f}"
                f" - loss: {history['loss'][-1]:.4f} - acc: {history['accuracy'][-1]:.4f}"
                f" - val_noisy_loss: {history['val_noisy_loss'][-1]:.4f} - val_noisy_acc: {history['val_noisy_accuracy'][-1]:.4f}"
                f" - val_clean_loss: {history['val_clean_loss'][-1]:.4f} - val_clean_acc: {history['val_clean_accuracy'][-1]:.4f}")


    run_time = time.time() - t0
    print(f"\nTrained in {run_time:.0f}s. Time per epoch = {run_time / (epoch + 1):.2f}s.")
    history["seconds"] = time.time() - t0
    return history





# Datasets labels from  https://github.com/UCSC-REAL/cifar-10-100n
noise_file = torch.load('./CIFAR-10_human.pt')

clean_label = noise_file['clean_label']
worst_label = noise_file['worse_label']
aggre_label = noise_file['aggre_label']
random_label1 = noise_file['random_label1']
random_label2 = noise_file['random_label2']
random_label3 = noise_file['random_label3']

label_sets = {
    "clean":   clean_label,
    "aggre":   aggre_label,
    "random1": random_label1,
    "random2": random_label2,
    "random3": random_label3,
    "worst":   worst_label,
}

# makes a dataset woith a specified label set, or the default labels if label_set_name is None
def make_dataset(label_set_name, seed, split=[0.6, 0.4]):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # Loads the dataset, images and original labels
    if label_set_name is not None:
        dataset.targets = torch.as_tensor(label_sets[label_set_name]).long().tolist() # replace label with different noisy label set

    generator = torch.Generator().manual_seed(seed) # seed for reproducibility
    train_set, val_set_noisy = torch.utils.data.random_split(dataset, split, generator=generator) # only using training and validation sets, test set is not needed when we are reaserching the transient of learning and not total model perfomance.


    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # Loads the dataset, images and original labels
    _        , val_set_clean = torch.utils.data.random_split(dataset, split, generator=generator)

    return train_set, val_set_noisy, val_set_clean




# CNN block that convolves with 3x3 kernel and uses relu and then pools into 2x2 blocks, aka half size downsampling, for easier reconfiguration of the model
def CNN_layer(input_channels, output_channels):
    return [torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)] 


# generic model, nothing fancy or special, just a simple CNN with 3 convolutional layers and 2 fully connected layers, for demonstration purposes
demo_model = torch.nn.Sequential(
    *CNN_layer(3, 32),   # 32x32 -> 16x16
    *CNN_layer(32, 64),  # 16x16   -> 8x8
    *CNN_layer(64, 128), # 8x8   -> 4x4
    torch.nn.Flatten(),
    torch.nn.Linear(4*4*128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)


# prints the number of parameters in the model, for sizing purposes
num_params = sum(p.numel() for p in demo_model.parameters())
print(f"Number of parameters in the model: {num_params:_}")




# demonstration experiment, trains modelB on the clean dataset and saves the history to a json file
if __name__ == "__main__":

    model = demo_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_set, val_set_noisy, val_set_clean = make_dataset("worst", seed=42, split=[0.8, 0.2]) # 42 because it is the answer to life, the universe and everything
    train_loader     = torch.utils.data.DataLoader(train_set,   batch_size=BATCH_SIZE, shuffle=True) # shuffle for better training so the model doesn't learn the order of the data
    val_noisy_loader = torch.utils.data.DataLoader(val_set_noisy,     batch_size=BATCH_SIZE, shuffle=False)
    val_clean_loader = torch.utils.data.DataLoader(val_set_clean,     batch_size=BATCH_SIZE, shuffle=False)

    history = train_model(model,
                          train_loader,
                          val_noisy_loader,
                          val_clean_loader,
                          torch.nn.CrossEntropyLoss(),
                          optimizer,
                          epochs=EPOCHS
    )

    with open("history.json", "w") as f:
        json.dump(history, f)