import json
import time

import torch
import torchvision

def device_to_run_on():
    # check if GPU is available, otherwise use CPU
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

def confusion_matrix(y_true, y_pred, num_classes):
    idx = y_true * num_classes + y_pred
    return torch.bincount(idx, minlength=num_classes**2).view(num_classes, num_classes)


def evaluate_model(model, data_loader, criterion):
    num_classes = 10
    model.eval()
    val_loss, val_correct, n_val = 0.0, 0, 0
    cm = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            val_loss += criterion(logits, y).item() * y.size(0)
            val_correct += (logits.argmax(dim=1) == y).sum().item()
            n_val += y.size(0)
            cm += confusion_matrix(y, logits.argmax(dim=1), num_classes).view(-1)
    cm = cm.view(num_classes, num_classes)

    loss = val_loss / n_val
    accuracy = val_correct / n_val
    return loss, accuracy, cm
        


#stolen from assignment 1, modified for readability, extra features and pruned.
def train_model(model, train_loader, val_noisy_loader, val_clean_loader, criterion, optimizer, epochs, max_time, experiment_name):
    model = model.to(device)
    history = {"train_loss": [], 
               "train_accuracy": [], 
               "val_noisy_loss": [], 
               "val_noisy_accuracy": [], 
               "val_clean_loss": [], 
               "val_clean_accuracy": [], 
               "lr": [], 
               "val_noisy_cm": [], 
               "val_clean_cm": []}

    t0 = time.time()

    for epoch in range(epochs):

        if max_time is not None and time.time() - t0 > max_time:
            print(f"\nMax time of {max_time} seconds reached, stopping training.")
            break

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

        training_loss = running_loss / n_seen
        training_accuracy = running_correct / n_seen


        val_noisy_loss, val_noisy_accuracy, val_noisy_cm = evaluate_model(model, val_noisy_loader, criterion)
        val_clean_loss, val_clean_accuracy, val_clean_cm = evaluate_model(model, val_clean_loader, criterion)

            
        history["train_loss"        ].append(training_loss)         # training
        history["train_accuracy"    ].append(training_accuracy)     # training
        history["val_noisy_loss"    ].append(val_noisy_loss)        # noisy validation
        history["val_noisy_accuracy"].append(val_noisy_accuracy)    # noisy validation
        history["val_clean_loss"    ].append(val_clean_loss)        # clean validation
        history["val_clean_accuracy"].append(val_clean_accuracy)    # clean validation
        history["val_noisy_cm"      ].append(val_noisy_cm.tolist()) # noisy validation confusion matrix
        history["val_clean_cm"      ].append(val_clean_cm.tolist()) # clean validation confusion matrix
        history["lr"].append(optimizer.param_groups[0]["lr"])       # learning rate


        print(f"Epoch {epoch + 1:3d}/{epochs}"
                f"   lr: {history['lr'][-1]:.1e}"
                f"   loss: {history['train_loss'][-1]:.4f}   acc: {history['train_accuracy'][-1]:.4f}"
                f"   val_noisy_loss: {history['val_noisy_loss'][-1]:.4f}   val_noisy_acc: {history['val_noisy_accuracy'][-1]:.4f}"
                f"   val_clean_loss: {history['val_clean_loss'][-1]:.4f}   val_clean_acc: {history['val_clean_accuracy'][-1]:.4f}"
                f"   running_time: {time.time() - t0:.0f}s/{max_time:.0f}s")
        
        with open(f"results/{experiment_name}_history.json", "w") as f:
                    json.dump(history, f)

    run_time = time.time() - t0
    epochs_run = len(history["train_loss"])
    print(f"\nTrained in {run_time:.0f}s. Time per epoch = {run_time / max(epochs_run, 1):.2f}s.")
    history["epochs_run"] = epochs_run
    history["seconds"] = time.time() - t0
    return history

def load_data():
    global label_sets
    # Datasets labels from  https://github.com/UCSC-REAL/cifar-10-100n
    noise_file = torch.load('./CIFAR-10_human.pt', weights_only=False) # explicit weight only = false because of pytorch warnings

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

    transform = torchvision.transforms.ToTensor()
    noisy_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    clean_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    if label_set_name is not None:
        noisy_dataset.targets = torch.as_tensor(label_sets[label_set_name]).long().tolist() # replace label with different noisy label set

    generator = torch.Generator().manual_seed(seed) # seed for reproducibility
    train_set, val_set_noisy = torch.utils.data.random_split(noisy_dataset, split, generator=generator)
    val_set_clean = torch.utils.data.Subset(clean_dataset, val_set_noisy.indices)

    return train_set, val_set_noisy, val_set_clean


# streamlining experiments in standardized way, so we can easily run multiple experiments with different parameters and models. intended for batching experminets for AFK science
def experiment(model, label_set_name, seed, split, epochs, learning_rate, batch_size, experiment_name="default", max_time=60, Number_of_workers = 6):
    try:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        loader_args = dict(batch_size=batch_size, num_workers=Number_of_workers, pin_memory=(device.type == "cuda"), persistent_workers=True) # common settings for the loaders, use more GPU to speed up compute


        train_set, val_set_noisy, val_set_clean = make_dataset(label_set_name, seed, split)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader     = torch.utils.data.DataLoader(train_set,       shuffle=True,   **loader_args,  generator=generator) # shuffle for better training so the model doesn't learn the order of the data
        val_noisy_loader = torch.utils.data.DataLoader(val_set_noisy,   shuffle=False,  **loader_args)
        val_clean_loader = torch.utils.data.DataLoader(val_set_clean,   shuffle=False,  **loader_args)

        history = train_model(model,
                            train_loader,
                            val_noisy_loader,
                            val_clean_loader,
                            torch.nn.CrossEntropyLoss(),
                            optimizer,
                            epochs=epochs,
                            max_time=max_time,
                            experiment_name=experiment_name
        )

        

        return history
    except:
        pass

def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

 # CNN block that convolves with 3x3 kernel and uses relu and then pools into 2x2 blocks, aka half size downsampling, for easier reconfiguration of the model
def CNN_layer(input_channels, output_channels):
    return [torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)] 

def model():
    # generic model, nothing fancy or special, just a simple CNN with 3 convolutional layers and 2 fully connected layers, for demonstration purposes
    return torch.nn.Sequential(
        *CNN_layer(3, 32),   # 32x32 -> 16x16
        *CNN_layer(32, 64),  # 16x16   -> 8x8
        *CNN_layer(64, 128), # 8x8   -> 4x4
        torch.nn.Flatten(),
        torch.nn.Linear(4*4*128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )


# demonstration experiment, trains model on the clean dataset and saves the history to a json file
if __name__ == "__main__":

    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 128
    MAX_TIME = 12
    SEED = 42
    WORKERS = 1

    device_to_run_on()
    load_data()

    torch.manual_seed(SEED) # seed random innit for reproducability
    demo_model = model()

    # prints the number of parameters in the model, for sizing purposes
    print(f"Number of parameters in the model: {get_model_parameters(demo_model):_}")

    experiment(demo_model, 
        "aggre", 
        seed=SEED, 
        split=[0.6, 0.4], 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE, 
        batch_size=BATCH_SIZE, 
        experiment_name="demo_aggre", 
        max_time=MAX_TIME,
        Number_of_workers = WORKERS)