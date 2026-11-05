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

def evaluate_accuracy(model, data_loader):
    correct = 0
    total = len(data_loader.dataset)

    if total == 0:
        return None

    model.eval()
    with torch.no_grad():
        for image, label in data_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model(image)
            correct += (logits.argmax(dim=1) == label).sum().item()

    return correct / total

#stolen from assignment 1, modified for readability, extra features and pruned.
def train_model(model, train_loader, val_noisy_loader, val_clean_loader, 
                correctly_labeled_loader, incorrectly_labeled_loader, 
                incorrectly_labeled_clean_loader, criterion, optimizer, epochs, max_time, experiment_name):
    
    model = model.to(device)
    history = {"train_loss": [], 
               "train_accuracy": [], 
               "val_noisy_loss": [], 
               "val_noisy_accuracy": [], 
               "val_clean_loss": [], 
               "val_clean_accuracy": [], 
               "lr": [], 
               "val_noisy_cm": [], 
               "val_clean_cm": [],
               "correctly_labeled_accuracy": [],
               "incorrectly_labeled_accuracy": [],
               "incorrectly_labeled_clean_accuracy": []}

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
        correctly_labeled_accuracy = evaluate_accuracy(model, correctly_labeled_loader)
        incorrectly_labeled_accuracy = evaluate_accuracy(model, incorrectly_labeled_loader)
        incorrectly_labeled_clean_accuracy = evaluate_accuracy(model, incorrectly_labeled_clean_loader)
            
        history["train_loss"        ].append(training_loss)         # training
        history["train_accuracy"    ].append(training_accuracy)     # training
        history["val_noisy_loss"    ].append(val_noisy_loss)        # noisy validation
        history["val_noisy_accuracy"].append(val_noisy_accuracy)    # noisy validation
        history["val_clean_loss"    ].append(val_clean_loss)        # clean validation
        history["val_clean_accuracy"].append(val_clean_accuracy)    # clean validation
        history["val_noisy_cm"      ].append(val_noisy_cm.tolist()) # noisy validation confusion matrix
        history["val_clean_cm"      ].append(val_clean_cm.tolist()) # clean validation confusion matrix
        history["correctly_labeled_accuracy"].append(correctly_labeled_accuracy) #accuracy when the lable is corect of the treaining data
        history["incorrectly_labeled_accuracy"].append(incorrectly_labeled_accuracy) #accuracy when the lable is not corect treaining data
        history["incorrectly_labeled_clean_accuracy"].append(incorrectly_labeled_clean_accuracy) #accuracy with the corect lable for the wrongly labald treaining data
        history["lr"].append(optimizer.param_groups[0]["lr"])       # learning rate


        print(f"Epoch {epoch + 1:3d}/{epochs}"
                f"   lr: {history['lr'][-1]:.1e}"
                f"   loss: {history['train_loss'][-1]:.4f}   acc: {history['train_accuracy'][-1]:.4f}"
                f"   val_noisy_loss: {history['val_noisy_loss'][-1]:.4f}   val_noisy_acc: {history['val_noisy_accuracy'][-1]:.4f}"
                f"   val_clean_loss: {history['val_clean_loss'][-1]:.4f}   val_clean_acc: {history['val_clean_accuracy'][-1]:.4f}"
                f"   correctly_labeled_accuracy: {format(history['correctly_labeled_accuracy'][-1])}"
                f"   incorrectly_labeled_accuracy: {format(history['incorrectly_labeled_accuracy'][-1])}"
                f"   incorrectly_labeled_clean_accuracy: {format(history['incorrectly_labeled_clean_accuracy'][-1])}"
                f"   running_time: {time.time() - t0:.0f}s/{max_time:.0f}s")
        
        
        with open(f"results/{experiment_name}_history.json", "w") as f:
                    json.dump(history, f)

    run_time = time.time() - t0
    epochs_run = len(history["train_loss"])
    print(f"\nTrained in {run_time:.0f}s. Time per epoch = {run_time / max(epochs_run, 1):.2f}s.")
    history["epochs_run"] = epochs_run
    history["seconds"] = time.time() - t0
    return history

def format(value):

    if value == None:
        string = "N/A"
    else:
        string = f"{value:.4f}"

    return string

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
    return label_sets

# makes a dataset woith a specified label set, or the default labels if label_set_name is None
def make_dataset(label_set_name, seed, split=[0.6, 0.4]):

    transform = torchvision.transforms.ToTensor()
    noisy_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    clean_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if label_set_name is not None:
        noisy_dataset.targets = torch.as_tensor(label_sets[label_set_name]).long().tolist() # replace label with different noisy label set

    generator = torch.Generator().manual_seed(seed) # seed for reproducibility
    train_set, val_set_noisy = torch.utils.data.random_split(noisy_dataset, split, generator=generator)
    val_set_clean = torch.utils.data.Subset(clean_dataset, val_set_noisy.indices)

    return train_set, val_set_noisy, val_set_clean, test_dataset

def make_evaluation_datasets(train_set, val_set_clean):

    noisy_dataset = train_set.dataset
    clean_dataset = val_set_clean.dataset

    correct_lable_index = []
    incorrect_label_index = []

    for index in train_set.indices:
        if noisy_dataset.targets[index] == clean_dataset.targets[index]:
            correct_lable_index.append(index)
        else:
            incorrect_label_index.append(index)

    correctly_labeled_data = torch.utils.data.Subset(noisy_dataset, correct_lable_index)
    incorrectly_labeled_data = torch.utils.data.Subset(noisy_dataset, incorrect_label_index)
    incorrectly_labeled_clean_data = torch.utils.data.Subset(clean_dataset, incorrect_label_index)

    return correctly_labeled_data, incorrectly_labeled_data, incorrectly_labeled_clean_data
    
# streamlining experiments in standardized way, so we can easily run multiple experiments with different parameters and models. intended for batching experminets for AFK science
def experiment(model, label_set_name, seed, split, epochs, learning_rate, batch_size, experiment_name="default", max_time=60, Number_of_workers = 6):
    try:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        loader_args = dict(batch_size=batch_size, num_workers=Number_of_workers, pin_memory=(device.type == "cuda"), persistent_workers=True) # common settings for the loaders, use more GPU to speed up compute


        train_set, val_set_noisy, val_set_clean, test_dataset = make_dataset(label_set_name, seed, split)
        correctly_labeled_data, incorrectly_labeled_data, incorrectly_labeled_clean_data = make_evaluation_datasets(train_set, val_set_clean)
        
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        test_loader = torch.utils.data.DataLoader(test_dataset,   shuffle=False,  **loader_args)
        train_loader     = torch.utils.data.DataLoader(train_set,       shuffle=True,   **loader_args,  generator=generator) # shuffle for better training so the model doesn't learn the order of the data
        val_noisy_loader = torch.utils.data.DataLoader(val_set_noisy,   shuffle=False,  **loader_args)
        val_clean_loader = torch.utils.data.DataLoader(val_set_clean,   shuffle=False,  **loader_args)
        correctly_labeled_loader = torch.utils.data.DataLoader(correctly_labeled_data,   shuffle=False,  **loader_args)
        incorrectly_labeled_loader = torch.utils.data.DataLoader(incorrectly_labeled_data,   shuffle=False,  **loader_args)
        incorrectly_labeled_clean_loader = torch.utils.data.DataLoader(incorrectly_labeled_clean_data,   shuffle=False,  **loader_args)

        criterion = torch.nn.CrossEntropyLoss()

        history = train_model(model,
                            train_loader,
                            val_noisy_loader,
                            val_clean_loader,
                            correctly_labeled_loader, 
                            incorrectly_labeled_loader, 
                            incorrectly_labeled_clean_loader,
                            criterion,
                            optimizer,
                            epochs=epochs,
                            max_time=max_time,
                            experiment_name=experiment_name
        )

        test_loss, test_accuracy, test_cm = evaluate_model(model, test_loader, criterion)
        
        history["test_loss"] = test_loss
        history["test_accuracy"] = test_accuracy
        history["test_cm"] = test_cm.tolist()

        with open(f"results/{experiment_name}_history.json", "w") as f:
            json.dump(history, f)

        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_accuracy}\n")

        return history
    except Exception as e:
        print(f"Experiment '{experiment_name}' failed: {e}")

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

