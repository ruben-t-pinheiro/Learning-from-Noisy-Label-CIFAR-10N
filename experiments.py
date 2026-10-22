from dataset_load import *
import torch

EPOCHS = 100
BATCH_SIZE = 128
MAX_TIME = 60*15
SEED = 42


def CNN_layer(input_channels, output_channels):
    return [torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)] 

torch.manual_seed(SEED) # seed random innit for reproducability

demo_model = torch.nn.Sequential(
    *CNN_layer(3, 32),   # 32x32 -> 16x16
    *CNN_layer(32, 64),  # 16x16   -> 8x8
    *CNN_layer(64, 128), # 8x8   -> 4x4
    torch.nn.Flatten(),
    torch.nn.Linear(4*4*128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)

print(f"Number of parameters in the model: {get_model_parameters(demo_model):_}")



experiment(demo_model, "aggre", seed=SEED, split=[0.6, 0.4], epochs=EPOCHS, learning_rate=0.01, batch_size=BATCH_SIZE, experiment_name="experiment1", max_time=MAX_TIME)
experiment(demo_model, "aggre", seed=SEED, split=[0.6, 0.4], epochs=EPOCHS, learning_rate=0.001, batch_size=BATCH_SIZE, experiment_name="experiment2", max_time=MAX_TIME)
experiment(demo_model, "aggre", seed=SEED, split=[0.6, 0.4], epochs=EPOCHS, learning_rate=0.0001, batch_size=BATCH_SIZE, experiment_name="experiment3", max_time=MAX_TIME)
experiment(demo_model, "aggre", seed=SEED, split=[0.6, 0.4], epochs=EPOCHS, learning_rate=0.00001, batch_size=BATCH_SIZE, experiment_name="experiment4", max_time=MAX_TIME)