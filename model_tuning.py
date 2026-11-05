from engine import *



modelA = torch.nn.Sequential(
    *CNN_layer(3, 32),   # 32x32 -> 16x16
    *CNN_layer(32, 64),  # 16x16   -> 8x8
    torch.nn.Flatten(),
    torch.nn.Linear(8*8*64, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)



modelTrue = torch.nn.Sequential(
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

print(get_model_parameters(modelTrue))
print(get_model_parameters(modelA))
print(get_model_parameters(modelC))
