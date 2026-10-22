from dataset_load import *
import torch


if __name__ == "__main__":
    EPOCHS = 100
    BATCH_SIZE = 128
    MAX_TIME = 60*15
    SEED = 42
    WORKERS = 4 #mellom 2 og 4 fungerte bra for ruben større antall WORKERS for mange EPOCHS

    device_to_run_on()
    load_data()

    learning_rates = [0.01, 0.001, 0.0001, 0.00001]

    for learning_rate in learning_rates:
        torch.manual_seed(SEED) # seed random innit for reproducability
        demo_model = model()

        print(f"Number of parameters in the model: {get_model_parameters(demo_model):_}")
        experiment(demo_model, 
            "aggre", 
            seed=SEED, 
            split=[0.6, 0.4], 
            epochs=EPOCHS, 
            learning_rate=learning_rate, 
            batch_size=BATCH_SIZE, 
            experiment_name=f"experiment_learning_rate_{learning_rate}", 
            max_time=MAX_TIME,
            Number_of_workers = WORKERS)
