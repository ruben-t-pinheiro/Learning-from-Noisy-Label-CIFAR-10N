from engine import *
import torch


if __name__ == "__main__":
    EPOCHS = 50
    BATCH_SIZE = [16, 32, 64, 128, 256, 512]
    MAX_TIME = 60*15
    SEEDS = [42]
    WORKERS = 6 #mellom 2 og 4 fungerte bra for ruben større antall WORKERS for mange EPOCHS

    device_to_run_on()
    label_sets = load_data()

    learning_rates = [0.0001]

    for seed in SEEDS:  # aproximately 1 hour per seed for 100 epochs with 4 workers on a rx 7900xt gpu, needed about 6 hours of compute time for all seeds 
        for noise in label_sets:
            for learning_rate in learning_rates:
                for batch_size in BATCH_SIZE:

                    torch.manual_seed(seed) # seed random innit for reproducability
                    demo_model = model()

                    print(f"Number of parameters in the model: {get_model_parameters(demo_model):_}")
                    experiment(demo_model, 
                        noise, 
                        seed=seed, 
                        split=[0.6, 0.4], 
                        epochs=EPOCHS, 
                        learning_rate=learning_rate, 
                        batch_size=batch_size, 
                        experiment_name=f"BATCH_TEST_{batch_size}_noise_{noise}_lr_{learning_rate}_seed{seed}_epochs_{EPOCHS}", 
                        max_time=MAX_TIME,
                        Number_of_workers = WORKERS)


    

