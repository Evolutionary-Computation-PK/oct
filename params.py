NUM_CLASSES = 4
DISEASE_CLASSES = [0, 1, 2]
N_FOLDS = 3
MAX_EPOCHS = 30
RANDOM_STATE = 42

# Optuna
N_TRIALS = 35
N_STARTUP_TRIALS = 5
N_WARMUP_STEPS = 3
INTERVAL_STEPS = 1

BEST_PARAMS = [{
    'lr': 0.00035045271826099973,
    'batch_size': 128,
    'weight_decay': 0.0006662399361671108,
    'dropout_rate': 0.2546729457892419,
    'optimizer': 'RMSprop',
    'num_unfrozen': 17,
    'dense_units': 256,
    'focal_loss_gamma': 2.830874617657475,
    'feature_extraction_epochs': 6,
    'patience': 7
},
{
    'lr': 0.0001901253376247502,
    'batch_size': 128,
    'weight_decay': 0.00046899671234917423,
    'dropout_rate': 0.2340627479127319,
    'optimizer': 'RMSprop',
    'num_unfrozen': 9,
    'dense_units': 160,
    'focal_loss_gamma': 2.637513676164258,
    'feature_extraction_epochs': 6,
    'patience': 7
},
{
    'lr': 0.0003475282161839681,
    'batch_size': 128,
    'weight_decay': 1.2963894940526187e-05,
    'dropout_rate': 0.6786759775515059,
    'optimizer': 'RMSprop',
    'num_unfrozen': 14,
    'dense_units': 32,
    'focal_loss_gamma': 1.031189599316912,
    'feature_extraction_epochs': 4,
    'patience': 6
}]
