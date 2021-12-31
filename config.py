EPOCHS = 50 # Number of the epochs
SAVE_EVERY = 5 # after how many epochs to save a checkpoint
LOG_EVERY = 1 #  log training and validation metrics every `LOG_EVERY` epochs
BATCH_SIZE = 2 
DEVICE = 'cuda'  
LR = 0.0001
ROOT_PATH = 'ADD HERE FULL PATH OF THE DATASET'

# the classes that we want to train
CLASSES_TO_TRAIN = ['Road', 'Background']
# DEBUG for visualizations
DEBUG = True