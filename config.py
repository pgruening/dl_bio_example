USER = 'david'

if USER == 'philipp':
    DO_NOT_COPY = []
    # dataset
    NAT_IM_BASE = 'data/natural_images'     
    DATA_FOLDER = 'data'
    DATASET = 'mnist'
    NUM_CLASSES = 10
    IN_DIM = 3

    EXP_FOLDER = './experiments'

    PRINT_FREQUENCY = 100


    # model
    USE_PRETRAINED = False
   
    WEIGHT_IDS = {
        }

    

    # default training values
    LR = 0.001
    WD = 0.00001
    BS = 16
    MOM = 0.9
    CS = 224
    MT = 'resnet18'
    OPT = 'Adam'

if USER == 'david':
    DO_NOT_COPY = []
    # dataset
    NAT_IM_BASE = 'data/natural_images'     
    DATA_FOLDER = 'data'
    DATASET = 'mnist'
    NUM_CLASSES = 10
    IN_DIM = 3

    EXP_FOLDER = './experiments'

    PRINT_FREQUENCY = 100


    # model
    USE_PRETRAINED = False
   
    WEIGHT_IDS = {
        'custom_net_layer5_dim8':'14M3uC29aAx2AMeCeidLQjqjkVpGqnb6k'
        }

    
    # default training values
    LR = 0.001
    WD = 0.00001
    BS = 16
    MOM = 0.9
    CS = 224
    MT = 'custom_net'
    OPT = 'Adam'


DO_NOT_COPY += [EXP_FOLDER]
