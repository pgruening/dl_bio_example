USER = 'david'

if USER == 'philipp':
    DO_NOT_COPY = []
    # dataset
    NAT_IM_BASE = 'data/natural_images'

    EXP_FOLDER = './experiments'

    PRINT_FREQUENCY = 100

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

    #DATASET = nat_im'

    EXP_FOLDER = './experiments'

    PRINT_FREQUENCY = 100

    # default training values
    LR = 0.001
    WD = 0.00001
    BS = 16
    MOM = 0.9
    CS = 224
    MT = 'resnet18'
    OPT = 'Adam'

    NAT_IM_PARAMS = {
        'dataset' : 'nat_im',
        'in_dim' : 3,
        'out_dim' : 8,
        'lr' : 0.001,
        'wd' : 0.0001,
        'mom' : 0.9,
        'cs' : 244,
        'bs' : 16,
        'opt' : 'Adam',
        'model_type' : 'resnet18'
    }

    MNIST_PARAMS = {        
        'dataset' : 'mnist',
        'in_dim' : 3,
        'out_dim' : 10,
        'lr' : 0.001,
        'wd' : 0.0001,
        'mom' : 0.9,
        'cs' : 244,
        'bs' : 16,
        'opt' : 'Adam',
        'model_type' : 'resnet18'
    }

DO_NOT_COPY += [EXP_FOLDER]
