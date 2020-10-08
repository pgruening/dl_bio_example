USER = 'philipp'

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

DO_NOT_COPY += [EXP_FOLDER]
