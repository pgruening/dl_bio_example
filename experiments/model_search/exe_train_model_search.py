"""
This module aims to create several training processes that are defined by 
specific keywords. Each process is essentially a system-call 
'python [EXE_FILE] --argument1 value1 --argument2 value2 ...'. Here, the 
EXE_FILE is 'run_training.py'; read it to see which arguments can be passed 
to the file.

Each training process creates a folder defined by the 'folder' property where 
different files a stored. During and especially after a training process is 
done, you can find the following files:

'opt.json': a dictionary that contains all the keywords that defined the 
training process
'log.json': a dictionary that contains loss values and metrics for each epoch. 
From this file, you can, for example, extract the training loss curve.
'model_specs.json': saves other model-specific values. E.g., the number of 
model parameters.
'model.pt': the PyTorch state-dictionary, which can be used to load the trained 
model.
'src_copy': A folder with a copy of all source files at the specific moment the 
training was started. These files are merely meant as emergency backups if 
you've forgotten how exactly the training was programmed.

Several training processes may be run in parallel, either on the same GPU or 
multiple GPUs.
"""
import copy
from os.path import join

from DLBio import pt_run_parallel
from DLBio.kwargs_translator import to_kwargs_str
from DLBio.pytorch_helpers import get_device
from helpers import log_tensorboard
from run_training import get_data, load_model
import itertools

# If you have multiple GPUs, you can determine which shall be used for training
AVAILABLE_GPUS = [0]
BASE_FOLDER = 'experiments/model_search'  # the folder of the experiment
EXE_FILE = 'run_training.py'  # will be called as a subprocess
CREATE_TBOARD_RUNS = True  # create tensorboard files to look at the model
RUN_TRAINING = True  # run the training processes

# If there are problems with the GPU memory prediction, you can disable it here.
# Instead, 'DEFAULT_GPU_MEM' is the predicted memory for each run.
DO_PREDICT_GPU_MEM = True
DEFAULT_GPU_MEM = 1000  # MegaByte

# NOTE: Be careful with this parameter:
# Defines how many parallel processes can be trained on one gpu. A number > 4,
# can cause CUDA to crash. Furthermore, if each run loads a large dataset into
# RAM, your computer may start swapping.
MAX_N_PROCESSES_PER_GPU = 1

# These keywords are passed directly to run_training.py
DEFAULT_KWARGS = {
    'lr': 0.01,  # learning rate
    'wd': 0.0001,  # weight decay
    'bs': 128,  # batch size
    'model_type': 'custom_net',  # which kind of model is used
    'dataset': 'mnist',  # which dataset is used
    'in_dim': 1,  # the input dimension for the model
    'out_dim': 10,  # the output dimension (number of classes)
    'nw': 0,  # number of workers the load the data, 0: debugging possible
    # save the trained models each sv_int-th step. If 0, save when training is
    # done. If -1, the model is not saved.
    'sv_int': -1,
    'epochs': 10,  # train the model for this many epochs
    # reduce the learning each time after epochs/lr_steps epochs by multiplying
    # the learning rate with 0.1
    'lr_steps': 2,
    # random seed controlling, for example, the random weight initialization
    # or random batch order
    'seed': 0
}

# These keywords may change with every different model you use. Thus, they are
# combined to a string by the kwargs_translator. Note that, all values need
# to be lists. See the models/model_getter.py module for more information
DEFAULT_MODEL_KWARGS = {
    'num_layers': [2],  # number of conv-layers for our costum model
    'kernel_size': [3],  # the kernel size of each conv-layer
    'hidden_dim': [32]  # the first conv-layer has an output dimension of 32
}

# These keywords are also combined to a string. See the dataset/data_getter.py
# module for more information
DEFAULT_DATA_KWARGS = {
    'convert_to_rgb': [False],
}


# A training process is called as a subprocess (look up subprocess.call).
# It is similar to running the following command in a terminal:
# python run_training.py --lr 0.001 --bs 128 --wd 0.0001 ...
class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        self.start_time = -1
        self.device = -1
        self.mem_used = None

        self.__name__ = 'Model_Search_training_process'
        self.module_name = EXE_FILE
        self.kwargs = kwargs


def run():
    """
    The function that starts all training runs. It requires a param_generator,
    which is a iterable that returns/yields dictionaries with keyword value
    pairs, e.g., the default values above. Usually the param_generator is
    a generator.
    It is possible to run several processes on a GPU and several GPUs at the
    same time. The function run_bin_packing aims to run as many processes as
    possible in parallel.
    Note that, in order to use this functionality, the keyword mem_used must
    be set in the parameters. The functions 
    'pt_run_parallel.predict_needed_gpu_memory' is meant to approximate how
    much GPU memory a training process will need.
    """
    make_object = pt_run_parallel.MakeObject(TrainingProcess)
    pt_run_parallel.run_bin_packing(
        param_generator(),  # in a for loop, generates dictionaries
        make_object,
        available_gpus=AVAILABLE_GPUS,  # which GPU can be used by the function
        # a log file is saved to this path to see which processes are running
        log_file=join(BASE_FOLDER, 'parallel_train_log.txt'),
        # running too many processes at once may cause errors. Use this
        # argument to restrict the number of parallel processes
        # Also, if the machine you're using does not have a lot of RAM
        # (<8 GByte) you might run into trouble.
        max_num_processes_per_gpu=MAX_N_PROCESSES_PER_GPU
    )


def check_tensorboard():
    """
    It is important to visualize the used data and models. Tensorboard is a
    great tool to achieve this. Here, for each parameter dictionary a
    tensorboard file is created in 'tboard/experiments/model_search/...'
    To look at the files, simply run "tensorboard --logdir tb_out" in the 
    terminal
    """
    print('creating tensorboard files...')
    is_first_sample = True
    for parameters in param_generator():
        # do not create redundant files
        if parameters['seed'] != 0:
            continue
        folder_name = parameters['folder'].split('/')[-1]
        model = load_model(parameters, get_device())

        # only for the first set of parameters we save some images as well
        if is_first_sample:
            data_loaders = get_data(parameters)
            is_first_sample = False
        else:
            data_loaders = None

        log_tensorboard(
            parameters['folder'],
            join('tboard', BASE_FOLDER, folder_name),
            data_loaders, 3, model,
            input_shape=(1, 1, 28, 28)
        )

    print('done!')


def param_generator():
    """
    An iterable that returns a dictionary with training parameters at each 
    iteration.
    """
    def setup_training(values, to_search):
        parameters = copy.deepcopy(DEFAULT_KWARGS)
        for key, val in zip(to_search.keys(), values):
            if key in DEFAULT_KWARGS.keys():
                parameters[key] = val

        return parameters

    def get_model_kw(values, to_search):
        model_kw = copy.deepcopy(DEFAULT_MODEL_KWARGS)
        for key, val in zip(to_search.keys(), values):
            if key in DEFAULT_MODEL_KWARGS.keys():
                # will be transformed to string -> needs to be a list
                model_kw[key] = [val]

        return to_kwargs_str(model_kw)

    def get_ds_kwargs(values, to_search):
        ds_kwargs = copy.deepcopy(DEFAULT_DATA_KWARGS)
        for key, val in zip(to_search.keys(), values):
            if key in DEFAULT_DATA_KWARGS.keys():
                # will be transformed to string -> needs to be a list
                ds_kwargs[key] = [val]

        return to_kwargs_str(ds_kwargs)

    ctr = 0  # used to create the folders for each model
    # create an iterator over all value combinations we want to train with
    to_search = {
        'seed': [0, 173, 43],
        'lr': [.001, .0001],  # try different learning rates
        # try different model parameters
        'num_layers': [1, 2, 3],
        'kernel_size': [3, 5],
        'hidden_dim': [8, 16]
    }
    product = itertools.product(*list(to_search.values()))
    for values in product:
        # setup the dictionary with the usual training parameters
        parameters = setup_training(values, to_search)
        # add model-specific parameters
        parameters['model_kw'] = get_model_kw(values, to_search)
        # add dataset-specific parameters
        parameters['ds_kwargs'] = get_ds_kwargs(values, to_search)

        if DO_PREDICT_GPU_MEM:
            # try to predict how much memory the model will use on the gpu
            parameters['mem_used'] = pt_run_parallel.predict_needed_gpu_memory(
                parameters, input_shape=(parameters['bs'], 1, 28, 28),
                load_model_fcn=load_model, device=AVAILABLE_GPUS[0]
            )
        else:
            parameters['mem_used'] = DEFAULT_GPU_MEM

        # define where all training data will be saved
        parameters['folder'] = join(
            BASE_FOLDER, 'trained_models', str(ctr).zfill(4)
        )
        ctr += 1

        yield parameters


if __name__ == "__main__":
    if RUN_TRAINING:
        run()

    if CREATE_TBOARD_RUNS:
        check_tensorboard()
