import argparse
import copy
import json
from os.path import isdir, join

import torch
from DLBio import pt_training
from DLBio.helpers import (check_mkdir, copy_source, dict_to_options,
                           save_options)
from DLBio.kwargs_translator import get_kwargs
from DLBio.pt_train_printer import Printer
from DLBio.pytorch_helpers import get_device, get_num_params

import config
from datasets.data_getter import get_data_loaders
from helpers import log_tensorboard
from models.model_getter import get_model
from train_interfaces import get_interface


def get_options():
    parser = argparse.ArgumentParser()

    # train hyperparams
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--wd', type=float, default=config.WD)
    parser.add_argument('--mom', type=float, default=config.MOM)
    parser.add_argument('--cs', type=int, default=config.CS)
    parser.add_argument('--bs', type=int, default=config.BS)
    parser.add_argument('--opt', type=str, default=config.OPT)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--folder', type=str, default='_debug')

    # model / ds specific params
    parser.add_argument('--in_dim', type=int, default=config.IN_DIM)
    parser.add_argument('--out_dim', type=int, default=config.NUM_CLASSES)
    parser.add_argument('--model_type', type=str, default=config.MT)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--comment', type=str, default='-1')

    # scheduling
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_steps', type=int, default=0)
    parser.add_argument('--fixed_steps', nargs='+', default=None)

    # dataset
    parser.add_argument('--dataset', type=str, default=config.DATASET)
    parser.add_argument('--ds_kwargs', type=str, default=None)
    parser.add_argument('--nw', type=int, default=0)

    # model saving
    parser.add_argument('--model_kw', type=str, default=None)

    parser.add_argument('--sv_int', type=int, default=0)
    parser.add_argument('--early_stopping', action='store_true')

    parser.add_argument('--do_overwrite', action='store_true')

    parser.add_argument('--log_tb', action='store_true', default=config.LOG_TB)

    return parser.parse_args()


def run(options):
    if options.device is not None:
        pt_training.set_device(options.device)

    device = get_device()

    pt_training.set_random_seed(options.seed)

    folder = options.folder

    if not options.do_overwrite:
        if abort_due_to_overwrite_safety(folder):
            print('Process aborted.')
            return

    check_mkdir(folder)

    if options.comment is None:
        print('You forgot to add a comment to your experiment. Please add something!')
        options.comment = input('Comment: ')

    save_options(join(
        folder, 'opt.json'),
        options)

    copy_source(folder, do_not_copy_folders=config.DO_NOT_COPY)

    _train_model(options, folder, device)


def _train_model(options, folder, device):
    model_out = join(folder, 'model.pt')
    log_file = join(folder, 'log.json')
    check_mkdir(log_file)

    model = load_model(options, device)
    write_model_specs(folder, model)

    optimizer = pt_training.get_optimizer(
        options.opt, model.parameters(),
        options.lr,
        momentum=options.mom,
        weight_decay=options.wd
    )

    if options.lr_steps > 0 or options.fixed_steps is not None:
        scheduler = pt_training.get_scheduler(
            options.lr_steps, options.epochs, optimizer,
            fixed_steps=options.fixed_steps
        )
    else:
        print('no scheduling used')
        scheduler = None

    print(f'ds_{options.dataset}')

    data_loaders = get_data(options)

    if options.early_stopping:
        assert options.sv_int == -1
        early_stopping = pt_training.EarlyStopping(
            options.es_metric, get_max=True, epoch_thres=options.epochs
        )
    else:
        early_stopping = None

    train_interface = get_interface(
        'classification',
        model, device, Printer(config.PRINT_FREQUENCY, log_file),
    )

    training = pt_training.Training(
        optimizer, data_loaders['train'], train_interface,
        scheduler=scheduler, printer=train_interface.printer,
        save_path=model_out, save_steps=options.sv_int,
        val_data_loader=data_loaders['val'],
        early_stopping=early_stopping,
        save_state_dict=True,
        test_data_loader=data_loaders['test'],
    )

    training(options.epochs)


def write_model_specs(folder, model):

    print(f'#train params: {get_num_params(model, True):,}')

    with open(join(folder, 'model_specs.json'), 'w') as file:
        json.dump({
            'num_trainable': float(get_num_params(model, True)),
            'num_params': float(get_num_params(model, False))
        }, file)


def abort_due_to_overwrite_safety(folder):
    abort = False
    print('OVERWRITE SAFETY OFF')
    return False
    if isdir(folder):
        print(f'The folder {folder} already exists. Overwrite it?')
        print('Y: overwrite')
        print('Any key: stop')
        char = input('Overwrite?')
        if char != 'Y':
            abort = True

    return abort


def get_data(options):
    options = copy.deepcopy(options)
    if isinstance(options, dict):
        options = dict_to_options(options)
    if options.ds_kwargs is not None:
        ds_kwargs = get_kwargs(options.ds_kwargs)
    else:
        ds_kwargs = {}
    return get_data_loaders(
        options.dataset, options.bs, options.nw,
        **ds_kwargs
    )


def load_model(options, device, new_model_path=None):
    if isinstance(options, dict):
        options = dict_to_options(options)

    model_kwargs = get_kwargs(options.model_kw)

    model = get_model(
        options.model_type,
        options.in_dim,
        options.out_dim,
        device,
        **model_kwargs
    )

    if hasattr(model, 'model_path') and options.model_path is not None:
        model_sd = torch.load(options.model_path).state_dict()
        model.load_state_dict(model_sd, strict=True)

    if new_model_path is not None:
        model_sd = torch.load(new_model_path).state_dict()
        model.load_state_dict(model_sd, strict=True)

    return model


if __name__ == "__main__":
    OPTIONS = get_options()
    run(OPTIONS)
