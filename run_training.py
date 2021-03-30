import argparse
from exe_log_tb import log_tensorboard
import json
from os.path import isdir, join

import torch
from DLBio import pt_training
from DLBio.helpers import check_mkdir, copy_source, save_options
from DLBio.pt_train_printer import Printer
from DLBio.pytorch_helpers import get_device, get_num_params
from DLBio.kwargs_translator import get_kwargs


import config
from train_interfaces import get_interface

from datasets.data_getter import get_data_loaders
from models.model_getter import get_model


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
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--dataset', type=str, default=config.DATASET)

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

    folder = join(config.EXP_FOLDER, options.folder)

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

    
    model_kwargs = get_kwargs(options.model_kw)

    model = get_model(
        options.model_type,
        options.in_dim,
        options.out_dim,
        device,
        config.USE_PRETRAINED,
        **model_kwargs
    )

    if options.model_path is not None:
        model_sd = torch.load(options.model_path).state_dict()
        model.load_state_dict(model_sd, strict=False)

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

    data_loaders = get_data_loaders(
        options.dataset, options.bs,
        split_index=options.split_index
    )

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

    if options.log_tb:
        log_tensorboard(folder, join("runs", options.folder), data_loaders, 3, model)


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


if __name__ == "__main__":
    OPTIONS = get_options()
    run(OPTIONS)
