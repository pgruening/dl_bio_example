
import argparse
from os.path import basename, dirname, join

from DLBio.kwargs_translator import get_kwargs
from DLBio.pytorch_helpers import get_device, load_model_with_opt

from utils import log_tensorboard

import config
from datasets.data_getter import get_data_loaders
from models.model_getter import get_model

FILE_PATH="experiments/_debug/log.json"


def get_options():
	
	parser = argparse.ArgumentParser()

	parser.add_argument("-log_file", type=str, default=FILE_PATH)
	parser.add_argument("-add_model", action="store_true")
	parser.add_argument('-model_path', type=str, default=None)
	parser.add_argument("-tb", type=str, default=None)

	return parser.parse_args()


def run(options):
	workdir = dirname(options.log_file)
	
	tb_out = options.tb
	if options.tb is None:
		tb_out = join('runs', basename(workdir))

	data_loaders = get_data_loaders('mnist', config.BS,
		split_index=0
	)

	model = None
	if options.add_model:
		model = load_model(workdir, options.model_path)		

	log_tensorboard(workdir, tb_out, data_loaders, 3, model)


def load_model(wrk_dir, model_path=None):
	if model_path is None:
			model_path = join(wrk_dir, 'model.pt')
	opt = join(wrk_dir, "opt.json")
	device = get_device()
	model = load_model_with_opt(model_path, options=opt, get_model_fcn=load_model_from_opt, device=device)
	return model


def load_model_from_opt(options, device):
	"""wrapper for get_model to work with only options and device as parameters
	needed for dlbio's 'load_model_with_opt'

	Returns
	-------
	pytorch model
		the pytorch model from get_model
	"""	
	mt = options.model_type
	in_dim = options.in_dim
	out_dim = options.out_dim
	model_kw = get_kwargs(options.model_kw)


	if hasattr(options,"model_kw"):
		kw = options.model_kw
		model_kw = get_kwargs(kw)
	else:
		model_kw = dict()

	return get_model(mt, in_dim, out_dim, device, False, **model_kw)


if __name__ == "__main__":
	OPTIONS = get_options()
	run(OPTIONS)
