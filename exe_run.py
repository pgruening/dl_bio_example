from copy import deepcopy
from os.path import join

import argparse
from DLBio import pt_run_parallel
from DLBio.kwargs_translator import to_kwargs_str

DEFAULT_TYPE = 'eval_custom_models'
AVAILABLE_GPUS = [0]




def get_options():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', type=str, default=DEFAULT_TYPE)

	return parser.parse_args()


class TrainingProcess(pt_run_parallel.ITrainingProcess):
	def __init__(self, **kwargs):
		super(TrainingProcess, self).__init__()
		self.module_name = 'run_training.py'
		self.kwargs = kwargs
		self.__name__ = kwargs['folder']


def run():
	available_gpus = [int(x) for x in AVAILABLE_GPUS]

	options = get_options()
	
	param_generator = get_param_generator(options.mode)
	print(param_generator)

	make_object = pt_run_parallel.MakeObject(TrainingProcess)
	pt_run_parallel.run(param_generator(), make_object,
							available_gpus=available_gpus,
							shuffle_params=True
							)




def get_param_generator(name):
	if name == LR_SEARCH.name:
			return LR_SEARCH()
	elif name == EVAL_CUSTOM_MODELS.name:
		return EVAL_CUSTOM_MODELS()
	
	raise ValueError(f'unkown input: {name}')


class LR_SEARCH():
	name = 'lr_search'

	def __init__(self):
		self.default_values = PARAMS.MNIST_PARAMS
		self.default_values.update ({
			'sv_int': -1,
			'lr' : [0.1, 0.01, 0.001, 0.0001, 0.00001]
		})
		self.default_folder = LR_SEARCH.name

	def __call__(self):
		comment = "training with different learning rates"
		learn_rates = self.default_values['lr']
		
		for lr in learn_rates:
			out = deepcopy(self.default_values)
			out['lr'] = lr
			out['folder'] = join(self.default_folder, f'lr_{lr}/')

			yield out

	
class EVAL_CUSTOM_MODELS():
	name = 'eval_custom_models'
	
	def __init__(self):		
		self.default_values = PARAMS.MNIST_PARAMS
		self.default_values.update({
			'model_type' : 'custom_net',
			'sv_int' : [-1]

		})
		self.default_folder = EVAL_CUSTOM_MODELS.name

	def __call__(self):

		num_layers = [3,5,7]
		init_dims = [4,8]
		k = 3
		seeds = [0,23,42]


		for seed in seeds:
			for num_layer in num_layers:
				for init_dim in init_dims:				
					model_kwargs = to_kwargs_str({
						"num_layer" : [num_layer],
						"init_dim" : [init_dim],
						"kernel_size" : [k]						
					})

					out = deepcopy(self.default_values)
					out['seed'] = seed
					out['model_kw'] = model_kwargs
					out['folder'] = join(self.default_folder, f'layer{num_layer}_dim{init_dim}_seed{seed}')

					yield out


class PARAMS():
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
		'model_type' : 'custom_net'
	}
		
				

		







if __name__ == '__main__':
	run()