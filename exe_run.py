from copy import copy, deepcopy
from os.path import join

import config
from DLBio import pt_run_parallel

DEFAULT_TYPE = 'lr_search'
AVAILABLE_GPUS = [0]

class TrainingProcess(pt_run_parallel.ITrainingProcess):
	def __init__(self, **kwargs):
		super(TrainingProcess, self).__init__()
		self.module_name = 'run_training.py'
		self.kwargs = kwargs
		self.__name__ = kwargs['folder']


def run():
	available_gpus = [int(x) for x in AVAILABLE_GPUS]
	
	param_generator = get_param_generator(DEFAULT_TYPE)
	print(param_generator)

	make_object = pt_run_parallel.MakeObject(TrainingProcess)
	pt_run_parallel.run(param_generator(), make_object,
							available_gpus=available_gpus,
							shuffle_params=True
							)




def get_param_generator(name):
	if name == LR_SEARCH.name:
			return LR_SEARCH()
	
	raise ValueError(f'unkown input: {name}')

	

class LR_SEARCH():
	name = 'lr_search'

	def __init__(self):
		self.default_values = config.MNIST_PARAMS
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




if __name__ == '__main__':
	run()