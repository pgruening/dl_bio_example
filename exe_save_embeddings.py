'''
Save the model's feature representation to npy.
This can be used in `plot_embeddings.py' for visualization the feature space with visdom
'''



import argparse
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from DLBio.helpers import check_mkdir
from DLBio.pytorch_helpers import get_device, load_model_with_opt

import config
from datasets.ds_mnist import get_dataloader
from models.model_getter import load_model_from_opt


SAVE_SMALL = True
MODEL = 'experiments/eval_custom_models_backup/layer5_dim8_seed0'


class EmbeddingModel(nn.Module):
	def __init__(self, model):
		super(EmbeddingModel, self).__init__()
		self.layers = model.layers

	def forward(self,x):
		x = self.layers(x)
		x = x.mean(-1).mean(-1)
		return x


def get_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_path', type=str, default=config.FEATURES_OUT)
	parser.add_argument('--model', type=str, default=MODEL)
	parser.add_argument('--save_small', action='store_true', default=SAVE_SMALL)

	return parser.parse_args()

def run(options):
	device = get_device()

	m = load_model(options.model, device)

	model = EmbeddingModel(m).eval()
	dl = get_dataloader(False, batch_size=config.BS)
	dataset, labels =  get_embeddings(dl, model, device)

	print("embeddings received")

	if options.save_small:
		filename = join(options.out_path, "train_features_short")
		ds = dataset[:1000,...]
		lbs = labels[:1000,...]
		save_np(ds, lbs, filename)

	filename = join(options.out_path, "train_features_full")
	save_np(dataset, labels, filename)


def get_embeddings(dataloader, model, device):
	
	with torch.no_grad():
		features = [model(x.to(device)).cpu().numpy()
				for x,_ in dataloader]
	features = np.concatenate(features, 0)
	with torch.no_grad():
		labels = [y.cpu().numpy() 
				for _, y in dataloader]
	labels = np.concatenate(labels, 0)
	
	return features, labels

def save_np(data, labels, path):
	filename = f'{path}.npy'
	check_mkdir(filename)
	np.save(filename, data)
	print(f'saving to {filename} finished')

	filename = f'{path}_labels.npy'
	check_mkdir(filename)
	np.save(filename, labels)
	print(f'saving to {filename} finished')

def load_model(wrk_dir, device=None, model_path=None):
	if model_path is None:
			model_path = join(wrk_dir, 'model.pt')
	opt = join(wrk_dir, "opt.json")
	if device is None:
		device = get_device()
	model = load_model_with_opt(model_path, options=opt, get_model_fcn=load_model_from_opt, device=device)
	return model



if __name__ == '__main__':
	options = get_options()
	run(options)
