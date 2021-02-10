from torchvision.models import resnet18, vgg16_bn, alexnet
import torch.nn as nn

from models.costum_archs import CustomNet


def get_model(model_type, input_dim, output_dim, device, pretrained, **kwargs):
	if model_type == 'resnet18':
		assert input_dim == 3
		model = resnet18(pretrained=pretrained)
		in_dim = model.fc.in_features
		model.fc = nn.Linear(in_dim, output_dim)
		return model.to(device)

	elif model_type == 'vgg16':
		assert kwargs['crop_size'] == 224
		assert input_dim == 3
		model = vgg16_bn(pretrained=pretrained)
		in_dim = model.classifier[-1].in_features
		model.model.classifier[-1] = nn.Linear(in_dim, output_dim)
		return model.to(device)

	elif model_type == 'alexnet':
		assert kwargs['crop_size'] == 256
		assert input_dim == 3
		model = alexnet(pretrained=pretrained)
		in_dim = model.classifier[-1].in_features
		model.model.classifier[-1] = nn.Linear(in_dim, output_dim)
		return model.to(device)

	elif model_type == 'custom_net':
		model = CustomNet(input_dim, output_dim, **kwargs)

		
		layer = int(kwargs.get('num_layer', [5])[0])		
		dim = int(kwargs.get('init_dim', [8])[0])

		model_name = f'custom_net_layer{layer}_dim{dim}'
		
		if pretrained:
			weights = get_weights(model_name)
			model.load_state_dict(weights)

		return model.to(device)
	


def get_weights(model_name):
	"""
	get pretrained weights or download them from google drive first 

	usage of drive links to download files:
		share the corrsponding document such that anyone with the link can see it
		copy the link and use only the file id for
			download_file_from_google_drive from torchvision.datasets.utils
			https://github.com/pytorch/vision/blob/1b7c0f54e2913e159394a19ac5e50daa69c142c7/torchvision/datasets/utils.py#L169

		example:
			link: https://drive.google.com/file/d/14M3uC29aAx2AMeCeidLQjqjkVpGqnb6k/view?usp=sharing
			with id = 14M3uC29aAx2AMeCeidLQjqjkVpGqnb6k


	Parameters
	----------
	model_name : str
		name of file_id saved in config.WEIGHT_IDS

	Returns
	-------
	state_dict
		state_dict containing pretrained weights
	"""	
	from os.path import join, exists
	import config
	import torch
	from torchvision.datasets.utils import download_file_from_google_drive

	
	pretrained_path = join(config.DATA_FOLDER, 'pretrained_models')	
	assert model_name in config.WEIGHT_IDS, f'no weights for model: {model_name}'

	file_name = f'{model_name}.pt'
	file_dest = join(pretrained_path, file_name)

	if not exists(file_dest):
		id = config.WEIGHT_IDS[model_name]
		download_file_from_google_drive(file_id=id, root=pretrained_path, filename=file_name)
		print(f'weights downloaded & saved at: {file_dest}')
		
	return torch.load(file_dest) 

	
