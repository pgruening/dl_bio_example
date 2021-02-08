from torchvision.models import resnet18, vgg16_bn, alexnet
import torch.nn as nn

from models.costum_archs import CustomNet


def get_model(model_type, input_dim, output_dim, device, **kwargs):
	if model_type == 'resnet18':
		assert input_dim == 3
		model = resnet18(pretrained=True)
		in_dim = model.fc.in_features
		model.fc = nn.Linear(in_dim, output_dim)
		return model.to(device)

	elif model_type == 'vgg16':
		assert kwargs['crop_size'] == 224
		assert input_dim == 3
		model = vgg16_bn(pretrained=True)
		in_dim = model.classifier[-1].in_features
		model.model.classifier[-1] = nn.Linear(in_dim, output_dim)
		return model.to(device)

	elif model_type == 'alexnet':
		assert kwargs['crop_size'] == 256
		assert input_dim == 3
		model = alexnet(pretrained=True)
		in_dim = model.classifier[-1].in_features
		model.model.classifier[-1] = nn.Linear(in_dim, output_dim)
		return model.to(device)

	elif model_type == 'custom_net':
		model = CustomNet(input_dim, output_dim, **kwargs)
		load_pretrained = True
		if load_pretrained:
			weights = get_weights(model_type)
			model.load_state_dict(weights)

		return model.to(device)
	


def get_weights(model_name):
	from os.path import join, exists
	import config
	import torch
	from torchvision.datasets.utils import download_file_from_google_drive
	
	pretrained_path = join(config.DATA_FOLDER, 'pretrained_model')	
	assert model_name in config.WEIGHT_IDS, f'Unkown model: {model_name}'

	file_name = f'{model_name}.pt'
	file_dest = join(pretrained_path, file_name)

	if not exists(file_dest):
		id = config.WEIGHT_IDS[model_name]
		download_file_from_google_drive(file_id=id, root=pretrained_path, filename=file_name)
		print(f'weights downloaded & saved at: {file_dest}')
		
	return torch.load(file_dest) 
