import argparse
import base64 as b64
from io import BytesIO

import numpy as np
import visdom
from PIL import Image

from datasets.ds_mnist import get_dataloader

type = 'short' 
features_path = f'data/mnist_embeddings/embed_train_{type}.npy'
labels_path = f'data/mnist_embeddings/embed_train_{type}_labels.npy'


def get_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--features', type=str, default=features_path)
	parser.add_argument('--labels', type=str, default=labels_path)
	parser.add_argument('--images', type=str, default=None)

	return parser.parse_args()

def run(options):
	features = np.load(options.features)
	labels = np.load(options.labels)

	if options.images is None:
		dl = get_dataloader(False, batch_size=16)
		images = dl.dataset.data.numpy()
	else:
		images = np.load(options.images)

	labels_cvt =[int(y) for y in labels]


	global images_enc
	images_enc = encode_images(images)


	vis = visdom.Visdom()
	vis.embeddings(features, labels_cvt, data_getter=get_images, data_type='html')


	
	input('Waiting for callbacks, press enter to quit.')


def encode_images(images):
	image_datas = []
	for img in images:
		im = Image.fromarray(img)
		buf = BytesIO()
		im.save(buf, format='PNG')
		b64encoded = b64.b64encode(buf.getvalue()).decode('utf-8')
		image_datas.append(b64encoded)
	return image_datas

def get_images(id):
	image_data = images_enc[id]
	display_data = 'data:image/png;base64,' + image_data
	return "<img src='" + display_data + "' />"

if __name__ == '__main__':
	options = get_options()
	run(options)




