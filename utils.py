import json
from os.path import join
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter

def log_tensorboard(workdir, tb_out, dataloaders, numBatches=1, model=None, to_uint_8=True):
	""" write data for visualization by tensorboard

	Start TensorBoard with 
		%tensorboard --logdir tb_out

	Parameters
	----------
	workdir : str
		the working directory which should contain a 'log.json' file
		where the training values are stored
	tb_out : str
		out_path to store the tensorboard data in
	dataloaders : dict
		the dataloaders used in training for visualization of some image_batches
	numBatches : int, optional
		number of image batches to display, by default 1
	model : pytorch model, optional
		the model used in trainig, will be displayed as graph in TensorBoard, by default None
	to_uint_8 : bool, optional
		convert images to uint8 to display them, by default True
	"""	

	with open(join(workdir, 'log.json'), 'r') as file:
		log = json.load(file)


	try:	
		tb_writer = SummaryWriter(tb_out) 

		for key, values in log.items():
			[tb_writer.add_scalar(key, v, ep) for ep,v in enumerate(values)]	
		
		for key, dl in dataloaders.items():
			if dl is not None:
				di = iter(dl)
				for i in range(numBatches):
					images, _ = di.next()
					grid = torchvision.utils.make_grid(images)	
					if to_uint_8:
						grid -= grid.min()
						grid /= grid.max()
						grid *= 255
						grid = np.array(grid.detach().cpu()).astype('uint8')

					tb_writer.add_image(f"{key}_batch{i}", grid)
			
		# load model if needed
		if model is not None:		
			tb_writer.add_graph(model, images.cuda())
	
		print(f'finished writing to {tb_out}')	
	except Exception as e:
		print("Exception thrown while logging to SummaryWriter:")
		print(e.__class__)
		print(e)
	finally:
		tb_writer.flush()
		tb_writer.close()




