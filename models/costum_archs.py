import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
	def __init__(self, in_dim, out_dim, **kwargs):
		super(CustomNet, self).__init__()
		self.out_dim = out_dim
		self.in_dim = in_dim

		self.num_layers = int(kwargs.get('num_layer', [5])[0])

		self.conv_layers = nn.ModuleList()
		self.batch_norms = nn.ModuleList()


		kernel_size = int(kwargs.get('kernel_size', [3])[0])
		
		in_channels = int(kwargs.get('init_dim', [8])[0])
		self.layers = nn.Sequential(
			nn.Conv2d(in_channels=in_dim, out_channels=in_channels, kernel_size=kernel_size, bias=False),
			nn.BatchNorm2d(in_channels)
		)
		

		for i in range(1, self.num_layers):			
			self.layers.add_module(f"conv_{i}", nn.Conv2d(in_channels, 2*in_channels, kernel_size=kernel_size, bias=False))
			self.layers.add_module(f"batch_norm_{i}", nn.BatchNorm2d(2*in_channels))
			self.layers.add_module(f"reLU{i}", nn.ReLU(inplace=True))

			in_channels = 2*in_channels

		self.out = nn.Linear(in_channels, self.out_dim)


	def forward(self, x):
		x = self.layers(x)
		x = x.mean(-1).mean(-1)
		x = self.out(x)

		return x
