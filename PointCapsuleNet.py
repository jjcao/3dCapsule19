import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def squash(s, dim=-1):
	'''
	"Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
	Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
	
	Args:
		s: 	Vector before activation
		dim:	Dimension along which to calculate the norm
	
	Returns:
		Squashed vector
	'''
	squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
	return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)

class PrimaryPointCapsules(nn.Module):
    def __init__(self, in_channels=128, num_caps=1024, dim_caps=16):
        """
        Initialize the layer.

        Args:
            dim_caps:		Dimensionality, i.e. length, of the output capsule vector. 16

        """        
        super(PrimaryPointCapsules, self).__init__()    
        self.in_channels = in_channels
        self.num_caps = num_caps
        self.dim_caps = dim_caps

        self.conv = nn.ModuleList([nn.Conv1d(self.in_channels, self.num_caps, 1) for i in range(0,self.dim_caps)])		
        self.bn = nn.ModuleList([nn.BatchNorm1d(self.num_caps) for i in range(0,self.dim_caps)])

    def forward(self, x):
        '''
        x(B,128,N)
        '''
        outs = []
        for i in range(0,self.dim_caps):   
            y = F.relu(self.bn[i](self.conv[i](x))) # B,1024,N
            y,_ = torch.max(y, 2) # B,1024
            outs.append(y)

        out = torch.cat(outs,1).contiguous() # B, 16*1024
        out = out.view(x.size(0), self.dim_caps, self.num_caps) # B, 16, 1024

        return squash(out, 1)

class RoutingCapsules(nn.Module):
	def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing, device: torch.device):
		"""
		Initialize the layer.

		Args:
			in_dim: 		Dimensionality (i.e. length) of each capsule vector. 8
			in_caps: 		Number of input capsules if digits layer. 1152
			num_caps: 		Number of capsules in the capsule layer. 10
			dim_caps: 		Dimensionality, i.e. length, of the output capsule vector. 16
			num_routing:	Number of iterations during routing algorithm. 3
		"""
		super(RoutingCapsules, self).__init__()
		self.in_dim = in_dim
		self.in_caps = in_caps
		self.num_caps = num_caps
		self.dim_caps = dim_caps
		self.num_routing = num_routing
		self.device = device

		self.W = nn.Parameter( 0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim ) )
	
	def __repr__(self):
		tab = '  '
		line = '\n'
		next = ' -> '
		res = self.__class__.__name__ + '('
		res = res + line + tab + '(' + str(0) + '): ' + 'CapsuleLinear('
		res = res + str(self.in_dim) + ', ' + str(self.dim_caps) + ')'
		res = res + line + tab + '(' + str(1) + '): ' + 'Routing('
		res = res + 'num_routing=' + str(self.num_routing) + ')'
		res = res + line + ')'
		return res

	def forward(self, x):
		batch_size = x.size(0)
		# (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
		x = x.unsqueeze(1).unsqueeze(4)
		#
		# W @ x =
		# (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
		# (batch_size, num_caps, in_caps, dim_caps, 1)
		u_hat = torch.matmul(self.W, x)
		# (batch_size, num_caps, in_caps, dim_caps)
		u_hat = u_hat.squeeze(-1)
		# detach u_hat during routing iterations to prevent gradients from flowing
		temp_u_hat = u_hat.detach()

		'''
		Procedure 1: Routing algorithm
		'''
		b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).to(self.device)

		for route_iter in range(self.num_routing-1):
			# (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
			c = F.softmax(b, dim=1)

			# element-wise multiplication
			# (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
			# (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
			# (batch_size, num_caps, dim_caps)
			s = (c * temp_u_hat).sum(dim=2)
			# apply "squashing" non-linearity along dim_caps
			v = squash(s)
			# dot product agreement between the current output vj and the prediction uj|i
			# (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
			# -> (batch_size, num_caps, in_caps, 1)
			uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
			b += uv
		
		# last iteration is done on the original u_hat, without the routing weights update
		c = F.softmax(b, dim=1)
		s = (c * u_hat).sum(dim=2)
		# apply "squashing" non-linearity along dim_caps
		v = squash(s)

		return v


class PointCapsuleNet(nn.Module):
    def __init__(self, primary_dim = 16, primary_num = 1024):
        super(PointCapsuleNet, self).__init__()
        self.primary_dim = primary_dim
        self.primary_num = primary_num
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)        
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        

if __name__ == '__main__':    
    N = 200 # num_points
    pts_feat = torch.rand(2,128,N) #

    ppc = PrimaryPointCapsules()
    out = ppc(pts_feat)

    tmp = out[0,:,0] 
    torch.sum(tmp**2)
    print('test')
    