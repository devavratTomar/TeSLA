
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head
		self.head.fc.weight = nn.parameter.Parameter(F.normalize(self.head.fc.weight.data, dim=-1))

	def forward(self, x, return_feats=False):
		feats = self.ext(x)
		norm_feats = F.normalize(feats, dim=-1)
		scores = F.softmax(self.head(norm_feats) / 0.05, dim=1)
		
		if return_feats:
			return scores, feats

		return scores

