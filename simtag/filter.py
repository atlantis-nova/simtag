from .utils.prep import prep
from .utils.search import search
from .utils.validate import validate

# functions
from sentence_transformers import SentenceTransformer

class simtag_filter(prep, search, validate):
	
	def __init__(self, sample_list=None, tag_list=None, model_name=None, quantization=None):

		# process samples
		if sample_list is not None:
			self.sample_list = sample_list

		if tag_list is not None:
			self.tag_list = tag_list
		else:
			self.tag_list = sorted(list(set([x for xs in self.sample_list for x in xs])))

		if model_name is not None:
			self.model = SentenceTransformer(model_name, device='cpu')

		if quantization is not None:
			self.quantization = 'int8'