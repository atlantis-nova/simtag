from .utils.prep import prep
from .utils.search import search
from .utils.validate import validate
from .utils.clustering import clustering
from .utils.encode import encode
from sklearn.neighbors import NearestNeighbors

# functions
from sentence_transformers import SentenceTransformer

class simtag_filter(prep, encode, clustering, search, validate):
	
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

	
	def encode(self, input):
		"""
		This function output a numpy array as a vector
		"""
		output = self.model.encode(input)
		return output
	

	def compute_index(self, data, k):
		"""
		This function computes HNSW for vector search
		"""
		index = NearestNeighbors(n_neighbors=k, metric='cosine').fit(data)
		return index
	

	def search_index(self, query_vector, index, k=None):
		"""
		This function searches through a complete index using a query vector as input, output a list of indices
		"""
		distances, indices = index.kneighbors([query_vector])
		indices = indices[0].tolist()
		return indices