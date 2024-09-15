from .utils.prep import prep
from .utils.search import search
from .utils.validate import validate

# functions
from sentence_transformers import SentenceTransformer

class simtag_filter(prep, search, validate):
	
	def __init__(self, covariate_vector_length, sample_list=None, tag_list=None, model_name=None):
		
		# vector lengths
		# TODO : as of now covariate_vector_length must equal the vector size input with df_M
		self.covariate_vector_length = covariate_vector_length
		
		# process samples
		if sample_list is not None:
			self.sample_list = sample_list

		if tag_list is not None:
			self.tag_list = tag_list
		else:
			self.tag_list = sorted(list(set([x for xs in self.sample_list for x in xs])))

		if model_name is not None:
			self.model = SentenceTransformer(model_name, device='cpu')

		# define transformation type on oneshot_tag_vector
		# we compress/expand the one_shot tag list into a covariate vector of arbitrary size
		if len(self.tag_list) < self.covariate_vector_length:
			# if # tags < desired length, we expand the vector
			self.adjust_transformation_type = 'expand'
			
		elif len(self.tag_list) > self.covariate_vector_length:
			# if # tags > desired length, we compress the vector
			self.adjust_transformation_type = 'compress'

		elif len(self.tag_list) == self.covariate_vector_length:
			self.adjust_transformation_type = None