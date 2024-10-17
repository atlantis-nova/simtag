import numpy as np
from tqdm import tqdm
from functools import reduce

class search():


	def show_similar(self, tag):

		# shows top-k neighbors tags
		return self.df_M[[tag]].sort_values(tag, ascending=False)[0:10]
	

	def compute_search_indexes(self, vector_list, k):
	
		# compute index_tags, necessary to find tags that are not present in our list using semantic similarity
		# we use df_M, because M might have been expaned or compressed: the knn is on self.M is NOT VALID
		# self.index_tags = NearestNeighbors(n_neighbors=1, metric='cosine').fit(self.df_M['vector_tags'].tolist())
		self.index_tags = self.compute_index(data=self.df_M['vector_tags'].tolist(), k=1)

		# customized nbrs used for search
		# index_covariate = NearestNeighbors(n_neighbors=k, metric='cosine').fit(vector_list)
		index_covariate = self.compute_index(data=vector_list, k=k)
		return index_covariate
		

	def classic_search(self, tag2index, indexed_sample_list, query_tag_list, search_type):

		target_values = [tag2index[x] for x in query_tag_list]

		# Apply the filter using NumPy's vectorized operations
		if search_type == 'AND':
			mask = reduce(np.logical_and, (np.isin(indexed_sample_list, val).any(axis=1) for val in target_values))

		elif search_type == 'OR':
			mask = reduce(np.logical_or, (np.isin(indexed_sample_list, val).any(axis=1) for val in target_values))

		indices = np.where(mask)[0]
		return indices


	def covariate_search(self, index_covariate=None, sample_list=None, query_tag_dict=None, query_tag_list=None, allow_new_tags=False, print_new_tags=False, skip_adjust=False, k=None):
	
		if query_tag_list is not None:
			query_vector = self.encode_query(list_tags=query_tag_list, allow_new_tags=allow_new_tags, print_new_tags=print_new_tags, skip_adjust=skip_adjust)
		elif query_tag_dict is not None:
			query_vector = self.encode_query(dict_tags=query_tag_dict, allow_new_tags=allow_new_tags, print_new_tags=print_new_tags, skip_adjust=skip_adjust)

		# distances, indices = index_covariate.kneighbors([query_vector])
		# indices = indices[0].tolist()

		indices = self.search_index(query_vector, index_covariate, k=k)
		search_results = [sample_list[x] for x in indices]

		return indices, search_results
	

	def semantic_covariate_search(self, index_covariate=None, sample_list=None, query=None, k=None):
	
		# distances, indices = index_covariate.kneighbors([self.encode(query)])
		# indices = indices[0].tolist()

		if self.covariate_transformation != 'dot_product':
			raise BaseException('semantic-covariate search is not compatible with PCA transformation')

		indices = self.search_index(self.encode(query), index_covariate, k=k)
		search_results = [sample_list[x] for x in indices][0:k]

		return indices, search_results