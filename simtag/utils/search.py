from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

class search():

	def encode_samples(self, sample_list):

		row_list = list()
		for row_index in tqdm(range(len(sample_list)), desc="processing samples"):
			indexes = [x for x in range(len(self.M.columns)) if self.M.columns[x] in sample_list[row_index]]
			one_hot = np.zeros((1, len(self.tag_list)))[0]
			for k in indexes:
				one_hot[int(k)] = 1
			row_list.append(one_hot)

		return row_list
	

	def encode_query(self, query_tag_list=None, query_tag_dict=None, negative_score=False, j=5):

		def encode_tag(tag, j):
			arr = np.array([x[0] for x in self.M[[tag]].values.tolist()])
			# create a new array with the top 5 values, and 0 for the rest
			top_5_indices = np.argsort(arr)[-j:]
			vector = np.zeros_like(arr)
			vector[top_5_indices] = arr[top_5_indices]
			return vector

		if query_tag_list is not None:

			# compute all vectors
			tags_vectors = list()
			for tag in query_tag_list:
				vector = encode_tag(tag, j)
				tags_vectors.append(vector)
		
		if query_tag_dict is not None:
			
			# compute all vectors
			tags_vectors = list()
			for tag in [*query_tag_dict.keys()]:
				vector = encode_tag(tag, j)
				tags_vectors.append(vector * query_tag_dict[tag])
				
		# sum everything into a single vector
		vector = sum(tags_vectors)
		if negative_score:
			vector = np.array([(x, -1)[x==0] for x in vector])
		# set minimum and maximum
		vector = np.array([(x, 1)[x>1] for x in vector])
		vector = np.array([(x, -1)[x<-1] for x in vector])
		return vector


	def show_similar(self, tag):

		# shows top-k neighbors tags
		return self.M[[tag]].sort_values(tag, ascending=False)[0:10]
	

	def compute_nbrs(self, vector_list, k):

		nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(vector_list)
		return nbrs
	

	def hard_tag_filtering(self, sample_list, query_tag_list):

		indices = [index for index in range(len(sample_list)) if all(x in sample_list[index] for x in query_tag_list)]
		search_results = [sample_list[x] for x in indices]

		return indices, search_results


	def soft_tag_filtering(self, nbrs, sample_list, query_vector):

		distances, indices = nbrs.kneighbors([query_vector])
		indices = indices[0].tolist()

		search_results = [sample_list[x] for x in indices]
		return indices, search_results