from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
from functools import reduce

class search():


	def quantize(self, arr):
		# scale the array to the range of int8 (-127 to 127)
		scaled_arr = (arr - arr.min()) / (arr.max() - arr.min()) * 254 - 127
		rounded_arr = np.round(scaled_arr)
		quantized_arr = rounded_arr.astype(np.int8)
		return quantized_arr


	def encode_samples(self, sample_list, quantize_samples=False, show_progress=True):

		def encode_sample(list_tags):
			
			# assign one_hot index to each tag
			vector_length = len(self.tag_list)
			onehot_covariate_vector = np.zeros(vector_length)
			
			indexes = [self.tag_list.index(x) for x in list_tags]
			for index in indexes:
				onehot_covariate_vector[index] = 1

			# adjust vector
			onehot_covariate_vector = self.adjust_oneshot_vector(onehot_covariate_vector)
			
			return onehot_covariate_vector
		
		if show_progress==True:
			disable_progress=False
		elif show_progress==False:
			disable_progress=True
		
		samples_encoded = list()
		for sample_encoded in tqdm(sample_list, desc="processing samples", disable=disable_progress):
			samples_encoded.append(encode_sample(sample_encoded))

		if quantize_samples:
			samples_encoded = [self.quantize(x) for x in samples_encoded]

		return samples_encoded
	

	def encode_query(self, list_tags=None, dict_tags=None, allow_new_tags=False, print_new_tags=False):

		def find_closest_index(tag, allow_new_tags):

			if tag in self.tag_list:
				index = self.tag_list.index(tag)
			else:
				if allow_new_tags:
					# for each non-existing tag, find the closest one
					_, index = self.nbrs_tags.kneighbors([self.model.encode(tag)])
					index = int(index[0][0])

					if print_new_tags:
						print(tag, '->', self.tag_list[index])

				else:
					raise Exception('input tag is not in list')
			return index
		
		# assign one_hot index to each tag
		vector_length = len(self.tag_list)
		onehot_covariate_vector = np.zeros(vector_length)
		tags_index = list()

		if list_tags is not None:
			for tag in list_tags:
				index = find_closest_index(tag, allow_new_tags)
				onehot_covariate_vector[index] = 1
				tags_index.append(index)

		elif dict_tags is not None:
			for tag in [*dict_tags.keys()]:
				index = find_closest_index(tag, allow_new_tags)
				onehot_covariate_vector[index] = 1 * dict_tags[tag]
				tags_index.append(index)

		# adjust vector: compress or expand
		onehot_covariate_vector = self.adjust_oneshot_vector(onehot_covariate_vector)
		
		# TODO : this operation is obsolete, we do not need to maintain the full relationship matrix
		# M_product = self.M + onehot_covariate_vector
		# M_mean = np.mean(M_product, axis=0) # column average
		
		return onehot_covariate_vector


	def show_similar(self, tag):

		# shows top-k neighbors tags
		return self.df_M[[tag]].sort_values(tag, ascending=False)[0:10]
	

	def compute_nbrs(self, vector_list, k):
	
		# compute nbrs_tags, necessary to find tags that are not present in our list using semantic similarity
		# we use df_M, because M might have been expaned or compressed: the knn is on self.M is NOT VALID
		nbrs_tags = NearestNeighbors(n_neighbors=1, metric='cosine').fit(self.df_M['vector_tags'].tolist())
		self.nbrs_tags = nbrs_tags

		nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(vector_list)

		return nbrs
		

	def hard_tag_filtering(self, tag2index, indexed_sample_list, query_tag_list, search_type):

		target_values = [tag2index[x] for x in query_tag_list]

		# Apply the filter using NumPy's vectorized operations
		if search_type == 'AND':
			mask = reduce(np.logical_and, (np.isin(indexed_sample_list, val).any(axis=1) for val in target_values))

		elif search_type == 'OR':
			mask = reduce(np.logical_or, (np.isin(indexed_sample_list, val).any(axis=1) for val in target_values))

		indices = np.where(mask)[0]
		return indices


	def jaccard_tag_filtering(self, sample_list, query_tag_list):

		def jaccard_similarity(list1, list2):
			set1 = set(list1)
			set2 = set(list2)
			intersection = set1.intersection(set2)
			union = set1.union(set2)
			return len(intersection) / len(union)

		def search_most_similar(sample_list, query_tag_list):
			similarity_samples = list()
			for index in range(len(sample_list)):
				sample = sample_list[index]
				similarity = jaccard_similarity(query_tag_list, sample)
				similarity_samples.append([similarity, index, sample])
			return similarity_samples

		similarity_samples = search_most_similar(sample_list, query_tag_list)
		similarity_samples = [x for x in similarity_samples if x[0] > 0]

		indices = [x[1] for x in similarity_samples]
		search_results = [x[2] for x in sorted(similarity_samples)[::-1]]

		return indices, search_results


	def soft_tag_filtering(self, nbrs, sample_list, query_vector):

		distances, indices = nbrs.kneighbors([query_vector])
		indices = indices[0].tolist()

		search_results = [sample_list[x] for x in indices]
		return indices, search_results