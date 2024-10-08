from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

class search():

	def encode_samples(self, sample_list):

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
		
		row_list = list()
		for sample in tqdm(sample_list, desc="processing samples"):
			row_list.append(encode_sample(sample))

		return row_list
	

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

		# adjust vector
		onehot_covariate_vector = self.adjust_oneshot_vector(onehot_covariate_vector)
		
		# TODO : this operation is obsolete, we do not need to maintain the full relationship matrix
		# M_product = self.M + onehot_covariate_vector
		# M_mean = np.mean(M_product, axis=0) # column average

		M_mean = self.M_mean + onehot_covariate_vector
		
		return M_mean


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
		

	def hard_tag_filtering(self, sample_list, query_tag_list):

		indices = [index for index in range(len(sample_list)) if all(x in sample_list[index] for x in query_tag_list)]
		search_results = [sample_list[x] for x in indices]

		return indices, search_results


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