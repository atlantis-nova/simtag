import numpy as np
from tqdm import tqdm

class encode():

	def quantize(self, arr):
		# scale the array to the range of int8 (-127 to 127)
		scaled_arr = (arr - arr.min()) / (arr.max() - arr.min()) * 254 - 127
		rounded_arr = np.round(scaled_arr)
		quantized_arr = rounded_arr.astype(np.int8)
		return quantized_arr


	def encode_samples(self, sample_list, quantize_samples=False, show_progress=True):

		def encode_sample(list_tags):

			def find_closest_index(tag):
				if self.tag_pointers is not None:
					index = self.tag_pointers[tag]
				else:
					index = self.tag_list.index(tag)	
				return index
			
			# assign one_hot index to each tag
			vector_length = self.M.shape[0]
			onehot_covariate_vector = np.zeros(vector_length)

			for tag in list_tags:
				index = find_closest_index(tag)
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

		# we convert list to numpy array
		return np.array(samples_encoded)
		

	def encode_query(self, list_tags=None, dict_tags=None, allow_new_tags=False, print_new_tags=False, skip_adjust=False):

		def find_closest_index(tag, allow_new_tags, tag_pointers=None):

			if self.tag_pointers is not None and tag in self.tag_pointers:
				index = tag_pointers[tag]
			elif tag in self.tag_list:
				index = self.tag_list.index(tag)
			else:
				if allow_new_tags:
					# for each non-existing tag, find the closest one
					# _, index = self.index_tags.kneighbors([self.encode(tag)]) # TODO: obsolete
					index = self.search_index(self.encode(tag), self.index_tags, k=1)
					index = int(index[0])

					if print_new_tags:
						if self.tag_pointers is not None:
							cluster_tags = self.list_cluster_tags(index)
							print(tag, '->', cluster_tags)
						else:
							print(tag, '->', self.tag_list[index])
							

				else:
					raise Exception('input tag is not in list')
			return index
		
		# assign one_hot index to each tag
		vector_length = self.M.shape[0]
		onehot_covariate_vector = np.zeros(vector_length)
		tags_index = list()

		if list_tags is not None:
			for tag in list_tags:
				index = find_closest_index(tag, allow_new_tags, self.tag_pointers)
				onehot_covariate_vector[index] = 1
				tags_index.append(index)

		elif dict_tags is not None:
			for tag in [*dict_tags.keys()]:
				index = find_closest_index(tag, allow_new_tags, self.tag_pointers)
				onehot_covariate_vector[index] = 1 * dict_tags[tag]
				tags_index.append(index)

		# adjust vector: compress or expand
		if skip_adjust == False:
			onehot_covariate_vector = self.adjust_oneshot_vector(onehot_covariate_vector)
		
		return onehot_covariate_vector