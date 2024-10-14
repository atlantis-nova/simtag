import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
tqdm.pandas()

class prep():

	# TODO : deprecated
	# def compute_T(self, shortest_vector, longest_vector):
	# 	"""
	# 	T is the linear transformation to make any query/sample vector which is smaller than the relationship vectors fit its size.
	# 	"""
		
	# 	index_list = list()
	# 	for k in range(1, shortest_vector+1):
	# 		value = int(k*longest_vector/shortest_vector)
	# 		index_list.append(value)

	# 	index_list.insert(0, 0)
	# 	index_list[-1] = longest_vector
	# 	index_list

	# 	T_indexes = list()
	# 	for k in range(shortest_vector):
	# 		T_indexes.append([index_list[k], index_list[k+1]])

	# 	return T_indexes

	
	def compute_M(self, method=None):
		
		if method == 'encoding':
			df_M = pd.DataFrame(self.tag_list)
			df_M.columns = ['tags']
			df_M['vector_tags'] = df_M['tags'].progress_apply(lambda x : self.model.encode(x).tolist())
			M = np.array(df_M['vector_tags'].tolist())

		elif method == 'co-occurrence':
			# compute co-occurence matrix
			M = list()
			for tag_col in tqdm(self.tag_list, desc="Processing tags"):
				M_col = []
				for tag_row in self.tag_list:
					if tag_row == tag_col:
						M_col.append(1)
					else:
						# get all samples that have this tag
						total_A = len([x for x in self.sample_list if tag_row in x])
						total_B = len([x for x in self.sample_list if tag_col in x])
						total_AB = len([x for x in self.sample_list if (tag_row in x) and (tag_col in x)])
						score = total_AB/(total_A + total_B - total_AB)
						# print(tag_row, tag_col, score)
						M_col.append(score)
				M.append(M_col)
			# raw vectors
			M = np.array(M)
			# dataframe with tags
			df_M = pd.DataFrame([self.tag_list, M]).T
			df_M.columns = ['tags', 'vector_tags']

		# store data
		self.M = M
		self.df_M = df_M
		return self.M, self.df_M
	

	def compute_optimal_components(self, threshold=0.95):
		
		# ex. 0.95 for 95% explained variance
		pca = PCA().fit(self.M.T)
		np.cumsum(pca.explained_variance_ratio_)
		cum_sum = np.cumsum(pca.explained_variance_ratio_)
		optimal_components = np.argmax(cum_sum >= threshold) + 1

		return optimal_components

	
	def index_samples(self, sample_list):

		tag2index = {self.tag_list[index]:index+1 for index in range(len(self.tag_list))}

		indexed_sample_list = [[tag2index[x] for x in sample] for sample in sample_list]
		max_len = max([len(x) for x in sample_list])

		padded_data = []
		for sample in indexed_sample_list:
			sample = [0 if x is None else x for x in sample]  # replace None with 0
			padded_data.append(sample + [0] * (max_len - len(sample)))  # pad with 0 to length 20
			
		indexed_sample_list = np.array(padded_data)
		return tag2index, indexed_sample_list

	
	def load_M(self, df_M, covariate_transformation, quantize_M=False):

		self.df_M = df_M
		self.M = np.array(df_M['vector_tags'].tolist())
		
		if quantize_M == True:
			self.M = quantize_embeddings(
				self.M,
				precision=self.quantization
			)

		# define transformation type on oneshot_tag_vector
		self.n_tags = len(self.tag_list)
		self.M_vector_length = self.M.shape[1]
		self.covariate_transformation = covariate_transformation

		# we might be able to use PCA
		# self.optimal_components = self.compute_optimal_components(threshold=0.95)

		# after we have decided on the transformation type, we set the transformation function
		if self.covariate_transformation == 'PCA':

			try:
				self.pca_vector_length = self.M_vector_length
				self.pca = PCA(n_components=self.pca_vector_length, svd_solver='full')
				self.pca.fit(self.M.T)
				# self.M_adjusted = self.pca.transform(self.M.T) # TODO : no need to change M, we can remove it
			except:
				raise Exception(f"PCA now allowed, n_tags < {self.M_vector_length}")

		elif self.covariate_transformation == 'dot_product':
			pass
		

	def adjust_oneshot_vector(self, onehot_covariate_vector):
		'This function is called each time we perform a one_hot encoding'
		
		# TODO : deprecated
		# if self.adjust_transformation_type == 'expand':
		# 	new_vector = np.zeros(self.T_indexes[-1][1])
		# 	for index, value in enumerate(onehot_covariate_vector):
		# 		for T_index in range(self.T_indexes[index][0], self.T_indexes[index][1]):
		# 			new_vector[T_index] = value

		if self.covariate_transformation == 'dot_product':
			new_vector = onehot_covariate_vector @ self.M
		
		elif self.covariate_transformation == 'PCA':
			new_vector = self.pca.transform(onehot_covariate_vector.reshape(1, len(self.tag_list)))[0]

		return new_vector
