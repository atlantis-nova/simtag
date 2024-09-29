import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
tqdm.pandas()

class prep():

	def compute_T(self, shortest_vector, longest_vector):
		"""
		T is the linear transformation to make any query/sample vector which is smaller than the relationship vectors fit its size.
		"""
		
		index_list = list()
		for k in range(1, shortest_vector+1):
			value = int(k*longest_vector/shortest_vector)
			index_list.append(value)

		index_list.insert(0, 0)
		index_list[-1] = longest_vector
		index_list

		T_indexes = list()
		for k in range(shortest_vector):
			T_indexes.append([index_list[k], index_list[k+1]])

		return T_indexes

	
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

	
	def load_M(self, df_M):

		self.df_M = df_M
		self.M = np.array(df_M['vector_tags'].tolist())
		# we either compress or expand M
		self.compute_adjusting_transformation()


	def compute_adjusting_transformation(self):

		self.T_indexes = None
		self.pca_vector_length = None

		if self.adjust_transformation_type == 'expand':
			self.T_indexes = self.compute_T(len(self.tag_list), self.covariate_vector_length)
			
		elif self.adjust_transformation_type == 'compress':
			self.pca_vector_length = self.covariate_vector_length
			self.pca = PCA(n_components=self.pca_vector_length)
			self.pca.fit(self.M.T)
			self.M = self.pca.transform(self.M.T)

		elif self.adjust_oneshot_vector is None:
			pass


	def adjust_oneshot_vector(self, onehot_covariate_vector):
		
		if self.adjust_transformation_type == 'expand':
			new_vector = np.zeros(self.T_indexes[-1][1])
			for index, value in enumerate(onehot_covariate_vector):
				for T_index in range(self.T_indexes[index][0], self.T_indexes[index][1]):
					new_vector[T_index] = value

		elif self.adjust_transformation_type == 'compress':
			new_vector = self.pca.transform(onehot_covariate_vector.reshape(1, len(self.tag_list)))

		elif self.adjust_transformation_type is None:
			new_vector = onehot_covariate_vector

		return new_vector
