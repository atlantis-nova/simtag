import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans
from collections import Counter
from sentence_transformers.quantization import quantize_embeddings
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.*")

class prep():
	
	def compute_optimal_components(self, threshold=0.95):
		
		# ex. 0.95 for 95% explained variance
		pca = PCA().fit(self.M.T)
		np.cumsum(pca.explained_variance_ratio_)
		cum_sum = np.cumsum(pca.explained_variance_ratio_)
		optimal_components = np.argmax(cum_sum >= threshold) + 1

		return optimal_components


	def compute_optimal_M(self, percentile_threshold=95, n_clusters=1000, quantize_M=False, visualize=False, verbose=False):
		"""
		Compress the one_hot vector into a smaller set of clusters
		"""
		tag_vectors = list()
		for tag_index in tqdm(range(len(self.tag_list)), disable=False):
			tag = self.tag_list[tag_index]
			vector = self.model.encode(tag)
			tag_vectors.append(vector)
		M = np.array(tag_vectors)
	
		# the process performs the clustering only the top filtered tags
		dict1 = dict(Counter(tag for game_tags in self.sample_list for tag in game_tags))

		if verbose : print('filering top tags by percentile_threshold')
		dict1_indexes = {self.tag2index[key]:dict1[key] for key in dict1.keys()}
		data_indexes = list(dict1_indexes.items())
		percentile = np.percentile([x[1] for x in data_indexes], percentile_threshold)
		filtered_data_indexes = [x for x in data_indexes if x[1] >= percentile]
		filtered_data_indexes.sort(key=lambda x: x[1], reverse=True)

		if visualize:
			dict1_tags = {key:dict1[key] for key in dict1.keys()}
			data_tags = list(dict1_tags.items())
			percentile = np.percentile([x[1] for x in data_tags], percentile_threshold)
			filtered_data_tags = [x for x in data_tags if x[1] >= percentile]
			filtered_data_tags.sort(key=lambda x: x[1], reverse=True)

			max_plot = 500
			fig = px.bar(x=[x[0] for x in filtered_data_tags[0:max_plot]], y=[x[1] for x in filtered_data_tags[0:max_plot]], 
				labels={'x': 'Category', 'y': 'Frequency'}, 
				title='Histogram of Frequencies (95th percentile and above)')
			fig.show()
		
		data = list(dict1_indexes.items())
		percentile = np.percentile([x[1] for x in data], percentile_threshold)
		filtered_data = [x for x in data if x[1] >= percentile]
		filtered_data.sort(key=lambda x: x[1], reverse=True)
		valid_tags = [x[0] for x in filtered_data]

		if len(valid_tags) > n_clusters:
			
			if verbose : print('clustering is efficient, computing k-means')
			X = M[valid_tags, :]
			kmeans = KMeans(n_clusters=n_clusters)
			kmeans.fit(X)

			# encode tags to the closest top n vectors: we create tag pointer
			if verbose : print('assinging a pointer to all tags')
			cluster_centers = kmeans.cluster_centers_
			index_cluster_centers = self.compute_index(data=cluster_centers, k=1)
			
			tag_pointers = dict()
			for tag_name in tqdm(self.tag_list, disable=False):
				tag_vector = M[self.tag2index[tag_name]]
				tag_index = self.search_index(tag_vector, index_cluster_centers, k=1)[0]
				tag_pointers[tag_name] = tag_index

		else:
			if verbose : print('clustering is not efficient, returning regular tags')
			tag_pointers = {self.tag_list[index]:index for index in range(len(self.tag_list))}

		M = cluster_centers
		if quantize_M == True:
			M = quantize_embeddings(
				M,
				precision=self.quantization
			)
   
		return M, self.tag_list, tag_pointers

	
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

	
	def load_M(self, M, tag_pointers, covariate_transformation, cluster_M=False, cluster_percentile_threshold=0.95, n_clusters=1000):

		self.M = M
		self.tag_pointers = tag_pointers

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

		if self.covariate_transformation == 'dot_product':
			new_vector = onehot_covariate_vector @ self.M
		
		elif self.covariate_transformation == 'PCA':
			new_vector = self.pca.transform(onehot_covariate_vector.reshape(1, len(self.tag_list)))[0]

		return new_vector
