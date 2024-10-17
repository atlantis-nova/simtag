import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
from tqdm import tqdm
# from sklearn.neighbors import NearestNeighbors
# import plotly.express as px
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.*")

class clustering():


	def compute_optimal_M(self, df_M, percentile_threshold=95, n_clusters=1000):
		"""
		Compress the one_hot vector into a smaller set of clusters
		"""

		dict1 = dict(Counter(tag for game_tags in self.sample_list for tag in game_tags))
		data = list(dict1.items())
		percentile = np.percentile([x[1] for x in data], percentile_threshold)
		filtered_data = [x for x in data if x[1] >= percentile]
		filtered_data.sort(key=lambda x: x[1], reverse=True)

		# Create a histogram using Plotly
		# max_plot = 500
		# fig = px.bar(x=[x[0] for x in filtered_data[0:max_plot]], y=[x[1] for x in filtered_data[0:max_plot]], 
		#             labels={'x': 'Category', 'y': 'Frequency'}, 
		#             title='Histogram of Frequencies (95th percentile and above)')
		# fig.show()

		valid_tags = [x[0] for x in filtered_data]
		df_M_filtered = df_M[df_M['tags'].isin(valid_tags)]

		X = np.array(df_M_filtered['vector_tags'].values.tolist())
		kmeans = KMeans(n_clusters=n_clusters)
		kmeans.fit(X)
		
		cluster_centers = kmeans.cluster_centers_
		return cluster_centers


	def assign_cluster_id(self, df_M, cluster_centers, show_progress=True):
		"""
		Assigns a cluster id to all the tags that are not from the top n
		"""
		# TODO : remove
		# def get_parent_number(tag_vector, index_cluster_centers):
			
		# 	# distances, indices = index_cluster_centers.kneighbors([tag_vector]) 
		# 	index = self.search_index(tag_vector, index_cluster_centers, k=1)
		# 	return index
		
		if show_progress==True:
			disable_progress = False
		elif show_progress==False:
			disable_progress = True

		# index_cluster_centers = NearestNeighbors(n_neighbors=1, metric='cosine').fit(cluster_centers) # TODO : remove
		index_cluster_centers = self.compute_index(data=cluster_centers, k=1)

		# encode tags to the closest top n vectors
		dict_clusters = dict()
		for tag_i in tqdm(df_M.values, disable=disable_progress):
			tag_name = tag_i[0]    
			tag_vector = np.array(df_M[df_M['tags']==tag_name]['vector_tags'].iloc[0])
			# tag_index = get_parent_number(tag_vector, index_cluster_centers) # TODO : remove
			tag_index = self.search_index(tag_vector, index_cluster_centers, k=1)[0]
			dict_clusters[tag_name] = tag_index

		df_M_clusters = pd.DataFrame([cluster_centers.tolist()]).T
		df_M_clusters.insert(0, 'tags', range(0, 1000))
		df_M_clusters.columns = ['tags', 'vector_tags']

		return dict_clusters, df_M_clusters
	

	def list_cluster_tags(self, cluster_n):
			
		clster_tags = list()
		for index, value in enumerate(self.dict_clusters):
			cluster_index = self.dict_clusters[value]
			if cluster_index == cluster_n:
				clster_tags.append(value)
						
		return clster_tags