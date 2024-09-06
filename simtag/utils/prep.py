import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

class prep():
	
	def compute_M(self):

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


		# values
		self.M = np.array(M)

		# dataframe with tags
		df_M = pd.DataFrame([self.tag_list, M]).T
		df_M.columns = ['tags', 'vector_tags']
		self.df_M = df_M

		return self.M, self.df_M

	
	def load_M(self, df_M):

		self.df_M = df_M
		self.M = np.array(df_M['vector_tags'].tolist())


	def apply_PCA(self, pca_vector_length):
		
		# initiate PCA
		self.pca_vector_length = pca_vector_length
		self.pca = PCA(n_components=self.pca_vector_length)
		self.pca.fit(self.M.T)
		self.M = self.pca.transform(self.M.T)