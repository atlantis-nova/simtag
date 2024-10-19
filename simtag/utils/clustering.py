class clustering():
	
	def list_cluster_tags(self, cluster_n):
			
		clster_tags = list()
		for index, value in enumerate(self.tag_pointers):
			cluster_index = self.tag_pointers[value]
			if cluster_index == cluster_n:
				clster_tags.append(value)
						
		return clster_tags