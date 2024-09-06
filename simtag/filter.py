from .utils.prep import prep
from .utils.search import search
from .utils.validate import validate

class simtag_filter(prep, search, validate):
    
    def __init__(self, sample_list=None, tag_list=None):

        self.pca_vector_length = None
        
        # process samples
        if sample_list is not None:
            self.sample_list = sample_list

        if tag_list is not None:
            self.tag_list = tag_list
        else:
            self.tag_list = sorted(list(set([x for xs in self.sample_list for x in xs])))