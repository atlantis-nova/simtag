from .utils.prep import prep
from .utils.search import search
from .utils.validate import validate

class simtag_filter(prep, search, validate):
    
    def __init__(self, sample_list):
        
        # process samples
        self.sample_list = sample_list
        self.tag_list = list(set([x for xs in self.sample_list for x in xs]))