import numpy as np
import statistics

class validate():

    def compute_neighbor_scores(self, search_results, query_tag_list, remove_max=False):

        # compute neighbor scores
        raw_scores = list()
        mean_scores = list()
        for tag in query_tag_list:
            tag_scores = list()
            for sample in search_results:
                score = self.M.loc[tag][sample]
                tag_scores.append(score)
            if remove_max:
                # we convert 1 to maximum neighbor scores to remove outliers
                tag_scores = [(x, -1)[x==1] for x in tag_scores]
                tag_scores = [(x, max(tag_scores))[x==-1] for x in tag_scores]
            raw_scores.append(tag_scores)
            mean_scores.append(statistics.mean(tag_scores))

        # average scores
        raw_scores = np.array(raw_scores)
        mean_scores = np.array(statistics.mean(mean_scores))

        return raw_scores, mean_scores