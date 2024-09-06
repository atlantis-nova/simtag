import pandas as pd
import numpy as np
import statistics
from IPython.display import HTML
from sentence_transformers.util import cos_sim

class validate():

    def compute_neighbor_scores(self, search_results, query_tag_list, remove_max=False):

        # compute neighbor scores
        raw_scores = list()
        mean_scores = list()
        for tag in query_tag_list:
            tag_scores = list()
            for sample in search_results:
                score = self.df_M.loc[tag][sample]
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
    
    def compute_neighbor_scores(self, search_results, query_tag_list, exp=1.5, remove_max=False):

        # compute a relationship matrix from the query
        # query_M = pd.DataFrame(query_tag_list, columns=['Tag'])
        relationship_matrix = pd.DataFrame(index=query_tag_list, columns=query_tag_list)
        for i in query_tag_list:
            for j in search_results:
                vector_i = self.df_M[self.df_M['tags']==i].vector_tags.tolist()[0]
                vector_j = self.df_M[self.df_M['tags']==j].vector_tags.tolist()[0]
                # we change the result for better coloring
                relationship_matrix.loc[i, j] = cos_sim(vector_i, vector_j).tolist()[0][0]**exp
        M_query = relationship_matrix
        # display(M_query)

        # compute neighbor scores
        raw_scores = list()
        mean_scores = list()
        for tag in query_tag_list:
            tag_scores = list()
            # tag_index = engine.tag_list.index(tag)
            for sample in search_results:
                # sample_index = engine.tag_list.index(sample)
                score = M_query.loc[tag][sample]

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


    def show_results(self, query_tag_list, raw_scores, filter_results, visualization_type, power=0.6, title='', visualize=False, return_html=False):

        if visualization_type == 'raw':

            for tag_index in range(len(query_tag_list)):
                data = list(zip(raw_scores[tag_index], filter_results))
                html_code = f"{title}:<br>"
                for intensity, word in data:
                    scaled_intensity = intensity ** power
                    g = int(20 * (1 - scaled_intensity))
                    r = int(255 * scaled_intensity * 0.7)  # adjust the green component
                    color = f"rgb({r},{g},0)"
                    html_code += f"<span style='background-color:{color}; color:white'>{word}</span> "
                    # html_code += f"<span style='color:{color}'>{word}</span> "

                if visualize: display(HTML(html_code))
                if return_html: return html_code

        elif visualization_type == 'mean':

            data = list(zip(np.mean(raw_scores, axis=0), filter_results))
            html_code = f"{title}:<br>"
            for intensity, word in data:
                scaled_intensity = intensity ** power
                g = int(20 * (1 - scaled_intensity))
                r = int(255 * scaled_intensity * 0.7)  # adjust the green component
                color = f"rgb({r},{g},0)"
                html_code += f"<span style='background-color:{color}; color:white'>{word}</span> "
                # html_code += f"<span style='color:{color}'>{word}</span> "

            if visualize: display(HTML(html_code))
            if return_html: return html_code