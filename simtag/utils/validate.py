import numpy as np
import statistics
from IPython.display import HTML

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


    def show_results(self, query_tag_list, raw_scores, filter_results, visualization_type, power=0.6, visualize=False, return_html=False):

        if visualization_type == 'raw':

            for tag_index in range(len(query_tag_list)):
                data = list(zip(raw_scores[tag_index], filter_results))
                tag = query_tag_list[tag_index]
                html_code = f"{tag}:<br>"
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
            tag = query_tag_list
            html_code = f"{tag}:<br>"
            for intensity, word in data:
                scaled_intensity = intensity ** power
                g = int(20 * (1 - scaled_intensity))
                r = int(255 * scaled_intensity * 0.7)  # adjust the green component
                color = f"rgb({r},{g},0)"
                html_code += f"<span style='background-color:{color}; color:white'>{word}</span> "
                # html_code += f"<span style='color:{color}'>{word}</span> "

            if visualize: display(HTML(html_code))
            if return_html: return html_code