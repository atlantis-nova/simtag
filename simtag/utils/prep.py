import pandas as pd
from tqdm import tqdm

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

        # add index and
        df_M = pd.DataFrame(M).T
        df_M.index = self.tag_list
        df_M.columns = self.tag_list[0:len(df_M.columns)]
        
        self.M = df_M