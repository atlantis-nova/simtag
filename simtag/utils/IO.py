import os
import pandas as pd
import json
import numpy as np

class IO():

    def parquet_load(self, batch_prefix, n=1):
        dfs = [pd.read_parquet(f'{batch_prefix}{i}.parquet') for i in range(n)]
        df = pd.concat(dfs, ignore_index=True)
        return df


    def parquet_save(self, df, batch_prefix, n=1):
        chunk_size = len(df) // n
        for i in range(n):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < (n-1) else len(df)
            df.iloc[start:end].to_parquet(f"{batch_prefix}{i}.parquet", engine='pyarrow')
    
    
    def json_save(self, my_dict, path_prefix):
        with open(f'{path_prefix}.json', 'w') as f: 
            json.dump(my_dict, f)
            
            
    def json_load(self, path_prefix):
        return json.loads(open(f'{path_prefix}.json').read())  
            
            
    def npy_save(self, array, path, n=1):
        # partition_size = len(array) // n  # TODO : remove?
        partitions = np.split(array, n)
        
        for i, partition in enumerate(partitions):
            if n == 1:
                np.save(f'{path}.npy', partition)
            elif n > 1:
                np.save(f'{path}_{i}.npy', partition)
            

    def npy_load(self, path, n=1):
        partitions = []
        for i in range(n):
            if n == 1:
                partition = np.load(f'{path}.npy')
            elif n > 1:
                partition = np.load(f'{path}_{i}.npy')
            partitions.append(partition)
        array = np.concatenate(partitions)
        return array