import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def find_rmse(file_path):
    if os.path.isfile(file_path):

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if 'RMSE:' in line:
                    return line.split(':')[1].strip()
        return None
    else:
        return None
filepath='/home/hy4080/wplyh/depthm/'
ts=['salt','temp']
df = pd.DataFrame()
for dir in ts:
    result = []
    for i in range(24):
        folder='depth'+str(i)
        depth=i
        subfolder=filepath+folder+'/'+dir+'/'
        file = 'output'+str(depth)+ '.txt'
        file_dir = subfolder + file
        rmse = find_rmse(file_dir)
        result.append(rmse)
    df[dir] = result
df.to_csv('/home/hy4080/wplyh/depthnew/ts.csv', index=True)
