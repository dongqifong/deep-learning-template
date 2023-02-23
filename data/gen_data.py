import pandas as pd
import numpy as np


x = np.random.random((100,7))
df = pd.DataFrame(data=x)
df.to_csv("data.csv",encoding="utf_8_sig",index=False)