import pandas as pd
import numpy as np

data = pd.read_csv("../data/ann.csv")

indet_ind = data.FINAL.isna() & ~data.IMAGE.isna()
print(indet_ind)

data_indet = data.loc[indet_ind]
print(data_indet)

data_indet = data_indet.loc[:,["PERSON", "IMAGE"]]
data_indet = data_indet.astype(np.int)
print(data_indet)

data_indet.to_csv("../data/indet.csv", index=False)
