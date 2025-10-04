import pandas as pd 

kepler_df = pd.read_csv("data/kepler.csv", comment="#")
tess_df = pd.read_csv("data/Tess.csv", comment="#")
k2_df= pd.read_csv("data/k2.csv", comment="#")

# print(k2_df.head())
print(k2_df.info())
# print(k2_df.describe())

# print(tess_df.head())
# print(tess_df.info())
# print(tess_df.describe())

# print(kepler_df.head())
# print(kepler_df.info())
# print(kepler_df.describe())

#Later USed for Confusion Matrix

# print(k2_df['disposition'].value_counts()) 
# print(kepler_df['koi_disposition'].value_counts())
# print(tess_df['tfopwg_disp'].value_counts())

#Check for null vars

print(k2_df.isnull().sum().sort_values(ascending=False).head(10))
# print(tess_df.isnull().sum())
# print(kepler_df.isnull().sum())

# k2 most incomplete columns:
# pl_orbeccenerr1    3778
# pl_orbeccenerr2    3778
# pl_bmasseerr2      3618
# pl_bmasseerr1      3618
# pl_bmassjerr1      3618
# pl_bmassjerr2      3618
# pl_orbeccen        3582
# pl_orbeccenlim     3582
# pl_bmasse          3576
# pl_bmassprov       3576

