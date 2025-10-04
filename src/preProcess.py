import  pandas as pd

k2_df=pd.read_csv("data/k2.csv", comment="#")

print(k2_df.info())
print(k2_df.isnull().sum().sort_values(ascending=False).head(10))

# Null Columns
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

#MVP Features Selelcting for modeling