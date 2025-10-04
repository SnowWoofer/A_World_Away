import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

k2_df=pd.read_csv("data/k2.csv", comment="#")

# print(k2_df.info())
# print(k2_df.isnull().sum().sort_values(ascending=False).head(10))

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

mvpCols = [
    "pl_name",                  # planet Id
    "disposition",              # Classifier = Positive, negative etc 
    "discoverymethod",          # Analyses Bias = Proximity bias between exoplanet classifications
    "disc_year",                # Year of discovery 
    "pl_orbper",                # oribital period = relates to planetary distances and temperature
    "pl_rade",                  # planet radius = used to classify gas giants vs earth-like
    "pl_bmasse",                # planet mass = radius+mass => density
    "pl_eqt",                   # equilibrim temp = amount of radiation
    "st_teff",                  # stellar temp = Host star type dteremined by temp 
    "st_rad",                   # stellar radius = como with temp => luminosity 
    "st_mass",                  # stellar mass = planet size and oribitalc chatratciestcs 
    "st_met"                    # stellar metalicty = metal rish vs non-metal richh planet formation rates 
]

k2_trim = k2_df[mvpCols].copy()
print(k2_trim.head())
#print(k2_trim.isnull().sum())

print(k2_trim.isnull().sum())

#Data visulaisation

# k2_cleaned["pl_rade"].hist(bins=20)
# plt.xlabel("Planet Radius (Earth Radii)")
# plt.ylabel("Count")
# plt.title("Distrubution Of Planet Radii In K2 Data")
# plt.show()

# missing data heat map 

# plt.figure(figsize=(12,6))
# sns.heatmap(k2_trim.isnull(), cbar=False, cmap='viridis')
# plt.title("Missing Data HeatMap")
# plt.show()

#   Missing NAN Handling:

k2_cleaned = k2_trim.dropna(subset=["pl_rade", "pl_eqt", "st_teff"]) # missing crutuical info and therefore dropped
k2_cleaned['pl_bmasse'] = k2_cleaned['pl_bmasse'].fillna(k2_cleaned['pl_bmasse'].median())
print("Cleaned shape:", k2_cleaned.shape)

# converison of disposition data into numerical format for machine learning

statusMap = {"Confirmed" : 2, "Candidate" : 1, "False Positive" : 0}
k2_cleaned['dispositionNum']= k2_cleaned["disposition"].map(statusMap)