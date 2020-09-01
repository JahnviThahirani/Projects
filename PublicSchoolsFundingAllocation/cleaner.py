import numpy as np
import pandas as pd

filePath = 'dataset.csv'
numGroups = 5
pd.options.mode.chained_assignment = None

# Find graduation rate based on given values
def find_avg(value):
	if '-' in value:
		value_split = value.split('-')
		return np.mean([int(value_split[0]), int(value_split[1])])
	elif 'GE' or 'GT' in value:
		return np.mean([100,int(value[2:])])
	elif 'LE' or 'LT' in value:
		return np.mean([0,int(value[2:])])
	else:
		return "ERROR"

# Group into equally sized groups
def create_groups(num_bins, df):
	bin_size = len(df)//num_bins + 1
	groups = [i//bin_size for i in range(len(df))]
	sorted_df = df.sort_values("ALL_RATE_1617")
	sorted_df["GROUP"] = groups
	return sorted_df

def convert_to_numeric(indices, df):
	for i in indices:
		df[i] = pd.to_numeric(df[i], errors='coerce')


df = pd.read_csv(filePath)
df = df.loc[(df["ALL_RATE_1617"].str.contains('\d+-\d+')) | (df["ALL_RATE_1617"].str.contains('[A-Z]+\d+'))]
df["ALL_RATE_1617"] = df["ALL_RATE_1617"].apply(find_avg)
convert_to_numeric(["ALL_COHORT_1617", "ALL_RATE_1617", "TOTALREV", "TFEDREV", "TSTREV", "TLOCREV", "TOTALEXP", "TCURINST", "TCURSSVC", "TCURONON", "TCAPOUT"], df)


# Create new columns using aggregates
df["NUM_SCH"] = df.groupby(['LEAID'])['TOTALEXP'].transform('count')
df["TOTALEXP_LN"] = np.log(df["TOTALEXP"] / df["NUM_SCH"])
df["TOTALEXP"] = df["TOTALEXP"] / df["NUM_SCH"]
df["CLASS_SIZE"] = df["ALL_COHORT_1617"] / (.01*df["ALL_RATE_1617"])
df["%_FEDREV"] = df["TFEDREV"] / df["TOTALREV"]
df["%_STREV"] = df["TSTREV"] / df["TOTALREV"]
df["%_LOCREV"] = df["TLOCREV"] / df["TOTALREV"]
df["TCURINST/CAP"] = df["TCURINST"]/ df["CLASS_SIZE"]
df["TCURSSVC/CAP"] = df["TCURSSVC"]/ df["CLASS_SIZE"]
df["TCURONON/CAP"] = df["TCURONON"]/ df["CLASS_SIZE"]
df["TCAPOUT/CAP"] = df["TCAPOUT"]/ df["CLASS_SIZE"]


filtered_df = df.filter(["YEAR", "STNAM", "LEAID", "CLASS_SIZE", "%_FEDREV", "%_STREV", "%_LOCREV", "TOTALEXP", "TOTALEXP_LN", "TCURINST/CAP", "TCURSSVC/CAP", "TCURONON/CAP", "TCAPOUT/CAP", "ALL_RATE_1617", "GROUP"]).dropna()
grouped = create_groups(numGroups, filtered_df)
grouped.to_csv("grouped_data.csv", index = False)