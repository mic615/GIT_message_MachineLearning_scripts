import pandas as pd
import os.path as path
import os
def parse_pair_file(filepath):
	df= pd.read_csv(filepath, header=None, skiprows=1)
	with open(filepath) as f:
		lines = f.readlines()
		header = lines[0].strip().replace("#", "").strip()
		columns = header.split
		f.close
	df.columns = ["bf_sha","bi_sha"]
	return df

inputdir = '/home/mike/QA_Project/sha_pairs/'
bfdir = '/home/mike/QA_Project/Bug_Fix_sha/'
bidir = '/home/mike/QA_Project/Bug_intro_sha/'

for file in os.listdir(inputdir):
	df=parse_pair_file(inputdir+file)
	dfFix=df['bf_sha'] 
	dfBug=df['bi_sha']
	name= file.split('.')[0]
	dfFix.to_csv(bfdir+name+".txt",index=False)
	dfBug.to_csv(bidir+name+".txt",index=False)
