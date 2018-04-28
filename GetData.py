import pandas as pd
import os.path as path
import os
import re
def parse_comment_file(filepath):
	df= pd.read_csv(filepath, header=None, delim_whitespace=True ,skiprows=1)
	with open(filepath) as f:
		lines = f.readlines()
		header = lines[0].strip().replace("#", "").strip()
		columns = header.split
		f.close
	#df.columns = ["sha","comment", "positive", "negative"]
	return df

bfinputdir = '/home/mike/QA_Project/BF_SentimentsDetection/'
biinputdir = '/home/mike/QA_Project/BI_SentimentsDetection/'
bfout = '/home/mike/QA_Project/Bug_Fix_sentiment_Data/'
biout = '/home/mike/QA_Project/Bug_intro_sentiment_Data/'

for file in os.listdir(bfinputdir):
	
	with open(bfinputdir+file, encoding="latin-1") as f:
			list= []
			name= file.split('.')[0] 
			i=0	
			for line in f:	
				line=line.rstrip()
				data = line.split() 
				pattern='".*"'				
				##quotes = line.split('\"') 
				#print(quotes)
				m= re.search(pattern,line)
				if m:
					comment=m.group(0)
					pos= data[len(data)-2]
					neg= data[len(data)-1]
					#print('neg'+neg)
					#print('pos'+pos)				
					sentiment= int(pos)+int(neg)
					list.append((comment, sentiment))
					i=i+1
					#print(m)
				else:	
					print(bfinputdir+file+' :(')
					print(i)
					comment = ' '	
				
				
			df=pd.DataFrame(list, columns=('comment','sentiment'))
			df.to_csv(bfout+name+".txt", sep=',',header=False,index=False)
	##bug intro comments
for file in os.listdir(biinputdir):
	
	with open(biinputdir+file, encoding="latin-1") as f:
			list= []
			name= file.split('.')[0] 
			i=0	
			for line in f:	
				line=line.rstrip()
				data = line.split() 
				pattern='".*"'
				m= re.search(pattern,line)
				if m:
					comment=m.group(0)
					pos= data[len(data)-2]
					neg= data[len(data)-1]
					#print('pos'+pos)
					#print('neg'+neg)				
					sentiment= int(pos)+int(neg)
					list.append((comment, sentiment))
					i=i+1
					#print(m)
				else:	
					print(biinputdir+file+' :(')
					print(i)
					comment = ' '	
				
				
			df=pd.DataFrame(list, columns=('comment','sentiment'))
			df.to_csv(biout+name+".txt", sep=',',header=False,index=False)

