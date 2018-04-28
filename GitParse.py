import subprocess
import pandas as pd
import os

bfdir = '/home/mike/QA_Project/Bug_Fix_sha/'
bidir = '/home/mike/QA_Project/Bug_intro_sha/'
bf_comments_dir = '/home/mike/QA_Project/Bug_Fix_comments/'
bi_comments_dir = '/home/mike/QA_Project/Bug_intro_comments/'
#for loop of all shas in file
##bug fix comments
for file in os.listdir(bfdir):
	with open (bfdir+file) as f:
		list= []
		name= file.split('.')[0] 
		os.chdir('/home/mike/QA_Project/repos/'+name)		
		for line in f:	
			line=line.rstrip()					
			p = subprocess.Popen("git log -1 "+line,stdout=subprocess.PIPE, shell=True) 
			(output, err) = p.communicate()
			p_status = p.wait()		
			commit= output.splitlines()
			if len(commit) >4 :
				comment= output.splitlines()[4].strip().decode('utf-8')
			else:
				print("no message for "+name +" : "+line)		
				comment= " "				
			list.append((line, comment))
		df=pd.DataFrame(list, columns=('sha','comment'))
		df.to_csv(bf_comments_dir+name+".txt", sep=' ',header=False,index=False)
##bug intro comments
for file in os.listdir(bidir):
	with open (bidir+file) as f:
		list= []
		name= file.split('.')[0] 
		os.chdir('/home/mike/QA_Project/repos/'+name)		
		for line in f:	
			line=line.rstrip()					
			p = subprocess.Popen("git log -1 "+line,stdout=subprocess.PIPE, shell=True) 
			(output, err) = p.communicate()
			p_status = p.wait()
			commit= output.splitlines()
			if len(commit) >4 :
				comment= output.splitlines()[4].strip().decode('utf-8')
			else:
				print("no message for "+name +" : "+line)		
				comment= " "
			list.append((line, comment))
		df=pd.DataFrame(list, columns=('sha','comment'))
		df.to_csv(bi_comments_dir+name+".txt", sep=' ',header=False,index=False)
