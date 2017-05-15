import os
import math
import os.path
import sys
import datetime
import time

if len(sys.argv) <> 2:
	print "python <pmb.py> <exec:gather>"   
	sys.exit()
	
procs = ["1", "2", "4", "8", "16"];
data = ["1M", "10M", "20M"];
data_speedup = ["10M", "20M"];
execs = ["pmb1_dp", "pmb3_dp"];
execs_further = ["pmb3_dp_numa", "pmb3_sp_numa"];
orders = ["", "_colperm", "_symamd"];
X = 3;

es1 = "OMP_NUM_THREADS=";
es2 = "; export OMP_SCHEDULE=dynamic,8; "
es3 = " /gandalf/data/Optimization/MovieLens/"

currentPath = os.path.dirname(os.path.abspath(sys.argv[0])) + "/"
option = sys.argv[1]
                                          
for di in range(len(data)):
       	dname = data[di]			
	for pi in range(len(procs)):		
		P = procs[pi];				
		for ei in range(len(execs)):
			exname = execs[ei];
			for x in range(X):
				expName = exname + "." + dname + "." + P + ".x" + str(x);
				toFile = currentPath + "Results/" + expName + ".result";                 				
				execStr = es1 + P + es2 + " " + currentPath + exname + es3 + dname + ".dat > " + toFile;
				if option == "exec":
					print execStr
					os.system(execStr);
				elif option == "gather":
					infile = open(toFile, 'r')
					firstLine = infile.readline()
					print expName + " " + firstLine,


for di in range(len(data)):
	dname = data[di]
	P = "16";
	for ei in range(len(execs_further)):
		exname = execs_further[ei];
		for oi in range(len(orders)):
			order = orders[oi];
			for x in range(X):
				expName = exname + "." + dname + order + "." + P + ".x" + str(x);
				toFile = currentPath + "Results/" + expName + ".result";
				execStr = es1 + P + es2 + " " + currentPath + exname + es3 + dname + order + ".dat > " + toFile;
				if option == "exec":
					print execStr
					os.system(execStr);
				elif option == "gather":
					infile = open(toFile, 'r')
                                        firstLine = infile.readline()
					print expName +" " + firstLine,

order = "_symamd";
for di in range(len(data)):
       	dname = data[di]			
	for pi in range(len(procs)):		
		P = procs[pi];				
		for ei in range(len(execs_further)):
			exname = execs_further[ei];
			for x in range(X):
				expName = exname + "." + dname + "." + P + ".x" + str(x);
				toFile = currentPath + "Results/" + expName + ".result";                 				
				execStr = es1 + P + es2 + " " + currentPath + exname + es3 + dname + order + ".dat > " + toFile;
				if option == "exec":
					print execStr
					os.system(execStr);
				elif option == "gather":
					infile = open(toFile, 'r')
                                        firstLine = infile.readline()
					print expName +" " + firstLine,
