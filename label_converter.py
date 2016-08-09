'''
JSON Label Generator for LSTM
author - Chethan Mysore Parameshwara
email - analogicalnexus159@gmail.com
'''
import csv
import json


def csv2json(s_fid, e_fid, label):
    data = {
    's_fid' : s_fid,
    'e_fid' : e_fid,
    'label' : label
	}
    return data

def jsonwrite(s_fid, e_fid, label):
     with open('al0_wo5_labels.json', 'a') as f:
            data = csv2json(s_fid, e_fid, label)
            json.dump(data, f, indent=3)
            f.write(',')
            f.write('\n')
            print (s_fid, e_fid, label)

def read_csv(filename):
	with open(filename) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		s_fid = []
		e_fid = []
		l_fid = []
		for row in readCSV:
			s = int(row[0])
			e = int(row[1])
			l = int(row[2])

			s_fid.append(s)
			e_fid.append(e)
			l_fid.append(l)
		return s_fid, e_fid, l_fid

#writer = csv.writer(open("label.csv", "w"), delimiter=',', lineterminator='\n')
def main():	
    [s_fid, e_fid, l_fid] = read_csv("al0_labels.csv")	
    jsonwrite(1, s_fid[0]-1, 5)
	
    for x in range(0, 100):
        jsonwrite(s_fid[x], e_fid[x], l_fid[x]) 
        if e_fid[x] == s_fid[x+1]:
            continue
        else: 
            jsonwrite(e_fid[x]+1,s_fid[x+1]-1, 5)

main()
