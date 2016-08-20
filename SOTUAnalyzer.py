# Mishaal Aleem

from __future__ import print_function
import numpy as np
from collections import Counter
import operator
import matplotlib.pyplot as plt
import string 
import csv 
import os 
import xlrd
import shutil as shutil
import mpld3
from mpld3 import plugins
from pylab import *

#-----------------------------
#Import
#-----------------------------
#Get categorization information
with open('Data.csv', 'rb') as f:
    reader = csv.reader(f)
    full_list = list(reader)[1:] #the 1 ignores the title row

presidents = [item[0] for item in full_list] #list of presidents
years = [item[1] for item in full_list] #list of years
parties = [item[2] for item in full_list] #list of president's parties

#Generate a list with the names of the speech .txt files (named by year)
docs = ["Speeches/" + s + ".txt" for s in years]

#Import Speeches
speeches = [];
for i in range (0,len(years)):
	with open (docs[i], "r") as myfile:
		data=myfile.read()
		data = data.translate(None, string.punctuation)
		data = data.lower()
		speeches = speeches + [data] 

#-----------------------------
#User Inputs
#-----------------------------
# Get user input about word
buzzword = raw_input('Enter a word to analyze: ')
output = raw_input('Enter an output file name (no extension): ')

#-----------------------------
#Analysis
#-----------------------------

# Create a temporary directory to save figures
tempdir_path = "temp_" + buzzword
if os.path.exists(tempdir_path) == False:
   os.makedirs(tempdir_path)

# Count how many times the buzzword is used and the total number of words in the speech
timesUsed = [];
totWords = [];
for i in range (0, len(years)):
	dat = speeches[i]
	words = dat.split()
	totWords = totWords + [len(words)]
	wordCount = Counter(words)
	timesUsed =  timesUsed + [wordCount[buzzword]]
	

# Get the min/max index and corresponding min/max value for the total number of words used and times used
total_min_index, total_min_value = min(enumerate(totWords), key=operator.itemgetter(1))
total_max_index, total_max_value = max(enumerate(totWords), key=operator.itemgetter(1))

# FIGURE: Total number of words in the speech
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
plt.sca(ax1)
ax1.plot(years, totWords,'k--') #black line

colors = [w.replace('Democratic', '1') for w in parties]
colors = [w.replace('Republican', '0') for w in colors]

plt.xlim([np.min(map(int, years)),np.max(map(int, years))])
plt.grid(b=True, which='both', color='0.65',linestyle='-')

label_dict = {0: 'r', 1: 'b'}
legend_dict = {0: 'Republican', 1: 'Democratic'}
marker_dict = {0: 'o', 1: '^'}
labels = map(int, colors)

seen = set()
for x, y, label in zip(years, totWords, labels):
    if label not in seen:
        plt.scatter(x, y, c=label_dict.get(label), s=50, label=legend_dict.get(label), marker=marker_dict.get(label))
    else:
        plt.scatter(x, y, c=label_dict.get(label), s=50, marker=marker_dict.get(label))
    seen.add(label)

plt.legend(loc=3,scatterpoints=1, mode="expand", ncol=2, bbox_to_anchor=(0., 1.0, 1., 0.0))

fig.savefig(tempdir_path + '/1.png')

# Calculate average lengths of all addresses and for each party
averageLength = int(np.average(totWords))
demTotLength = []
repTotLength = []
for i in range(0,len(years)):
   if parties[i] == 'Democratic':
      demTotLength = demTotLength + [totWords[i]]
   elif parties[i] == 'Republican':
      repTotLength = repTotLength + [totWords[i]]
demAverageLength = int(np.average(demTotLength))
repAverageLength = int(np.average(repTotLength))

# FIGURE: Overlap the speech length plot with historical events 
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
plt.sca(ax1)
plt.grid(b=True, which='both', color='0.65',linestyle='-')


def getYearsOf(yearsInt, start, end, getThis):
   event = []
   eventYears =[]
   for i in range(start,end+1):
      if i in yearsInt:
         eventYears = eventYears + [i]
         event = event + [operator.itemgetter(yearsInt.index(i))(getThis)]
   return event, eventYears


yearsInt = [int(numeric_string) for numeric_string in years]

ww1, ww1Years = getYearsOf(yearsInt, 1914, 1918, totWords)
ax1.plot(ww1Years, ww1,'c-', lw=8.0, label='World War I')

depression, depressionYears = getYearsOf(yearsInt, 1929, 1939, totWords)
ax1.plot(depressionYears, depression,'m-', lw=8.0, label='Great Depression')

ww2, ww2Years = getYearsOf(yearsInt, 1939, 1945, totWords)
ax1.plot(ww2Years, ww2,'y-', lw=8.0, label='World War II ')

recession, recessionYears = getYearsOf(yearsInt, 2007, 2009, totWords)
ax1.plot(recessionYears, recession,'r-', lw=8.0, label='Great Recession')

ax1.plot(years, totWords, 'k--', lw=1.5) # Total Words Lines
seen = set()
for x, y, label in zip(years, totWords, labels):
    if label not in seen:
        plt.scatter(x, y, s=50, label=legend_dict.get(label), marker=marker_dict.get(label),c='k')
    else:
        plt.scatter(x, y, s=50, marker=marker_dict.get(label),c='k')
    seen.add(label)



plt.xlim([np.min(map(int, years)),np.max(map(int, years))])
plt.legend(loc=9,scatterpoints=1, mode="expand", ncol=2, bbox_to_anchor=(0, -0.1,1.0,0))

fig.savefig(tempdir_path + '/5.png',bbox_inches="tight")

#Create a dictionary that holds all words used in all speeches and the count for each word
usageDictionary = [];
for i in range (0, len(years)):
	dat = speeches[i]
	words = dat.split()
	words = words
	counts = {}
	for w in words:
	   counts[w] = counts.get(w,0) + 1
	usageDictionary  = usageDictionary + [counts]

#Count the number of unique words in each speech 
uniquewords = []
for i in range (0, len(years)):
   uniquewords = uniquewords + [len(usageDictionary[i])]

#FIGURE: Plot the unique words used as bars
fig_TotalUniqueCount = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
plt.sca(ax1)
plt.xlim([np.min(map(int, years)),np.max(map(int, years))])

seen = set()
for x, y, label in zip(years, uniquewords, labels):
   if label not in seen:
      plt.bar([x], [y], 0.5, color=label_dict.get(label), label=legend_dict.get(label))
   else:
      plt.bar([x], [y], 0.5, color=label_dict.get(label))
   seen.add(label)


plt.legend(loc=3,scatterpoints=1, mode="expand", ncol=2, bbox_to_anchor=(0., 1.0, 1., 0.0))
fig_TotalUniqueCount.savefig(tempdir_path + '/4.png')

uniquewords_min_index, uniquewords_min_value = min(enumerate(uniquewords), key=operator.itemgetter(1))
uniquewords_max_index, uniquewords_max_value = max(enumerate(uniquewords), key=operator.itemgetter(1))

keys = []
for i in range(0,len(years)):
   keys = keys + usageDictionary[i].keys()

totalDict={}
repDict={}
demDict={}
for key in keys:
   tmp = 0
   tmpD = 0
   tmpR = 0
   for i in range(0,len(years)):
      tmp = tmp + usageDictionary[i].get(key, 0)
      if parties[i] == "Democratic":
         tmpD = tmpD + usageDictionary[i].get(key, 0)
      elif parties[i] == "Republican":
         tmpR = tmpR + usageDictionary[i].get(key, 0)
   totalDict[key] = tmp
   repDict[key] = tmpR
   demDict[key] = tmpD
   
repDict = {x:y for x,y in repDict.items() if y!=0}
demDict = {x:y for x,y in demDict.items() if y!=0}

totUniqueWords = len(totalDict)
repUniqueWords = len(repDict)
demUniqueWords = len(demDict)

fig_TotalUniquePartyCount = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
plt.sca(ax1)

plt.bar([1], [demUniqueWords], 0.5, color='b', label='Democratic')
plt.bar([2], [repUniqueWords], 0.5, color='r', label='Republican')
ax1.get_xaxis().set_visible(False)

ax1.text(1.15, demUniqueWords + 200, str(demUniqueWords), color='k', fontweight='bold')
ax1.text(2.15, repUniqueWords + 200, str(repUniqueWords), color='k', fontweight='bold')
plt.ylim([0,15000])      
plt.legend(loc=3,scatterpoints=1, mode="expand", ncol=2, bbox_to_anchor=(0., 1.0, 1., 0.0))
fig_TotalUniquePartyCount.savefig(tempdir_path + '/6.png')

#Unique Word Percentage - Speech Level
uniquePercent = np.asarray(uniquewords, dtype=np.float32)/np.asarray(totWords, dtype=np.float32)*100.

fig_UniquePer = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
plt.sca(ax1)

plt.xlim([np.min(map(int, years)),np.max(map(int, years))])

seen = set()
for x, y, label in zip(years, uniquePercent, labels):
    if label not in seen:
        plt.bar([x], [y], color=label_dict.get(label), label=legend_dict.get(label))
    else:
        plt.bar([x], [y], color=label_dict.get(label))
    seen.add(label)
    
plt.legend(loc=3,scatterpoints=1, mode="expand", ncol=2, bbox_to_anchor=(0., 1.0, 1., 0.0))
fig_UniquePer.savefig(tempdir_path + '/7.png')


demUniquePercent = (float(demUniqueWords)/float(np.sum(demTotLength)))*100.0
repUniquePercent = (float(repUniqueWords)/float(np.sum(repTotLength)))*100.0

fig_UniquePerParty = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
plt.sca(ax1)

plt.bar([1], [demUniquePercent], 0.5, color='b', label='Democratic')
plt.bar([2], [repUniquePercent], 0.5, color='r', label='Republican')
ax1.get_xaxis().set_visible(False)

ax1.text(1.25, demUniquePercent + .25, str('%.0f' % demUniquePercent), color='k', fontweight='bold')
ax1.text(2.25, repUniquePercent + .25, str('%.0f' % repUniquePercent), color='k', fontweight='bold')
#plt.ylim([0,15000])      
plt.legend(loc=3,scatterpoints=1, mode="expand", ncol=2, bbox_to_anchor=(0., 1.0, 1., 0.0))
fig_UniquePerParty.savefig(tempdir_path + '/8.png')

#-----------------
#Buzzword
#-----------------

# Get the min/max index and corresponding min/max value for the buzzword used and that value
buzz_min_index, buzz_min_value = min(enumerate(timesUsed), key=operator.itemgetter(1))
buzz_max_index, buzz_max_value = max(enumerate(timesUsed), key=operator.itemgetter(1))

#FIGURE: Total times buzzword used in the speech
fig_TotalWordsCountWithHistorical = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
plt.sca(ax1)
ax1.plot(years, timesUsed,'k--') #black line

colors = [w.replace('Democratic', '1') for w in parties]
colors = [w.replace('Republican', '0') for w in colors]


plt.xlim([np.min(map(int, years)),np.max(map(int, years))])
plt.ylim(ymin=0)
plt.grid(b=True, which='both', color='0.65',linestyle='-')

seen = set()
for x, y, label in zip(years, timesUsed, labels):
    if label not in seen:
        plt.scatter(x, y, c=label_dict.get(label), s=50, label=legend_dict.get(label), marker=marker_dict.get(label))
    else:
        plt.scatter(x, y, c=label_dict.get(label), s=50, marker=marker_dict.get(label))
    seen.add(label)
    
plt.legend(loc=3,scatterpoints=1, mode="expand", ncol=2, bbox_to_anchor=(0., 1.0, 1., 0.0))

fig_TotalWordsCountWithHistorical.savefig(tempdir_path + '/2.png')

# Percentage usage of buzzword        
useperTOTAL = np.asarray(timesUsed, dtype=np.float32)/np.asarray(totWords, dtype=np.float32)*100.

buzzp_min_index, buzzp_min_value = min(enumerate(useperTOTAL), key=operator.itemgetter(1))
buzzp_max_index, buzzp_max_value = max(enumerate(useperTOTAL), key=operator.itemgetter(1))


# FIGURE: Percentage buzzword used in the speech
fig3 = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
plt.sca(ax1)
ax1.plot(years, useperTOTAL,'k--') #black line



plt.xlim([np.min(map(int, years)),np.max(map(int, years))])
plt.ylim(ymin=0)
plt.grid(b=True, which='both', color='0.65',linestyle='-')

seen = set()
for x, y, label in zip(years, useperTOTAL, labels):
    if label not in seen:
        plt.scatter(x, y, c=label_dict.get(label), s=50, label=legend_dict.get(label), marker=marker_dict.get(label))
    else:
        plt.scatter(x, y, c=label_dict.get(label), s=50, marker=marker_dict.get(label))
    seen.add(label)
    
plt.legend(loc=3,scatterpoints=1, mode="expand", ncol=2, bbox_to_anchor=(0., 1.0, 1., 0.0))
fig3.savefig(tempdir_path + '/3.png')


#-----------------------------
#Output
#-----------------------------
from jinja2 import *
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template("myreport.html")

template_vars = {"title" : "SOTU - Word Analysis",
                 "buzzword": buzzword,
                 "fig_TotalWordsCount": tempdir_path + "/1.png",
                 "shortestYEAR": years[total_min_index],
                 "shortestPRES": presidents[total_min_index],
                 "shortestPARTY": parties[total_min_index][0:1],
                 "shortestWORDS": total_min_value,
                 "longestYEAR": years[total_max_index],
                 "longestPRES": presidents[total_max_index],
                 "longestPARTY": parties[total_max_index][0:1],
                 "longestWORDS": total_max_value,
                 "averageLength": averageLength,
                 "demAverageLength": demAverageLength,
                 "repAverageLength": repAverageLength,
                 "buzzwordTOTAL": sum(timesUsed),
                 "buzzwordMAX": buzz_max_value,
                 "buzzwordPRES": presidents[buzz_max_index],
                 "buzzwordMAXPARTY": parties[buzz_max_index][0:1],
                 "buzzwordMAXYEAR": years[buzz_max_index],
                 "fig_BuzzwordTotal": tempdir_path + "/2.png",
                 "fig_BuzzwordPer": tempdir_path + "/3.png",
                 "fig_TotalUniqueCount": tempdir_path + "/4.png",
                 "fig_TotalWordsCountWithHist": tempdir_path + "/5.png",
                 "fig_TotalUniquePartyCount": tempdir_path + "/6.png",
                 "fig_UniquePer": tempdir_path + "/7.png",
                 "fig_UniquePerParty": tempdir_path + "/8.png",
                 "buzzwordpMAX": np.round(buzzp_max_value,2),
                 "buzzwordpPRES": presidents[buzzp_max_index],
                 "buzzwordpMAXYEAR": years[buzzp_max_index],
                 "buzzwordpMAXPARTY": parties[buzzp_max_index][0:1],
                 "uniquewordsMIN": uniquewords_min_value,
                 "uniquewordsMINYEAR": years[uniquewords_min_index],
                 "uniquewordsMINPRES": presidents[uniquewords_min_index],
                 "uniquewordsMINPARTY": parties[uniquewords_min_index][0:1],
                 "uniquewordsMAX": uniquewords_max_value,
                 "uniquewordsMAXYEAR": years[uniquewords_max_index],
                 "uniquewordsMAXPRES": presidents[uniquewords_max_index],
                 "uniquewordsMAXPARTY": parties[uniquewords_max_index][0:1],
                 "totUniqueWords": totUniqueWords,
                 "presidents": presidents,
                 "years": years,
                 "parties": parties,
                 "appendixdata": zip(years, presidents, parties)}



html_out = template.render(template_vars)
from weasyprint import HTML
HTML(string=html_out,base_url=os.getcwd()).write_pdf(output + ".pdf")
shutil.rmtree(tempdir_path, ignore_errors=False, onerror=None)
