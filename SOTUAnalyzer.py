# Created by Mishaal Aleem
# September 2016

#---------------------------------------------------------------------------------------
# Program that analyzes word usage in the State of the Union Addresses
# delivered as speeches from 1913 - 2016. Accepts user input to analyze specific words.
#---------------------------------------------------------------------------------------

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
# Create a directory to save figures
#-----------------------------
dir_path = "output"
if os.path.exists(dir_path) == False:
   os.makedirs(dir_path)
   
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
buzzword = [];
buzzword = buzzword + [raw_input('Enter a word to analyze: ')]
listDone = False
while(listDone == False):
   done = raw_input('Enter another word? (yes/no ONLY): ')
   if(done == "yes"):
      buzzword = buzzword + [raw_input('Enter a word to analyze: ')]
   elif(done == "no"):
      listDone = True
   else:
      print("ERROR: Enter 'yes' or 'no' (without apostrophes) ONLY")

f = open('output.html','w')

f.write("""<html> <head> <title> SOTU Address Analysis </title></head> <body><h1>State of the Union Address - Word Analysis</h1>  <p>The State of the Union Address - Word Analysis program assess the content of the State of the Union Addresses and Messages which were delivered as speeches from the year 1913 onwards. The text from the speeches is sourced from <a href = "http://www.presidency.ucsb.edu/sou.php">The American Presidency Project.</a></p>""") 

f.write("""<h2>General Analysis</h2> <p> The following sections analyzes total word usage in the whole Sate of the Union (SOTU) address. First, we look at the overall length of the addresses. Note that there was not necessarily a SOTU address every year from 1913 onward (see Appendix for detailed list). The years where there was an address is marked, with a different marker denoting a Democratic president (Blue Triangle) or Republican (Red Circle).</p>""")

#-----------------------------
#General Analysis
#-----------------------------
totWords = [];
for i in range (0, len(years)):
	dat = speeches[i]
	words = dat.split()
	totWords = totWords + [len(words)]
	wordCount = Counter(words)
   
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

fig.savefig(dir_path + '/fig1.png')

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


write = """ <center> <h3> Total Number of Words </h3>
<img src = '"""
f.write(write)
f.write(dir_path + '/fig1.png')
write = """ '> </center>"""
f.write(write)

f.write("<b>Longest: </b>" + str(total_max_value) + " words by " + str(presidents[total_max_index]) + " (" + str(parties[total_max_index][0:1]) + ")" + " in " + str(years[total_max_index]) + "<br>")
f.write("<b>Shortest: </b>" + str(total_min_value) + " words by " +  str(presidents[total_min_index]) +" (" +  str(parties[total_min_index][0:1]) +") in " +  str(years[total_min_index]) + "<br>")
f.write("<b>Average: </b>" + str(averageLength) + " words <br>")
f.write("<b>Democratic Average: </b>" + str(demAverageLength) + " words <br>")
f.write("<b>Republican Average: </b>" + str(repAverageLength) + " words <br>")

f.write("""<p> Next, we look at the speechs in historical context by overlapping a plot of speech length with historical event lines. For ease of reading, the Democratic/Republican differentiating markers have been replaced with simple black circles for all speeches.</p>""")


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

fig.savefig(dir_path + '/fig2.png',bbox_inches="tight")

f.write("<center> <h3> Total Words Used and Hisorical Events </h3> <img src = '" + dir_path + '/fig2.png' + "'> </center>")

f.write("<p>Note that the SOTU is typically delivered at the beginning of the year, thus the events of the previous year impact the speech of the following year.</p>")

f.write("Next we consider the unique words used in each speech. Each speech is analyzed without punctuation and capitalization, to count the total number of unique words.")

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
fig_TotalUniqueCount.savefig(dir_path + '/fig3.png')

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

f.write("<center> <h3> Total Number of Unique Words Used </h3> <img src = '" + dir_path + "/fig3.png'> </center>")

f.write("<b>Least number of Unique Words in an address: </b>" + str(uniquewords_min_value) + " unique words by" + 
str(presidents[uniquewords_min_index]) + " (" + str(parties[uniquewords_min_index][0:1]) + ") in " + str(years[uniquewords_min_index]) + "<br>")

f.write("<b>Most number of Unique Words in an address: </b>" + str(uniquewords_max_value) + " unique words by" + 
str(presidents[uniquewords_max_index]) + " (" + str(parties[uniquewords_max_index][0:1]) + ") in " + str(years[uniquewords_max_index])  + "<br>")


f.write("<p>There have been " + str(totUniqueWords) + " total unique words in all the SOTU addresses. We can also look at the unique word usage on a party level.</p>")


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
fig_TotalUniquePartyCount.savefig(dir_path + '/fig4.png')


f.write(" <center> <h3> Total Unique Words Used by Each Political Party </h3> <img src = '" + dir_path + "/fig4.png'> </center>")

f.write("<p>It makes more sense to normalize unique word usage by the length of the speech. We can look at it as a percentage of the total speech length, both for individual addresses and for each party.</p>")

f.write("<blockquote><i> For Each Address: </i>Unique Words % = (Total Number of Unique Words) / (Total Number of Words) * 100</blockquote>")

f.write("<blockquote><i> For Each Party: </i>Unique Words % = (Total Number of Unique Words Used by Party) / (Total Number of Words Delivered by Party) * 100</blockquote>")

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
fig_UniquePer.savefig(dir_path + '/fig5.png')

write = """ <center> <h3> Unique Words Used as Percentage of Speech</h3>
<img src = '"""
f.write(write)
f.write(dir_path + '/fig5.png')
write = """ '> </center>"""
f.write(write)


uniquep_min_index, uniquep_min_value = min(enumerate(uniquePercent), key=operator.itemgetter(1))
uniquep_max_index, uniquep_max_value = max(enumerate(uniquePercent), key=operator.itemgetter(1))


f.write("<b>Highest % Unique Words Used:</b> " + str(np.round(uniquep_max_value,2)) + "% by " + str(presidents[uniquep_max_index]) + " (" + str(parties[uniquep_max_index][0:1]) + ") in " + str(years[uniquep_max_index]) + "<br>")
f.write("<b>Lowest % Unique Words Used:</b> " + str(np.round(uniquep_min_value,2)) + "% by " + str(presidents[uniquep_min_index]) + " (" + str(parties[uniquep_min_index][0:1]) + ") in " + str(years[uniquep_min_index]))



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
fig_UniquePerParty.savefig(dir_path + '/fig6.png')

f.write(" <center> <h3> Unique Words Used as Percentage of Speech for Each Party </h3> <img src = '" + dir_path + "/fig6.png'> </center>")


#-----------------------------
#Buzzword Analysis
#-----------------------------
for i in range(0,len(buzzword)):
   
   write = """ <h2> Word to analyze: '""" + buzzword[i]  + """'</h2> """
   f.write(write)
   
   timesUsed = [];
   for j in range (0, len(years)):
      dat = speeches[j]
      words = dat.split()
      wordCount = Counter(words)
      timesUsed =  timesUsed + [wordCount[buzzword[i]]]

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

   fig_TotalWordsCountWithHistorical.savefig(dir_path + '/' + str(i) + '_' + 'fig1.png')

   f.write("<center> <h3> Total Usage of '" + buzzword[i] + "'</h3>")
   f.write(" <img src = '" + dir_path + "/" + str(i) + "_" + "fig1.png'> </center>")
   
   #print(timesUsed)
   f.write("<b>Total Usage:</b> " + str(sum(timesUsed)) + " times <br>")
   f.write("<b>Most Used:</b> " + str(buzz_max_value) + " times by " + str(presidents[buzz_max_index]) + " (" + str(parties[buzz_max_index][0:1]) +") in " + str(years[buzz_max_index]))

   f.write("<p> Again, this is actually more sensible to look at normalized by the length of the speech. Once again we can calculate a percentage usage for each address for the selected word. </p>")

   f.write("<blockquote> '" + buzzword[i] + "' Use % = (Total Number of times '" + buzzword[i] + "' Used) / (Total Number of Words) * 100</blockquote>")
   
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
   fig3.savefig(dir_path + '/' + str(i) + '_' + 'fig2.png')
   
   f.write(" <center> <h3> Perentage of Total Speech Usage of '" + buzzword[i] + "'</h3>")
   f.write(" <img src = '" + dir_path + "/" + str(i) + "_" + "fig2.png'> </center>")

   f.write("<b>Highest % Used:</b> " + str(np.round(buzzp_max_value,2)) + "% by " + str(presidents[buzzp_max_index]) + " (" + str(parties[buzzp_max_index][0:1]) + ") in " + str(years[buzzp_max_index]))

f.write("<h2> Appendix </h2>")
f.write("<h3> Speeches Analyzed </h3>")


f.write("""<center><table style="width:50%">
  <tr>
    <th>Year</th>
    <th>President</th> 
    <th>Party</th>
  </tr>""")

for year, pres, party in list(zip(years, presidents, parties)):
   f.write("""<tr>
    <td>""" + year + """</td>
    <td>""" + pres + """</td>
    <td>""" + party +  """</td>
   </tr>""")

f.write("</table></center>")

write = """</body></html>"""
f.write(write)
f.close()
