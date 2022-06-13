# !pip install pandas requests BeautifulSoup4
# !pip install gmaps
import pandas
import pandas as pd
import requests
import gmaps
import plotly.graph_objects as go

import array
from pickle import NONE
import re
import nltk
import timeit
import folium as f
import csv
import random
from sys import maxsize
from itertools import permutations
import numpy

from google.colab import output
output.enable_custom_widget_manager()

API_key = "AIzaSyCFI1DAgc89wqpetVeDmY6Yql6i72VmL7Y"

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('wordnet')
from nltk.corpus import wordnet

nltk.download('sentiwordnet')
from nltk.stem import WordNetLemmatizer

#Problem 1
wordnet_lemmatizer = WordNetLemmatizer()

positive_counter = 0
negative_counter = 0
neutral_counter = 0


# clean text function
def clean(text):
    # Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text


# POS tagger dictionary and remove stop words
pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}


def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


def KMPSearch(pat, txt):
    P = len(pat)
    T = len(txt)
    # create lps[] to hold the longest proper prefix suffix
    lps = [0] * P
    j = 0  # index for pat[]
    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, P, lps)
    i = 0  # index for txt[]
    while i < T:
        if pat[j] == txt[i]:
            i += 1
            j += 1

            if j == P:
                j = lps[j - 1]
                return True

        # mismatch
        elif i < T and pat[j] != txt[i]:
            if j != 0:
                # change j to the portion that will not repeating checking the same prefix
                j = lps[j - 1]
            else:
                i += 1
    return False


def computeLPSArray(pat, P, lps):
    len = 0  # length of the previous lps
    lps[0]  # lps[0] is always 0
    i = 1
    # the loop calculates lps[i] for i = 1 to  P-1
    while i < P:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            if len != 0:
                # return to previous value until find the same one
                len = lps[len - 1]
            else:
                # fail to find any same value from previous value
                lps[i] = 0
                i += 1
                # start comparing with the first value


def readFile(wordFile):
    with open(wordFile, 'r', encoding="utf8") as file:
        data = file.read().replace('\n', '').replace(',', '')
        data.strip
    return data

def matchingCountryList(country,textFileList,countrylist):
    global positive_counter, negative_counter, neutral_counter
    positive_counter = 0
    negative_counter = 0
    neutral_counter = 0
    for j in range(len(textFileList)):
        textFile = textFileList[j]
        data = pd.read_csv(textFile, sep='t')
        # print(data.head())
        mydata = data.drop('Unnamed: 0', axis=1)
        # print(mydata.head())

        # clean text
        mydata['Cleaned words'] = mydata['words'].apply(clean)
        # print(mydata.head())

        # POS tagged and remove stop words
        mydata['POS tagged'] = mydata['Cleaned words'].apply(token_stop_pos)
        # print(mydata.head())

        mydata['Lemma'] = mydata['POS tagged'].apply(lemmatize)
        # print(mydata.head())

        # print(len(mydata.index))
        positive_words = readFile('positive word.txt')
        negative_words = readFile('negative word.txt')
        i = 0
        while i < len(mydata.index):
            val = mydata['Lemma'].values[i]
            if pd.isnull(mydata.loc[i, 'Lemma']):
                i += 1
                continue
            else:
                i += 1
                if (KMPSearch(val, positive_words)):
                    positive_counter += 1
                elif (KMPSearch(val, negative_words)):
                    negative_counter += 1
                else:
                    neutral_counter += 1
   
    sentiment_list = [positive_counter,negative_counter,neutral_counter]
    countrylist.append({"name": country,
        "Positive Word Count":sentiment_list[0],
        "Negative Word Count":sentiment_list[1],
        "Neutral Word Count":sentiment_list[2]
    })
    return countrylist
    
def matching(country,textFileList):
    global positive_counter, negative_counter, neutral_counter
    positive_counter = 0
    negative_counter = 0
    neutral_counter = 0
    for j in range(len(textFileList)):
        textFile = textFileList[j]
        data = pd.read_csv(textFile, sep='t')
        # print(data.head())
        mydata = data.drop('Unnamed: 0', axis=1)
        # print(mydata.head())

        # clean text
        mydata['Cleaned words'] = mydata['words'].apply(clean)
        # print(mydata.head())

        # POS tagged and remove stop words
        mydata['POS tagged'] = mydata['Cleaned words'].apply(token_stop_pos)
        # print(mydata.head())

        mydata['Lemma'] = mydata['POS tagged'].apply(lemmatize)
        # print(mydata.head())

        # print(len(mydata.index))
        positive_words = readFile('positive word.txt')
        negative_words = readFile('negative word.txt')
        i = 0
        while i < len(mydata.index):
            val = mydata['Lemma'].values[i]
            if pd.isnull(mydata.loc[i, 'Lemma']):
                i += 1
                continue
            else:
                i += 1
                if (KMPSearch(val, positive_words)):
                    positive_counter += 1
                elif (KMPSearch(val, negative_words)):
                    negative_counter += 1
                else:
                    neutral_counter += 1

    sentiment_list = [positive_counter,negative_counter,neutral_counter]
    return sentiment_list

def plotGraph(countrylist,count):
	country = [countrylist[0]['name'], countrylist[1]['name'], countrylist[2]['name'], countrylist[3]['name'], countrylist[4]['name']]
	dict_of_fig = dict({"data":[
		{"type": "bar", "x": [countrylist[0]['name'], countrylist[1]['name'], countrylist[2]['name'], countrylist[3]['name'], countrylist[4]['name']], "y": [countrylist[0][count], countrylist[1][count], countrylist[2][count], countrylist[3][count], countrylist[4][count]]}], "layout": {"title": {"text": count+ " Graph"}}})
	fig = go.Figure(dict_of_fig)
	fig.update_xaxes(title_text='Country')
	fig.update_yaxes(title_text='Counts')
	fig.show()

def plotMark(countrylist,percentage):
	country = [countrylist[0]['name'], countrylist[1]['name'], countrylist[2]['name'], countrylist[3]['name'], countrylist[4]['name']]
	dict_of_fig = dict({"data":[
		{"type": "bar", "x": [countrylist[0]['name'], countrylist[1]['name'], countrylist[2]['name'], countrylist[3]['name'], countrylist[4]['name']], "y": [percentage[0], percentage[1], percentage[2], percentage[3], percentage[4]]}], "layout": {"title": {"text": "Graph of Percentage of Positive Word Count with respect to Total Word Counts"}}})
	fig = go.Figure(dict_of_fig)
	fig.update_xaxes(title_text='Country')
	fig.update_yaxes(title_text='Percentage of positive word count(%)')
	fig.show()


# Problem 2
def findRandomStores(store, ranNum):
  rows, cols = (10, 2)
  coordinate = [[0] * cols] * rows
  coordinate_Name = [[0]*cols]*rows

  for i in range(0, 10):
      coordinate[i] = [store.Latitude[ranNum[i]], store.Longitude[ranNum[i]]]
      coordinate_Name[i] = store.Name[ranNum[i]]

  return coordinate, coordinate_Name

def findDistMatrix(coordinate):
  lat_long_list = "|".join([f"{l[0]},{l[1]}" for l in coordinate])
  distance_matrix=[]
  URL = ("https://maps.googleapis.com/maps/api/distancematrix/json?language=en-US&units=meters"
        +"&origins={}"+'&destinations={}'
        +'&key={}').format(lat_long_list, lat_long_list, API_key)

  response = requests.request("GET", URL)
  j_son = response.json() #convert the txt body of response into json format
  result = j_son['rows']

  for x in result:
      oriToDest=[]
      info = x["elements"]

      for dist in info:
          oriToDest.append(dist['distance']['value'])
      distance_matrix.append(oriToDest)
    
  return distance_matrix

def findCenter(coordinate, distance_matrix, coordinate_Name):
  min = 0
  center_index = 0

  for i in range(len(distance_matrix[0])):
      min += distance_matrix[0][i]

  for i in range(len(distance_matrix)):
      dis_travel = 0
      for j in range(len(distance_matrix[i])):
          dis_travel += distance_matrix[i][j]
      if dis_travel < min:
          min = dis_travel
          center_index = i
  
  centreName = ""
  for j in range(len(coordinate)):
      if j == center_index:
          center = coordinate[j]
          centreName = coordinate_Name[j]

  return center, center_index, centreName

def plotShortestPath(data):
    store = pd.read_csv(data)
    # we use this list to get non-repeating elemets
    list = range(0, len(store))
    ranNum = random.sample(list, 10)

    coordinate, coordinate_Name = findRandomStores(store, ranNum)
    distance_matrix = findDistMatrix(coordinate)
    center, center_index, centreName = findCenter(coordinate, distance_matrix, coordinate_Name)

    # implementation of traveling Salesman Problem
    # store all vertex apart from source vertex
    path = []
    vertex = []
    V = len(distance_matrix)
    for i in range(V):
        if i != center_index:
            vertex.append(i)

    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation = permutations(vertex)
    for i in next_permutation:

        # store current Path weight(cost)
        current_pathweight = 0

        # compute current path weight
        k = center_index
        for j in i:
            current_pathweight += distance_matrix[k][j]
            k = j
        current_pathweight += distance_matrix[k][center_index]

        # update minimum
        if min_path > current_pathweight:
            min_path = current_pathweight
            path_list = i

    path.append(center_index)
    for i in path_list:
        path.append(i)
    path.append(center_index)

    return coordinate, coordinate_Name, center, centreName, path, min_path, distance_matrix
    
# print the map with route
def mapWithRoute(path, coordinate, coordinate_Name, centreName, center):
  gmaps.configure(api_key=API_key)
  fig = gmaps.figure(map_type='ROADMAP')
    
  print("Path: ")
  for i in range(len(path)-1):
      ori = coordinate[path[i]]
      des = coordinate[path[i+1]]
      for j in range(len(coordinate)):
          if ori == coordinate[j]:
              storeName = coordinate_Name[j]
              print(storeName)
              break;
      fig.add_layer(gmaps.directions_layer(ori,des,stroke_color='red',show_markers=False, stroke_weight=2.0, stroke_opacity=1.0))

  print(centreName)

  markers = gmaps.marker_layer(coordinate, info_box_content=coordinate_Name)
  center_markers = gmaps.marker_layer([center],label='C', info_box_content=centreName)
  fig.add_layer(markers)
  fig.add_layer(center_markers)
    
  return fig

# Problem 3

# This function is used to calculate the probability of a country to be selected
# Weightage is the fraction that one factors hold in the total probability
# If we consider sentiment score (economic and social situation) is as important as delivery
# Then each of them will have a weigtage of 0.5
def min_max_normalisation(score_list, weightage):
    normalized_list = list()
    minimum = min(score_list)
    maximum = max(score_list)
    for i in range(len(score_list)):
        normalized_list.append(((score_list[i] - minimum) / (maximum - minimum)) * weightage)
        normalized_list[i]=round(normalized_list[i],4)
    return normalized_list


# Function to display array as table
def printList(country, sentiment, distance, total):
    score_table = list()
    for i in range(len(country_list)):
        score_table.append([country[i], sentiment[i], distance[i], total[i]])
    score_table = numpy.array(score_table)
    column_labels = ["Country", "Sentiment Score", "Distance Score", "Total"]
    row_labels = [1, 2, 3, 4, 5]
    score_table = pandas.DataFrame(score_table, columns=column_labels, index=row_labels)
    print(score_table)
    print()

# Function to sort the probability of a country to have good local economic and optimal delivery
def mergeSort(country, sentiment, distance, total):
    if len(total) > 1:
        # Finding the mid of the array
        mid = len(total) // 2
        # Dividing the array elements into 2 halves
        L0 = country[:mid]
        R0 = country[mid:]
        L1 = sentiment[:mid]
        R1 = sentiment[mid:]
        L2 = distance[:mid]
        R2 = distance[mid:]
        L3 = total[:mid]
        R3 = total[mid:]
        # Sorting the first half
        mergeSort(L0, L1, L2, L3)
        # Sorting the second half
        mergeSort(R0, R1, R2, R3)
        
        # i is pointer for L3
        # j is pointer for R3
        # k is pointer for output array
        i = j = k = 0
        
        # Before reaching either end of L3 or R3
        while i < len(L3) and j < len(R3):
            # Select the larger element and place into k position of output array
            if L3[i] > R3[j]:
                total[k] = L3[i]
                country[k] = L0[i]
                sentiment[k] = L1[i]
                distance[k] = L2[i]
                i += 1
            else:
                total[k] = R3[j]
                country[k] = R0[j]
                sentiment[k] = R1[j]
                distance[k] = R2[j]
                j += 1
            k += 1
            
        # Place the remaining element at index k of output array
        while i < len(L3):
            total[k] = L3[i]
            country[k] = L0[i]
            sentiment[k] = L1[i]
            distance[k] = L2[i]
            i += 1
            k += 1
        while j < len(R3):
            total[k] = R3[j]
            country[k] = R0[j]
            sentiment[k] = R1[j]
            distance[k] = R2[j]
            j += 1
            k += 1

# List to store country name
country_list = ["France", "Malaysia", "Singapore", "United Kingdom", "United States"]

#List to store country article file
MY_list = ["MY.txt","MY2.txt","MY3.txt","MY4.txt","MY5.txt"]
SG_list = ["SG.txt","SG2.txt","SG3.txt","SG4.txt","SG5.txt"]
UK_list = ["UK.txt","UK2.txt","UK3.txt","UK4.txt","UK5.txt"]
US_list = ["US.txt","US2.txt","US3.txt","US4.txt","US5.txt"]
FR_list = ["FR.txt","FR2.txt","FR3.txt","FR4.txt","FR5.txt"]

#matchingCountryList
countrylist= []
countrylist = matchingCountryList("France",FR_list,countrylist)
countrylist = matchingCountryList("Malaysia",MY_list,countrylist)
countrylist = matchingCountryList("Singapore",SG_list,countrylist)
countrylist = matchingCountryList("United Kingdom",UK_list,countrylist)
countrylist = matchingCountryList("United States",US_list,countrylist)

# plot Graphs        
plotGraph(countrylist,"Positive Word Count")
plotGraph(countrylist,"Negative Word Count")
plotGraph(countrylist,"Neutral Word Count")

# calculate the Percentage of Positive Word Count with respect to Total Positive and Negative Word Counts
percentage = []
for y in range(len(countrylist)):
    percentage.append((countrylist[y]["Positive Word Count"])/(countrylist[y]["Positive Word Count"]+countrylist[y]["Negative Word Count"]+countrylist[y]["Neutral Word Count"])*100)
# plot the Graph of Percentage of Positive Word Count with respect to Total Positive and Negative Word Counts
plotMark(countrylist,percentage)

# Calculate probability score for sentiment
# Assume sentiment is equally important with distance
# Weightage = 0.5
sentiment_score = list()
for i in range (len(sentiment_list)):
    probability = percentage[i]/100*0.5
    sentiment_score.append(round(probability,4))

# Store the shortest distance of each country (optimal delivery) in distance_list
distance_list = list()

print("FR")
data = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSL3hIEO010RdV9D7J5v4gFPtKrHecZE40ALyJvMClpzkOwLCjJ-0CxyJ1keJ1W3YrLRSFvHdMn-pPd/pub?gid=0&single=true&output=csv'
coordinate_FR, coordinate_Name_FR, center_FR, centreName_FR, path_FR, min_path_FR, distance_matrix_FR = plotShortestPath(data)
print("Coordinate : ", coordinate_FR, "\n", 
      "Coordinate Name: ", coordinate_Name_FR, "\n", 
      "Distance Matrix: ", distance_matrix_FR, "\n",
      "Distribution Centre Coordinate: ", center_FR, "\n", 
      "Distribution Centre Name: ", centreName_FR, "\n", 
      "Path: ", path_FR, "\n", 
      "Minimum Path Cost: ", min_path_FR,
      "\n")
distance_list.append(min_path_FR)
mapWithRoute(path_FR, coordinate_FR, coordinate_Name_FR, centreName_FR, center_FR)

print("GB")
data = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSdjNHDAM3ta6gFAAsg8kFksBHH8GFpo4bamO0xEl_mXtntW1gVZRvJmxG5fSavZ7QTetknE0T21z4g/pub?output=csv'
coordinate_GB, coordinate_Name_GB, center_GB, centreName_GB, path_GB, min_path_GB, distance_matrix_GB = plotShortestPath(data)
print("Coordinate : ", coordinate_GB, "\n", 
      "Coordinate Name: ", coordinate_Name_GB, "\n", 
      "Distance Matrix: ", distance_matrix_GB, "\n", 
      "Distribution Centre Coordinate: ", center_GB, "\n", 
      "Distribution Centre Name: ", centreName_GB, "\n", 
      "Path: ", path_GB, "\n", 
      "Minimum Path Cost: ", min_path_GB,
      "\n")
distance_list.append(min_path_GB)
mapWithRoute(path_GB, coordinate_GB, coordinate_Name_GB, centreName_GB, center_GB)

print("US")
data = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQY1F342p3QH2B0xmbPUFjddPe0RJmOCT_HmNWU7QR55FEwhvIbZSEadtJPQ1Ddj1bvaUcNgI_96_q-/pub?output=csv'
coordinate_US, coordinate_Name_US, center_US, centreName_US, path_US, min_path_US, distance_matrix_US = plotShortestPath(data)
print("Coordinate : ", coordinate_US, "\n", 
      "Coordinate Name: ", coordinate_Name_US, "\n", 
      "Distance Matrix: ", distance_matrix_US, "\n",
      "Distribution Centre Coordinate: ", center_US, "\n", 
      "Distribution Centre Name: ", centreName_US, "\n", 
      "Path: ", path_US, "\n", 
      "Minimum Path Cost: ", min_path_US,
      "\n")
distance_list.append(min_path_US)
mapWithRoute(path_US, coordinate_US, coordinate_Name_US, centreName_US, center_US)

print("MY")
data = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTrUpg5xoNjC9uzEJwVyHTAc06tv6gAJpJ-6w8_qH7A60fHdLoCFShBpb1-W8ZCQ6dC0mnUFMvyC9Lf/pub?output=csv'
coordinate_MY, coordinate_Name_MY, center_MY, centreName_MY, path_MY, min_path_MY, distance_matrix_MY  = plotShortestPath(data)
print("Coordinate : ", coordinate_MY, "\n", 
      "Coordinate Name: ", coordinate_Name_MY, "\n", 
      "Distance Matrix: ", distance_matrix_MY, "\n",
      "Distribution Centre Coordinate: ", center_MY, "\n", 
      "Distribution Centre Name: ", centreName_MY, "\n", 
      "Path: ", path_MY, "\n", 
      "Minimum Path Cost: ", min_path_MY,
      "\n")
distance_list.append(min_path_MY)
mapWithRoute(path_MY, coordinate_MY, coordinate_Name_MY, centreName_MY, center_MY)

print("SG")
data = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQVq6V5a9G5v86w4Ldn_wGvKrWELtsRvg9esjKF3-aa5M8kVM4BF7yI_tJxgu7QBhabZnjPatolz4Wk/pub?output=csv'
coordinate_SG, coordinate_Name_SG, center_SG, centreName_SG, path_SG, min_path_SG, distance_matrix_SG = plotShortestPath(data)
print("Coordinate : ", coordinate_SG, "\n", 
      "Coordinate Name: ", coordinate_Name_SG, "\n", 
      "Distance Matrix: ", distance_matrix_MY, "\n",
      "Distribution Centre Coordinate: ", center_SG, "\n", 
      "Distribution Centre Name: ", centreName_SG, "\n", 
      "Path: ", path_SG, "\n", 
      "Minimum Path Cost: ", min_path_SG,
      "\n")
distance_list.append(min_path_SG)
mapWithRoute(path_SG, coordinate_SG, coordinate_Name_SG, centreName_SG, center_SG)

# Calculate normalised probability score for distance
# Weightage = 0.5
distance_score = min_max_normalisation(distance_list,0.5)

# The shorter distance should have higher ranking but when 
# calculate the larger distance will have higher score
# Therefore, use the total probability that this factor hold 
# to deduct the score get to obtain the actual probability.
for i in range(len(distance_score)):
    distance_score[i] = round(0.5 - distance_score[i],4)

# List to store total score of each country
total_score = [0] * len(country_list)
for i in range(len(sentiment_score)):
    total_score[i] = round(sentiment_score[i] + distance_score[i],4)

# Print the unsorted list
print()
print("Problem 3: ")
print("Unsorted Ranking Table: ")
printList(country_list, sentiment_score, distance_score, total_score)

# Use merge sort to sort according to total_score
mergeSort(country_list, sentiment_score, distance_score, total_score)

# Print the sorted list
print("Sorted Ranking Table:")
printList(country_list, sentiment_score, distance_score, total_score)

# stop = timeit.default_timer()
# print('Time: ', stop - start)
