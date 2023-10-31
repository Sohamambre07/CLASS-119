#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np 

words=[]
classes=[]
word_tags_list=[]
ignore_words=["?","!",".",",",";",":"]
train_data_file=open("intents.json").read()
intents = json.loads(train_data_file)
#print(intents)

# function for appending stem words
def get_stem_words(words,ignore_words):
        stem_words=[]
        for word in words:
                if word not in ignore_words:
                        ww=stemmer.stem(word.lower())
                        stem_words.append(ww)
        return stem_words                   


    
        # Add all words of patterns to list
for intent in intents["intents"]:
        for pattern in intent["patterns"]:
                patternword = nltk.word_tokenize(pattern)
                words.extend(patternword)
                word_tags_list.append((patternword,intent['tag']))
        if intent['tag'] not in classes:
                classes.append(intent['tag'])      
                stem_words=get_stem_words(words,ignore_words)  
        # Add all tags to the classes list
         
print(classes)
#Create word corpus for chatbot
def create_bot_corpus(stem_words,classes):
        stem_words = sorted(list(set(stem_words)))
        classes = sorted(list(set(classes)))
        pickle.dump(stem_words,open("words.pkl","wb"))
        pickle.dump(classes,open("classes.pkl","wb"))
        return stem_words,classes

stem_words,classes = create_bot_corpus(stem_words,classes) 
print(stem_words)
print(classes)       
