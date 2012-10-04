import re
import collections
from operator import itemgetter
import numpy 
import math
import scipy
from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier




def read_training_data(training_file):
    f = open(training_file)
    f.readline()

    data = []
    labels = []
    for row in f:
        row = row.strip().split("\"\"\"")
        label = row[0].split(",")[0]
        text = row[1]       
        data.append(re.sub('_|\.',' ',re.sub("\n|,|\'|\"","",text)))
        labels.append(float(label))
    return {"data":data,"labels":labels}

def read_test_data(test_file):
    f = open(test_file)
    f.readline()
    
    test_data = []
    for row in f:
        row = row.strip().split("\"\"\"")
       
        text = row[1]
       
        test_data.append(re.sub('_|\.',' ',re.sub("\n|\'|\"","",text)))
    return {"data":test_data}

def read_final_test_data(test_file):
    f = open(test_file)
    f.readline()
    
    test_data = []
    ids = []
    for row in f:
        row = row.strip().split("\"\"\"")
       
        text = row[1]
        id = row[0].split(",")[0]
        test_data.append(re.sub('_|\.',' ',re.sub("\n|\'|\"","",text)))
        ids.append(int(id))
    return {"data":test_data, "ids" : ids}

def get_personal_refs(text):
    return (len(re.findall("you| hey | u | yo ",text.lower())))

def get_common_insults(text):
    return len(re.findall("moron|idiot|fool|assh|loser|retard|dick|stupid|sick|dumb|whore|faggot|slut|bigot|douche|turd|jerk|jackass",text.lower() ))
    
def get_common_swear_words(text):
    return len(re.findall("fuck|shit|bitch|pussy|frigg",text.lower() ))

def get_exaggeration(text):
    return (len(re.findall("\?|!",text )))
    
def get_words_upper(text):
    return sum(y.isupper() for  y in re.findall(r"\s", text) )

def get_letters_upper(text):
    return (len(re.findall("[A-Z]", text)))
                
def get_word_count(text):
    return len(re.findall(r"\s", text))+1

def extract_features(texts, feature_functions):
    return [[f(es) for f in feature_functions] for es in texts]



def main():
    
    print("Reading Training Data")
    training = read_training_data("../data/train_with_test.csv")
    nb = MultinomialNB()
    
    
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),token_pattern=ur'\b\w+\b',stop_words=None, min_df=3)
    tfidf_features = tfidf_vectorizer.fit_transform(training["data"])
        
    
    lr = LogisticRegression(C=.1, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, penalty='l2',
          tol=0.0001)
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=ur'\b\w+\b',stop_words=None, min_df=3,binary=True)
    text_features = bigram_vectorizer.fit_transform(training["data"]) 
    
    
    feature_functions = [get_words_upper,get_personal_refs,get_word_count,get_common_insults,get_common_swear_words,get_letters_upper,get_exaggeration]
    features = extract_features(training["data"],
                                    feature_functions)
                                    
    lr.fit(text_features,training["labels"])
    lr_preds = lr.predict_proba(text_features)
    
    nb.fit(tfidf_features, training["labels"])
    nb_preds = nb.predict_proba(tfidf_features)
    
    
   
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(features,training["labels"]) 
    rf_preds = rf.predict_proba(features)
    
    
    gb_features = numpy.empty((len(lr_preds),3))
    for i in range(len(lr_preds)):
       gb_features[i][0] = (lr_preds[i][1])
       gb_features[i][1] = (rf_preds[i][1])
       
       gb_features[i][2] = (nb_preds[i][1])
   
       
    
    gb = GradientBoostingClassifier(n_estimators=200)
    gb.fit(gb_features,training["labels"])
    
    print("Reading Test Data")
    test = read_final_test_data("../data/impermium_verification_set.csv")
    text_features_test = bigram_vectorizer.transform(test["data"])
   
    
    tfidf_features_test = tfidf_vectorizer.transform(test["data"])
    
    
    features = extract_features(test["data"],
                                    feature_functions)
    lr_preds = lr.predict_proba(text_features_test)
    rf_preds = rf.predict_proba(features)
    
    nb_preds = nb.predict_proba(tfidf_features_test)
    
    
    gb_features = numpy.empty((len(lr_preds),3))
    
    lr_pred = []
    rf_pred=[]
    
    gb_pred=[]
    nb_pred=[]
   
    for i in range(len(lr_preds)):
       gb_features[i][0] = (lr_preds[i][1])
       gb_features[i][1] = (rf_preds[i][1])
       gb_features[i][2] = (nb_preds[i][1])
    
       
       lr_pred.append(lr_preds[i][1])
       rf_pred.append(rf_preds[i][1])
      
       nb_pred.append(nb_preds[i][1])
    
       
    predictions = gb.predict_proba(gb_features)
    
    output_file = "submission.csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("id,insult\n")
   
    for i in range(len(test["data"])):
        f.write("%d,%f\n" % (test["ids"][i],predictions[i][1]))
        gb_pred.append(predictions[i][1])
    f.close()
    


if __name__=="__main__":
    main()