Insults
=======

Kaggle contest code - 
Detecting insults in Social Commentary at: http://www.kaggle.com/c/detecting-insults-in-social-commentary
My technique was a blend of Logistic Regression(L2) with binary(presence/absence) word grams(uni-grams and bi-grams),
Random Forest with chosen word feature counts(personal references, derogatory words, swear words, 
capitalized characters and words) and a Naive Bayes model using TF-IDF weights.
The blend was done using a Gradient boosted tree(the probabilities from the individual classifiers were features)