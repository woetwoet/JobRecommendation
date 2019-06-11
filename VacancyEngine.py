# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:31:20 2019

@author: woutv
"""

import pandas as pd
import time

from VacancyData import VacancyData
from VacancyHelper import helper

from lightfm.data import Dataset
from lightfm import LightFM
import pickle

from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import reciprocal_rank
import numpy as np

# Load data part

print ('--------------------------')
print ('--- Start Recommending ---')
print ('--------------------------')

qd = VacancyData();

matchings, vacancies, profiles, profilestest = qd.getData()
# Creating a dataset    
dataset = Dataset(user_identity_features=False, item_identity_features=False)
dataset.fit((x['ProfielId'] for x in qd.getMatchings()),
            (x['VacatureId'] for x in qd.getMatchings()))

# Check on items and users 
num_users, num_items = dataset.interactions_shape()
print('--- Interaction set : Num users: {}, num_items {}. ---'.format(num_users, num_items))

# Adding the features in the mix
dataset.fit_partial(items=(x['VacatureId'] for x in qd.getVacancies()),
                    item_features=(x['Naam'] for x in qd.getVacancies()),
                    )
'''dataset.fit_partial(items=(x['VacatureId'] for x in qd.getVacancies()),
                    item_features=(x['Taal'] for x in qd.getVacancies()),
                    )

dataset.fit_partial(items=(x['VacatureId'] for x in qd.getVacancies()),
                    item_features=(x['Functie'] for x in qd.getVacancies()),
                    )

dataset.fit_partial(users=(x['Id'] for x in qd.getProfiles()),
                    user_features=(x['Motivatie'] for x in qd.getProfiles())                    
                    )
'''
num_users, num_items = dataset.interactions_shape()
print('--- Total set : Num users: {}, num_items {}. ---'.format(num_users, num_items))


# creating the interaction matrix for the model
(interactions, weights) = dataset.build_interactions(((x['ProfielId'], x['VacatureId'])
                                                      for x in qd.getMatchings()))
#print(interactions.toarray())



# creating the item feature matrix for the model
'''item_features = dataset.build_item_features(((x['VacatureId'], [x['Naam'],x['Taal'],x['Functie']])
                                              for x in qd.getVacancies()),normalize=False)
'''
item_features = dataset.build_item_features(((x['VacatureId'], [x['Naam']])
                                              for x in qd.getVacancies()),normalize=False)


# print(item_features.toarray())

print(dataset.mapping())
'''
user_features = dataset.build_user_features(((x['Id'], [x['Motivatie']])
                                             for x in qd.getProfiles()))
print(user_features)
'''

# Creating a user fettu
# Split the set in train and test
test , train = random_train_test_split(interactions, test_percentage=0.2, random_state=None)

# Start training the model
print("--- Start model training ---")
model=LightFM(no_components=1,learning_rate=0.027,loss='warp')
model.fit(train,item_features=item_features, epochs=100,num_threads=4, verbose=False)
# model.fit(train,epochs=12,num_threads=4)

modelnofeatures=LightFM(no_components=1,learning_rate=0.027,loss='warp')
modelnofeatures.fit(train, epochs=100,num_threads=4, verbose=False)

# model.fit(train,epochs=12,num_threads=4)
'''

with open('saved_model','wb') as f:
     saved_model={'model':model}
     pickle.dump(saved_model, f)


model = pickle.load( open( "saved_model", "rb" ) )

'''

print(model.item_embeddings)
#print(model.user_embeddings)

'''
# Start evaluation of the model
print("--- Start model evaluation ---")
# Default k is 10. K is top N in which the precision or recall is measured.

topN = 5
start_time = time.time()

auc_train = auc_score(model, train,item_features=item_features).mean()
auc_test = auc_score(model, test,item_features=item_features).mean()
precision_train = precision_at_k(model, train, k=topN, item_features=item_features).mean()
precision_test = precision_at_k(model, test, k=topN, item_features=item_features).mean()
recall_train = recall_at_k(model, train,k=topN, item_features=item_features).mean()
recall_test = recall_at_k(model, test,k=topN, item_features=item_features).mean()

print("--- End model evaluation. Run time:  {} mins ---".format((time.time() - start_time)/60))

print('Auc: train %.2f, test %.2f.' % (auc_train, auc_test))
print('Precision: train %.2f, test %.2f.' % (precision_train, precision_test))
print('Recall: train %.2f, test %.2f.' % (recall_train, recall_test))

auc_trainnf = auc_score(modelnofeatures, train).mean()
auc_testnf = auc_score(modelnofeatures, test).mean()
precision_trainnf = precision_at_k(modelnofeatures, train, k=topN).mean()
precision_testnf = precision_at_k(modelnofeatures, test, k=topN).mean()
recall_trainnf = recall_at_k(modelnofeatures, train,k=topN).mean()
recall_testnf = recall_at_k(modelnofeatures, test,k=topN).mean()

print("--- End model evaluation model no features. Run time:  {} mins ---".format((time.time() - start_time)/60))

print('Auc: train %.2f, test %.2f.' % (auc_trainnf, auc_testnf))
print('Precision: train %.2f, test %.2f.' % (precision_trainnf, precision_testnf))
print('Recall: train %.2f, test %.2f.' % (recall_trainnf, recall_testnf))

'''
# Manual testing
ratingspd = pd.DataFrame(matchings)
ratingspd['rating']=ratingspd.apply(lambda row:'1', axis=1)

user_item_matrix = ratingspd.pivot(index='ProfielId', columns='VacatureId', values='rating')
user_item_matrix.fillna(0, inplace = True)
user_item_matrix = user_item_matrix.astype(np.int32)


# print(dataset.mapping())

itemspd = pd.DataFrame(vacancies)
user_dikt, item_dikt = helper.user_item_dikts(user_item_matrix, itemspd)

# Generate recommendations for the user
# helper.similar_recommendation(model, user_item_matrix, '5', user_dikt, item_dikt,threshold = 0)

# Generate predictions for the user and features. TODO clean up
# the num_items must be the list of items before partial fit. Dunno why but after the partial fit interaction matrix gets screwed up....
helper.similar_recommendation_features(model, user_item_matrix, '1', item_dikt,dataset,interactions,item_features,threshold = 0)





