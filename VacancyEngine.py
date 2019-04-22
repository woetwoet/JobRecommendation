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

from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
import numpy as np

# Load data part
qd = VacancyData();

matchings, vacancies, profiles, profilestest = qd.getData()
# Creating a dataset    
dataset = Dataset()
dataset.fit((x['ProfielId'] for x in qd.getMatchings()),
            (x['VacatureId'] for x in qd.getMatchings()))

# Check on items and users
num_users, num_items = dataset.interactions_shape()
print('--- Num users: {}, num_items {}. ---'.format(num_users, num_items))


# Adding the vacancy features in the mix
dataset.fit_partial(items=(x['VacatureId'] for x in qd.getVacancies()),
                    item_features=(x['Takenprofiel'] for x in qd.getVacancies()),
#                    user_features=(x['Motivatie'] for x in qd.getProfiles())                    
                    )

'''dataset.fit_partial(items=(x['ISBN'] for x in qd.getVacancies()),
                    item_features=(x['Book-Title'] for x in qd.getVacancies()))
'''

# creating the interaction matrix for the model
(interactions, weights) = dataset.build_interactions(((x['ProfielId'], x['VacatureId'])
                                                      for x in qd.getMatchings()))

#print(interactions)

# creating the item feature matrix for the model
item_features = dataset.build_item_features(((x['VacatureId'], [x['Takenprofiel']])
                                              for x in qd.getVacancies()))

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
start_time = time.time()
model=LightFM(no_components=115,learning_rate=0.027,loss='warp')
model.fit(train,item_features=item_features, epochs=12,num_threads=4, verbose=False)
# model.fit(train,epochs=12,num_threads=4)


# with open('saved_model','wb') as f:
#     saved_model={'model':model}
#     pickle.dump(saved_model, f)

# Start evaluation of the model
print("--- Start model evaluation ---")
auc_train = auc_score(model, train,item_features=item_features).mean()
auc_test = auc_score(model, test,item_features=item_features).mean()

# auc_train = auc_score(model, train).mean()
# auc_test = auc_score(model, test).mean()


print("--- End model evaluation. Run time:  {} mins ---".format((time.time() - start_time)/60))
print("--- Train AUC Score: {} --- ".format(auc_train))
print("--- Test AUC Score: {} --- ".format(auc_test))

# Manual testing
ratingspd = pd.DataFrame(matchings)
ratingspd['rating']=ratingspd.apply(lambda row:'1', axis=1)

user_item_matrix = ratingspd.pivot(index='ProfielId', columns='VacatureId', values='rating')
user_item_matrix.fillna(0, inplace = True)
user_item_matrix = user_item_matrix.astype(np.int32)

# print(user_item_matrix)

itemspd = pd.DataFrame(vacancies)
user_dikt, item_dikt = helper.user_item_dikts(user_item_matrix, itemspd)

# Generate recommendations for the user
helper.similar_recommendation(model, user_item_matrix, '10', user_dikt, item_dikt,threshold = 0)