# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:31:20 2019

@author: woutv

https://towardsdatascience.com/solving-business-usecases-by-recommender-system-using-lightfm-4ba7b3ac8e62


"""

import pandas as pd
import time
import matplotlib.pyplot as plt

from VacancyData import VacancyData
from VacancyHelper2 import *

from lightfm.data import Dataset
from lightfm import LightFM


from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
import numpy as np

# Load data part
qd = VacancyData();

matchings, vacancies, profiles, profilestest = qd.getDataAsPandas()
matchings['rating']=matchings.apply(lambda row:1, axis=1)

#print(matchings.groupby(['ProfielId']))
#print(matchings.groupby(['ProfielId']).sum())
matchings.groupby(['VacatureId']).sum().hist('rating', range=[0,150])
#matchings.hist('rating')
plt.show()

'''
# Creating interaction matrix using rating data
interactions = create_interaction_matrix(df = matchings,
                                         user_col = 'ProfielId',
                                         item_col = 'VacatureId',
                                         rating_col = 'rating')
#print(interactions)

# Create User Dict
profile_dict = create_user_dict(interactions=interactions)
# Create Item dict
vacancy_dict = create_item_dict(df = vacancies,
                               id_col = 'VacatureId',
                               name_col = 'Naam')

mf_model = runMF(interactions = interactions,
                 n_components = 30,
                 loss = 'warp',
                 epoch = 30,
                 n_jobs = 4)

## Calling 10 movie recommendation for user id 11
rec_list = sample_recommendation_user(model = mf_model, 
                                      interactions = interactions, 
                                      user_id = 11, 
                                      user_dict = profile_dict,
                                      item_dict = vacancy_dict, 
                                      threshold = 4,
                                      nrec_items = 10,
                                      show = True)

'''

