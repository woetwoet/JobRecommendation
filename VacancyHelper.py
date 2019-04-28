# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:18:03 2019

@author: woutv

See 
- https://github.com/nxs5899/Recommender-System-LightFM/blob/master/informed_train-test_Recommender.ipynb


"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix 


class helper:

    def user_item_dikts(interaction_matrix, items_df):
        user_ids = list(interaction_matrix.index)
        user_dikt = {}
        counter = 0 
        for i in user_ids:
            user_dikt[i] = counter
            counter += 1
    
        item_dikt ={}
        for i in range(items_df.shape[0]):
            item_dikt[(items_df.loc[i,'VacatureId'])] = items_df.loc[i,'Naam']
        
        return user_dikt, item_dikt    
    
    def similar_recommendation(model, interaction_matrix, user_id, user_dikt, 
                               item_dikt,threshold = 0,number_rec_items = 5):

        #Function to produce user recommendations
    
        n_users, n_items = interaction_matrix.shape
        #print('--- Num users: {}, num_items {}. ---'.format(n_users, n_items))

        user_x = user_dikt[user_id]
        scores = pd.Series(model.predict(user_x,np.arange(n_items)))
        #print(scores)
        scores.index = interaction_matrix.columns
        #print(scores)
        scores = list(pd.Series(scores.sort_values(ascending=False).index))
        
        known_items = list(pd.Series(interaction_matrix.loc[user_id,:][interaction_matrix.loc[user_id,:] > threshold].index).sort_values(ascending=False))
                
        scores = [x for x in scores if x not in known_items]
        score_list = scores[0:number_rec_items]
        known_items = list(pd.Series(known_items).apply(lambda x: item_dikt[x]))
        scores = list(pd.Series(score_list).apply(lambda x: item_dikt[x]))
    
        print("Jobs that are chosen by the user:")
        counter = 1
        for i in known_items[:25]:
            print(str(counter) + '- ' + i)
            counter+=1
    
        print("\n Recommended Jobs:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
        #  return score_list
        
    def similar_recommendation_features(model, interaction_matrix, user_id, 
                               item_dikt, dataset, interactions,item_features, threshold = 0,number_rec_items = 3):
    
        # Get an overview on the entire map
        n_users, n_items =dataset.interactions_shape()
        
        # Get the mappings
        user_map = dataset.mapping()[0]
        item_map = dataset.mapping()[2]
        
        # get the user x from the map
        user_x = user_map[user_id]
        scores = pd.Series(model.predict(user_x,np.arange(n_items),item_features=item_features))
        
        # get the list of items from the map
        item_map_list = list(item_map.keys()) 
        scores.index = item_map_list[:n_items]
        scores = list(pd.Series(scores.sort_values(ascending=False).index))
        
        # instantiate an empty list
        known_items = list()
        
        # Check if the user allready had an interaction, If not no need to check the interaction matrix 
        if (interaction_matrix.index == user_id).any():
            known_items = list(pd.Series(interaction_matrix.loc[user_id,:][interaction_matrix.loc[user_id,:] > threshold].index).sort_values(ascending=False))
            known_items_mapped = list(pd.Series(known_items).apply(lambda x: item_dikt[x]))
            print("Jobs that are chosen by the user:")
            counter = 1
            for i in known_items_mapped[:25]:
                print(str(counter) + '- ' + i)
                counter+=1
        else:
            print("The user has not applied for a job yet...")
        # Get the scores and filter out those which are allready known        
        scores = [x for x in scores if x not in known_items]
        # Limit the scores to the max nr of items
        score_list = scores[0:number_rec_items]
        scores = list(pd.Series(score_list).apply(lambda x: item_dikt[x]))
    
        # Print the recommended Jobs
        print("Recommended Jobs:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
        #  return score_list 
          
    
    def users_for_item(model,interaction_matrix,vacancyid,number_of_user):
  
        #Funnction to produce a list of top N interested users for a given item
    
        n_users, n_items = interaction_matrix.shape
        x = np.array(interaction_matrix.columns)
        scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(vacancyid),n_users)))
        user_list = list(interaction_matrix.index[scores.sort_values(ascending=False).head(number_of_user).index])
        return user_list 
    

    def item_emdedding_distance_matrix(model,interaction_matrix):
    
    #     Function to create item-item distance embedding matrix
    
        df_item_norm_sparse = csr_matrix(model.item_embeddings)
        similarities = cosine_similarity(df_item_norm_sparse)
        item_emdedding_distance_matrix = pd.DataFrame(similarities)
        item_emdedding_distance_matrix.columns = interaction_matrix.columns
        item_emdedding_distance_matrix.index = interaction_matrix.columns
        return item_emdedding_distance_matrix
    
    def also_bought_recommendation(item_emdedding_distance_matrix, item_id, 
                                 item_dikt, n_items = 4):
    
    #     Function to create item-item recommendation
    
        recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                      sort_values(ascending = False).head(n_items+1). \
                                      index[1:n_items+1]))
        
        print("Item of interest :{}".format(item_dikt[item_id]))
        print("Items that are frequently bought together:")
        counter = 1
        for i in recommended_items:
            print(str(counter) + '- ' +  item_dikt[i])
            counter+=1
        return recommended_items