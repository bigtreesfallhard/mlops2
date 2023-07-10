
import os
import pandas as pd

def total_lift(X):
    
    df = X.copy()
    
    df['norm_dl'] = df['deadlift']/df['weight']
    df['norm_j'] = df['candj']/df['weight']
    df['norm_s'] = df['snatch']/df['weight']
    df['norm_bs'] = df['backsq']/df['weight']

    df['total_lift'] = df['norm_dl']+df['norm_j']+df['norm_s']+df['norm_bs']
    
    return df    
    

