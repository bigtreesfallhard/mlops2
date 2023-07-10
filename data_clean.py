
import os
import pandas as pd
import numpy as np

def clean_data(v1):
    v2 = v1.copy()
    
    #Removing less relevant columns
    v2 = v2.dropna(subset=['region','age','weight','height','howlong','gender','eat','train','background','experience','schedule','howlong','deadlift','candj','snatch','backsq','experience','background','schedule','howlong']) 
    v2 = v2.drop(columns=['affiliate','team','name','athlete_id','fran','helen','grace','filthy50','fgonebad','run400','run5k','pullups','train']) 
    
    #Removing Outliers

    v2 = v2[v2['weight'] < 1500] 
    v2 = v2[v2['gender']!='--'] 
    v2 = v2[v2['age']>=18] 
    v2 = v2[(v2['height']<96)&(v2['height']>48)]

    v2 = v2[(v2['deadlift']>0)&(v2['deadlift']<=1105)|((v2['gender']=='Female')&(v2['deadlift']<=636))] 
    v2 = v2[(v2['candj']>0)&(v2['candj']<=395)]
    v2 = v2[(v2['snatch']>0)&(v2['snatch']<=496)]
    v2 = v2[(v2['backsq']>0)&(v2['backsq']<=1069)]
    
    #Cleaning Survey Data

    decline_dict = {'Decline to answer|':np.nan}
    v2 = v2.replace(decline_dict)
    v2 = v2.dropna(subset=['background','experience','schedule','howlong','eat'])

    #encoding background data 

    #encoding background questions 
    v2['rec'] = np.where(v2['background'].str.contains('I regularly play recreational sports'), 1, 0)
    v2['high_school'] = np.where(v2['background'].str.contains('I played youth or high school level sports'), 1, 0)
    v2['college'] = np.where(v2['background'].str.contains('I played college sports'), 1, 0)
    v2['pro'] = np.where(v2['background'].str.contains('I played professional sports'), 1, 0)
    v2['no_background'] = np.where(v2['background'].str.contains('I have no athletic background besides CrossFit'), 1, 0)

    #delete nonsense answers
    v2 = v2[~(((v2['high_school']==1)|(v2['college']==1)|(v2['pro']==1)|(v2['rec']==1))&(v2['no_background']==1))] 


    #encoding experience questions

    #create encoded columns for experience reponse
    v2['exp_coach'] = np.where(v2['experience'].str.contains('I began CrossFit with a coach'),1,0)
    v2['exp_alone'] = np.where(v2['experience'].str.contains('I began CrossFit by trying it alone'),1,0)
    v2['exp_courses'] = np.where(v2['experience'].str.contains('I have attended one or more specialty courses'),1,0)
    v2['life_changing'] = np.where(v2['experience'].str.contains('I have had a life changing experience due to CrossFit'),1,0)
    v2['exp_trainer'] = np.where(v2['experience'].str.contains('I train other people'),1,0)
    v2['exp_level1'] = np.where(v2['experience'].str.contains('I have completed the CrossFit Level 1 certificate course'),1,0)

    #delete nonsense answers
    v2 = v2[~((v2['exp_coach']==1)&(v2['exp_alone']==1))] 

    #creating no response option for coaching start
    v2['exp_start_nr'] = np.where(((v2['exp_coach']==0)&(v2['exp_alone']==0)),1,0)

    #other options are assumed to be 0 if not explicitly selected

    #creating encoded columns with schedule data
    v2['rest_plus'] = np.where(v2['schedule'].str.contains('I typically rest 4 or more days per month'),1,0)
    v2['rest_minus'] = np.where(v2['schedule'].str.contains('I typically rest fewer than 4 days per month'),1,0)
    v2['rest_sched'] = np.where(v2['schedule'].str.contains('I strictly schedule my rest days'),1,0)

    v2['sched_0extra'] = np.where(v2['schedule'].str.contains('I usually only do 1 workout a day'),1,0)
    v2['sched_1extra'] = np.where(v2['schedule'].str.contains('I do multiple workouts in a day 1x a week'),1,0)
    v2['sched_2extra'] = np.where(v2['schedule'].str.contains('I do multiple workouts in a day 2x a week'),1,0)
    v2['sched_3extra'] = np.where(v2['schedule'].str.contains('I do multiple workouts in a day 3\+ times a week'),1,0)

    #removing/correcting problematic responses 
    v2 = v2[~((v2['rest_plus']==1)&(v2['rest_minus']==1))] 

    #points are only assigned for the highest extra workout value (3x only vs. 3x and 2x and 1x if multi selected)
    v2['sched_0extra'] = np.where((v2['sched_3extra']==1),0,v2['sched_0extra'])
    v2['sched_1extra'] = np.where((v2['sched_3extra']==1),0,v2['sched_1extra'])
    v2['sched_2extra'] = np.where((v2['sched_3extra']==1),0,v2['sched_2extra'])
    v2['sched_0extra'] = np.where((v2['sched_2extra']==1),0,v2['sched_0extra'])
    v2['sched_1extra'] = np.where((v2['sched_2extra']==1),0,v2['sched_1extra'])
    v2['sched_0extra'] = np.where((v2['sched_1extra']==1),0,v2['sched_0extra'])

    #adding no response columns
    v2['sched_nr'] = np.where(((v2['sched_0extra']==0)&(v2['sched_1extra']==0)&(v2['sched_2extra']==0)&(v2['sched_3extra']==0)),1,0)
    v2['rest_nr'] = np.where(((v2['rest_plus']==0)&(v2['rest_minus']==0)),1,0)
    #schedling rest days is assumed to be 0 if not explicitly selected


    # encoding howlong (crossfit lifetime)
    v2['exp_1to2yrs'] = np.where((v2['howlong'].str.contains('1-2 years')),1,0)
    v2['exp_2to4yrs'] = np.where((v2['howlong'].str.contains('2-4 years')),1,0)
    v2['exp_4plus'] = np.where((v2['howlong'].str.contains('4\+ years')),1,0)
    v2['exp_6to12mo'] = np.where((v2['howlong'].str.contains('6-12 months')),1,0)
    v2['exp_lt6mo'] = np.where((v2['howlong'].str.contains('Less than 6 months')),1,0)

    #keeping only higest repsonse 
    v2['exp_lt6mo'] = np.where((v2['exp_4plus']==1),0,v2['exp_lt6mo'])
    v2['exp_6to12mo'] = np.where((v2['exp_4plus']==1),0,v2['exp_6to12mo'])
    v2['exp_1to2yrs'] = np.where((v2['exp_4plus']==1),0,v2['exp_1to2yrs'])
    v2['exp_2to4yrs'] = np.where((v2['exp_4plus']==1),0,v2['exp_2to4yrs'])
    v2['exp_lt6mo'] = np.where((v2['exp_2to4yrs']==1),0,v2['exp_lt6mo'])
    v2['exp_6to12mo'] = np.where((v2['exp_2to4yrs']==1),0,v2['exp_6to12mo'])
    v2['exp_1to2yrs'] = np.where((v2['exp_2to4yrs']==1),0,v2['exp_1to2yrs'])
    v2['exp_lt6mo'] = np.where((v2['exp_1to2yrs']==1),0,v2['exp_lt6mo'])
    v2['exp_6to12mo'] = np.where((v2['exp_1to2yrs']==1),0,v2['exp_6to12mo'])
    v2['exp_lt6mo'] = np.where((v2['exp_6to12mo']==1),0,v2['exp_lt6mo'])


    #encoding dietary preferences 
    v2['eat_conv'] = np.where((v2['eat'].str.contains('I eat whatever is convenient')),1,0)
    v2['eat_cheat']= np.where((v2['eat'].str.contains('I eat 1-3 full cheat meals per week')),1,0)
    v2['eat_quality']= np.where((v2['eat'].str.contains('I eat quality foods but don\'t measure the amount')),1,0)
    v2['eat_paleo']= np.where((v2['eat'].str.contains('I eat strict Paleo')),1,0)
    v2['eat_cheat']= np.where((v2['eat'].str.contains('I eat 1-3 full cheat meals per week')),1,0)
    v2['eat_weigh'] = np.where((v2['eat'].str.contains('I weigh and measure my food')),1,0)


    #encoding location as US vs non-US
    US_regions = ['Southern California', 'North East', 'North Central','South East', 'South Central', 'South West', 'Mid Atlantic','Northern California','Central East', 'North West']
    v2['US'] = np.where((v2['region'].isin(US_regions)),1,0)


    #encoding gender
    v2['gender_'] = np.where(v2['gender']=='Male',1,0)
    
    
    return v2
    
