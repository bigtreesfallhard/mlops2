
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

test_percent = 0.2
random_id = 12345

if __name__ == "__main__": 
    
    base_dir = "/opt/ml/processing"
    
    df = pd.read_csv(
         f"{base_dir}/input/data.csv",
        )

    train_set, test_set = train_test_split(df, test_size = test_percent, random_state = random_id)
    
    X_train, y_train = train_set.drop(columns=['total_lift']), train_set['total_lift']
    X_test, y_test = test_set.drop(columns=['total_lift']), test_set['total_lift']
    
    pd.DataFrame(train_set).to_csv(f"{base_dir}/train/train.csv", index=False)
    pd.DataFrame(test_set).to_csv(f"{base_dir}/test/test.csv", index=False)
    
