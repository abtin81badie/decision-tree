import math 
from collections import Counter
from graphviz import Digraph
import pydotplus
import numpy as np
import pandas as pd
import random
import os
import json
class DataProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(file_name).dropna().drop(['nameOrig','nameDest'],axis=1)

        self.min_max_values = self.calculate_min_max()
        self.train=[]
        self.test=[] 
        self.label_test=[]
        self.label_train=[]

        # self.process_data()
        self.temp=self.calculate_min_max()
    

    def calculate_min_max(self):
        return {
            'amount': (self.data['amount'].min(), self.data['amount'].max()),
            'oldbalanceOrg': (self.data['oldbalanceOrg'].min(), self.data['oldbalanceOrg'].max()),
            'newbalanceOrig': (self.data['newbalanceOrig'].min(), self.data['newbalanceOrig'].max()),  
            'oldbalanceDest': (self.data['oldbalanceDest'].min(), self.data['oldbalanceDest'].max()),
            'newbalanceDest': (self.data['newbalanceDest'].min(), self.data['newbalanceDest'].max()),  
        }

    def discretize(self, value, min_val, max_val, length):
        return np.floor((value - min_val) / ((max_val - min_val) / length))

    def edit_row(self, row):
        label = row.pop('isFraud')

        name = 'type'

        # Define a dictionary to map string values to integer values
        type_mapping = {
                'PAYMENT': 0,
                'TRANSFER': 1,
                'CASH_OUT': 2,
                'DEBIT': 3,
                'CASH_IN': 4
            }

        # Check if the value in row[name] exists in the mapping, then assign the corresponding integer value
        if row[name] in type_mapping:
            row[name] = type_mapping[row[name]]
        else:
            # Handle the case when the value is not found in the mapping
            row[name] = type_mapping['CASH_IN']  # Default to 'CASH_IN'

        for name in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            min_val, max_val = self.min_max_values[name]
            row[name] = self.discretize(row[name], min_val, max_val, 8 if name != 'amount' else 6)
    
        return row, int(label)

    def read_csv(self):
        pos_list = self.data[self.data['isFraud'] == 1].sample(frac=1).to_dict('records')
        neg_list = self.data[self.data['isFraud'] == 0].sample(frac=1).to_dict('records')

        selected_data = pos_list[:211] + neg_list[:9788]
        selected_test = pos_list[-211 + 100:-1] + neg_list[-9788 + 100:-1]


        return selected_data, selected_test

    def process_data(self):
        selected_data, selected_test = self.read_csv()
      
        for row in selected_data:
            xt1,yt1=self.edit_row(row) 
            self.train.append(xt1)
            self.label_train.append(yt1)
        for row in selected_test:
            xt1,yt1=self.edit_row(row) 
            self.test.append(xt1)
            self.label_test.append(yt1)
    

        return 

