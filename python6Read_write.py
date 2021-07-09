# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:50:55 2021

@author: RM
"""

import pandas

# Reading CSV Files with Pandas:
df = pandas.read_csv('C:/Users/RM/Downloads/User_Data (1).csv')
print(df)

# Writing CSV Files with Pandas:
df.to_csv('C:/Users/RM/Downloads/User_Data (1).csv')

# Reading Excel Files with Pandas
df1 = pandas.read_excel('C:/Users/RM/Downloads/User_Data.xlsx')

df1 = pandas.read_excel('User_Data.xlsx')
print(df1)

# Writing Excel Files with Pandas 
df1.to_excel('IIT-B.xlsx')
df2 = pandas.DataFrame(df1)
print (df2)
