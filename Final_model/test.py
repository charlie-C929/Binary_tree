# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
#
# component_train_file_path = "G:\Fundamental\Final_Dance\Material_Produce\Final_data.csv"
# composite_train_file_path = 'G:\Fundamental\Final_Dance\Material_Produce\\results\\result.csv'
#
# data = pd.read_csv(component_train_file_path)
# data_values = data.values
# print(data)
# print(data_values)
#
# ans = pd.read_csv(composite_train_file_path)
# ans_values = ans.values
# print(ans)
#
# x_train_all, x_test, y_train_all, y_test = train_test_split(
#      data_values, ans_values, random_state = 10)
# x_train, x_valid, y_train, y_valid, = train_test_split(
#     x_train_all, y_train_all, random_state = 1)
# print(len(x_train_all))
# print(len(x_train))

a = [1,2,3,4,5]
for item in a[:-1]:
    print(item)