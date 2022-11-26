#encoding:utf-8
import random
import csv
import os
import numpy as np
component_polymer_path = 'Polymer.csv'
component_matrix_path = 'Matrix.csv'
composite_result_path ='result_modify.csv'
p_data = []
m_data = []
r_data = []

#polymer
with open(component_polymer_path,'r',encoding='UTF-8-sig') as f:
    row = csv.reader(f, delimiter=',')
    for item in row:
        real = []
        for thing in item:
            num =float(thing)
            real.append(num)
        matrix = []
        matrix.append(real[0:6])
        matrix.append(real[6:12])
        matrix.append(real[12:18])
        matrix.append(real[18:24])
        matrix.append(real[24:30])
        matrix.append(real[30:36])
        p_data.append(matrix)
# hula = np.array(data[0])
# print(hula)

#matrix
with open(component_matrix_path,'r',encoding='UTF-8-sig') as f:
    row = csv.reader(f, delimiter=',')
    for item in row:
        real = []
        for thing in item:
            num =float(thing)
            real.append(num)
        matrix = []
        matrix.append(real[0:6])
        matrix.append(real[6:12])
        matrix.append(real[12:18])
        matrix.append(real[18:24])
        matrix.append(real[24:30])
        matrix.append(real[30:36])
        m_data.append(matrix)

#result
with open(composite_result_path,'r',encoding='UTF-8-sig') as f:
    row = csv.reader(f, delimiter=',')
    for item in row:
        real = []
        for thing in item:
            num =float(thing)
            real.append(num)
        matrix = []
        matrix.append(real[0:6])
        matrix.append(real[6:12])
        matrix.append(real[12:18])
        matrix.append(real[18:24])
        matrix.append(real[24:30])
        matrix.append(real[30:36])
        r_data.append(matrix)


a = np.array(p_data[0])*0.0075+np.array(m_data[0])*0.9925
b = np.linalg.inv(np.array(p_data))
print(a)
print(np.array(r_data[0]))