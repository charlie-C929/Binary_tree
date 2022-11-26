import random
import csv
import os

#get_path函数用来获取某个路径下所有路径
def get_path(file_dir):
    file_path=[]
    for path in os.listdir(file_dir):
        real_path = os.path.join(file_dir, path)
        file_path.append(real_path)
    return file_path


#polymer
filepath = 'G:\Fundamental\Final_Dance\Material_Produce\Polymer'
# print(get_path(filepath))
paths = get_path(filepath)
for i in range(7):
    with open(paths[i],'r') as f:
        content = f.readlines()[0]
        print(content)
        E1 = float(content.split(',')[0])
        E2 = float(content.split(',')[1])
        E3 = float(content.split(',')[2])
        u12 = float(content.split(',')[3])
        u13 = float(content.split(',')[4])
        u23 = float(content.split(',')[5])
        G12 = float(content.split(',')[6])
        G23 = float(content.split(',')[7])
        G31 = float(content.split(',')[8])
        z = open("Polymer.csv", "a+")
        csv_writer = csv.writer(z)
        csv_writer.writerow([1 / E1, -u12 / E2, -u13 / E3,
                             -u12 / E1, 1 / E2, -u23 / E3,
                             -u13 / E1, -u23 / E2, 1 / E3,
                             1 / G23,1 / G31,1 / G12])
        z.close()

#matrix
m_filepath = 'G:\Fundamental\Final_Dance\Material_Produce\Matrix'
m_paths = get_path(m_filepath)
for i in range(7):
    with open(m_paths[i],'r') as f:
        content = f.readlines()[0]
        #print(content)
        E = float(content.split(',')[0])
        u = float(content.split(',')[1])
        G = E / (2 * (1 + u))
        z = open("Matrix.csv", "a+")
        csv_writer = csv.writer(z)
        csv_writer.writerow([1 / E, -u / E, -u / E,
                             -u / E, 1 / E, -u / E,
                             -u / E, -u / E, 1 / E,
                             1 / G,1 / G,1 / G])
        z.close()

