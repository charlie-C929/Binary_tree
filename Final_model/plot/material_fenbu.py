import xlrd
import matplotlib.pyplot as plt

data = xlrd.open_workbook("material_pic.xlsx")
table = data.sheet_by_index(0)
#print(table.name, table.nrows, table.ncols)
x = table.col_values(0)
y = table.col_values(1)



plt.scatter(x[0:300:2],y[0:300:2],label=r'$Training\rm$ $Dataset\rm$')
plt.scatter(x[301::3],y[301::3],label=r'$Testing\rm$ $Dataset\rm$',marker='^')
#plt.plot(step[1::17],validation_value[1::17],label='validation_accuracy')
plt.legend()
plt.xlabel(r'$log_{10}(E_{22}/E_{11})\rm$')
plt.ylabel(r'$log_{10}(E_{33}/E_{11})\rm$')
plt.ylim((-1.1,1.1))
plt.xlim((-1.1,1.1))
plt.show()
