import xlrd
import matplotlib.pyplot as plt

data = xlrd.open_workbook("accuracy.xlsx")
table = data.sheet_by_index(0)
#print(table.name, table.nrows, table.ncols)
step = table.col_values(0)
train_value = table.col_values(1)
validation_value = table.col_values(3)


data_loss = xlrd.open_workbook("loss.xlsx")
table_loss = data_loss.sheet_by_name("Sheet1")
# print(table_loss)
step_loss = table_loss.col_values(0)
train_loss_value = table_loss.col_values(2)
# print(train_loss_value)
validation_loss_value = table_loss.col_values(5)

plt.plot(step[1::22],train_value[1::22],".-",label='train_accuracy')
plt.plot(step[1::22],validation_value[1::22],"*-",label='validation_accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(step_loss[1::17],train_loss_value[1::17],".-",label='train_loss')
plt.plot(step_loss[1::17],validation_loss_value[1::17],".-",label='validation_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


