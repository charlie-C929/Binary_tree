import torch
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt
#
model=resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
mode='cosineAnnWarm'
if mode=='cosineAnn':
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
elif mode=='cosineAnnWarm':
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2)
    '''
    以T_0=5, T_mult=1为例:
    T_0:学习率第一次回到初始值的epoch位置.
    T_mult:这个控制了学习率回升的速度
        - 如果T_mult=1,则学习率在T_0,2*T_0,3*T_0,....,i*T_0,....处回到最大值(初始学习率)
            - 5,10,15,20,25,.......处回到最大值
        - 如果T_mult>1,则学习率在T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,.....,(1+T_mult+T_mult**2+...+T_0**i)*T0,处回到最大值
            - 5,15,35,75,155,.......处回到最大值
    example:
        T_0=5, T_mult=1
    '''
plt.figure()
max_epoch=50
iters=200
cur_lr_list = []
for epoch in range(max_epoch):
    for batch in range(iters):
        '''
        这里scheduler.step(epoch + batch / iters)的理解如下,如果是一个epoch结束后再.step
        那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
        则可以在这里进行.step
        '''
        #scheduler.step(epoch + batch / iters)
        optimizer.step()
    scheduler.step()
    cur_lr=optimizer.param_groups[-1]['lr']
    cur_lr_list.append(cur_lr)
    print('cur_lr:',cur_lr)
x_list = list(range(len(cur_lr_list)))
plt.plot(x_list, cur_lr_list,'o-',label=r'$current$ $learning$ $rate\rm$')
plt.legend()
plt.xlabel(r'$epoch\rm$')
plt.ylabel(r'$current$ $learning$ $rate\rm$')
plt.show()