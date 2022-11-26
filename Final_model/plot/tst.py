import matplotlib.pyplot as plt

# 对matplotlib一知半解的不用往下看了


# 下述是为保证中文宋体，英文stix（近似Times New Roman），更具体说明见https://zhuanlan.zhihu.com/p/118601703


# 实际上做的是$$内应用LaTex语法，采用字体为mathtext.fontset=stix，其外默认的字体为宋体SimSun


plt.style.use('classic')

plt.rcParams['legend.framealpha'] = 0  # 图例框完全透明

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

config = {

    "font.family": 'serif',

    "font.size": 24,

    "mathtext.fontset": 'stix',

    "font.serif": ['SimSun'],

}

plt.rcParams.update(config)

fig = plt.figure()

ax = fig.add_subplot(111)

fig.add_axes(ax)

u = list(range(100))

p = [i ** 2 for i in u]

ax.plot(u, p, color="k", lw=1)

ax.set_ylabel(r"荷载$P\rm{/kN}$")

# 实际上要解决的就这句，怎么把其中的Delta转为斜体，且保证“位移”和“/mm”为正体


ax.set_xlabel(r'位移$\Delta\rm{/mm}$')

plt.show()

