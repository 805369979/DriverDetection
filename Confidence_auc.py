import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 生成数据（与之前相同）
np.random.seed(42)
n_groups = 10
n_subgroups = 1
n_samples = 100

groups = ['Adjust\nRadio','Drink','Drive\nSafe','Hair\nMakeup','Reach\nBehind','Talk\nLeft','Talk\nPassenger','Talk\nRight','Text\nLeft','Text\nRight']
subgroups = [f"Sub_{chr(65+j)}" for j in range(n_subgroups)]

data = []
aaa = ['./confidenceFile/aaa.csv','./confidenceFile/auc1.csv','./confidenceFile/my.csv','./confidenceFile/my.csv','./confidenceFile/my.csv']
# top_class_index,top_class_probability,x_real,real_class_probability
for index,group in enumerate(groups):
    print(group)
    for key,subgroup in enumerate(subgroups):
        print(subgroup)
        aucCondidence = pd.read_csv(aaa[key])
        # print(aucCondidence['top_class_index'])
        select = aucCondidence.loc[aucCondidence['x_real'] == index]
        # print(select)
        # select = select.loc[select['top_class_index'] == index]
        # print(select['real_class_probability'])
        select = select['real_class_probability'].to_numpy()
        print(select.shape)
        data.extend([{"Group": group, "Subgroup": subgroup, "Value": v} for v in select])

df =pd.DataFrame(data)
# 设置样式和配色方案
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 10))

# 定义淡雅的配色方案（莫兰迪色系）
pastel_palette = [
        "#FF5E70",  # 灰蓝色
        "#F0C3FF",  # 浅橙
        "#84FF68",  # 灰绿
        "#FFFC32",  # 粉棕
        "#58C3FF"   # 淡紫
]

# pastel_palette = [
#     "#FF5E70",  # 灰蓝色
#     "#F0C3FF",  # 浅橙
#     "#84FF68",  # 灰绿
#     "#FFFC32",  # 粉棕
#     "#58C3FF"   # 淡紫
# ]

# 绘制箱线图（使用新配色）
box = sns.violinplot(
    data=df,
    # showfliers=False,
    x="Group",
    # palette="pastel",
    y="Value",
    cut=0,
    linecolor="red",
    hue="Subgroup",
    # scale="count",
    inner="stick",
    palette=pastel_palette,  # 替换为淡雅配色
    notch=True,
    linewidth=1.8,
    width=0.6,
    dodge=True,
    flierprops={"marker": ".", "markersize": 2, "markerfacecolor": "none", "markeredgecolor": "#666666"},  # 离群点颜色
    medianprops={"color": "black", "linewidth": 4}  # 中位数线颜色
)

# for line in box.get_lines():
#     line.set_color("#EB7280")  # 修改内部竖线颜色
# 遍历每个小提琴对象
# for violin in box.collections:
    # violin.set_edgecolor("#707070")  # 边框颜色
    # violin.set_facecolor("#EB7280")  # 填充颜色
# 计算均值
means = df.groupby(["Group", "Subgroup"])["Value"].mean().reset_index()

# 计算子组偏移量
n_hue = len(subgroups)
width = 0.8
hue_offsets = (np.arange(n_hue) - (n_hue - 1)/2) * (width / n_hue)

# 绘制五角星（调整颜色与整体风格一致）
for group_idx, group in enumerate(groups):
    for sub_idx, subgroup in enumerate(subgroups):
        x = group_idx + hue_offsets[sub_idx]
        mean_val = means[(means["Group"] == group) & (means["Subgroup"] == subgroup)]["Value"].values
        plt.scatter(
            x, mean_val,
            marker="*",
            s=200,
            color="#ffff00",  # 淡红色五角星（与配色协调）
            edgecolor="black",  # 边缘深红色
            linewidth=1,
            zorder=10
        )

# 添加主组间的垂直虚线分隔线
for i in range(1, n_groups):
    plt.axvline(
        x=i - 0.5,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        zorder=0
    )

# 调整横坐标标签
plt.xticks(
    ticks=np.arange(n_groups),
    labels=groups,
    rotation=0,
    ha="center",
    fontsize=18
)

plt.yticks(fontsize=18)
plt.xlim(-0.5, n_groups - 0.5)
# plt.ylim(0, 1)
# 优化图例（使用新配色）
handles = [Patch(facecolor=color) for color in pastel_palette]
# plt.legend(
#     handles=[]
#     # handles, subgroups,
#     # title="Subgroup",
#     # bbox_to_anchor=(1.01, 1),
#     # loc="upper left",
#     # frameon=True,
#     # framealpha=0.9
# )
plt.legend().set_visible(False)
# 调整标题和标签
plt.title("Classification  Confidence", fontsize=24, pad=20)
plt.xlabel("", fontsize=24)
plt.ylabel("AUC  test  set  classification  confidence", fontsize=20)
plt.tight_layout()

plt.savefig('confidence.png')
plt.savefig('confidence.eps')
plt.savefig('confidence.pdf')
plt.show()