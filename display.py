import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("result.csv")
image_name = "img014499"
# 找到file_name列为image_name的行,取出pos_label_1,pos_label_2,pos_label_3
pose_label = df[df["file_name"] == image_name + ".jpg"][["pos_label_1", "pos_label_2", "pos_label_3"]].values[0]
# 找到file_name列为image_name的行,取出pos_pred_1,pos_pred_2,pos_pred_3
pose_pre = df[df["file_name"] == image_name + ".jpg"][["pos_pred_1", "pos_pred_2", "pos_pred_3"]].values[0]

# 用matplotlib画3D图
# 分别绘制从原点到pose_label和pose_pre的向量
# pose_label用实线表示,pose_pre用虚线表示
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')
ax.plot([0, pose_label[0]], [0, pose_label[1]], [0, pose_label[2]], label="label", linewidth=4)
ax.plot([0, pose_pre[0]], [0, pose_pre[1]], [0, pose_pre[2]], label="pred", linestyle="--", linewidth=4)
ax.set_xlabel('X', fontsize=15)
ax.set_ylabel('Y', fontsize=15)
ax.set_zlabel('Z', fontsize=15)
# 设置坐标字号变大
ax.tick_params(labelsize=13)
# 紧凑布局,删除周围白边
plt.tight_layout()
# 保存图片
plt.savefig(f"{image_name}_pos_result.png", bbox_inches='tight')
plt.show()


print(pose_label, pose_pre)