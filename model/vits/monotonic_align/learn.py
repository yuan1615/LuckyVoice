import numpy as np

value = np.load("/home/admin/yuanxin/Bil-vits/attn.npy")  # 这个是保存的《似然矩阵》
value = value[0].T

t_x, t_y = value.shape

path = np.zeros([t_x, t_y])

Q = float('-inf') * np.ones_like(value)

for y in range(t_y):
    for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
        if y == 0:
            Q[x, 0] = value[x, 0]
        else:
            if x == 0:
                v_prev = float('-inf')
            else:
                v_prev = Q[x-1, y-1]
            v_cur = Q[x, y-1]
            Q[x, y] = value[x, y] + max(v_prev, v_cur)

# Backtrack from last observation
index = t_x - 1
for y in range(t_y - 1, -1, -1):
    path[index, y] = 1
    if index !=0 and (index == y or Q[index, y-1] < Q[index-1, y-1]):
        index = index - 1

# np.save("/home/admin/yuanxin/Bil-vits/path.npy", path)


