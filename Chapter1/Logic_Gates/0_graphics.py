import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

offset = 0.03

# =========================================================
# AND
# =========================================================
labels_and = np.array([0, 0, 0, 1])
ax = axes[0]

red = X[labels_and == 0]
green = X[labels_and == 1]

ax.scatter(red[:, 0], red[:, 1], color='red', s=120)
ax.scatter(green[:, 0], green[:, 1], color='green', s=150)

for i, txt in enumerate(["(0,0)", "(0,1)", "(1,0)", "(1,1)"]):
    ax.text(X[i,0] + offset, X[i,1] + offset, txt)

x = np.linspace(-0.5, 1.5, 100)
y = -x + 1.5  ### line for AND
ax.plot(x, y, color='orange')

ax.set_title("AND")
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.grid(True)

# =========================================================
# OR
# =========================================================
labels_or = np.array([0, 1, 1, 1])
ax = axes[1]

red = X[labels_or == 0]
green = X[labels_or == 1]

ax.scatter(red[:, 0], red[:, 1], color='red', s=120)
ax.scatter(green[:, 0], green[:, 1], color='green', s=150)

for i, txt in enumerate(["(0,0)", "(0,1)", "(1,0)", "(1,1)"]):
    ax.text(X[i,0] + offset, X[i,1] + offset, txt)

y = -x + 0.5  ### line for OR
ax.plot(x, y, color='orange')

ax.set_title("OR")
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.grid(True)

# =========================================================
# XOR
# =========================================================
labels_xor = np.array([0, 1, 1, 0])
ax = axes[2]

red = X[labels_xor == 0]
green = X[labels_xor == 1]

ax.scatter(red[:, 0], red[:, 1], color='red', s=120)
ax.scatter(green[:, 0], green[:, 1], color='green', s=150)

for i, txt in enumerate(["(0,0)", "(0,1)", "(1,0)", "(1,1)"]):
    ax.text(X[i,0] + offset, X[i,1] + offset, txt)

### lines for XOR
y1 = -x + 0.5
y2 = -x + 1
y3 = x
y4 = np.full_like(x, 0.5)

ax.plot(x, y1, linestyle='--', dashes=(7, 12), color='orange')
ax.plot(x, y2, linestyle='--', dashes=(7, 12), color='orange')
ax.plot(x, y3, linestyle='--', dashes=(7, 12), color='orange')
ax.plot(x, y4, linestyle='--', dashes=(7, 12), color='orange')

ax.set_title("XOR")
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.grid(True)

# -------------------------
# LEYEND
# -------------------------
handles = [
    plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Output = 0'),
    plt.Line2D([], [], color='green', marker='o', linestyle='None', markersize=8, label='Output = 1')
]

fig.legend(handles=handles, loc='upper center', ncol=2)

plt.suptitle("", fontsize=14)

plt.tight_layout()
plt.show()