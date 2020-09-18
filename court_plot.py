from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as patches
import numpy as np

squash_court_lines = [
        [(0, 261), 160, 5],
        [(160, 261), 5, 165],
        [(0, 426), 640, 5],
        [(317.5, 0), 5, 426],
        [(475, 261), 5, 165],
        [(480, 261), 160, 5],
        ]

n = "001"
that = "data/frames/match_1_rally_1_1080_60fps/image-{0}.jpeg".format(n)

# figure(num=None, figsize=(6.4, 9.75), dpi=100, facecolor='k', edgecolor='k')
figure(num=None, figsize=(6.4, 9.75), dpi=100)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.xlim(0, 640)
plt.ylim(0, 975)
rect = plt.Rectangle( (0, 0), 640, 975, fill=True, facecolor='grey',
        edgecolor='black')
plt.gca().add_patch(rect)
for xy, w, h in squash_court_lines:
    line_color = 'gold'
    rect = plt.Rectangle( xy, w, h, fill=True, facecolor=line_color,
            edgecolor=line_color)
    plt.gca().add_patch(rect)
plt.savefig("court_plot.png", bbox_inches='tight', transparent=
        True)


# fig,ax = plt.subplots(1)

# for xy, w, h in squash_court_lines:
    # rect = plt.Rectangle( xy, w, h, fill=True, edgecolor='red')
    # ax.add_patch(rect)

# plt.savefig("court_plot.png")

