import numpy as np
import matplotlib.pyplot as plt

gsr = (9.68, 9.85, 9.91, 5.43, 0.90)

ann = (95.73, 83.67, 10.02, 5.00, 1.00)

cnn = (99.37, 92.11, 74.71, 55.21, 41.97)

labels = ("mnist_d", "mnist_f", "cifar_10", "cifar_100_c", "cifar_100_f")

plt.bar(labels, gsr)
plt.show()
# gsr_figure = plt.figure()
# gsr_ax = gsr_figure.add_axes([0, 0, 1, 1])
# gsr_ax.bar(labels, gsr)
# plt.show()
#
# ann_figure = plt.figure()
# ann_ax = ann_figure.add_axes([0, 0, 1, 1])
# ann_ax.bar(labels, ann)
# plt.show()
#
# cnn_figure = plt.figure()
# cnn_ax = cnn_figure.add_axes([0, 0, 1, 1])
# cnn_ax.bar(labels, cnn)
# plt.show()
