import matplotlib.pyplot as plt
"""

fig = plt.figure()

ax0 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax0.set_title('ax0')

ax1 = fig.add_axes([0.175, 0.5, 0.3, 0.3])
ax1.set_title('ax1')
ax1.set_facecolor('lightgray')
ax1.set_xlim(2, 3)
ax1.set_ylim(2, 3)

 """


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


def enter_axes(event):
    print(event.inaxes)


fig.canvas.mpl_connect('axes_enter_event', enter_axes)

plt.show()
