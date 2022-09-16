import matplotlib.pyplot as plt

import matplotlib.widgets as mwidgets

fig, ax = plt.subplots()

ax.plot([1, 2, 3], [10, 50, 100])


def onselect(eclick, erelease):

    print(eclick.xdata, eclick.ydata)

    print(erelease.xdata, erelease.ydata)


props = dict(facecolor='blue', alpha=0.5)

rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,
                                  props=props)

fig.show()
