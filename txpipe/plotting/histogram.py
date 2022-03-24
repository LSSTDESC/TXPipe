def manual_step_histogram(edges, counts, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    x = [edges[0]]
    y = [0]

    for i, c in enumerate(counts):
        x.append(edges[i])
        y.append(c)
        x.append(edges[i + 1])
        y.append(c)
    x.append(edges[-1])
    y.append(0)

    p = plt.Line2D(x, y, **kwargs)
    ax.add_line(p)
    ax.autoscale_view()
    ax.set_ylim(0, None)
