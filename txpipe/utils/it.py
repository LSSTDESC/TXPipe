
def combined_iterators(stage, inputs, rows):

    iterators = zip(*[stage.iterate_hdf(tag, section, cols, rows) 
                 for tag, section, cols in inputs])
    # breakpoint()
    # 1
    # 2
    # 3
    for its in iterators:
        data = {}
        for (s, e, d) in its:
            data.update(d)
        yield s, e, data
