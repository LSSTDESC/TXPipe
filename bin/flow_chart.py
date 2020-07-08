import sys

sys.path.append('.')
import ceci
import txpipe
import yaml
import pygraphviz

config = yaml.safe_load(open(sys.argv[1]))

# Get all the stage objects
stages = [ceci.PipelineStage.get_stage(stage['name']) for stage in config['stages']]

# Make a graph object
graph = pygraphviz.AGraph(directed=True)

# Nodes we have already added
seen = set()

# Add overall pipeline inputs
inputs = config['inputs']
for inp in inputs.keys():
    graph.add_node(inp, shape='box', color='gold', style='filled')
    seen.add(inp)

for stage in stages:
    # add the stage itself
    graph.add_node(stage.name, shape='ellipse', color='orangered', style='filled')
    # connect that stage to its inputs
    for inp, _ in stage.inputs:
        if inp not in seen:
            graph.add_node(inp, shape='box', color='skyblue', style='filled')
            seen.add(inp)
        graph.add_edge(inp, stage.name)
    # and to its outputs
    for out, _ in stage.outputs:
        if out not in seen:
            graph.add_node(out, shape='box', color='skyblue', style='filled')
            seen.add(out)
        graph.add_edge(stage.name, out)

# finally, output the stage to file
graph.draw(sys.argv[2], prog='dot')
