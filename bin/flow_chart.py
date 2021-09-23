import sys
sys.path.append('.')
import ceci
import txpipe
import yaml
import argparse
import pygraphviz

parser = argparse.ArgumentParser(description="Make a flow chart from a pipeline file")
parser.add_argument("input", help="Input YML pipeline file")
parser.add_argument("output", help="Output image file")
parser.add_argument("--highlight", default="", nargs="*", help="Highlight inputs and outputs from stage(s)")
args = parser.parse_args()
config = yaml.safe_load(open(args.input))

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
        color = 'darkorange' if (stage.name in args.highlight) or (inp in args.highlight) else 'black'
        graph.add_edge(inp, stage.name, color=color)
    # and to its outputs
    for out, _ in stage.outputs:
        if out not in seen:
            graph.add_node(out, shape='box', color='skyblue', style='filled')
            seen.add(out)
        color = 'darkorchid' if (stage.name in args.highlight) or (out in args.highlight) else 'black'
        graph.add_edge(stage.name, out, color=color)

# finally, output the stage to file
graph.draw(args.output, prog='dot')
