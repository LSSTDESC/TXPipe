import sys
sys.path.append('.')
import ceci
import txpipe
import yaml
import argparse
import importlib
import pygraphviz

parser = argparse.ArgumentParser(description="Make a flow chart from a pipeline file")
parser.add_argument("input", help="Input YML pipeline file")
parser.add_argument("output", help="Output image file")
parser.add_argument("--highlight", default="", nargs="*", help="Highlight inputs and outputs from stage(s)")
args = parser.parse_args()
config = yaml.safe_load(open(args.input))

# Import any extra modules that the pipeline file says it needs, e.g. to find RAIL stages
for module in str(config.get('modules', '')).split():
    importlib.import_module(module)

# Get all the stage objects. A stage entry can use a different class from
# its name (classname) and rename its inputs and outputs (aliases)
stages = []
for stage in config['stages']:
    cls = ceci.PipelineStage.get_stage(stage.get('classname', stage['name']))
    aliases = stage.get('aliases', {})
    stages.append((
        stage['name'],
        [aliases.get(tag, tag) for tag, _ in cls.inputs],
        [aliases.get(tag, tag) for tag, _ in cls.outputs],
    ))

# Make a graph object
graph = pygraphviz.AGraph(directed=True)

# Nodes we have already added
seen = set()

# Add overall pipeline inputs
inputs = config['inputs']
for inp in inputs.keys():
    graph.add_node(inp, shape='box', color='gold', style='filled')
    seen.add(inp)

for stage_name, stage_inputs, stage_outputs in stages:
    # add the stage itself
    graph.add_node(stage_name, shape='ellipse', color='orangered', style='filled')
    # connect that stage to its inputs
    for inp in stage_inputs:
        if inp not in seen:
            graph.add_node(inp, shape='box', color='skyblue', style='filled')
            seen.add(inp)
        color = 'darkorange' if (stage_name in args.highlight) or (inp in args.highlight) else 'black'
        graph.add_edge(inp, stage_name, color=color)
    # and to its outputs
    for out in stage_outputs:
        if out not in seen:
            graph.add_node(out, shape='box', color='skyblue', style='filled')
            seen.add(out)
        color = 'darkorchid' if (stage_name in args.highlight) or (out in args.highlight) else 'black'
        graph.add_edge(stage_name, out, color=color)

# finally, output the stage to file
graph.draw(args.output, prog='dot')
