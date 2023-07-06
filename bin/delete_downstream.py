"""
This script prints out the commands to delete all files generated by a pipeline,
downstream of a specified stage.

If one stage was wrong, and you need to re-run everything it affected, this script
will print out the commands to delete the relevant files to that re-running the pipeline
with resume=True will re-run the correct stages.
"""
import sys
sys.path.append('.')
import ceci
import txpipe
import yaml
import collections
import os

config_file = sys.argv[1]
stage_to_delete = sys.argv[2]

# get the stages we need
pipe_config = ceci.Pipeline.build_config(config_file, ["resume=False"], dry_run=True)

with ceci.main.prepare_for_pipeline(pipe_config):
    pipeline = ceci.Pipeline.create(pipe_config)

stage_names = [s['name'] for s in pipe_config['stages']]
stages = pipeline.stages

# build the mapping tag => stages depending on that tag
dependencies = collections.defaultdict(list)
for stage in stages:
    for tag in stage.find_inputs(pipeline.pipeline_files):
        dependencies[tag].append(stage)


# initialize with deletng one stage and the tags it makes
tags_to_delete = ceci.PipelineStage.get_stage(stage_to_delete).output_tags()
stages_to_delete = {stage}

# loop through nstage times (the maximum it could be)
for i in range(len(stage_names)):
    # take all tags we currently know we have to delete
    for tag in tags_to_delete[:]:
        # find out which stages to clear because they need
        # this tag which we are deleting
        deps = set(dependencies[tag])
        for s in stages:
            if s in deps:
                # if we need to delete this stage,
                # add its outputs to the tags to delete
                tags_to_delete += s.output_tags()
                # and it to the stages to delete
                stages_to_delete.add(s)
    tags_to_delete = list(set(tags_to_delete))

# now at the end we delete all tags output by stage to delete
for s in stages_to_delete:
    for f in s.find_outputs(pipe_config['output_dir']).values():
        print(f"rm -f {f}")
