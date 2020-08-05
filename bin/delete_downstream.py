import sys
sys.path.append('.')
import ceci
import txpipe
import yaml
import collections
import os

config = yaml.safe_load(open(sys.argv[1]))
stage_to_delete = sys.argv[2]

stage_names = [s['name'] for s in config['stages']]
pipeline = ceci.Pipeline(config['stages'], None)
stages = [ceci.PipelineStage.get_stage(stage_name) for stage_name in stage_names]


dependencies = collections.defaultdict(list)
for stage in stages:
    for tag in stage.input_tags():
        dependencies[tag].append(stage)

tags_to_delete = ceci.PipelineStage.get_stage(stage_to_delete).output_tags()
stages_to_delete = set()

for i in range(len(stage_names)):
    for tag in tags_to_delete:
        deps = set(dependencies[tag])
        for s in stages:
            if s in deps:
                tags_to_delete += s.output_tags()
                stages_to_delete.add(s)
    tags_to_delete = list(set(tags_to_delete))

for s in stages_to_delete:
    for f in pipeline.find_outputs(s, config).values():
        print(f"rm {f}")