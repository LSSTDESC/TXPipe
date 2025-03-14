import ceci

pipeline_files = [
    "examples/2.2i/pipeline.yml",
    "examples/buzzard/pipeline.yml",
    "examples/clmm/pipeline.yml",
    "examples/cosmodc2/pipeline.yml",
    "examples/desy1/pipeline.yml",
    "examples/desy3/pipeline.yml",
    "examples/dp0.2/pipeline.yml",
    "examples/kids-1000/pipeline.yml",
    "examples/lensfit/pipeline.yml",
    "examples/lognormal/pipeline.yml",
    "examples/metacal/pipeline.yml",
    "examples/metadetect/pipeline.yml",
    "examples/metadetect_source_only/pipeline.yml",
    "examples/mock_shear/pipeline.yml",
    "examples/redmagic/pipeline.yml",
    "examples/skysim/pipeline.yml",
]

def get_tags(pipeline_file):
    tags = set()

    pipe_config = ceci.Pipeline.build_config(pipeline_file, dry_run=True)

    with ceci.prepare_for_pipeline(pipe_config) as pipeline:
        p = ceci.Pipeline.create(pipe_config)

        # First pass, get the classes for all the stages
        stage_classes = []
        for stage_name in p.stage_names:
            sec = p.stage_execution_config[stage_name]
            stage_classes.append(sec.build_stage_class())

            stage_aliases = sec.aliases
            stage_class = sec.stage_class
            for tag,ftype in stage_class.outputs:
                aliased_tag = stage_aliases.get(tag, tag)
                if ftype.suffix is not None:
                    tags.add((ftype.suffix, aliased_tag))
                # if ftype.suffix is None:
                #     print("NONE TAG TYPE:", stage_name, pipeline_file, stage_class)
            for tag, ftype in stage_class.inputs:
                aliased_tag = stage_aliases.get(tag, tag)
                if ftype.suffix is not None:
                    tags.add((ftype.suffix, aliased_tag))
                # if ftype.suffix is None:
                #     print("NONE TAG TYPE:", stage_name, pipeline_file, stage_class)
                
    return tags




def main():
    tags = set()
    for pipeline_file in pipeline_files:
        try:
            tags.update(get_tags(pipeline_file))
        except ceci.errors.StageNotFound:
            print(f"Error: {pipeline_file} old")
            continue
        except ValueError:
            print(f"Error: {pipeline_file} broken")
            continue
    # print(tags)
    for tag in sorted(tags):
        print(tag)
            



if __name__ == "__main__":
    main()