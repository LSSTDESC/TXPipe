import ceci

def get_ancestors(dag, job):
    for parent in dag[job]:
        yield parent
        yield from get_ancestors(dag, parent)


def main(pipeline_config_file, target):
    with open(pipeline_config_file) as f:
        pipe_config = yaml.safe_load(f)

    pipe_config['resume'] = False

    with ceci.prepare_for_pipeline(pipe_config):        
        pipe = ceci.Pipeline.create(pipe_config)

    jobs = pipe.run_info[0]
    dag = pipe.build_dag(jobs)

    if target in jobs:
        job = jobs[target]
    else:
        for stage in pipe.stages:
            if target in stage.output_tags():
                job = jobs[stage.instance_name]
                break
        else:
            raise ValueError(f"Could not find job or output tag {target}")

    for ancestor in get_ancestors(dag, job):
        print(ancestor.name)


if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline_config_file')
    parser.add_argument('stage_name_or_output_tag')
    args = parser.parse_args()

    main(args.pipeline_config_file, args.stage_name_or_output_tag)
