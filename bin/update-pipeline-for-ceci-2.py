#!/usr/bin/env python
import ruamel.yaml
import sys


def update_pipeline_file(pipeline_info, config_info):
    for stage_info in pipeline_info["stages"]:
        name = stage_info["name"]

        stage_config = config_info.get(name, None)

        if stage_config is None:
            continue

        aliases = stage_config.get("aliases", None)

        if aliases is not None:
            aliases = {k:v.strip() for k, v in aliases.items()}
            stage_info["aliases"] = aliases

def update_config_file(config_info):
    for stage_info in config_info.values():
        stage_info.pop("aliases", None)


def main(pipeline_files):
    # configure yaml - these are the approximate
    # settings we have mostly used in TXPipe
    yaml = ruamel.yaml.YAML()
    yaml.indent(sequence=4, offset=4, mapping=4)
    yaml.allow_duplicate_keys = True

    # Read all the pipeline files
    pipeline_infos = []
    for pipeline_file in pipeline_files:

        with open(pipeline_file) as f:
            yaml_str = f.read()

        pipeline_info = yaml.load(yaml_str)
        config_file = pipeline_info["config"]
        pipeline_infos.append(pipeline_info)

    # Check that all the pipeline files use the same config file
    for pipeline_info in pipeline_infos:
        if not pipeline_info["config"] == config_file:
            raise ValueError("All pipeline files supplied to this script should use the same config file. Run the script multiple times on different files otherwise.")

    # Read the config file
    with open(config_file) as f:
        yaml_str = f.read()
    config_info = yaml.load(yaml_str)

    # Update all the pipeline files.
    for pipeline_info in pipeline_infos:
        update_pipeline_file(pipeline_info, config_info)
 
    # Only now can we delete the alias information
    update_config_file(config_info)

    # Update all the files in-place
    for pipeline_file, pipeline_info in zip(pipeline_files, pipeline_infos):
        with open(pipeline_file, "w") as f:
            yaml.dump(pipeline_info, f)

    with open(config_file, "w") as f:
        yaml.dump(config_info, f)

if __name__ == "__main__":
    input_files = sys.argv[1:]
    main(input_files)
