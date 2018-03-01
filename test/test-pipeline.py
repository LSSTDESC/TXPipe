from pipette import Pipeline
import descTX.selector
import descTX.photoz
import pathlib
import os
import yaml
import sys
import parsl

localIPP = {
    "sites": [
        {"site": "Local_IPP",
         "auth": {
             "channel": None,
         },
         "execution": {
             "executor": "ipp",
             "provider": "local",  # LIKELY SHOULD BE BOUND TO SITE
             "block": {  # Definition of a block
                 "taskBlocks": 4,       # total tasks in a block
                 "initBlocks": 4,
             }
         }
         }]
}


def test_pipeline(config_filename="./test/test.yml"):
    parsl.set_file_logger('log.txt')
    config = yaml.load(open(config_filename))
    # Required configuration information

    # List of stage names, must be imported somewhere
    stages = config['stages']

    # parsl execution/launcher configuration information
    # launcher_config = localIPP
    launcher_config = config['launcher']

    # Inputs and outputs
    output_dir = config['output_dir']
    inputs = config['inputs']

    # Create and run pipeline
    pipeline = Pipeline(launcher_config, stages)
    pipeline.run(inputs, output_dir)


if __name__ == '__main__':
    narg = len(sys.argv)
    if narg>1:
        test_pipeline(sys.argv[1])
    else:
        test_pipeline()

