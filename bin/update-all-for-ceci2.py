import yaml
import os
import collections
groups = collections.defaultdict(list)

for dirname in os.listdir("examples/"):

    for filename in os.listdir("examples/" + dirname):
        if not (filename.endswith(".yaml") or filename.endswith(".yml")):
            continue

        filename = "examples/" + dirname + "/" + filename 
        with open(filename) as f:
            yaml_str = f.read()
        info = yaml.safe_load(yaml_str)

        if "config" in info:
            config = info["config"]
            groups[config].append(filename)


for config_filename, group in groups.items():
    try:
        with open(config_filename) as f:
            yaml_str = f.read()
    except FileNotFoundError:
        print('# missing', config_filename)
        continue
    if not "alias" in yaml_str:
        continue

    print("bin/update-pipeline-for-ceci-2.py", " ".join(group))
