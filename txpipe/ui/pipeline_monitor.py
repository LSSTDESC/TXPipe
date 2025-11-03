import ceci
import pygraphviz
from paramiko import SSHClient
import io
import warnings
import os
import time
import IPython.display
import PIL.Image
from .nersc_monitor import NerscMonitor
import paramiko
import yaml


class PipelineMonitor:
    def __init__(self, username, key_filename, remote_dir, config_file):
        # get the config data from the remote_dir
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            key_filename = os.path.expanduser(key_filename)
            client.connect("cori.nersc.gov", username=username, key_filename=key_filename)
            config_remote_path = os.path.join(remote_dir, config_file)

            cmd = f"cat {config_remote_path}"

            stdin, stdout, stderr = client.exec_command(cmd)
            err = stderr.read()
        if err:
            raise ValueError(f"Remote config file {config_file} not found on NERSC: {err}")
        config_data = stdout.read()

        config = yaml.safe_load(config_data)
        # d/l file

        # Info we need from the pipeline
        self.stages = [ceci.PipelineStage.get_stage(stage["name"]) for stage in config["stages"]]
        self.inputs = config["inputs"]

        # determine which directories to watch - log dir and output dir
        log_dir = os.path.join(remote_dir, config["log_dir"])
        output_dir = os.path.join(remote_dir, config["output_dir"])

        # start monitor watching them
        self.monitor = NerscMonitor([log_dir, output_dir], client=client)

        # build initial dag - no files completed yet
        self.dag = self.build_dag([], [], [], [])

    def draw_pil(self):
        s = io.BytesIO()
        png_data = self.dag.draw(format="png", prog="dot")
        s.write(png_data)
        s.seek(0)
        return PIL.Image.open(s)

    def draw_ipython(self):
        IPython.display.clear_output(wait=True)
        png_data = self.dag.draw(format="png", prog="dot")
        img = IPython.display.Image(png_data)
        IPython.display.display(img)

    def update(self):
        # ask monitor for files
        results = self.monitor.update()

        # if monitor is not ready yet, just return False
        if results is None:
            return False
        # otherwise, parse these files and see what they mean for the pipeline
        log_files, output_files = results

        # check file statuses
        complete_files, running_files = self.check_file_statuses(output_files)

        # check stage statuses
        complete_stages, running_stages = self.check_stage_statuses(complete_files, log_files)

        # Build the dag
        self.dag = self.build_dag(running_stages, complete_stages, running_files, complete_files)

        return True

    def main_loop(self, interval, count=1000000000):
        for i in range(count):
            try:
                self.update()
                time.sleep(interval)
            except KeyboardInterrupt:
                break

    def check_file_statuses(self, output_files):
        running_files = set()
        complete_files = set()
        mark = "inprogress_"
        for filename in output_files:
            if filename.startswith(mark):
                running_files.add(filename[len(mark) :])
            else:
                complete_files.add(filename)

        return complete_files, running_files

    def check_stage_statuses(self, complete_files, log_files):
        running_stages = set()
        complete_stages = set()
        for stage in self.stages:
            # Check if all outputs from the stage are present.
            # If so, stage is complete
            for tag, ftype in stage.outputs:
                name = ftype.make_name(tag)
                if name not in complete_files:
                    break
            else:
                # this happens if we never "break" above, i.e. if all output
                # files are present
                complete_stages.add(stage.name)
                # If job us complete we do not count it as
                # running also
                continue
            # Job is running if log file is located
            if stage.name + ".log" in log_files:
                running_stages.add(stage.name)

        return complete_stages, running_stages

    def build_dag(self, running_stages, complete_stages, running_files, complete_files):
        G = pygraphviz.AGraph(
            directed=True,
            rankdir="LR",
            nodesep=0.1,
            len=0.2,
            minlen=0.01,
            mindist=0.1,
            ranksep=0.01,
        )
        names = {}

        # Draw overall inputst as plum boxes
        for tag, name in self.inputs.items():
            names[tag] = f"{tag}:\n {name}"
            G.add_node(names[tag], color="plum", style="filled", shape="box")

        for stage in self.stages:
            # Draw files input/output from the pipeline
            for tag, ftype in stage.inputs + stage.outputs:
                if tag in self.inputs:
                    name = names[tag]
                else:
                    name = ftype.make_name(tag)
                # Do not re-add overall inputs or multiple
                # inputs to different pipelines
                if tag not in names:
                    # Choose colour for file
                    if name in complete_files:
                        color = "mediumspringgreen"
                    elif name in running_files:
                        color = "lightskyblue"
                    else:
                        color = "white"
                    # Build up nodeds
                    G.add_node(name, fillcolor=color, style="filled", shape="box")
                names[tag] = name

        for stage in self.stages:
            # Choose colour from status
            if stage.name in complete_stages:
                color = "palegreen2"
            elif stage.name in running_stages:
                color = "orange"
            else:
                color = "whitesmoke"
            G.add_node(stage.name, fillcolor=color, style="filled")

            # Draw edges to/from this stage from/to input/output files
            for tag, _ in stage.inputs:
                G.add_edge(names[tag], stage.name, style="bold")
            for tag, _ in stage.outputs:
                G.add_edge(stage.name, names[tag], style="bold")
        return G
