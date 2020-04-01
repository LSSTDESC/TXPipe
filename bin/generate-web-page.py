import sys
from pathlib import Path
import shutil
import argparse
import string
from textwrap import dedent
import yaml
import os
import argparse
import ceci

# # If we have v1 of ceci use that
# try:
#     from ceci.main import override_config
# except ImportError:
#     def override_config(a,b):
#         raise ValueError("Cannot use override_config with older versions of ceci")

parser = argparse.ArgumentParser("Make a nice web page from outputs")
# These two args ar ethe same as ceci
parser.add_argument('name', help='A name for this run')
parser.add_argument('pipeline_config', help='Pipeline configuration file in YAML format.')
parser.add_argument('extra_config', nargs='*', help='Over-ride the main pipeline yaml file '
                    'e.g. launcher.name=cwl')

# These args are specific to this script
parser.add_argument('--www', default="/project/projectdirs/lsst/www/txpipe/runs",
                    help='Directory to put results in')
parser.add_argument('--all', action='store_true', help="Copy large files as well as small")
parser.add_argument('--group', default='lsst', help="Group to change ownership to")



class PageMaker:
    def __init__(self, run_name, run_dir, www_base_dir, copy_all=False, group='lsst'):
        self.run_name = run_name.replace('/','_').replace('\\', '_')
        self.in_dir = Path(run_dir)
        self.out_dir = Path(www_base_dir) / run_name
        self.copy_all = copy_all
        self.group = group

    def header(self):
        return dedent(f"""\
            <HEAD>
                <TITLE>TXPipe Output {self.run_name}</TITLE>
                <style>
                table {{
                  font-family: arial, sans-serif;
                  border-collapse: collapse;
                  width: 50%;
                  margin-left:auto;
                  margin-right:auto;
                }}

                td, th {{
                  border: 1px solid #dddddd;
                  text-align: left;
                  padding: 8px;
                }}

                tr:nth-child(1) {{
                  background-color: #dddddd;
                }}
                </style>
            </HEAD>
            <BODY>
                <H1>
                    <CENTER>TXPipe Output {self.run_name}</CENTER>
                </H1>""")

    def find_pipeline_outputs(self, pipeline):
        files = []
        for stage_name in pipeline.stage_names:
            files += self.find_stage_outputs(pipeline, stage_name)
        return files

    def find_stage_outputs(self, pipeline, stage_name):
        files = []
        stage = ceci.PipelineStage.get_stage(stage_name)
        outputs = pipeline.find_outputs(stage, self.in_dir)
        for output in outputs:
            files.append([stage_name, Path(output)])
        return files


    def should_copy_file(self, size):
        return self.copy_all or (size < 1_000_000_000)


    def assemble(self, files):
        files2 = []
        for stage_name, path in files:
            exists = path.exists()
            if exists:
                size = path.stat().st_size
                size_txt = '{:,.2f}'.format(size / 1_000_000)
            else:
                size_txt = ''
            if exists and self.should_copy_file(size):
                out_path = self.out_dir / path.name
                shutil.copy(path, self.out_dir)
                os.system(f"chgrp {self.group} {out_path}")
                os.system(f"chmod g+rw {out_path}")
                os.system(f"chmod o+r {out_path}")
                files2.append([stage_name, path, out_path, size_txt, 'copied'])
            elif exists:
                files2.append([stage_name, path, '', size_txt, 'uncopied'])
            else:
                files2.append([stage_name, path, '', size_txt, 'missing'])
        return files2

    def row(self, info):
        stage_name, path, out_path, size, status = info
        name = path.name
        if status == 'copied':
            link = f'<A href="{name}">{name}</A>'
        elif status == 'uncopied':
            link = f'{name} [too large to copy here]'
        else:
            link = f'{name} [missing]'
        return dedent(f"""\

            <tr>
                <th>{link}</th>
                <th>{stage_name}</th>
                <th>{size}</th>
            </tr>""")


    def body(self, files):
        lines = [
            "<TABLE>",
            "<TR>"
            "<TH>File</TH>",
            "<TH>Stage</TH>",
            "<TH>Size / MB</TH>",
            "</TR>",
            ]
        for info in files:
            lines.append(self.row(info))
        lines.append("</TABLE>")
        return '\n'.join(lines)



    def tail(self):
        return dedent("""
            </BODY>
            </HTML>
            """)

    def run(self, pipeline):
        self.out_dir.mkdir(exist_ok=True)
        index = (self.out_dir / 'index.html').open('w')
        files = self.find_pipeline_outputs(pipeline)
        files = self.assemble(files)
        index.write(self.header())
        index.write(self.body(files))
        index.write(self.tail())


def main():
    args = parser.parse_args()

    pipe_config = yaml.safe_load(open(args.pipeline_config))
    # if args.extra_config:
    #     override_config(pipe_config, args.extra_config)

    for module in pipe_config['modules'].split():
        __import__(module)

    maker = PageMaker(
            args.name,
            pipe_config['output_dir'],
            args.www,
            copy_all=args.all,
            group=args.group)


    # need to update this when we merge ceci 1.0 - much cleaner
    for stage in pipe_config['stages']:
        stage['site'] = 'local'
        stage['mpi_command'] = ''
    pipeline = ceci.Pipeline(pipe_config['stages'])

    maker.run(pipeline)


if __name__ == '__main__':
    main()