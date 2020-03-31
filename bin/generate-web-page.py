import sys
from pathlib import Path
import shutil
import argparse
import string
import os


parser = argparse.ArgumentParser(description='Make a TXPipe webpage')
parser.add_argument("run_name", help='A name for the new run directory')
parser.add_argument("run_dir", help='The directory containing the files you want to share')
parser.add_argument("--all", "-a", action='store_true', help='Include files > 1GB')

args = parser.parse_args()

www_base_dir = Path("/project/projectdirs/lsst/www/txpipe/runs")

# remove punctuation
run_name = args.run_name.replace('/','_').replace('\\', '_')
out_dir = www_base_dir / args.run_name
run_dir = Path(args.run_dir)

# if out_dir.exists():
#     raise ValueError(f"Run directory {out_dir} already exists.")

try:
    out_dir.mkdir()
except IOError:
    pass

html_path = out_dir / 'index.html'
html = html_path.open('w')
html.write(f"""
<HEAD>
<TITLE>TXPipe Output {run_name}</TITLE>
</HEAD>
<BODY>
<H1>TXPipe Output {run_name}</H1>
""")

def include_file(p):
    return (args.all or p.stat().st_size < 1_000_000_000)


for f in run_dir.glob("*"):
    if f.name.startswith('inprogress'):
        continue
    if include_file(f):
        print(f"Copying {f.name}")
        shutil.copy(f, out_dir)
        os.system(f"chgrp lsst {out_dir}/{f.name}")
        os.system(f"chmod g+rw {out_dir}/{f.name}")
        os.system(f"chmod o+r {out_dir}/{f.name}")
        html.write(f'<P><A href="{f.name}">{f.name}</A></P>\n')
    else:
        html.write(f"<P>{f.name} (not copied here)")

html.write("""
</BODY>
</HTML>
""")

print(out_dir)
