import os
import argparse
import h5py
import tabulate
import sacc


def summarize_sacc(filename):
    try:
        s = sacc.Sacc.load_fits(filename)
    except IOError:
        return ""
    lines = ["Data points: " + str(len(s))]
    lines.append(f"\n\nTracers:\n")
    for name in s.tracers:
        lines.append("- " + name)
    lines.append("\n")
    lines.append("Data:\n")
    for dt in s.get_data_types():
        ncombo = len(s.get_tracer_combinations(dt))
        ndata = s.get_mean(dt).size
        lines.append(f"* {dt}: {ncombo} tracer combinations, {ndata} data points")


    return "\n".join(lines)


def summarize_hdf5(filename):
    table_rows = []
    def summary(name, obj):
        if isinstance(obj, h5py.File):
            return
        bits = name.split("/")
        row = []
        if (bits[0] == "provenance"):
            return
        indent = name.count("/")
        name = bits[-1]

        row = ["" for _ in range(indent)]
        if isinstance(obj, h5py.Group):
            row.append(f"[{name}]")
        elif isinstance(obj, h5py.Dataset):
            row.append(name)
            row.append(str(obj.shape))
            row.append(str(obj.dtype))
        table_rows.append(row)

    with h5py.File(filename, "r") as f:
        f.visititems(summary)

    if len(table_rows) > 40:
        table_rows = table_rows[:40] + [["..."]]

    if table_rows:
        n = max([len(row) for row in table_rows])
        for row in table_rows:
            row.extend(["" for _ in range(n - len(row))])
        header = ["Group"] + [""] * (n - 4) + ["Dataset", "Shape", "Type"]
    else:
        header = []
    

    return tabulate.tabulate(table_rows, headers=header, tablefmt="github")
    
def summarize_txt(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = ["    " + line.rstrip() for line in lines]
    return "\n".join(lines)

def summarize_png(filename):
    return f"![{filename}]({filename}){{ width=90% }}\ "

def main(input_directory, output_file):
    unused_files = []
    with open(output_file, "w") as out:

        if not os.path.exists(input_directory) or not os.path.listdir(input_directory):
            out.write("# No report generated\n\n")
            out.write("No input directory, or directory is empty.")

        for filename in os.listdir(input_directory):
            if filename.startswith("inprogress"):
                continue
            path = os.path.join(input_directory, filename)
            if filename.endswith(".png"):
                text = summarize_png(path)
            elif filename.endswith(".hdf5"):
                text = summarize_hdf5(path) 
            elif filename.endswith(".txt"):
                text = summarize_txt(path)
            elif filename.endswith(".sacc"):
                text = summarize_sacc(path)
            else:
                unused_files.append(filename)
            out.write(f"# {filename}\n\n")
            out.write(text + "\n\\newpage\n\n")
        if unused_files:
            out.write("# Unused files\n\n")
            out.write("\n- ".join(unused_files))



parser = argparse.ArgumentParser(description="Summarize a directory of files")
parser.add_argument("input_directory", help="Directory to summarize")
parser.add_argument("output_file", help="Output file")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.input_directory, args.output_file)
