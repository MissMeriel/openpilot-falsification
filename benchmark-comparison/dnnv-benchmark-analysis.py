import argparse
import ast
from pathlib import Path
import onnx
import os
import onnx_graphsurgeon as gs

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath", type=Path)
    return parser.parse_args()

def analyze_network(path):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

    graph = gs.import_onnx(onnx_model)
    ops = {}
    for node in graph.nodes:
        if node.op in ops.keys():
            ops[node.op] += 1
        else:
            ops[node.op] = 1

    file_size = os.path.getsize(path)
    input_size = graph.inputs
    output_size = graph.outputs
    print(f"{path.name}")
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Unique operators: {len(ops.keys())}")
    print(f"Filesize on disk: {file_size / 1000000:.1f}Mb")
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    print(ops)

def analyze_networks(args):
    parentdir = args.modelpath #os.path.join(args.modelpath, "camera")
    h5files = [f for f in os.listdir(parentdir) if ".onnx" in f]
    for f in h5files:
        analyze_network(Path(os.path.join(parentdir, f)))
        print()

def get_metas(filename):
    activations = [' Sigmoid', ' Relu', ' Atan', ' Tanh']
    metas = {"ops": [], "nodes":[], "sizes":[], "optypes": set(),
             "unique_acts":[], "input_dim":[], "output_dim":[]}
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Total nodes: " in line:
            line = line.replace("Total nodes: ", "")
            metas["nodes"].append(int(line))
        if "Unique operators: " in line:
            line = line.replace("Unique operators: ", "")
            metas["ops"].append(int(line))
        if "Filesize on disk: " in line:
            line = line.replace("Filesize on disk: ", "")
            line = line.replace("Mb\n", "")
            metas["sizes"].append(float(line))
        if "{" in line:
            line = line.replace("{", "").replace("}", "")
            line = line.replace("\'", "")
            line = line.split(",")
            unique_acts_count = 0
            for s in line:
                s = s.split(":")[0]
                metas["optypes"].add(s)
                if s in activations:
                    unique_acts_count += 1
            metas["unique_acts"].append(unique_acts_count)
        # if "Input size: " in line:
        #     line = line.split("shape=")[-1]
        #     line = line.split(", dtype")[0]
        #     metas["input_dim"] = ast.literal_eval(line)
        # if "Output size: " in line:
        #     line = line.split("shape=")[-1]
        #     line = line.split(", dtype")[0]
        #     metas["output_dim"] = ast.literal_eval(line)
    print(f"Total networks: {len(metas['nodes'])}")
    return metas


def summarize_network_metadata(filename, summary="avg"):
    metas = get_metas(filename)
    for key in metas.keys():
        if isinstance(metas[key], list):
            if summary is "avg":
                avg = sum(metas[key]) / len(metas[key])
                print(f"Average {key}: {avg:.2f}")
            elif summary is "max":
                mx = max(metas[key])
                print(f"Maximum {key}: {mx:.2f}")
        else:
            print(f"All {key}: {metas[key]}")


if __name__ == '__main__':
    args = parse_args()
    summarize_network_metadata("DNNF-benchmark-metadata.txt", summary="max")
    # analyze_networks(args)
    # analyze_network(args.modelpath)
