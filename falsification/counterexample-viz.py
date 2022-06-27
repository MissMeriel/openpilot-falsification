import argparse
import math
import time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import onnx
import torch
import cv2
import os
import onnxruntime as ort
from dataset import Dataset
from dnnv.nn import parse
from dnnf.pytorch import convert
import re
from lane_image_space import transform_points
from supercombo_parser import parser

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath", type=Path)
    parser.add_argument("datapath", type=Path)
    return parser.parse_args()

# https://github.com/littlemountainman/modeld/blob/master/lane_visulaizer_dynamic.py
def plot_path(output, output_type=""):
    x_left = x_right = x_path = np.linspace(0, 192, 192)
    parsed = parser(output)
    new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
    new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])
    new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])
    plt.plot(new_x_left, new_y_left, label='transformed', color='w')
    plt.plot(new_x_right, new_y_right, label='transformed', color='w')
    plt.plot(new_x_path, new_y_path, label='transformed', color='green')
    plt.pause(0.001)

def plot_output(output, output_type="", filename=None, orig_array=None):
    fig, ax = plt.subplots()
    plt.clf()
    plt.subplot(1, 3, 1)

    plt.title(f"{output_type}Lane & Road Edge \nPredictions")
    # plt.plot(output['plans']['plan1']['means']['x_pos'], range(0, 33), "m-", linewidth=1)
    plt.plot(output['lanelines']['far_left']['means']['y_pos'], range(0, 33), "b-", linewidth=1)
    plt.plot(output['lanelines']['near_left']['means']['y_pos'], range(0, 33), "y-", linewidth=1)
    plt.plot(output['lanelines']['near_right']['means']['y_pos'], range(0, 33), "y-", linewidth=1)
    plt.plot(output['lanelines']['far_right']['means']['y_pos'], range(0, 33), "b-", linewidth=1)
    plt.plot(output['road_edges']['left']['means']['y_pos'], range(0, 33), "r-", linewidth=1)
    plt.plot(output['road_edges']['right']['means']['y_pos'], range(0, 33), "r-", linewidth=1)
    if orig_array is not None:
        # plt.plot([orig_array[5020], orig_array[5219]], [32, 32], marker="o", markersize=3, markeredgecolor="purple")
        plt.plot([orig_array[5019], orig_array[5217]], [32, 32], marker="o", markersize=3, markeredgecolor="purple")
        plt.plot([orig_array[5556], orig_array[5621]], [32, 32], marker="o", markersize=3, markeredgecolor="orange")
        # 5087 laneline nearleft
    plt.xlim((-30, 30))

    plt.subplot(1, 3, 2)
    plt.title(f"{output_type}Lane Detection \nConfidence")
    plt.plot(0, output['laneline_prob'][0], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
    plt.plot(0, output['laneline_prob'][1], marker="o", markersize=10, markeredgecolor="orange", markerfacecolor="yellow")
    plt.plot(0, output['laneline_prob'][2], marker="o", markersize=10, markeredgecolor="orange", markerfacecolor="yellow")
    plt.plot(0, output['laneline_prob'][3], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
    plt.xlim((-1, 1))
    plt.ylim((0, 1))
    # plt.plot(output['plans']['plan1']['means']['z_pos'], range(0, 33), "m-", linewidth=1)

    plt.subplot(1, 3, 3)
    plt.title("Lead Car \nProbabilties")
    plt.plot(0, output['lead_prob'][1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    plt.plot(0, output['lead_prob'][2], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
    plt.plot(0, output['lead_prob'][0], marker="o", markersize=10, markeredgecolor="orange", markerfacecolor="orange")
    plt.xlim((-1, 1))
    plt.ylim((0, 1))

    fig.tight_layout()
    # plt.axis('square')
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # display image with opencv or any operation you like
    # cv2.imshow(f"{output_type} output", img)
    if filename is not None:
        cv2.imwrite(filename, img)
    cv2.waitKey(1)
    plt.close('all')


def plot_output_stylized(output, output_type="", filename=None):
    fig, ax = plt.subplots()
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.title(f"{output_type}Lane Detection \nConfidence")
    plt.plot(0, output['laneline_prob'][1], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
    plt.plot(0, output['laneline_prob'][2], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
    plt.xlim((-1, 1))
    plt.ylim((0, 1))

    plt.subplot(1, 3, 2)
    plt.title(f"{output_type}Lane & Road Edge \nPredictions")
    plt.plot(output['lanelines']['near_left']['means']['y_pos'], range(0, 33), "g-", linewidth=1)
    plt.plot(output['lanelines']['near_right']['means']['y_pos'], range(0, 33), "g-", linewidth=1)
    plt.xlim((-5, 5))

    plt.subplot(1, 3, 3)
    plt.title("Lead Car \nProbabilties")
    plt.plot(0, output['lead_prob'][1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    plt.plot(0, output['lead_prob'][2], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
    plt.plot(0, output['lead_prob'][0], marker="o", markersize=10, markeredgecolor="orange", markerfacecolor="orange")
    plt.xlim((-1, 1))
    plt.ylim((0, 1))

    fig.tight_layout()
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if filename is not None:
        cv2.imwrite(filename, img)
    cv2.waitKey(1)
    plt.close('all')



def plot_output_with_orig(ce_output, orig_output, filename, prop=1):
    fig, ax = plt.subplots()
    plt.clf()

    if prop == 1:
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image") #\nLane Detection \nConfidence")
        # plt.plot(0, orig_output['laneline_prob'][0], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
        plt.plot(0, orig_output['laneline_prob'][1], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
        plt.plot(0, orig_output['laneline_prob'][2], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
        # plt.plot(0, orig_output['laneline_prob'][3], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
        plt.xlim((-1, 1))
        plt.ylim((0, 1))

        plt.subplot(1, 2, 2)
        plt.title(f"Counterexample") #\nLane Detection \nConfidence")
        # plt.plot(0, ce_output['laneline_prob'][0], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
        plt.plot(0, ce_output['laneline_prob'][1], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
        plt.plot(0, ce_output['laneline_prob'][2], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
        # plt.plot(0, ce_output['laneline_prob'][3], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
        plt.xlim((-1, 1))
        plt.ylim((0, 1))

    elif prop == 2:
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image") #\nLane & Road Edge \nPredictions")
        # plt.plot(orig_output['lanelines']['far_left']['means']['y_pos'], range(0, 33), "b-", linewidth=1)
        plt.plot(orig_output['lanelines']['near_left']['means']['y_pos'], range(0, 33), "g-", linewidth=1)
        plt.plot(orig_output['lanelines']['near_right']['means']['y_pos'], range(0, 33), "g-", linewidth=1)
        # plt.plot(orig_output['lanelines']['far_right']['means']['y_pos'], range(0, 33), "b-", linewidth=1)
        # plt.plot(orig_output['road_edges']['left']['means']['y_pos'], range(0, 33), "r-", linewidth=1)
        # plt.plot(orig_output['road_edges']['right']['means']['y_pos'], range(0, 33), "r-", linewidth=1)
        plt.xlim((-5, 5))

        plt.subplot(1, 2, 2)
        plt.title(f"Counterexample") #\nLane & Road Edge \nPredictions")
        # plt.plot(ce_output['lanelines']['far_left']['means']['y_pos'], range(0, 33), "b-", linewidth=1)
        plt.plot(ce_output['lanelines']['near_left']['means']['y_pos'], range(0, 33), "g-", linewidth=1)
        plt.plot(ce_output['lanelines']['near_right']['means']['y_pos'], range(0, 33), "g-", linewidth=1)
        # plt.plot(ce_output['lanelines']['far_right']['means']['y_pos'], range(0, 33), "b-", linewidth=1)
        # plt.plot(ce_output['road_edges']['left']['means']['y_pos'], range(0, 33), "r-", linewidth=1)
        # plt.plot(ce_output['road_edges']['right']['means']['y_pos'], range(0, 33), "r-", linewidth=1)
        plt.xlim((-5, 5))

    elif prop == 3:
        plt.subplot(1, 2, 1)
        plt.title("Original Image") #\nLead Car \nProbabilties")
        plt.plot(0, orig_output['lead_prob'][1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
        plt.plot(0, orig_output['lead_prob'][2], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
        plt.plot(0, orig_output['lead_prob'][0], marker="o", markersize=10, markeredgecolor="orange",
                 markerfacecolor="orange")
        plt.xlim((-1, 1))
        plt.ylim((0, 1))

        plt.subplot(1, 2, 2)
        plt.title("Counterexample") #\nLead Car\nProbabilties")
        plt.plot(0, ce_output['lead_prob'][1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
        plt.plot(0, ce_output['lead_prob'][2], marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="blue")
        plt.plot(0, ce_output['lead_prob'][0], marker="o", markersize=10, markeredgecolor="orange", markerfacecolor="orange")
        plt.xlim((-1, 1))
        plt.ylim((0, 1))

    fig.tight_layout()
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if filename is not None:
        cv2.imwrite(filename, img)
    cv2.waitKey(1)
    plt.close('all')


def change_and_convert_model_via_DNNV(args):
    onnx_model = onnx.load(args.modelpath)
    onnx.checker.check_model(onnx_model)
    # nosplit_model, changed_modelpath = replace_split_ops(onnx_model)

    print("Parsing onnx model.... ", end="")
    op_graph = parse(args.modelpath)
    print("Done")
    print("Converting operator graph.... ", end="")
    start = time.time()
    pytorch_model = convert(op_graph.output_operations)
    print("Done")
    print(f"Time to convert:{time.time()-start:.6f}")

    ds = Dataset(args.datapath)
    dataset, img_pairs = ds.load_dataset()
    ort_sess = ort.InferenceSession(
        args.modelpath.__str__(),
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    desire = np.zeros((1, 8)).astype('float32')
    traffic_convention = np.zeros((1, 2)).astype('float32')
    initial_state = np.zeros((1, 512)).astype('float32')
    errors = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model.eval().to(device)

    for x,pair in zip(dataset, img_pairs):

        result_orig_model = ort_sess.run(["outputs"], {"input_imgs": x,
                                           "desire": desire,
                                           "traffic_convention": traffic_convention,
                                            "initial_state": initial_state
                                           })[0][0]
        result_converted_model = pytorch_model(torch.from_numpy(x).to(device), #input.to(device), #
                                               torch.from_numpy(desire).to(device),
                                               torch.from_numpy(traffic_convention).to(device),
                                               torch.from_numpy(initial_state).to(device))
        result_converted_model = result_converted_model.numpy()[0]
        result_orig_model = np.array(result_orig_model)

        error = np.mean(result_orig_model - result_converted_model)
        errors.append(error)
        print(f"Conversion error = {error}\n")

        frame = cv2.resize(np.array(cv2.cvtColor(pair[0], cv2.COLOR_BGR2RGB)), (640, 420))
        cv2.imshow("image 0", frame)
        x_deparsed = ds.deparseInput(x)
        x_deparsed0 = cv2.resize(np.array(cv2.cvtColor(x_deparsed[0], cv2.COLOR_BGR2RGB)), (640, 420))
        cv2.imshow("image 0 deparsed", x_deparsed0)

        # orig_output = ds.parseOutput3(result_orig_model)
        # converted_output = ds.parseOutput(result_converted_model)
        # plot_output(orig_output)
        # plot_output(orig_output, output_type="Converted")
        # plot_path(orig_output)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    overall_error = sum(errors) / len(errors)
    print(f"average overall error: {overall_error}")
    print(f"total image pairs: {len(errors)}")
    cv2.destroyAllWindows()


def write_dataset_to_npy(args):
    ds = Dataset(args.datapath)
    dataset, img_pairs = ds.load_dataset()
    start = 100
    end = 1001
    count = 0
    for x,pair in zip(dataset[start:end], img_pairs[start:end]):
        with open(f"H:/GitHub/testing-openpilot/model-manipulation/dataset_imgs/imageset{count:04d}.npy", "wb") as f:
            arr = np.array(x)
            np.save(f, arr)
        count += 1
        frame = cv2.resize(np.array(cv2.cvtColor(pair[0], cv2.COLOR_BGR2RGB)), (640, 420))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cv2.imshow("image 0", frame)
    cv2.destroyAllWindows()


def generate_baseline():
    count = 0
    images = ["005", "102", "104", "199", "314", "390", "475", "448", "597", "680"]
    for img in images:
        origimg = np.load(f"H:/GitHub/testing-openpilot/model-manipulation/dataset_imgs/imageset0{img}.npy")
        for i in range(100):
            noise = np.random.normal(loc=0.0, scale=1.75, size=origimg.shape)
            arr = np.array(origimg + noise, dtype="float32")
            arr = np.around(arr, decimals=0)
            arr = np.clip(arr, 0, 255)
            norm_linf = np.max(np.array(origimg) - arr)
            rnorm = round(norm_linf, 1)
            if rnorm == 10.0:
                with open(f"H:/GitHub/testing-openpilot/model-manipulation/baseline/imageset0{img}-{count}-{rnorm}.npy", "wb") as f:
                    np.save(f, arr)
            count += 1


def view_counterexamples(args):
    ds = Dataset(args.datapath)
    dir = "H:/GitHub/testing-openpilot/model-manipulation/counterexamples"
    fileExt = r".npy"
    ce_files = [_ for _ in os.listdir(dir) if _.endswith(fileExt)]
    for file in ce_files:
        ce = np.load("/".join([dir, file]))
        x_deparsed = ds.deparseInput(ce)
        x_deparsed0 = cv2.resize(np.array(cv2.cvtColor(x_deparsed[0], cv2.COLOR_BGR2RGB)), (640, 420))
        cv2.imshow(file, x_deparsed0)
        cv2.imwrite(f'{dir}/{file.replace(".npy", ".jpg")}', x_deparsed0)
        cv2.waitKey(0)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def view_counterexample_output(args, src="counterexamples", dest="figures", orig="model-manipulation/dataset_imgs/imageset0025.npy"):

    sys.setrecursionlimit(5000)
    onnx_model = onnx.load(args.modelpath)
    onnx.checker.check_model(onnx_model)

    print("Parsing onnx model.... ", end="")
    op_graph = parse(args.modelpath)
    print("Done")
    print("Converting operator graph.... ", end="")
    start = time.time()
    pytorch_model = convert(op_graph.output_operations)
    print("Done")
    print(f"Time to convert:{time.time()-start:.6f}")

    desire = np.zeros((1, 8)).astype('float32')
    traffic_convention = np.zeros((1, 2)).astype('float32')
    initial_state = np.zeros((1, 512)).astype('float32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model.eval().to(device)

    ds = Dataset(args.datapath)
    dir = f"H:/GitHub/testing-openpilot/model-manipulation/{src}"
    fileExt = r".npy"
    ce_files = [_ for _ in os.listdir(dir) if _.endswith(fileExt)]

    # if "baseline" not in src:
    orig_img_file = orig
    orig_img = np.load(orig_img_file)
    orig_result = pytorch_model(torch.from_numpy(orig_img).to(device),  # input.to(device), #
                                           torch.from_numpy(desire).to(device),
                                           torch.from_numpy(traffic_convention).to(device),
                                           torch.from_numpy(initial_state).to(device))
    orig_result = orig_result.numpy()[0]
    orig_output = ds.parseOutput(orig_result)

    def get_orig_img(file):
        m = re.search("imageset[0-9]+", file)
        return f"H:/GitHub/testing-openpilot/model-manipulation/dataset_imgs/{m.group(0)}.npy"

    def get_prop(file):
        f = file.replace(".npy", "")
        f = f.split("_")
        f = [s for s in f if "prop" in s]
        if len(f) > 0:
            return int(f[0].replace("prop", ""))
        return 2

    for file in ce_files:
        print(f"{file}")
        orig_filepath = get_orig_img(file)
        orig_img = np.load(orig_filepath)

        orig_result = pytorch_model(torch.from_numpy(orig_img).to(device),
                                    torch.from_numpy(desire).to(device),
                                    torch.from_numpy(traffic_convention).to(device),
                                    torch.from_numpy(initial_state).to(device))
        orig_result = orig_result.numpy()[0]
        orig_output = ds.parseOutput(orig_result)

        x = np.load("/".join([dir, file]))

        result_converted_model = pytorch_model(torch.from_numpy(x).to(device),
                                               torch.from_numpy(desire).to(device),
                                               torch.from_numpy(traffic_convention).to(device),
                                               torch.from_numpy(initial_state).to(device))
        result_converted_model = result_converted_model.numpy()[0]

        x_deparsed = ds.deparseInput(x)
        x_deparsed0 = cv2.resize(np.array(cv2.cvtColor(x_deparsed[0], cv2.COLOR_BGR2RGB)), (640, 420))
        x_deparsed1 = cv2.resize(np.array(cv2.cvtColor(x_deparsed[1], cv2.COLOR_BGR2RGB)), (640, 420))
        # cv2.imshow(f"{file} deparsed 0", x_deparsed0)
        cv2.imwrite(f'{dir.replace(src, dest)}/{file.replace(".npy", "-0.jpg")}', x_deparsed0)
        cv2.imwrite(f'{dir.replace(src, dest)}/{file.replace(".npy", "-1.jpg")}', x_deparsed1)

        ce_output = ds.parseOutput(result_converted_model)
        if "baseline" in src or "orig" in dest:
            plot_output_stylized(ce_output, filename=f'{dir.replace(src, dest)}/{file.replace(".npy", "plot.jpg")}')
        else:
            # plot_output(ce_output, filename=f'{dir.replace(src, dest)}/{file.replace(".npy", "plot.jpg")}')
            plot_output_with_orig(ce_output, orig_output, filename=f'{dir.replace(src, dest)}/{file.replace(".npy", "plot.jpg")}', prop=get_prop(file))

        print("Property 1:")
        for i, label in zip(range(5484, 5491, 2), ["far left", "near left", "near right", "far right"]):
            print(label, f"index={i}", result_converted_model[i], orig_result[i])
            print(f"lb={sigmoid(orig_result[i]) - sigmoid(orig_result[i]) * 0.1:.3f}, ub={sigmoid(orig_result[i]) + sigmoid(orig_result[i]) * 0.1:.3f}")
            print(f"orig={sigmoid(orig_result[i]):.3f}, counter={sigmoid(result_converted_model[i]):.3f}")

        print("Property 2:")
        print("far left", f"orig={orig_result[5019]:.3f} counter={result_converted_model[5019]:.3f} delta={orig_result[5019]-result_converted_model[5019]:.3f}")
        print("near left", f"orig={orig_result[5217]:.3f} counter={result_converted_model[5217]:.3f} delta={orig_result[5217]-result_converted_model[5217]:.3f}")
        print("near right", f"orig={orig_result[5556]:.3f} counter={result_converted_model[5556]:.3f} delta={orig_result[5556]-result_converted_model[5556]:.3f}")
        print("far right", f"orig={orig_result[5621]:.3f} counter={result_converted_model[5621]:.3f} delta={orig_result[5621]-result_converted_model[5621]:.3f}")

        print("Property 3:")
        print(result_converted_model[5857], result_converted_model[5858], orig_result[5859])
        print(f"lead1={sigmoid(orig_result[5857]):.3f} lead2={sigmoid(orig_result[5858]):.3f} lead3={sigmoid(orig_result[5859]):.3f}")
        print(f"lead1={sigmoid(result_converted_model[5857]):.3f} lead2={sigmoid(result_converted_model[5858]):.3f} lead3={sigmoid(result_converted_model[5859]):.3f}")
        print()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

def parse_logfile(logfile, samples=10):
    with open(logfile, "r") as f:
        lines = f.readlines()
        avgs = {}
        for p in range(1,4):
            avgs[f"PROP{i+1}"] = {"results":[], "restarts": [], "f_time":[], "total_time":[]}
            for i in range(samples):
                start = p*5*samples + i*5
                outcome = lines[start + 1].split("result: ")[-1]
                if "unsafe" in outcome:
                    avgs["results"].append(1)
                else:
                    avgs["results"].append(0)
                avgs["f_time"].append(float(lines[start + 2].split("falsification time: ")[-1]))
                avgs["total_time"].append(float(lines[start + 3].split("total time: ")[-1]))
                avgs["restarts"].append(int(lines[start + 4].split("restarts: ")[-1]))
            print(f"PROPERTY {p+1}")
            for k in avgs:
                print("\t", k, "\t\t\t", round(sum(avgs[k]) / len(avgs[k]), 1), "\t\t\t", [round(a,1) for a in avgs[k]],)
            print()
    return avgs

def parse_new_logfile(logfile, samples=10):
    import re
    with open(logfile, "r") as f:
        lines = f.readlines()
        avgs = {}
        for i in range(1,4):
            avgs[f"PROP{i}"] = {"results":[], "restarts": [], "f_time":[], "total_time":[]}
            prop = ""
        for i in range(len(lines)):
            if "imageset" in lines[i]:
                m = re.search("property [0-9]", lines[i])
                prop = m.group(0).replace("property ", "")
            elif "result: " in lines[i]:
                if "unsafe" in lines[i]:
                    avgs[f"PROP{prop}"]["results"].append(1)
                else:
                    avgs[f"PROP{prop}"]["results"].append(0)
            elif "falsification time: " in lines[i]:
                avgs[f"PROP{prop}"]["f_time"].append(float(lines[i].split("falsification time: ")[-1]))
            elif "total time: " in lines[i]:
                avgs[f"PROP{prop}"]["total_time"].append(float(lines[i].split("total time: ")[-1]))
            elif "restarts: " in lines[i]:
                avgs[f"PROP{prop}"]["restarts"].append(int(lines[i].split("restarts: ")[-1]))
    return avgs

def parse_logfiles(dir="H:/GitHub/testing-openpilot/model-manipulation"):
    files = os.listdir(dir)
    files = [f for f in files if "forTable2" in f]
    avgs = {}
    imgs = {}
    for i in range(1,4):
        avgs[f"PROP{i}"] = {"results":[], "restarts": [], "f_time":[], "total_time":[]}
    for f in files:
        print(f)
        temp = parse_new_logfile(f)
        imgs[f] = temp
        for i in range(1,4):
            for k in avgs[f"PROP{i}"].keys():
                avgs[f"PROP{i}"][k].extend(temp[f"PROP{i}"][k])
    for i in range(1,4):
        print(f"PROPERTY {i}")
        for k in avgs[f"PROP{i}"].keys():
            print(f"Avg. {k}:", sum(avgs[f"PROP{i}"][k]) / len(avgs[f"PROP{i}"][k]))

    for i in range(1,4):
        count = 0
        for img in imgs.keys():
            if sum(imgs[img][f"PROP{i}"]["results"]) > 0:
                print(f"{img} falsified for Property {i}")
                count += 1
        print(f"Property {i} falsified on {count} images")
    print(avgs)
    print(imgs)


if __name__ == '__main__':
    args = parse_args()
    # write_dataset_to_npy(args)
    # generate_baseline()
    view_counterexample_output(args, src="counterexamples", dest="figures", orig="H:/GitHub/testing-openpilot/model-manipulation/dataset_imgs/imageset0104.npy")
    # parse_logfile("H:/GitHub/testing-openpilot/model-manipulation/dnnf_test_eps1_imageset104_pm10percent.log")
    # parse_logfiles()