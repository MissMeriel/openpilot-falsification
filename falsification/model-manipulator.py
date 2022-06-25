import argparse
import time

import onnx
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnx
import h5py
import torch
from torchvision import utils
from torchvision import transforms
from PIL import Image
import cv2
import os
import onnxruntime as ort
from dataset import Dataset
from dnnv.nn import parse
from dnnf.pytorch import convert

from lane_image_space import transform_points
from supercombo_parser import parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath", type=Path)
    parser.add_argument("datapath", type=Path)
    # parser.add_argument("--direction", type=str, choices=["left", "right"], default="right")
    return parser.parse_args()

def test_model_on_dataset(args):
    # load model
    onnx_model = onnx.load(args.modelpath)
    onnx.checker.check_model(onnx_model)
    ds = Dataset(args.datapath)
    dataset = ds.load_dataset()

    ort_sess = ort.InferenceSession(
        args.modelpath.__str__(),
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    desire = np.zeros((1, 8)).astype('float32')
    initial_state = np.zeros((1, 512)).astype('float32')
    traffic_convention = np.zeros((1, 2)).astype('float32')

    for x in dataset:
        result = ort_sess.run(["outputs"], {"input_imgs": x,
                                           "desire": desire,
                                           "initial_state": initial_state,
                                           "traffic_convention": traffic_convention
                                           })[0][0]
        print(f"{result=}, {len(result)=}")

def convert_model_via_DNNV(args):
    print("Parsing onnx model...")
    op_graph = parse(args.modelpath)
    print("Converting operator graph....")
    start = time.time()
    try:
        pytorch_model = convert(op_graph)
    except Exception as e:
        print(e)
        print(e.with_traceback())
        print(f"Time to fail:{time.time() - start:.2f}")
    print(f"Model converted!\n{pytorch_model=}")
    print(f"Time to convert:{time.time()-start:.2f}")

def remove_node(graph, node_name="Split_236"):
    remove_node = [node for node in graph.nodes if node.name == node_name][0]

    # Get the input node of the fake node
    # Node provides i() and o() functions that can optionally
    # be provided an index (default is 0)
    # These serve as convenience functions for the alternative,
    # which would be to fetch the input/output
    # tensor first, then fetch the input/output node of the tensor.
    # For example, node.i() is equivalent to node.inputs[0].inputs[0]
    inp_node = remove_node.i()

    # Reconnect the input node to the output tensors of the fake node,
    # so that the first identity node in the example graph now
    # skips over the fake node.
    inp_node.outputs = remove_node.outputs
    remove_node.outputs.clear()

    # Remove the fake node from the graph completely
    graph.cleanup()

    h = graph.inputs[0].shape[2]
    w = graph.inputs[0].shape[3]

    scale = 0
    if graph.inputs[0].shape[1] == 4:
        scale = 2
    else:
        scale = 3

    graph.outputs[0].shape = [1, 3, h * scale, w * scale]
    print(graph.outputs)
    return graph

def remove_node_Split_236(graph):
    node_name = "Split_236"
    remove_node = [node for node in graph.nodes if node.name == node_name][0]
    new_connecting_node = [node for node in graph.nodes if node.name == "Add_240"][0]
    # Get the input node of the fake node
    # Node provides i() and o() functions that can optionally
    # be provided an index (default is 0)
    # These serve as convenience functions for the alternative,
    # which would be to fetch the input/output
    # tensor first, then fetch the input/output node of the tensor.
    # For example, node.i() is equivalent to node.inputs[0].inputs[0]
    inp_node = remove_node.i()

    # Reconnect the input node to the output tensors of the fake node,
    # so that the first identity node in the example graph now
    # skips over the fake node.
    # inp_node.outputs = remove_node.outputs
    inp_node.outputs = new_connecting_node.inputs
    remove_node.outputs.clear()

    # Remove the fake node from the graph completely
    graph.cleanup()

    h = graph.inputs[0].shape[2]
    w = graph.inputs[0].shape[3]

    scale = 0
    if graph.inputs[0].shape[1] == 4:
        scale = 2
    else:
        scale = 3

    graph.outputs[0].shape = [1, 3, h * scale, w * scale]
    print(graph.outputs)
    return graph

def remove_node_Split_237(graph):
    node_name = "Split_237"
    remove_node = [node for node in graph.nodes if node.name == node_name][0]
    new_connecting_node = [node for node in graph.nodes if node.name == "Add_238"][0]
    # Get the input node of the fake node
    # Node provides i() and o() functions that can optionally
    # be provided an index (default is 0)
    # These serve as convenience functions for the alternative,
    # which would be to fetch the input/output
    # tensor first, then fetch the input/output node of the tensor.
    # For example, node.i() is equivalent to node.inputs[0].inputs[0]
    inp_node = remove_node.i()

    # Reconnect the input node to the output tensors of the fake node,
    # so that the first identity node in the example graph now
    # skips over the fake node.
    # inp_node.outputs = remove_node.outputs
    inp_node.outputs = new_connecting_node.inputs
    remove_node.outputs.clear()

    # Remove the fake node from the graph completely
    graph.cleanup()

    h = graph.inputs[0].shape[2]
    w = graph.inputs[0].shape[3]

    scale = 0
    if graph.inputs[0].shape[1] == 4:
        scale = 2
    else:
        scale = 3

    graph.outputs[0].shape = [1, 3, h * scale, w * scale]
    print(graph.outputs)
    return graph

# reference: https://zenn.dev/pinto0309/articles/8cb106569c9c3e
def replace_split_ops(onnx_model):
    import onnx_graphsurgeon as gs
    graph = gs.import_onnx(onnx_model)

    # for i in graph.nodes:
    #     if "Split" in i.name:
    #         print(f"\nNODE:{i}")
    #         print("OUTPUTS:")
    #         for output in i.outputs:
    #             print(output)
    #     elif "Sigmoid_241" in i.name or "Tanh_244" in i.name:
    #         print(f"\nNODE:{i}")
    #         print("INPUTS:")
    #         for input in i.inputs:
    #             print(input)
    #     elif "Gemm_234" in i.name or "Gemm_235" in i.name or "Add_243" in i.name:
    #         print(f"\nNODE:{i}")
    #         print("OUTPUTS:")
    #         for output in i.outputs:
    #             print(output)
    graph = remove_node_Split_236(graph)
    graph = remove_node_Split_237(graph)

    nosplit_model = gs.export_onnx(graph)
    changed_modelpath = Path("../models/supercombo_nosplit.onnx")
    onnx.save(nosplit_model, changed_modelpath)
    return nosplit_model, changed_modelpath

def change_and_convert_model_via_DNNV_dmonitoring_model(args):
    onnx_model = onnx.load(args.modelpath)
    onnx.checker.check_model(onnx_model)

    print("Parsing onnx model.... ", end="")
    op_graph = parse(args.modelpath)
    print("Done")
    print("Converting operator graph.... ", end="")
    start = time.time()
    # try:
    pytorch_model = convert(op_graph.output_operations)
    # except Exception as e:
    #     print(f"CRASHED\nTime to fail:{time.time() - start:.0f}s ({(time.time() - start)/60:.2f}min)")
    #     print(e)
    #     return
    print("Done")
    print(f"Model converted!\n{pytorch_model=}")
    print(f"Time to convert:{time.time()-start:.2f}")

    ort_sess = ort.InferenceSession(
        args.modelpath.__str__(),
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    desire = np.zeros((1, 8)).astype('float32')
    initial_state = np.zeros((1, 512)).astype('float32')
    traffic_convention = np.zeros((1, 2)).astype('float32')
    pytorch_model.eval()
    for x in range(1000):
        # input = [x,desire,initial_state,traffic_convention]

        # compare against onnx model
        input = torch.rand((1, 6, 320, 160), dtype=torch.float32)  # np.random.random(size=(1,6,320,160))
        input_np = input.numpy()
        # result = ort_sess.run(["outputs"], {"input_img": input_np,
        #                                    "desire": desire,
        #                                    "initial_state": initial_state,
        #                                    "traffic_convention": traffic_convention
        #                                    })[0][0]
        pytorch_model.zero_grad()
        result = pytorch_model(input)
        print(f"{result=}, {len(result)=}")

# https://github.com/littlemountainman/modeld/blob/master/lane_visulaizer_dynamic.py
def plot_path(output, output_type="Orig."):
    x_left = x_right = x_path = np.linspace(0, 192, 192)
    parsed = parser(output)
    new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
    new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])
    new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])
    plt.plot(new_x_left, new_y_left, label='transformed', color='w')
    plt.plot(new_x_right, new_y_right, label='transformed', color='w')
    plt.plot(new_x_path, new_y_path, label='transformed', color='green')
    plt.pause(0.001)

def plot_output(output, output_type="Orig."):
    fig, ax = plt.subplots()
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title(f"{output_type} Lane and Path Predictions")

    plt.plot(output['plans']['plan1']['means']['x_pos'], range(0, 33), "m-", linewidth=1)

    # plt.plot(output['lanelines']['far_left']['means']['y_pos'], range(0, 33), "b-", linewidth=1)
    # plt.plot(output['lanelines']['near_left']['means']['y_pos'], range(0, 33), "y-", linewidth=1)
    # plt.plot(output['lanelines']['near_right']['means']['y_pos'], range(0, 33), "y-", linewidth=1)
    # plt.plot(output['lanelines']['far_right']['means']['y_pos'], range(0, 33), "b-", linewidth=1)
    # plt.plot(output['road_edges']['left']['means']['y_pos'], range(0, 33), "r-", linewidth=1)
    # plt.plot(output['road_edges']['right']['means']['y_pos'], range(0, 33), "r-", linewidth=1)
    # plt.xlim((-30, 30))

    plt.subplot(1, 2, 2)
    plt.title(f"{output_type} Lane Detection Confidence")
    # plt.plot(0, output['laneline_prob'][1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    # plt.plot(0, output['laneline_prob'][2], marker="o", markersize=10, markeredgecolor="blue",
    #          markerfacecolor="blue")
    # plt.xlim((-1, 1))
    # plt.ylim((0, 1))
    plt.plot(output['plans']['plan1']['means']['z_pos'], range(0, 33), "m-", linewidth=1)

    # plt.axis('square')
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # display image with opencv or any operation you like
    cv2.imshow(f"{output_type} output", img)
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

        orig_output = ds.parseOutput3(result_orig_model)
        # converted_output = ds.parseOutput(result_converted_model)
        # plot_output(orig_output)
        plot_output(orig_output, output_type="Converted")
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
    end = 201
    count = 0
    for x,pair in zip(dataset[start:end], img_pairs[start:end]):
        with open(f"dataset/imageset/imageset{count:04d}.npy", "wb") as f:
            arr = np.array(x)
            np.save(f, arr)
        count += 1
        frame = cv2.resize(np.array(cv2.cvtColor(pair[0], cv2.COLOR_BGR2RGB)), (640, 420))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cv2.imshow("image 0", frame)
    cv2.destroyAllWindows()


def mini_supercombo_tester():
    import onnx.helper as h
    # The required constants:
    c1 = h.make_node('Constant', inputs = [], outputs = ["c1"], name = "c1 - node",
    value = h.make_tensor(name="c1v", data_type = tp.FLOAT,
                                                  dims = training_results["barX"].shape,
                                                  vals = training_results["barX"].flatten()))
    c2 = h.make_node("Constant", inputs = [], outputs = ["c2"], name = "c2 - node",
    value = h.make_tensor(name="c2v", data_type = tp.FLOAT,
                                                  dims = training_results["sdX"].shape,
                                                  vals = training_results["sdX"].flatten()))
    # The functional nodes:
    n1 = h.make_node("Sub", inputs = ["x", "c1"], outputs = ["xmin"], name ="n1")
    n2 = h.make_node("Div", inputs = ["xmin", "c2"], outputs = ["zx"], name ="n2")
    # Create the graph
    g1 = h.make_graph([c1, n1, c2, n2], "preprocessing",
    [h.make_tensor_value_info("x", tp.FLOAT, [3])],
    [h.make_tensor_value_info("zx", tp.FLOAT, [3])])
    # Create the model and check
    m1 = h.make_model(g1, producer_name= "scailable - demo")
    onnx.checker.check_model(m1)
    # Save the model
    onnx.save(m1, "pre-processing.onnx")
    return


def simplify():
    print("Parsing onnx model.... ", end="")
    op_graph = parse(args.modelpath)
    op_graph = op_graph.simplify()
    op_graph.export_onnx("models/supercombo_simplified.onnx")
    print("Done")


if __name__ == '__main__':
    args = parse_args()
    # simplify()
    # mini_supercombo_tester()
    write_dataset_to_npy(args)
    exit(0)
    if "supercombo" in args.modelpath.name:
        print("Converting supercombo model")
        # import sys
        # sys.setrecursionlimit(5000)
        # test_model_on_dataset(args)
        change_and_convert_model_via_DNNV(args)
    else:
        print("Converting dmonitoring model")
        change_and_convert_model_via_DNNV_dmonitoring_model(args)
