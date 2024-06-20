"""Automatic calibration script."""

import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import papermill as pm
from qililab.calibration import CalibrationController, CalibrationNode
from qililab.calibration.comparison_models import (
    IQ_norm_root_mean_sqrt_error,
    ssro_comparison_2D,
)

# CHANGE RELATIVE PATH
ABSPATH = os.path.abspath(__file__)
DNAME = os.path.dirname(ABSPATH)
os.chdir(DNAME)

RESET_TIME = 30


def custom_draw(graph: nx.DiGraph):
    """
    Draw the graph. It disposes the nodes in topological sort and colors
    them by layers.

    Args:
        graph (nx.DiGraph): Directed Acyclic Graph to draw.
    """
    random.seed(12)
    # A layer represents a level in the graph in topological sort
    for layer, nodes in enumerate(nx.topological_generations(graph)):
        # `multipartite_layout` expects the layer as a node attribute,
        # so add the numeric layer value as a node attribute
        # add colors by layer
        node_color = plt.cm.viridis(random.random())
        for node in nodes:
            graph.nodes[node]["layer"] = layer
            graph.nodes[node]["color"] = node_color

    color = [data["color"] for _, data in graph.nodes(data=True)]
    options = {
        "node_size": 200,
        "node_color": color,
        "with_labels": False,
        "font_size": 6,
        "font_color": "black",
        "font_weight": "bold",
    }
    pos = nx.multipartite_layout(graph, subset_key="layer", align="vertical")

    nx.draw(graph, pos, **options)

    first_layer_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]
    for idx, node in enumerate(first_layer_nodes):
        plt.text(pos[node][0] - 0.1, pos[node][1], s=f"q{idx}", horizontalalignment="center")

    plt.savefig("calibration_graph")


def find_last_created_log_file():
    """
    Finds the last created log file in the directory.

    Returns:
        str: The name of the most recently created log file, or None if no log files are found.
    """
    file_names = [entry.name for entry in os.scandir(DNAME) if entry.is_file() and ".log" in entry.name]
    # Get the last created file, most recent one
    return max(file_names, key=lambda fname: os.path.getctime(os.path.join(DNAME, fname))) if file_names != [] else None


def get_filenames_for_tables(log_file: str):
    """
    Create filenames for the tables with the same timestamp.

    Args:
        log_file (str): The name of the log file.

    Returns:
        tuple: Filenames for 1qb and 2qb tables.
    """
    base_filename = log_file.split(".log")[0]
    fn_1qb = base_filename + "_1qb_table.csv"
    fn_2qb = base_filename + "_2qb_table.csv"
    return fn_1qb, fn_2qb


def execute_reset_nb(reset: str, platform_path: str, partition: str):
    """
    Execute the reset notebook with the given parameters.

    Args:
        reset (str): Elements to reset.
        platform_path (str): Path to the platform.
        partition (str): Partition to use.
    """
    reset_nb_path = os.path.join(DNAME, "autocalibration_notebooks/reset.ipynb")
    reset_nb_out_path = os.path.join(DNAME, "autocalibration_notebooks/reset_out.ipynb")
    reset_params = {
        "elems_to_reset": reset,
        "PLATFORM_PATH": platform_path,
        "partition": partition,
        "RESET_TIME": RESET_TIME,
    }

    pm.execute_notebook(reset_nb_path, reset_nb_out_path, reset_params)
    os.remove(reset_nb_out_path)


def main(
    draw: bool = False,
    run_fidelities: bool = False,
    partition: str = "",
    runcard: str = "",
    reset: str = "",
    linear: bool = False,
):
    """
    Run the automatic calibration.

    Args:
        draw (bool, optional): If True, draws the graph. Defaults to False.
        run_fidelities (bool, optional): If True, computes fidelities after calibration. Defaults to False.
        partition (str, optional): Partition to calibrate. Defaults to "".
        runcard (str, optional): Runcard name to use. Defaults to "".
        reset (str, optional): Resets instruments before calibration. Defaults to "".
        linear (bool, optional): If True, calibrates the graph linearly. Defaults to False.
    """
    # GET RUNCARD PATH

    platform_path = os.path.join(os.path.dirname(DNAME), f"runcards/{runcard}.yml")

    # NODE MAPPING TO THE GRAPH {key <- name in graph, value <- node object}:
    nodes = {}
    G = nx.DiGraph()

    # BUILD THE GRAPH
    for qubit in range(5):
        # CREATE NODES:
        two_tone = CalibrationNode(
            nb_path="notebooks/2tone.ipynb",
            qubit_index=qubit,
            in_spec_threshold=0.1,
            bad_data_threshold=0.3,
            comparison_model=IQ_norm_root_mean_sqrt_error,
            drift_timeout=3600 * 5,
            input_parameters={"PLATFORM_PATH": platform_path, "partition": partition},
        )
        nodes[two_tone.node_id] = two_tone

        all_xy = CalibrationNode(
            nb_path="notebooks/allxy.ipynb",
            qubit_index=qubit,
            in_spec_threshold=0.05,
            bad_data_threshold=0.1,
            comparison_model=ssro_comparison_2D,
            drift_timeout=3600 * 5,
            input_parameters={"PLATFORM_PATH": platform_path, "partition": partition},
        )
        nodes[all_xy.node_id] = all_xy

        # BUILD EDGES:
        G.add_edge(two_tone.node_id, all_xy.node_id)

    # DRAW THE GRAPH
    if draw:
        custom_draw(G)

    # RESET ALL THE REQUIRED ELEMENTS IF ANY
    if reset:
        execute_reset_nb(reset, platform_path, partition)

    controller = CalibrationController(G, nodes, linear=linear)
    controller.run_calibration_loop()


if __name__ == "__main__":
    parser = ArgumentParser(description="Automatic calibration script")
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draws the calibration graph",
    )
    parser.add_argument(
        "--run_fidelities",
        action="store_true",
        help="Computes fidelities after calibration",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="",
        help="Partition to calibrate",
    )
    parser.add_argument(
        "--runcard",
        type=str,
        default="",
        help="Runcard name to use",
    )
    parser.add_argument(
        "--reset",
        type=str,
        default="",
        help="Resets instruments before calibration",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Calibrates the graph linearly",
    )
    args = parser.parse_args()
    main(
        draw=args.draw,
        run_fidelities=args.run_fidelities,
        partition=args.partition,
        runcard=args.runcard,
        reset=args.reset,
        linear=args.linear,
    )
