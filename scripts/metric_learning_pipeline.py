#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import argparse
import pprint
import click

import acts.examples
import acts

from acts.examples.reconstruction import *
from acts.examples.simulation import *

from acts import UnitConstants as u

from common_pipeline import common_pipeline

@click.command()
@click.option('--data', help="Path to the ROOT file with dumped data", default=None)
@click.option('--gnn', help="Path to the GNN model file (*.pt)", default=None)
@click.option('--filter', help="Path to the filter model", default=None)
@click.option('--metric-learning', help="path to metric learning script", default=None)
@click.option('--truth', help="Use truth tracking instead of GNN", default=False, is_flag=True)
@click.option('--debug','-v', default=False, is_flag=True)
@click.option('--verbose','-vv', default=False, is_flag=True)
@click.option('--select', default=False, is_flag=True)
@click.option('--output','-o', type=str, default=".")
@click.option('--no-phi-ovl-sps', is_flag=True, default=False)
def main(data, gnn, filter, metric_learning, truth, debug, verbose, select, output, no_phi_ovl_sps):
    print("Configuration:")
    pprint.pprint(locals())
    print()

    if not truth:
        assert os.path.exists(data)
        assert os.path.exists(gnn)
        assert os.path.exists(filter)
        assert os.path.exists(metric_learning)

    logLevel = acts.logging.INFO
    if debug:
        logLevel = acts.logging.DEBUG
    if verbose:
        logLevel = acts.logging.VERBOSE

    e = acts.examples.NodeFeature

    feature_list = [
        (e.R,           1000.0),
        (e.Phi,         3.14),
        (e.Z,           1000.0),
        (e.Cluster1X,   1000.0),
        (e.Cluster1Y,   1000.0),
        (e.Cluster1Z,   1000.0),
        (e.Cluster2X,   1000.0),
        (e.Cluster2Y,   1000.0),
        (e.Cluster2Z,   1000.0),
    ]

    for n in [1,2]:
        feature_list += [
            (getattr(e, f"CellCount{n}"),   1),
            (getattr(e, f"ChargeSum{n}"),   1),
            (getattr(e, f"LocEta{n}"),      3.14),
            (getattr(e, f"LocPhi{n}"),      3.14),
            (getattr(e, f"LocDir0{n}"),     1),
            (getattr(e, f"LocDir1{n}"),     1),
            (getattr(e, f"LocDir2{n}"),     1),
            (getattr(e, f"LengthDir0{n}"),  1),
            (getattr(e, f"LengthDir1{n}"),  1),
            (getattr(e, f"LengthDir2{n}"),  1),
            (getattr(e, f"GlobEta{n}"),     3.14),
            (getattr(e, f"GlobPhi{n}"),     3.14),
            (getattr(e, f"EtaAngle{n}"),    3.14),
            (getattr(e, f"PhiAngle{n}"),    3.14),
        ]

    metricLearningConfig = {
        "level": logLevel,
        "modelPath": metric_learning,
        "knnVal": 50, #knn_val is 300
        "rVal": 0.1,
        "numFeatures": len(feature_list)
    }
    graphConstructor = acts.examples.TorchMetricLearning(**metricLearningConfig)

    gnnConfig = {
        "level": logLevel,
        "cut": 0.5,
        "numFeatures": len(feature_list),
        "undirected": False,
        "modelPath": gnn,
    }

    filterConfig = {
        "level": logLevel,
        "cut": 0.5,
        "numFeatures": len(feature_list),
        "undirected": False,
        "modelPath": filter,
        "nChunks": 10,
    }

    edgeClassifiers = [
        acts.examples.TorchEdgeClassifier(**filterConfig),
        acts.examples.TorchEdgeClassifier(**gnnConfig),
    ]
    trackBuilder = acts.examples.BoostTrackBuilding(logLevel)

    gnn_alg_config = dict(
        graphConstructor=graphConstructor,
        edgeClassifiers=edgeClassifiers,
        trackBuilder=trackBuilder,
        nodeFeatures=[ f for f, s in feature_list ],
        featureScales = [ s for f, s in feature_list ],
        filterShortTracks = True,
    )

    common_pipeline(data, gnn_alg_config, no_phi_ovl_sps, output, select, logLevel, truth)


if "__main__" == __name__:
    main()
