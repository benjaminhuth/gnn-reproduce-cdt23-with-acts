#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import argparse

import click

import acts.examples
import acts

from acts import UnitConstants as u

@click.command()
@click.option('--data', help="Path to the ROOT file with dumped data")
@click.option('--modulemap', help="Path to the module map file")
@click.option('--gnn', help="Path to the GNN model file (*.pt)")
@click.option('-v','--verbose', default=False, is_flag=True)
def main(data, modulemap, gnn, verbose):
    assert os.path.exists(data)
    assert os.path.exists(modulemap)
    assert os.path.exists(gnn)

    print("verbose", verbose)

    # Setup pipeline
    modelDir = Path(__file__).parent / "torchscript_models"
    print("model dir:", modelDir)

    gnnLogLevel = acts.logging.VERBOSE if verbose else acts.logging.DEBUG

    metricLearningConfig = {
        "level": gnnLogLevel,
        "moduleMapPath": modulemap
    }

    gnnConfig = {
        "level": gnnLogLevel,
        "cut": 0.5,
        "numFeatures": 3,
        "undirected": False,
        "modelPath": gnn,
    }

    graphConstructor = acts.examples.ModuleMapCpp(**metricLearningConfig)
    edgeClassifiers = [
        acts.examples.TorchEdgeClassifier(**gnnConfig),
    ]
    trackBuilder = acts.examples.BoostTrackBuilding(gnnLogLevel)

    s = acts.examples.Sequencer(
        events=1,
        numThreads=1,
        outputDir = str(Path.cwd())
    )

    # Read Athena input space points and clusters from root file
    athReader = acts.examples.RootAthenaDumpReader(
        level=acts.logging.INFO,
        treename  = "GNN4ITk",
        inputfile = data,
        outputSpacePoints = "spacepoints",
        outputClusters = "clusters",
    )

    s.addReader(athReader)

    findingAlg = acts.examples.TrackFindingAlgorithmExaTrkX(
        level=gnnLogLevel,
        inputSpacePoints="spacepoints",
        outputProtoTracks="gnn_prototracks",
        graphConstructor=graphConstructor,
        edgeClassifiers=edgeClassifiers,
        trackBuilder=trackBuilder,
        useXYZ=True,
    )

    s.addAlgorithm(findingAlg)

    s.run()

if "__main__" == __name__:
    main()
