#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import time
import argparse
import pprint
import click
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import acts.examples
import acts

from acts.examples.reconstruction import *
from acts.examples.simulation import *

from acts import UnitConstants as u

@click.command()
@click.option("--output", "-o", type=str)
@click.option("--modulemap", type=str)
@click.option("--input-file", type=str)
@click.option("--gnn", type=str)
@click.option("--exclude-phi-ovl-sps", type=bool)
def main(output, modulemap, input_file, gnn, exclude_phi_ovl_sps):
    outputDir = Path(output)
    outputDir.mkdir(exist_ok=True, parents=True)
    logLevel=acts.logging.INFO

    s = acts.examples.Sequencer(
        events=1,
        skip=0,
        numThreads=1,
        outputDir = outputDir,
    )

    # Read Athena input space points and clusters from root file
    s.addReader(
        acts.examples.RootAthenaDumpReader(
            level=logLevel,
            treename  = "GNN4ITk",
            inputfile = [input_file,],
            outputSpacePoints = "spacepoints",
            outputClusters = "clusters",
            outputMeasurements = "measurements",
            outputMeasurementParticlesMap = "measurement_particles_map",
            outputParticles = "particles",
            onlyPassedParticles = False,
            skipOverlapSPsPhi = exclude_phi_ovl_sps,
            skipOverlapSPsEta = False,
            absBoundaryTolerance = 2 * u.mm,
        )
    )

    addParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(1*u.GeV, None),
            rho=(0, 26*u.cm),
            measurements=(7, None),
            excludeAbsPdgs=[11,],
            removeSecondaries=True,
            removeNeutral=True,
        ),
        inputParticles="particles",
        outputParticles="particles_selected",
        inputMeasurementParticlesMap="measurement_particles_map",
    )

    moduleMapConfig = {
        "level": logLevel,
        "moduleMapPath": modulemap,
        "rScale": 1000.0,
        "phiScale": 3.141592654,
        "zScale": 1000.0,
        "useGpu": True,
        "gpuDevice": 0,
        "gpuBlocks": 512,
    }
    graphConstructor = acts.examples.ModuleMapCpp(**moduleMapConfig)

    gnnConfig = {
        "level": logLevel,
        "cut": 0.5,
        "undirected": False,
        "modelPath": gnn,
        "useEdgeFeatures": True,
    }

    EdgeClassifier = None
    if ".pt" in gnn:
        EdgeClassifier = acts.examples.TorchEdgeClassifier
    elif ".so" in gnn:
        EdgeClassifier = acts.examples.TorchEdgeClassifierAOT
    elif ".onnx" in gnn:
        EdgeClassifier = acts.examples.OnnxEdgeClassifier
        del gnnConfig["useEdgeFeatures"]
        del gnnConfig["undirected"]
    elif ".engine" in gnn:
        EdgeClassifier = acts.examples.TensorRTEdgeClassifier
        gnnConfig["doSigmoid"] = True
        del gnnConfig["undirected"]
        del gnnConfig["useEdgeFeatures"]
    assert EdgeClassifier is not None

    edgeClassifiers = [
        EdgeClassifier(**gnnConfig),
    ]

    builderCfg = {
        "level": logLevel,
        "doWalkthrough": False,
    }
    trackBuilder = acts.examples.BoostTrackBuilding(**builderCfg)

    e = acts.examples.NodeFeature

    gnn_alg_config = dict(
        graphConstructor=graphConstructor,
        edgeClassifiers=edgeClassifiers,
        trackBuilder=trackBuilder,
        nodeFeatures=[
            e.R, e.Phi, e.Z, e.Eta,
            e.Cluster1R, e.Cluster1Phi, e.Cluster1Z, e.Cluster1Eta,
            e.Cluster2R, e.Cluster2Phi, e.Cluster2Z, e.Cluster2Eta,
        ],
        featureScales = [1000.0, 3.14159265359, 1000.0, 1.0] * 3,
    )

    s.addAlgorithm(
        acts.examples.TrackFindingAlgorithmExaTrkX(
            level=logLevel,
            inputSpacePoints="spacepoints",
            inputClusters="clusters",
            outputProtoTracks="gnn_prototracks",
            minMeasurementsPerTrack = 3,
            **gnn_alg_config,
        )
    )

    s.addAlgorithm(
        acts.examples.TruthTrackFinder(
            level=logLevel,
            inputParticles="particles_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputProtoTracks="truth_prototracks"
        )
    )

    if True:
        for key in ["gnn_prototracks", "truth_prototracks"]:
            thisOutputDir = outputDir / key
            thisOutputDir.mkdir(exist_ok=True)
            s.addWriter(
                acts.examples.CsvProtoTrackWriter(
                    level=logLevel,
                    inputPrototracks=key,
                    inputSpacepoints="spacepoints",
                    outputDir=thisOutputDir,
                )
            )

    s.run()

if __name__ == "__main__":
    main()
