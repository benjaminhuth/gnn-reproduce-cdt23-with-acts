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
    assert os.path.exists(modulemap + ".triplets.root")
    assert os.path.exists(modulemap + ".doublets.root")
    assert os.path.exists(gnn)

    print("verbose", verbose)

    # Setup pipeline
    modelDir = Path(__file__).parent / "torchscript_models"
    print("model dir:", modelDir)

    gnnLogLevel = acts.logging.VERBOSE if verbose else acts.logging.DEBUG

    metricLearningConfig = {
        "level": gnnLogLevel,
        "moduleMapPath": modulemap,
        "rScale": 1000.0,
        "phiScale": 3.141592654,
        "zScale": 1000.0,
    }

    gnnConfig = {
        "level": gnnLogLevel,
        "cut": 0.5,
        "numFeatures": 12,
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
        outputMeasurements = "measurements",
        outptuMeasurementsParticlesMap = "meas_part_map",
        outputParticles = "particles"
    )

    s.addReader(athReader)

    e = acts.examples.NodeFeature

    s.addAlgorithm(
        acts.examples.TrackFindingAlgorithmExaTrkX(
            level=gnnLogLevel,
            inputSpacePoints="spacepoints",
            inputClusters="clusters",
            outputProtoTracks="gnn_prototracks",
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
    )

    s.addAlgorithm(
        acts.examples.PrototracksToTracks(
            inputPrototracks="gnn_prototracks",
            inputMeasurements="measurements",
            outputTracks="tracks",
        )
    )

    self.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles",
            inputMeasurementParticlesMap="meas_part_map",
            outputTrackParticleMatching="tpm",
            outputParticleTrackMatching="ptm",
            doubleMatching=True,
        )
    )


    self.addWriter(
        acts.examples.CKFPerformanceWriter(
            level=acts.logging.WARNING,
            inputParticles=self.target_particles_key,
            inputTrackParticleMatching="tpm",
            inputParticleTrackMatching="ptm",
            inputTracks="tracks",
            filePath="performance.root",
            # effPlotToolConfig=acts.examples.EffPlotToolConfig(self.binningCfg),
            # duplicationPlotToolConfig=acts.examples.DuplicationPlotToolConfig(
            #     self.binningCfg
            # ),
            # fakeRatePlotToolConfig=acts.examples.FakeRatePlotToolConfig(
            #     self.binningCfg
            # ),
        )
    )

    s.run()

if "__main__" == __name__:
    main()
