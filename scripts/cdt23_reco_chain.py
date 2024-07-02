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
@click.option('--data', help="Path to the ROOT file with dumped data", default="@@@")
@click.option('--modulemap', help="Path to the module map file", default="@@@")
@click.option('--gnn', help="Path to the GNN model file (*.pt)", default="@@@")
@click.option('--truth', help="Use truth tracking instead of GNN", default=False, is_flag=True)
@click.option('-v','--verbose', default=False, is_flag=True)
def main(data, modulemap, gnn, truth, verbose):
    if not truth:
        assert os.path.exists(data)
        assert os.path.exists(modulemap + ".triplets.root")
        assert os.path.exists(modulemap + ".doublets.root")
        assert os.path.exists(gnn)

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
        outputMeasurementParticlesMap = "meas_part_map",
        outputParticles = "particles"
    )

    s.addReader(athReader)

    if not truth:
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
    else:
        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=acts.logging.VERBOSE,
                inputParticles="particles",
                inputMeasurementParticlesMap="meas_part_map",
                outputProtoTracks="gnn_prototracks"
            )
        )

    s.addAlgorithm(
        acts.examples.PrototracksToTracks(
            level=acts.logging.DEBUG,
            inputProtoTracks="gnn_prototracks",
            inputMeasurements="measurements",
            outputTracks="tracks",
        )
    )

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=acts.logging.DEBUG,
            inputTracks="tracks",
            inputParticles="particles",
            inputMeasurementParticlesMap="meas_part_map",
            outputTrackParticleMatching="tpm",
            outputParticleTrackMatching="ptm",
            doubleMatching=True,
        )
    )


    s.addWriter(
        acts.examples.CKFPerformanceWriter(
            level=acts.logging.VERBOSE,
            inputParticles="particles",
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
