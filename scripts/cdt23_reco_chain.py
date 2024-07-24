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

@click.command()
@click.option('--data', help="Path to the ROOT file with dumped data", default=None)
@click.option('--modulemap', help="Path to the module map file", default=None)
@click.option('--gnn', help="Path to the GNN model file (*.pt)", default=None)
@click.option('--metric-learning', help="path to metric learning script", default=None)
@click.option('--truth', help="Use truth tracking instead of GNN", default=False, is_flag=True)
@click.option('--debug','-v', default=False, is_flag=True)
@click.option('--verbose','-vv', default=False, is_flag=True)
@click.option('--select', default=False, is_flag=True)
@click.option('--output','-o', type=str, default=".")
@click.option('--no-phi-ovl-sps', is_flag=True, default=False)
def main(data, modulemap, gnn, metric_learning, truth, debug, verbose, select, output, no_phi_ovl_sps):
    print("Configuration:")
    pprint.pprint(locals())
    print()

    if not truth:
        assert (modulemap is None) != (metric_learning is None)
        use_modulemap = metric_learning is None

        assert os.path.exists(data)
        assert os.path.exists(gnn)

        if use_modulemap:
            assert os.path.exists(modulemap + ".triplets.root")
            assert os.path.exists(modulemap + ".doublets.root")
            print("INFO: Use ModuleMap")
        else:
            assert os.path.exists(metric_learning)
            print("INFO: Use Metric Learning")

    logLevel = acts.logging.INFO
    if debug:
        logLevel = acts.logging.DEBUG
    if verbose:
        logLevel = acts.logging.VERBOSE

    outputDir=Path(output)
    outputDir.mkdir(exist_ok=True)
    outputDirCsv = outputDir / "csv"
    outputDirCsv.mkdir(exist_ok=True)

    s = acts.examples.Sequencer(
        events=1,
        numThreads=1,
        outputDir = str(Path.cwd())
    )

    # Read Athena input space points and clusters from root file
    s.addReader(
        acts.examples.RootAthenaDumpReader(
            level=max(logLevel, acts.logging.DEBUG),
            treename  = "GNN4ITk",
            inputfile = data,
            outputSpacePoints = "spacepoints",
            outputClusters = "clusters",
            outputMeasurements = "measurements",
            outputMeasurementParticlesMap = "meas_part_map",
            outputParticles = "particles",
            onlyPassedParticles = False,
            skipOverlapSPsPhi = no_phi_ovl_sps,
            skipOverlapSPsEta = False,
        )
    )

    addParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(1*u.GeV, None),
            rho=(0, 26*u.cm),
            measurements=(3,None),
            excludeAbsPdgs=[11,],
            removeSecondaries=True,
            removeNeutral=True,
        ),
        inputParticles="particles",
        outputParticles="particles_selected",
        inputMeasurementParticlesMap="meas_part_map",
    )

    if not truth:
        if True:
            s.addAlgorithm(
                acts.examples.TruthGraphBuilder(
                    level=logLevel,
                    inputParticles="particles_selected",
                    inputSpacePoints="spacepoints",
                    inputMeasurementParticlesMap="meas_part_map",
                    outputGraph="truth_graph",
                    targetMinPT=1.0*u.GeV,
                    targetMinSize=3,
                    uniqueModules=True,
                )
            )
        else:
            s.addReader(
                acts.examples.CsvExaTrkXGraphReader(
                    level=logLevel,
                    inputDir=outputDirCsv,
                    inputStem="acorn-graph",
                    outputGraph="truth_graph",
                )
            )

        if use_modulemap:
            moduleMapConfig = {
                "level": logLevel,
                "moduleMapPath": modulemap,
                "rScale": 1000.0,
                "phiScale": 3.141592654,
                "zScale": 1000.0,
            }
            graphConstructor = acts.examples.ModuleMapCpp(**moduleMapConfig)
        else:
            metricLearningConfig = {
                "level": logLevel,
                "modelPath": metric_learning,
                "knnVal": 50, #knn_val is 300
                "rVal": 0.1,
                "numFeatures": 12
            }
            graphConstructor = acts.examples.TorchMetricLearning(**metricLearningConfig)

        gnnConfig = {
            "level": logLevel,
            "cut": 0.5,
            "numFeatures": 12,
            "undirected": False,
            "modelPath": gnn,
        }

        edgeClassifiers = [
            acts.examples.TorchEdgeClassifier(**gnnConfig),
        ]
        trackBuilder = acts.examples.BoostTrackBuilding(logLevel)

        e = acts.examples.NodeFeature

        s.addAlgorithm(
            acts.examples.TrackFindingAlgorithmExaTrkX(
                level=logLevel,
                inputSpacePoints="spacepoints",
                inputClusters="clusters",
                outputProtoTracks="gnn_prototracks",
                inputTruthGraph="truth_graph",
                graphConstructor=graphConstructor,
                edgeClassifiers=edgeClassifiers,
                trackBuilder=trackBuilder,
                nodeFeatures=[
                    e.R, e.Phi, e.Z, e.Eta,
                    e.Cluster1R, e.Cluster1Phi, e.Cluster1Z, e.Cluster1Eta,
                    e.Cluster2R, e.Cluster2Phi, e.Cluster2Z, e.Cluster2Eta,
                ],
                featureScales = [1000.0, 3.14159265359, 1000.0, 1.0] * 3,
                filterShortTracks = True,
            )
        )
    else:
        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=logLevel,
                inputParticles="particles",
                inputMeasurementParticlesMap="meas_part_map",
                outputProtoTracks="gnn_prototracks"
            )
        )

    s.addAlgorithm(
        acts.examples.PrototracksToTracks(
            level=logLevel,
            inputProtoTracks="gnn_prototracks",
            inputMeasurements="measurements",
            outputTracks="tracks",
        )
    )

    if select:
        addTrackSelection(
            s,
            TrackSelectorConfig(nMeasurementsMin=7),
            inputTracks="tracks",
            outputTracks="tracks_selected",
            logLevel=logLevel,
        )
    else:
        s.addWhiteboardAlias("tracks_selected", "tracks")

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=max(logLevel, acts.logging.DEBUG),
            inputTracks="tracks_selected",
            inputParticles="particles_selected",
            inputMeasurementParticlesMap="meas_part_map",
            outputTrackParticleMatching="tpm",
            outputParticleTrackMatching="ptm",
            doubleMatching=True,
        )
    )


    s.addWriter(
        acts.examples.CKFPerformanceWriter(
            level=max(logLevel, acts.logging.DEBUG),
            inputParticles="particles_selected",
            inputTrackParticleMatching="tpm",
            inputParticleTrackMatching="ptm",
            inputTracks="tracks_selected",
            filePath=outputDir/"performance.root",
        )
    )

    s.addWriter(
        acts.examples.CsvSpacepointWriter(
            level=logLevel,
            inputSpacepoints="spacepoints",
            outputDir=outputDirCsv,
        )
    )

    if not truth:
        s.addWriter(
            acts.examples.CsvExaTrkXGraphWriter(
                level=logLevel,
                inputGraph="truth_graph",
                outputStem="truth-graph",
                outputDir=outputDirCsv,
            )
        )

    s.run()

if "__main__" == __name__:
    main()
