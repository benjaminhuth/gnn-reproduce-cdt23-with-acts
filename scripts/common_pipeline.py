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

def plot_gpu_memory(inputFile, outputDir):
    assert inputFile.exists()

    data = pd.read_csv(
        inputFile,
        converters={"timestamp": str},
        skipinitialspace=True,
        skip_blank_lines=True,
        on_bad_lines="skip",
    )

    # Remove last line that can be corrupted
    data.drop(data.tail(1).index,inplace=True)

    data["timestamp"] = data["timestamp"].apply(
        lambda tp: datetime.strptime(tp, "%Y/%m/%d %H:%M:%S.%f")
    )
    data["time"] = data["timestamp"].apply(
        lambda tp: (tp - data.at[0, "timestamp"]).total_seconds()
    )

    gpu_ids = np.unique(data["index"])

    fig, ax = plt.subplots()

    for gpu in gpu_ids:
        df = data[data["index"] == gpu]
        ax.plot(df["time"], df["memory.used [MiB]"], label="GPU{}".format(gpu))

    ax.set_xlabel("wall clock time [s]")
    ax.set_ylabel("GPU used memory [MiB]")
    ax.set_title("GPU memory usage")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outputDir / "gpu_memory_profile.png")



class ItkEnvironment:
    def __init__(self, pixelDatabase, stripDatabase, materialMap, logLevel=acts.logging.INFO):
        from itk_from_geomodel_gen1 import ItkBuilder

        self.logLevel = logLevel

        assert Path(pixelDatabase).exists()
        assert Path(stripDatabase).exists()
        self.gctx = acts.GeometryContext()
        self.itkBuilder = ItkBuilder(
            pixelDatabase, stripDatabase,
            applyModuleSplit=True, gctx=self.gctx,
            logLevel=self.logLevel
        )

        mdec = acts.examples.RootMaterialDecorator(
            fileName=materialMap,
            level=self.logLevel,
        )

        self.trackingGeometry = self.itkBuilder.finalize(mdec)
        self.field = acts.ConstantBField(
            acts.Vector3(0, 0, 2 * acts.UnitConstants.T)
        )

    def get_geoid_map(self, events, input_file):
        s = acts.examples.Sequencer(
            events=events,
            numThreads=1,
        )

        geometryIdMap = acts.examples.GeometryIdMapActsAthena()
        s.addReader(
            acts.examples.RootAthenaDumpGeoIdCollector(
                level = self.logLevel,
                treename  = "GNN4ITk",
                inputfile = input_file,
                geometryIdMap = geometryIdMap,
                trackingGeometry = self.trackingGeometry,
            )
        )
        s.run()
        return geometryIdMap


def common_pipeline(
    input_file, gnn_alg_config, no_phi_ovl_sps,
    output, select_tracks, logLevel, truth, events=1,
    profile=False, itkEnvironment=None,
):
    outputDir=Path(output)
    outputDir.mkdir(exist_ok=True)
    outputDirCsv = outputDir / "csv"
    outputDirCsv.mkdir(exist_ok=True)

    # Make geo id mapping if we do the fitting
    geometryIdMap = None
    if itkEnvironment is not None:
        geometryIdMap = itkEnvironment.get_geoid_map(events, input_file)
    
    s = acts.examples.Sequencer(
        events=events,
        numThreads=1,
        outputDir = str(Path.cwd())
    )

    # Read Athena input space points and clusters from root file
    s.addReader(
        acts.examples.RootAthenaDumpReader(
            level=logLevel,
            treename  = "GNN4ITk",
            inputfile = input_file,
            outputSpacePoints = "spacepoints",
            outputClusters = "clusters",
            outputMeasurements = "measurements",
            outputMeasurementParticlesMap = "measurement_particles_map",
            outputParticles = "particles",
            onlyPassedParticles = False,
            skipOverlapSPsPhi = no_phi_ovl_sps,
            skipOverlapSPsEta = False,
            geometryIdMap = geometryIdMap,
            trackingGeometry = None if itkEnvironment is None else itkEnvironment.trackingGeometry
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
        inputMeasurementParticlesMap="measurement_particles_map",
    )

    if not truth:
        if True:
            print("WARNING:      Use all particles as reference for truth graph, not target")
            s.addAlgorithm(
                acts.examples.TruthGraphBuilder(
                    level=logLevel,
                    inputParticles="particles_selected",
                    inputSpacePoints="spacepoints",
                    inputMeasurementParticlesMap="measurement_particles_map",
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

        s.addAlgorithm(
            acts.examples.TrackFindingAlgorithmExaTrkX(
                level=logLevel,
                inputSpacePoints="spacepoints",
                inputClusters="clusters",
                outputProtoTracks="gnn_prototracks",
                inputTruthGraph="truth_graph",
                geometryIdMap = geometryIdMap,
                **gnn_alg_config,
            )
        )
    else:
        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=logLevel,
                inputParticles="particles",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputProtoTracks="gnn_prototracks"
            )
        )

    if itkEnvironment is None:
        s.addAlgorithm(
            acts.examples.PrototracksToTracks(
                level=logLevel,
                inputProtoTracks="gnn_prototracks",
                inputMeasurements="measurements",
                outputTracks="tracks",
            )
        )
    else:
        s.addAlgorithm(
            acts.examples.PrototracksToParameters(
                level=logLevel,
                inputProtoTracks="gnn_prototracks",
                inputSpacePoints="spacepoints",
                outputProtoTracks="prototracks_with_params",
                outputParameters="estimatedparameters",
                magneticField=itkEnvironment.field,
                geometry=itkEnvironment.trackingGeometry,
                buildTightSeeds=True,
            )
        )

        addKalmanTracks(
            s,
            itkEnvironment.trackingGeometry,
            itkEnvironment.field,
            inputProtoTracks="prototracks_with_params",
        )

    if select_tracks:
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
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching="tpm",
            outputParticleTrackMatching="ptm",
            doubleMatching=True,
        )
    )

    s.addWriter(
        acts.examples.CKFPerformanceWriter(
            level=max(logLevel, acts.logging.INFO),
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

    if profile:
        profile_file = outputDir / "gpu_memory_profile.csv"
        gpu_profiler_args = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.total,memory.reserved,memory.free,memory.used",
            "--format=csv,nounits",
            "--loop-ms=10",
            "--filename={}".format(profile_file),
        ]
        gpu_profiler = subprocess.Popen(gpu_profiler_args)

    s.run()
    del s

    if profile:
        time.sleep(0.5)
        gpu_profiler.kill()
        plot_gpu_memory(profile_file, outputDir)

if "__main__" == __name__:
    main()
