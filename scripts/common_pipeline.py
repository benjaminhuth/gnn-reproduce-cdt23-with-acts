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




def common_pipeline(input_file, gnn_alg_config, no_phi_ovl_sps, output, select_tracks, logLevel, truth, events=1, profile=False):
    outputDir=Path(output)
    outputDir.mkdir(exist_ok=True)
    outputDirCsv = outputDir / "csv"
    outputDirCsv.mkdir(exist_ok=True)

    s = acts.examples.Sequencer(
        events=events,
        numThreads=1,
        outputDir = str(Path.cwd())
    )

    # Read Athena input space points and clusters from root file
    s.addReader(
        acts.examples.RootAthenaDumpReader(
            level=max(logLevel, acts.logging.DEBUG),
            treename  = "GNN4ITk",
            inputfile = input_file,
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
            print("WARNING:      Use all particles as reference for truth graph, not target")
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

        s.addAlgorithm(
            acts.examples.TrackFindingAlgorithmExaTrkX(
                level=logLevel,
                inputSpacePoints="spacepoints",
                inputClusters="clusters",
                outputProtoTracks="gnn_prototracks",
                inputTruthGraph="truth_graph",
                **gnn_alg_config,
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

    try:
        s.run()
    except Exception as e:
        print("ERROR in s.run()")
        print(e)
    del s

    if profile:
        time.sleep(0.5)
        gpu_profiler.kill()
        plot_gpu_memory(profile_file, outputDir)

if "__main__" == __name__:
    main()
