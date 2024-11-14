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
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()

    fig.tight_layout()
    fig.savefig(outputDir / "gpu_memory_profile.png")




class ItkEnvironment:
    def __init__(self, file1, file2, materialMap, logLevel=acts.logging.INFO):
        from itk_from_geomodel_gen1 import ItkBuilderGeomodel
        from itk_from_json_gen1 import ItkBuilderJson

        self.logLevel = logLevel

        assert Path(file1).exists()
        assert Path(file2).exists()
        self.gctx = acts.GeometryContext()
      
        if "FRANKENSTEIN_ITK" in os.environ:
            print("WARNING: Use Frankenstein ITk geometry!!!")
            p = Path("/root/itk_gen1/itk_geometry")
            
            self.gmBuilder = ItkBuilderGeomodel(
                str(p / "ITKPixels.db"), str(p / "ITKStrips.db"), True, self.gctx, self.logLevel
            )
            print("- Done with GeoModel part")


            self.gmBuilder.index_hierarchy.settings.allow_insertion = True

            self.jsonBuilder = ItkBuilderJson(
                str(p / "athena_surfaces.json"),
                str(p / "athena_transforms.csv"),
                self.gctx, self.logLevel
            )
            print("- Done with JSON part")

            from itk_frankenstein_gen1 import ItkBuilderFrankenstein
            self.itkBuilder = ItkBuilderFrankenstein(
                index_hierarchy=self.jsonBuilder.index_hierarchy,
                index_hierarchy_endcaps=self.gmBuilder.index_hierarchy,
                gctx=self.gctx,
                logLevel=self.logLevel
            )
            print("- Done with Frankenstein merging", flush=True)
        else:
            if ".json" in file1:
                ItkBuilder = ItkBuilderJson
                kwargs = {}
            else:
                ItkBuilder = ItkBuilderGeomodel
                kwargs = { "applyModuleSplit": True }

            self.itkBuilder = ItkBuilder(
                file1, file2, gctx=self.gctx, logLevel=self.logLevel, **kwargs
            )

        mdec = acts.examples.RootMaterialDecorator(
            fileName=materialMap,
            level=self.logLevel,
        )

        self.trackingGeometry = self.itkBuilder.finalize(mdec)
        self.field = acts.ConstantBField(
            acts.Vector3(0, 0, 2 * acts.UnitConstants.T)
        )

    def get_geoid_map(self, events, skip, input_file):
        s = acts.examples.Sequencer(
            events=events,
            numThreads=1,
            skip=skip,
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
    output, logLevel, finding_mode, events=1, skip=0,
    profile=False, itkEnvironment=None, use_ckf=False,
    timing_mode=False,
):
    outputDir = Path(output)
    outputDir.mkdir(exist_ok=True, parents=True)
    outputDirCsv = outputDir / "csv"
    outputDirCsv.mkdir(exist_ok=True)

    writeIntermediateOutput = False
    if timing_mode:
        writeIntermediateOutput = False

    useTightSeeds = False

    filenamePrefix = f"n{events}"
    filenamePrefix += finding_mode 
    if use_ckf:
        filenamePrefix += "_ckf"
    else:
        filenamePrefix += "_kf"

    if "," in input_file:
        input_file = input_file.split(',')
    else:
        input_file = [input_file]


    def write_performance(s, tracks_key, require_ref_surface):    
        def match_and_write(doubleMatching, tag, reweight):
            s.addAlgorithm(
                acts.examples.TrackTruthMatcher(
                    level=max(logLevel, acts.logging.INFO),
                    inputTracks=tracks_key,
                    inputParticles="particles_selected",
                    inputMeasurementParticlesMap="measurement_particles_map",
                    outputTrackParticleMatching=tracks_key + "_" + tag + "_tpm",
                    outputParticleTrackMatching=tracks_key + "_" + tag + "_ptm",
                    doubleMatching=doubleMatching,
                    reweightVolumes=reweight,
                )
            )

            s.addWriter(
                acts.examples.CKFPerformanceWriter(
                    level=max(logLevel, acts.logging.INFO),
                    inputParticles="particles_selected",
                    inputTrackParticleMatching=tracks_key + "_" + tag + "_tpm",
                    inputParticleTrackMatching=tracks_key + "_" + tag + "_ptm",
                    inputTracks=tracks_key,
                    filePath=outputDir / f"performance_{filenamePrefix}_{tag}_{tracks_key}.root",
                )
            )

        pixelVolumeWeights = { v: 2.0 for v in [4,5,6,9,10,11,14,15,16]}
        match_and_write(False, "atlas_matching", pixelVolumeWeights)
        #match_and_write(False, "single_matching", {})
        #match_and_write(True, "double_matching", {})
       
        if writeIntermediateOutput:
            s.addWriter(
                acts.examples.JsonTrackWriter(
                    level=acts.logging.INFO,
                    inputTracks=tracks_key,
                    inputMeasurementParticlesMap="measurement_particles_map",
                    outputDir=outputDir,
                    fileStemp=tracks_key,
                )
            )
 
    for f in input_file:
        assert Path(f).exists()
    print("input file list:",input_file,flush=True)

    # Make geo id mapping if we do the fitting
    geometryIdMap = None
    if itkEnvironment is not None:
        geometryIdMap = itkEnvironment.get_geoid_map(events, skip, input_file)
    
    s = acts.examples.Sequencer(
        events=events,
        skip=skip,
        numThreads=1,
        outputDir = outputDir,
    )

    # Read Athena input space points and clusters from root file
    s.addReader(
        acts.examples.RootAthenaDumpReader(
            level=max(logLevel, acts.logging.DEBUG),
            treename  = "GNN4ITk",
            inputfiles = input_file,
            outputSpacePoints = "spacepoints",
            outputClusters = "clusters",
            outputMeasurements = "measurements",
            outputMeasurementParticlesMap = "measurement_particles_map",
            outputParticles = "particles",
            onlyPassedParticles = False,
            skipOverlapSPsPhi = no_phi_ovl_sps,
            skipOverlapSPsEta = False,
            geometryIdMap = geometryIdMap,
            trackingGeometry = None if itkEnvironment is None else itkEnvironment.trackingGeometry,
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
    
    if writeIntermediateOutput:
        s.addWriter(
            acts.examples.CsvSpacepointWriter(
                level=logLevel,
                inputSpacepoints="spacepoints",
                outputDir=outputDirCsv,
            )
        )

    if finding_mode == "full-gnn":
        if not timing_mode:
            s.addAlgorithm(
                acts.examples.TruthGraphBuilder(
                    level=logLevel,
                    inputParticles="particles_selected",
                    inputSpacePoints="spacepoints",
                    inputMeasurementParticlesMap="measurement_particles_map",
                    outputGraph="truth_graph",
                    targetMinPT=1.0*u.GeV,
                    targetMinSize=7,
                    uniqueModules=True,
                )
            )

        s.addAlgorithm(
            acts.examples.TrackFindingAlgorithmExaTrkX(
                level=logLevel,
                inputSpacePoints="spacepoints",
                inputClusters="clusters",
                outputProtoTracks="gnn_prototracks",
                inputTruthGraph="" if timing_mode else "truth_graph",
                geometryIdMap = geometryIdMap,
                minMeasurementsPerTrack = 7,
                **gnn_alg_config,
            )
        )
    elif finding_mode == "full-truth":
        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=logLevel,
                inputParticles="particles_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputProtoTracks="gnn_prototracks"
            )
        )
    elif finding_mode == "gc-only":
        del gnn_alg_config["edgeClassifiers"]
        s.addAlgorithm(
            acts.examples.GraphConstructionWithTruthClassifier(
                level=logLevel,
                inputSpacePoints="spacepoints",
                inputMeasurementParticlesMap="measurement_particles_map",
                inputClusters="clusters",
                outputProtoTracks="gnn_prototracks",
                geometryIdMap=geometryIdMap,
                minMeasurementsPerTrack = 7,
                **gnn_alg_config,
            )
        )

    if False:
        s.addWriter(
            acts.examples.CsvProtoTrackWriter(
                level=logLevel,
                inputProtoTracks="gnn_prototracks",
                inputSpacepoints="spacepoints",
                outputDir=outputDirCsv,
            )
        )

    s.addAlgorithm(
        acts.examples.PrototracksToTracks(
            level=logLevel,
            inputProtoTracks="gnn_prototracks",
            inputMeasurements="measurements",
            outputTracks="gnn_only_tracks",
        )
    )

    addTrackSelection(
        s,
        TrackSelectorConfig(nMeasurementsMin=7, requireReferenceSurface=False),
        inputTracks="gnn_only_tracks",
        outputTracks="gnn_only_tracks_selected",
        logLevel=logLevel,
    )
 
    write_performance(
        s,
        tracks_key="gnn_only_tracks_selected",
        require_ref_surface=False,
    )

    if itkEnvironment is not None:
        s.addAlgorithm(
            acts.examples.PrototracksToParameters(
                level=logLevel,
                inputProtoTracks="gnn_prototracks",
                inputSpacePoints="spacepoints",
                outputProtoTracks="prototracks_with_params",
                outputParameters="estimatedparameters",
                outputSeeds="gnn_seeds",
                magneticField=itkEnvironment.field,
                geometry=itkEnvironment.trackingGeometry,
                buildTightSeeds=useTightSeeds,
                stripVolumes={18,19,20},
                minSpacepointDist=20*u.mm,
                initialVarInflation=6*[100.0],
            )
        )

        if writeIntermediateOutput:
            s.addWriter(
                acts.examples.RootSeedWriter(
                    level=acts.logging.INFO,
                    inputSeeds="gnn_seeds",
                    filePath=outputDir/"seeds.root",
                    writingMode="big",
                )
            )

        tag = "tight_seeds" if useTightSeeds else "spread_seeds"  
        s.addWriter(
            acts.examples.SeedingPerformanceWriter(
                level=logLevel,
                inputSeeds="gnn_seeds",
                filePath=outputDir/f"seeding_performance_{tag}.root",
                inputParticles="particles_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
            )
        )

        kalmanOptions = {
            "multipleScattering": True,
            "energyLoss": True,
            "reverseFilteringMomThreshold": 0,
            "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
            "level": logLevel,
            "chi2Cut": 100000000.0,
        }

        if not use_ckf:
            s.addAlgorithm(
                acts.examples.TrackFittingAlgorithm(
                    level=logLevel,
                    inputMeasurements="measurements",
                    inputProtoTracks="prototracks_with_params",
                    inputInitialTrackParameters="estimatedparameters",
                    inputClusters="",
                    outputTracks="fitted_tracks",
                    pickTrack=-1,
                    fit = acts.examples.makeKalmanFitterFunction(
                        itkEnvironment.trackingGeometry, itkEnvironment.field, **kalmanOptions
                    ),
                    calibrator=acts.examples.makePassThroughCalibrator(),
                )
            )
        else:
            s.addAlgorithm(
                acts.examples.TrackFindingFromPrototrackAlgorithm(
                    level=logLevel,
                    inputProtoTracks="prototracks_with_params",
                    inputMeasurements="measurements",
                    inputInitialTrackParameters="estimatedparameters",
                    outputTracks="fitted_tracks", #"raw_tracks",
                    measurementSelectorCfg=acts.MeasurementSelector.Config(
                        #[(acts.GeometryIdentifier(), ([], [chi2Cut], [1], []))]
                        [(acts.GeometryIdentifier(), acts.MeasurementSelectorCuts([], [kalmanOptions["chi2Cut"],], [1], []))]
                    ),
                    trackingGeometry=itkEnvironment.trackingGeometry,
                    magneticField=itkEnvironment.field,
                    findTracks=acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction(
                        itkEnvironment.trackingGeometry,
                        itkEnvironment.field,
                        acts.logging.INFO,
                    ),
                    onlyPrototrackMeasurements=False,
                )
            )
            
            if False:
                s.addAlgorithm(
                    acts.examples.RefittingAlgorithm(
                        level=logLevel,
                        inputTracks="raw_tracks",
                        outputTracks="fitted_tracks",
                        fit = acts.examples.makeKalmanFitterFunction(
                            itkEnvironment.trackingGeometry, itkEnvironment.field, **kalmanOptions
                        ),
                    )
                )


        # Do not select here, so we see tracks that are shorter after KF in the plots
        write_performance(
            s,
            tracks_key="fitted_tracks",
            require_ref_surface=True,
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
