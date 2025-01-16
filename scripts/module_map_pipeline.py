#!/usr/bin/env python3
from pathlib import Path
import os
import copy
import sys
import argparse
import pprint
import click
import yaml

import acts.examples
import acts

from acts.examples.reconstruction import *
from acts.examples.simulation import *

from acts import UnitConstants as u

from common_pipeline import *

@click.command()
@click.option('--data', help="Path to the ROOT file with dumped data", default=None)
@click.option('--modulemap', help="Path to the module map file", default=None)
@click.option('--gnn', help="Path to the GNN model file", default=None)
@click.option('--finding-mode', default="full-gnn", type=click.Choice(['full-gnn', 'full-truth', 'gc-only']))
@click.option('--debug','-v', default=False, is_flag=True)
@click.option('--verbose','-vv', default=False, is_flag=True)
@click.option('--output','-o', type=str, default=".")
@click.option('--no-phi-ovl-sps', is_flag=True, default=False)
@click.option('--events', '-n', default=1, type=int)
@click.option('--skip', '-s', default=0, type=int)
@click.option('--ckf/--no-ckf', default=False)
@click.option('--profile/--no-profile', default=False)
@click.option('--fit/--no-fit', default=False)
@click.option('--timing-mode/--no-timing-mode', default=False)
@click.option('--walkthrough/--no-walkthrough', default=False)
@click.option('--itk-pixel-data', default=None)
@click.option('--itk-strip-data', default=None)
@click.option('--itk-material-map', default=None)
@click.option('--gpu-blockdim', default=512)
@click.option('--jobs', '-j', default=1)
@click.option('--module-map-dynamic-alloc/--no-module-map-dynamic-alloc', default=False)
@click.option('--tensorrt-exec-contexts', default=1)
def main(data, modulemap, gnn, finding_mode, debug, verbose,
         output, no_phi_ovl_sps, events, skip, ckf, profile,
         fit, timing_mode, walkthrough, itk_pixel_data, itk_strip_data, itk_material_map,
         gpu_blockdim, jobs, module_map_dynamic_alloc, tensorrt_exec_contexts):
    print("Configuration:")
    config = copy.deepcopy(locals())
    pprint.pprint(config)
    print(flush=True)
    with open(Path(output) / "configuration.yaml", 'w') as f:
        yaml.dump(config, f)

    if finding_mode == "full-gnn":
        assert os.path.exists(gnn)

    if finding_mode == "full-gnn" or finding_mode == "gc-only":
        assert os.path.exists(modulemap + ".triplets.root")
        assert os.path.exists(modulemap + ".doublets.root")

    logLevel = acts.logging.INFO
    if debug:
        logLevel = acts.logging.DEBUG
    if verbose:
        logLevel = acts.logging.VERBOSE

    itkEnvironment = None
    if fit:
        itkEnvironment = ItkEnvironment(
            itk_pixel_data,
            itk_strip_data,
            itk_material_map,
            logLevel,    
        )

    
    moduleMapConfig = {
        "level": logLevel,
        "moduleMapPath": modulemap,
        "rScale": 1000.0,
        "phiScale": 3.141592654,
        "zScale": 1000.0,
        "gpuDevice": 0,
        "gpuBlocks": gpu_blockdim,
        "maxEdgesAllocate": 0 if module_map_dynamic_alloc else 3000
    }
    print(moduleMapConfig, flush=True)
    graphConstructor = acts.examples.ModuleMapCuda(**moduleMapConfig)

    gnnConfig = {
        "level": logLevel,
        "cut": 0.01 if walkthrough else 0.5,
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
        gnnConfig["doSigmoid"] = not ("sigmoid" in gnn)
        del gnnConfig["undirected"]
        del gnnConfig["useEdgeFeatures"]
        gnnConfig["numExecutionContexts"] = tensorrt_exec_contexts
    assert EdgeClassifier is not None

    edgeClassifiers = [
        EdgeClassifier(**gnnConfig),
    ]

    if False:
        builderCfg = {
            "level": logLevel,
            "doWalkthrough": walkthrough,
        }
        trackBuilder = acts.examples.BoostTrackBuilding(**builderCfg)
    else:
        trackBuilder = acts.examples.CudaTrackBuilding(level=logLevel)

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

    common_pipeline(
        input_file=data,
        gnn_alg_config=gnn_alg_config,
        no_phi_ovl_sps=no_phi_ovl_sps,
        output=output,
        logLevel=logLevel,
        finding_mode=finding_mode,
        events=events,
        skip=skip,
        use_ckf=ckf,
        profile=profile,
        itkEnvironment=itkEnvironment,
        timing_mode=timing_mode,
        jobs=jobs,
    )

if "__main__" == __name__:
    main()
