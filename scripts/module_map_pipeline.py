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

from common_pipeline import *

@click.command()
@click.option('--data', help="Path to the ROOT file with dumped data", default=None)
@click.option('--modulemap', help="Path to the module map file", default=None)
@click.option('--gnn', help="Path to the GNN model file (*.pt)", default=None)
@click.option('--truth', help="Use truth tracking instead of GNN", default=False, is_flag=True)
@click.option('--debug','-v', default=False, is_flag=True)
@click.option('--verbose','-vv', default=False, is_flag=True)
@click.option('--select', default=False, is_flag=True)
@click.option('--output','-o', type=str, default=".")
@click.option('--no-phi-ovl-sps', is_flag=True, default=False)
@click.option('--events', '-n', default=1, type=int)
@click.option('--profile/--no-profile', default=False)
@click.option('--fit/--no-fit', default=False)
@click.option('--itk-pixel-data', default=None)
@click.option('--itk-strip-data', default=None)
@click.option('--itk-material-map', default=None)
def main(data, modulemap, gnn, truth, debug, verbose,
         select, output, no_phi_ovl_sps, events, profile,
         fit, itk_pixel_data, itk_strip_data, itk_material_map):
    print("Configuration:")
    pprint.pprint(locals())
    print()

    if not truth:
        assert os.path.exists(data)
        assert os.path.exists(gnn)
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

    assert ".pt" in gnn or ".so" in gnn
    EdgeClassifier = acts.examples.TorchEdgeClassifier if ".pt" in gnn else acts.examples.TorchEdgeClassifierAOT

    edgeClassifiers = [
        EdgeClassifier(**gnnConfig),
    ]
    trackBuilder = acts.examples.BoostTrackBuilding(logLevel)

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
        filterShortTracks = True,
    )

    common_pipeline(data, gnn_alg_config, no_phi_ovl_sps, output,
                    select, logLevel, truth, events, profile, itkEnvironment)

if "__main__" == __name__:
    main()
