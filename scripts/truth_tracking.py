from pathlib import Path
import click
from common_pipeline import ItkEnvironment

import acts
import acts.examples
from acts.examples.reconstruction import *


@click.command()
@click.option("--events", "-n", type=int, default=1)
@click.option("--input-data", "-i", type=str)
@click.option("--output-dir", "-o", type=str)
@click.option("--material-map", type=str)
@click.option("--itk-file1", type=str)
@click.option("--itk-file2", type=str)
def main(events, input_data, output_dir, material_map, itk_file1, itk_file2):

    itkEnvironment = ItkEnvironment(
        itk_file1, itk_file2, material_map, acts.logging.INFO
    )
    geometryIdMap = itkEnvironment.get_geoid_map(events, 0, input_data)
    rnd = acts.examples.RandomNumbers(seed=34509)

    s = acts.examples.Sequencer(
        events=events,
    )

    s.addReader(
        acts.examples.RootAthenaDumpReader(
            level=acts.logging.ERROR,
            treename="GNN4ITk",
            inputfile=input_data,
            outputSpacePoints="spacepoints",
            outputClusters="clusters",
            outputMeasurements="measurements",
            outputMeasurementParticlesMap="measurement_particles_map",
            outputParticles="particles",
            onlyPassedParticles=True,
            geometryIdMap=geometryIdMap,
            trackingGeometry=itkEnvironment.trackingGeometry,
        )
    )

    addSeeding(
        s,
        itkEnvironment.trackingGeometry,
        itkEnvironment.field,
        inputParticles="particles",
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        particleHypothesis=acts.ParticleHypothesis.muon,
        truthSeedRanges=TruthSeedRanges(nHits=(7, None)),
        rnd=rnd,
    )

    addKalmanTracks(
        s,
        itkEnvironment.trackingGeometry,
        itkEnvironment.field,
        directNavigation=False,
        reverseFilteringMomThreshold=0.0,
    )

    s.addWriter(
        acts.examples.CKFPerformanceWriter(
            level=acts.logging.INFO,
            inputParticles="particles",
            inputTrackParticleMatching="kf_track_particle_matching",
            inputParticleTrackMatching="kf_particle_track_matching",
            inputTracks="kf_tracks",
            filePath=Path(output_dir) / "performance_truth_tracking.root",
        )
    )

    s.run()


if __name__ == "__main__":
    main()
