import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import atlasify
import re

atlasify.monkeypatch_axis_labels()

algorithms = [
    "Algorithm:TrackFindingMLBasedAlgorithm",
    "Algorithm:PrototracksToParsAndSeeds",
    "Algorithm:TrackFittingAlgorithm",
]

timing = pd.read_csv("timing.csv")
timing = timing[timing.identifier.isin(algorithms)]
timing["cum_sum"] = np.cumsum(timing["time_perevent_s"])
print(timing)

finding_subtimings = []
finding_subparts = []

pattern = r"^.*TrackFinding\s+INFO\s+- (preprocessing|graph building|classifier|track building|postprocessing):\s+([0-9.]+) \+- [0-9.]+ \[[0-9.]+, [0-9.]+\]$"
with open("logfile.log") as f:
    for line in f:
        if result := re.findall(pattern, line):
            finding_subparts.append(result[0][0])
            finding_subtimings.append(float(result[0][1]) * 1e-3)  # ms to seconds

print(finding_subparts)
finding_subparts[0] = "pre- and postprocessing"
finding_subparts[-1] = "_" + finding_subparts[-1]

print(finding_subtimings)
assert len(finding_subparts) == 5

algorithm_timings = timing["time_perevent_s"].to_list()

# Distribute remaining difference on pre and postprocessing
delta = algorithm_timings[0] - sum(finding_subtimings)
print(
    f"Found delta of {delta}s, this corresponds to {100.0*delta/algorithm_timings[0]}% of algorithm timing"
)
finding_subtimings[0] += 0.5 * delta
finding_subtimings[1] += 0.5 * delta

xs = np.cumsum(
    [
        0.0,
    ]
    + algorithm_timings[:-1]
)
print(xs)
print(algorithm_timings)

fig, ax = plt.subplots(figsize=(8, 4))
items1 = ax.barh(
    0.5, width=algorithm_timings, left=xs, color=["tab:blue", "tab:orange", "tab:green"]
)

xs2 = np.cumsum([0.0, *finding_subtimings[:-1]])

items2 = ax.barh(
    -0.5,
    width=finding_subtimings,
    left=xs2,
    alpha=1.0,
    color=["lightgrey", "cornflowerblue", "steelblue", "deepskyblue", "lightgrey"],
)

ax.vlines(algorithm_timings[0], -0.9, 0.9, colors="black", linestyles="dashed", lw=0.5)

ax.set_xlabel("walltime [s]")
ax.set_yticks([-0.5, 0.5])
ax.set_yticklabels(["Track finding", "Full chain"])
ax.set_xlim(0, 3.2)

leg1 = ax.legend(
    items1,
    ["Track finding", "Parameter Estimation", "Kalman Filter"],
    loc=(0.6, 0.05),
    title="Full chain",
)
leg2 = ax.legend(items2, finding_subparts, loc=(0.3, 0.05), title="Track finding")

ax.add_artist(leg1)

atlasify.atlasify(
    "Internal",
    "Average full chain timing of the ACTS standalone GNN workflow",
    outside=True,
    sub_font_size=11,
)
fig.savefig("timing.png")
