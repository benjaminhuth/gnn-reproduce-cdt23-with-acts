import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
import sys

inputFile = Path(sys.argv)
assert inputFile.exists()

data = pd.read_csv(
    inputFile,
    converters={"timestamp": str},
    skipinitialspace=True,
    skip_blank_lines=True,
    on_bad_lines="skip",
)

# Remove last line that can be corrupted
data.drop(data.tail(1).index, inplace=True)

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
