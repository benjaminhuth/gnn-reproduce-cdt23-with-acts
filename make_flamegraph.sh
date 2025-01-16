#!/bin/bash

perf script > perf.script
cat perf.script | c++filt > perf.script.filt
~/FlameGraph/stackcollapse-perf.pl perf.script.filt > perf.script.filt.folded
~/FlameGraph/flamegraph.pl perf.script.filt.folded > flamegraph.svg

rm -f perf.script perf.script.filt perf.script.filt.folded
