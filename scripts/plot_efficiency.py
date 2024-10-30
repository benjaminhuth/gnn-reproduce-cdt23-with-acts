#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import uproot
import ROOT

plt.rcParams.update({'font.size': 12})

if True: #try:
    timing = pd.read_csv("timing.csv")
    timing["cum_sum"] = np.cumsum(timing["time_perevent_s"])
    timing

    timing_per_event = timing["time_perevent_s"].to_numpy()

    finding_subtimings = [0.572, 0.312, 0.023]
    finding_subparts = ["ModuleMap", "GNN", "CC"]

    used_timings = [timing_per_event[3], timing_per_event[7], timing_per_event[8]]

    xs = np.cumsum([0.0, *used_timings[:-1]])
    print(xs)
    print(used_timings)

    plt.barh(3*[0.5], width=used_timings, left=xs, color=["tab:blue","tab:orange","tab:green"], label=["Track finding", "Parameter Estimation", "Kalman Filter"])

    xs2 = np.cumsum([0.0, *finding_subtimings[:-1]])

    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(3*[-0.5], width=finding_subtimings, left=xs2, alpha=0.8,
             color=["cornflowerblue", "steelblue","deepskyblue"], 
             label=["Graph building","GNN","Track building"]) 

    ax.set_xlabel("walltime [s]")
    ax.set_yticks([-0.5, 0.5])
    ax.set_yticklabels(["Track finding", "Full chain"])

    ax.set_title("Full chain timings per event")
    ax.legend()

    fig.tight_layout()
    fig.savefig("timing.png")
#except Exception as e:
#    print(f"Cannot plot timing: {e}")


class TEfficiency:
    def __init__(self, tefficency):
        th1 = tefficency.GetTotalHistogram()

        bins = [i for i in range(th1.GetNbinsX()) if th1.GetBinContent(i) > 0.0]
        bins = bins[1:]

        self.x = [th1.GetBinCenter(i) for i in bins]

        self.x_lo = [th1.GetBinLowEdge(i) for i in bins]
        self.x_width = [th1.GetBinWidth(i) for i in bins]
        self.x_hi = np.add(self.x_lo, self.x_width)
        self.x_err_lo = np.subtract(self.x, self.x_lo)
        self.x_err_hi = np.subtract(self.x_hi, self.x)

        self.y = [tefficency.GetEfficiency(i) for i in bins]
        self.y_err_lo = [tefficency.GetEfficiencyErrorLow(i) for i in bins]
        self.y_err_hi = [tefficency.GetEfficiencyErrorUp(i) for i in bins]

    def errorbar(self, ax, **errorbar_kwargs):
        ax.errorbar(
            self.x, self.y, yerr=(self.y_err_lo, self.y_err_hi), xerr=(self.x_err_lo, self.x_err_hi), **errorbar_kwargs
        )
        return ax

    def step(self, ax, **step_kwargs):
        ax.step(self.x_hi,self.y, **step_kwargs)
        return ax


    def bar(self, ax, **bar_kwargs):
        ax.bar(self.x, height=self.y, yerr=(self.y_err_lo, self.y_err_hi), **bar_kwargs)
        return ax



try:
    perf_kf = ROOT.TFile.Open("performance_kf_tracks.root")
except:
    perf_kf = None

perf_gnn_only = ROOT.TFile.Open("performance_non_fitted_tracks.root")

for metric in ["trackeff", "duplicationRate", "fakerate"]:
    for var in ["eta", "pT"]:
        key = metric + "_vs_" + var
        fig, ax = plt.subplots()
        
        gnn_only_teff = TEfficiency(perf_gnn_only.Get(key))
        gnn_only_teff.errorbar(ax, fmt='none', label="track finding only")

        if perf_kf is not None:
            kf_teff = TEfficiency(perf_kf.Get(key))
            kf_teff.errorbar(ax, fmt='none', color="tab:orange", label="after track fit")

        ax.set_ylabel(metric)
        ax.set_xlabel(f"true {var}")
        ax.set_title(f"ACTS performance writer: {key}")
        
        if metric == "trackeff":
            ax.set_ylim(0.59,1.01)
            ax.legend(loc='lower right')
        else:
            ax.set_ylim(0.0,0.41)
            #ax.set_yscale('log')
            ax.legend(loc='upper right')

        fig.savefig(f"{key}.png")

