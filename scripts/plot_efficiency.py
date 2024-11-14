#!/usr/bin/env python
import sys
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


import uproot
import ROOT

import os

import atlasify 
atlasify.monkeypatch_axis_labels()

class TEfficiency:
    def __init__(self, tefficency):
        try:
            th1 = tefficency.GetTotalHistogram()
        except:
            th1 = tefficency

        bins = [i for i in range(th1.GetNbinsX()) if th1.GetBinContent(i) > 0.0]
        bins = bins[1:]

        self.x = [th1.GetBinCenter(i) for i in bins]

        self.x_lo = [th1.GetBinLowEdge(i) for i in bins]
        self.x_width = [th1.GetBinWidth(i) for i in bins]
        self.x_hi = np.add(self.x_lo, self.x_width)
        self.x_err_lo = np.subtract(self.x, self.x_lo)
        self.x_err_hi = np.subtract(self.x_hi, self.x)
        
        try:
            self.y = [tefficency.GetEfficiency(i) for i in bins]
            self.y_err_lo = [tefficency.GetEfficiencyErrorLow(i) for i in bins]
            self.y_err_hi = [tefficency.GetEfficiencyErrorUp(i) for i in bins]
        except:
            self.y = [tefficency.GetBinContent(i) for i in bins]
            self.y_err_lo = [tefficency.GetBinError(i) for i in bins]
            self.y_err_hi = [tefficency.GetBinError(i) for i in bins]

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


with open('configuration.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if len(sys.argv) == 1:
    files = [ f for f in os.listdir(".") if ("performance" in f) and ("tracks" in f) ]
else:
    files = sys.argv[1:]

for f in files:
    if "gnn_only" in f:
        gnn_only_file = f
    elif "fitted_tracks" in f:
        perf_fitted_file = f

try:
    perf_kf = ROOT.TFile.Open(perf_fitted_file)
except:
    perf_kf = None

perf_gnn_only = ROOT.TFile.Open(gnn_only_file)

label_dict = {
    "eta": "$\eta$", 
    "pT": "$p_T$",
    "trackeff": "Technical Efficiency",
    "duplicationRate": "Duplication Rate",
    "fakerate": "Fake Rate",
    "nMeasurements": "Number of Measurements",
}

for metric in ["trackeff", "duplicationRate", "fakerate", "nMeasurements"]:
    for var in ["eta", "pT"]:
        key = metric + "_vs_" + var
        fig, ax = plt.subplots(figsize=(10,7))
        
        gnn_only_teff = TEfficiency(perf_gnn_only.Get(key))
        gnn_only_teff.errorbar(ax, fmt='o', color='blue', label="GNN track candidates")

        if perf_kf is not None:
            kf_teff = TEfficiency(perf_kf.Get(key))
            kf_teff.errorbar(ax, fmt='None', color="red", label="Fitted tracks")

        ax.set_ylabel(label_dict[metric])
        ax.set_xlabel(f"Truth {label_dict[var]}")
        
        if metric == "trackeff":
            ax.set_ylim(0.9,1.0)
            ax.legend(loc='lower right')
        elif "rate" in metric.lower():
            ax.set_ylim(0.0,0.41)
            #ax.set_yscale('log')
            ax.legend(loc='upper right')

        if "double_matching" in gnn_only_file:
            matching = "double matching"
        elif "atlas_matching" in gnn_only_file:
            matching = "ATLAS standard matching"
        else:
            matching = "<unkknown matching>"

        #f"{config['events']} HL-LHC, ITk Layout: 03-00-00 $t\bar{t}$, HS, #LT#mu#GT = 200, #sqrt{s}=14 TeV\n"
        atlasify.atlasify("Simulation Internal",
                 "HL-LHC, ACTS standalone GNN workflow, ITk Layout: 03-00-00\n"
                 "$t\\bar{t}$, $\\langle\\mu\\rangle = 200$, $\sqrt{s}=14 TeV$\n"
                 "Truth $p_{T}$ > 1 GeV, Truth hits $\geq$ 7, Truth $|\eta| < 4$, no electrons\n"
                 "Hits on track $\geq$ 7",
                 font_size=20,
                 label_font_size=18,
                 sub_font_size=14,
                 subtext_distance=0.2)
        atlasify.enlarge_yaxis(ax, 1.2)


        fig.savefig(f"{key}.png")

