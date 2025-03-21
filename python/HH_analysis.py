#!/usr/bin/env python
# coding: utf-8

# # HH 

# ## To do
# 
# - ...

# ## Imports

# In[ ]:


# # %pip install uproot awkward numpy particle vector networkx matplotlib
# print()
# print("###############")
# print("conda packages:")
# print("###############")
# print()
# %conda list uproot
# %conda list awkward
# %conda list numpy
# %conda list particle
# %conda list vector
# %conda list networkx
# %conda list matplotlib
# print()
# print("#############")
# print("pip packages:")
# print("#############")
# print()
# %pip show uproot awkward numpy particle vector networkx matplotlib


# In[ ]:


# system and misc
import os
import sys
from types import MappingProxyType as immutable_dict # immutable dictionary

# numpy and scipy
import numpy as np
np.set_printoptions(linewidth=120)
from scipy import stats

# uproot and awkward
import uproot
import awkward as ak
import vector
vector.register_awkward()

# gen particle stuff
from particle import Particle
import networkx as nx

# histogramminng and plotting
import matplotlib.pyplot as plt
from plothist import make_hist, plot_model
import boost_histogram as bh
import mplhep as hep
hep.style.use("CMS")

# from common.py
from common import build_decay_graph, print_hierarchy, export_decay_hierarchy, draw_decay_graph, process_gen_objects, process_event_data
from common import print_summary, print_matching_base, print_matching
from common import filter_events
from common import objects, base_objects, gen_objects, jet_objects, tau_objects, muon_objects
from common import L1Jet_objects, L1DJet_objects, L1Tau_objects, L1TauP2_objects, HLT_objects, L1Mu_objects
from common import delta_phi, geometric_matching_base, geometric_matching
from common import object_matching_base, object_matching, hlt_matching_base, hlt_matching
from common import L1T_passing, HLT_passing
from common import clopper_pearson_interval, plot_sig_eff_vs_jet_rank, plot_perf_vs_pt


# ## Common

# ### Settings

# In[ ]:


# DEBUG MODE
debug = False

settings_ = immutable_dict({
    # Misc
    "debug":debug, 
    "nevents":10000 if not debug else 10,
    "skip":0 if not debug else 0,
    "verbosity":0 if not debug else 3,
    # Total integrated luminosity
    "lumi":300,
    # Kinematic thresholds
    "gen_pt_min":10.,
    "gen_eta_max":2.5,
    "off_pt_min":35.,
    "off_eta_max":2.5,
    "off_btag_min":0.55,
    "sct_pt_min":10.,
    "sct_eta_max":2.5,
    # Use only di-tau trigger, ParkingHH trigger, or the OR of both
    "option":["tautau","bb","bbtautau"][2],
    # Match L1 and HLT objects to GEN
    "use_matched":False,
})


# ## bbbb (Run 3)

# ### LOAD

# In[ ]:


#############################################################################################
#
def qcd_gen_pt_hat_range(sample):
    lower = float(sample.split("-")[1].split("To")[0])
    upper = float(sample.split("To")[1].replace("Inf", str(np.inf)))
    return lower,upper

#############################################################################################
#
def load_datasets(samples, labels=None, path="../data/", cross_sections=None, lumi=300., nevents=None, skip=0, verbosity=0):

    branches = [
        "GenPart_pt", "GenPart_eta", "GenPart_phi", # GEN-level kinematics
        "nGenPart","GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_statusFlags", # GEN-level information
        #"L1_HTT280er", # L1 seeds
        #"HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55", # HLT paths
        #"nTrigObj", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id", "TrigObj_filterBits", # Trigger objects
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagPNetB", # Offline jets
        "nL1Jet", "L1Jet_pt", "L1Jet_eta", "L1Jet_phi", # L1 jets 
        #"nL1Tau", "L1Tau_pt", "L1Tau_eta", "L1Tau_phi", # L1 taus
    ]

    total = 0
    datasets = {}
    for i,sample in enumerate(samples):
        events = uproot.concatenate(f"{path}{sample}/data_?.root:Events",branches,library="ak")
        events = events[skip:nevents+skip] if nevents is not None else events[skip:]
        nevents = len(events)
        weight = lumi * cross_sections[i] / nevents
        events["weight"] = ak.full_like(events["nGenPart"],weight,dtype="float32")
        print(f"Sample: {sample:15s} MC events: {nevents:8.2e} Weighted: {ak.sum(events.weight):8.2e}")

        if "QCD" in sample:
            #lower,upper = qcd_gen_pt_hat_range(sample)
            #events["lower_pt_hat"] = ak.full_like(events["nGenPart"], lower, dtype="float32")
            #events["upper_pt_hat"] = ak.full_like(events["nGenPart"], upper, dtype="float32")
            gen = gen_objects(events)
            mask = ((abs(gen.id) <= 6)|(gen.id == 21)) & gen.is_hard_process & gen.is_prompt # identify quarks and gluons from the hard process
            gen = ak.mask(gen,mask)
            gen_pt_hat = ak.max(gen.pt,axis=-1) #@@ Why max and not sum ?
            gen_pt_hat = ak.fill_none(gen_pt_hat,0.)
            events["gen_pt_hat"] = gen_pt_hat
        elif "HHTo4B" in sample:
            gen = gen_objects(events)
            mask = ((abs(gen.id) == 25)) & gen.is_hard_process & gen.is_prompt # identify quarks and gluons from the hard process
            #is_first_copy = (gen.first_copy == 1) & (gen.is_hard_process == 1)
            #mask = (gen.id == 5) & is_last_copy & ~is_first_copy & gen.is_prompt # identify b quarks from Higgs decay
            ##is_last_copy = (gen.last_copy == 1) & (gen.from_hard_process == 1)
            gen = ak.mask(gen,mask)
            gen_pt_hat = ak.sum(gen.pt,axis=-1)
            gen_pt_hat = ak.fill_none(gen_pt_hat,0.)
            events["gen_pt_hat"] = gen_pt_hat
        else:
            events["gen_pt_hat"]   = ak.full_like(events["nGenPart"], 0., dtype="float32")

        datasets[sample] = {"label":labels[sample], "events":events}

    events = ak.concatenate([datasets[sample]["events"] for sample in samples])
    print(f"Totals: {' '*15} MC events: {len(events):8.2e} Weighted: {ak.sum(events.weight):8.2e}")
    print()

    if verbosity>=3:
        keys = events.fields
        print()
        print("[load_data_bbbb]")
        print("All branches:")
        for key in keys:
            print(f"  {key}")
        print()
        print("L1 seeds:")
        for key in keys:
            if key.startswith("L1_") : print(f"  {key}")
        print()
        print("HLT paths:")
        for key in keys:
            if key.startswith("HLT_") : print(f"  {key}")

    return events, datasets

#############################################################################################
#
def load_datasets_signal(lumi=300., nevents=None, skip=0, verbosity=0):

    # Open ROOT file with uproot
    samples = [
        "HHTo4B",
#        "HHTo2B2Tau",
    ]

    cross_sections_pb = [
        0.01575,
#        0.01575,
    ]
    cross_sections_fb = [ xs * 1.e3 for xs in cross_sections_pb]

    labels = [
        "HH${\\to}$bbbb",
 #       "HH${\\to}$bb$\\tau\\tau$",
    ]
    labels = dict(zip(samples,labels))

    return load_datasets(
        samples,
        labels=labels,
        path="../data/Phase2/",
        cross_sections=cross_sections_fb,
        lumi=lumi,
        nevents=nevents, skip=skip, verbosity=verbosity)

#############################################################################################
#
def load_datasets_qcd(lumi=300., nevents=None, skip=0, verbosity=0):

    # Open ROOT file with uproot
    samples = [
        "QCD_Pt-20To30",
        "QCD_Pt-30To50",
        "QCD_Pt-50To80",
        "QCD_Pt-80To120",
        "QCD_Pt-120To170",
        "QCD_Pt-170To300",
        "QCD_Pt-300To470",
        "QCD_Pt-470To600",
        "QCD_Pt-600ToInf",
    ][::-1] # reversed !!!

    cross_sections_pb = [
        432900000., # == ~0.4 mb
        117200000.,
        17490000.,
        2657000.,
        467800.,
        120300.,
        8157.,
        683.1,
        241.6,
    ][::-1] # reversed !!!
    cross_sections_fb = [ xs * 1.e3 for xs in cross_sections_pb]
    
    labels = [
        "QCD (20-30)",
        "QCD (30-50)",
        "QCD (50-80)",
        "QCD (80-120)",
        "QCD (120-170)",
        "QCD (170-300)",
        "QCD (300-470)",
        "QCD (470-600)",
        "QCD (600-Inf)",
    ][::-1] # reversed !!!
    labels = dict(zip(samples,labels))

    return load_datasets(
        samples,
        labels=labels,
        path="../data/Phase2/QCD/",
        cross_sections=cross_sections_fb,
        lumi=lumi,
        nevents=nevents, skip=skip, verbosity=verbosity)


# ### PLOT

# In[ ]:


# Useful documentation
# https://plothist.readthedocs.io/en/latest/index.html (super useful)
# https://hist.readthedocs.io/en/latest/ (hmmm...)
# https://boost-histogram.readthedocs.io/en/latest/index.html (backend for histograms)
# https://mplhep.readthedocs.io/en/latest/ (matplotlib styles for CMS)

#############################################################################################
#
def histos_1d(
    datasets,
    xvar,
    **kwargs): 

    nbins = 50 if "nbins" not in kwargs else kwargs["nbins"]
    start = 0. if "start" not in kwargs else kwargs["start"]
    stop = 100. if "stop" not in kwargs else kwargs["stop"]
    
    histos = {}
    for dataset,dict in datasets.items():
        events = dict["events"]
        label = dict["label"]
        var = events[xvar]
        mask = ~ak.is_none(var)
        var = ak.drop_none(var[mask])
        wei = ak.drop_none(events["weight"][mask])
        histo = make_hist(var, bins=nbins, range=[start,stop], weights=wei)
        histos[dataset] = {"label":label, "histo":histo}
    return histos

#############################################################################################
#
def plot_1d(
        background_histos,
        signal_histos=None,
        lumi=300.,
        **kwargs):

    nbins = 50 if "nbins" not in kwargs else kwargs["nbins"]
    start = 0. if "start" not in kwargs else kwargs["start"]
    stop = 100. if "stop" not in kwargs else kwargs["stop"]
    xlabel = "Unknown" if "xlabel" not in kwargs else kwargs["xlabel"]
    ylabel = "Arbitrary" if "ylabel" not in kwargs else kwargs["ylabel"] 
    bins = np.linspace(start, stop, nbins+1)
    centers = (bins[:-1] + bins[1:]) / 2
    year = 2023 if "year" not in kwargs else kwargs["year"]
    com = 13.6 if "com" not in kwargs else kwargs["com"]
    lumi = lumi if "lumi" not in kwargs else kwargs["lumi"]
    ymin = 0.1 if "ymin" not in kwargs else kwargs["ymin"]
    ymax = 1.e3 if "ymax" not in kwargs else kwargs["ymax"]

    bkg_histos = [background_histos[dataset]["histo"] for dataset in background_histos.keys() if "QCD" in dataset]
    bkg_labels = [background_histos[dataset]["label"] for dataset in background_histos.keys() if "QCD" in dataset]
    bkg_colors = ["red","blue","green","orange","purple","brown","pink","cyan","magenta"][:len(bkg_labels)]

    sig_histos = [signal_histos[dataset]["histo"] for dataset in signal_histos.keys() if "HHTo4B" in dataset] if signal_histos is not None else []
    sig_labels = [signal_histos[dataset]["label"] for dataset in signal_histos.keys() if "HHTo4B" in dataset] if signal_histos is not None else None
    sig_colors = ["black","gray"][:len(sig_labels)] if signal_histos is not None else None
    sig_kwargs = [{"linestyle": "dotted", "linewidth":2},{"linestyle": "dashed"},][:len(sig_labels)] if signal_histos is not None else []

    plt.style.use([hep.style.CMS, hep.style.firamath])
    fig, ax = plot_model(
        stacked_components=bkg_histos,
        stacked_labels=bkg_labels,
        stacked_colors=bkg_colors,
        unstacked_components=sig_histos,
        unstacked_labels=sig_labels,
        unstacked_colors=sig_colors,
        unstacked_kwargs_list=sig_kwargs,
        xlabel=xlabel,
        ylabel=ylabel,
        model_sum_kwargs={"show": True, "label": "Total", "color": "black"},
        model_uncertainty_label="Stat. unc.",
    )
    plt.legend(fontsize=15)

    bkg_total = make_hist(bins=nbins, range=[start,stop])
    for h in bkg_histos: bkg_total.view().value += h.view().value
    print(f"Total expected number of events in {lumi}/fb for the bkgd process:  ",bkg_total.view().sum())

    if signal_histos is not None:
        sig_total = make_hist(bins=nbins, range=[start,stop])
        for h in sig_histos: sig_total.view().value += h.view().value
        print(f"Total expected number of events in {lumi}/fb for the signal process:",sig_total.view().sum())

    # y axis range
    ymax = 2. * max(bkg_total.view().value)
    ax.set_ylim(bottom=0.1,top=ymax)
    ax.set_yscale('log')
    
    hep.cms.label("Preliminary",data=False, year=year, com=com, lumi=lumi)


# In[ ]:


#############################################################################################
#
def plot_1d_gen_pt_hat(bkg_datasets, sig_datasets=None, lumi=300., nevents=None, skip=0, verbosity=0):
    nbins, start, stop = 100, 0., 1000.; width = (stop - start) / nbins
    kwargs = {
        "nbins":nbins, "start":start, "stop":stop,
        "xlabel":"GEN $\hat{p_{T}}$ [GeV]", 
        "ylabel":f"Entries / {width:.0f} GeV",
        "year":None, "com":14, "lumi":lumi}
    sig_histos = histos_1d(sig_datasets,"gen_pt_hat",**kwargs)
    qcd_histos = histos_1d(bkg_datasets,"gen_pt_hat",**kwargs)
    plot_1d(qcd_histos,sig_histos,**kwargs)

#############################################################################################
#
def plot_1d_reco_ht_lead4jets(bkg_datasets, sig_datasets=None, lumi=300., nevents=None, skip=0, verbosity=0):

    # Calculate HT from the leading 4 jets
    for _,dataset in bkg_datasets.items():
        events = dataset["events"]
        mask = (events.Jet_pt > 10.) & (abs(events.Jet_eta) < 2.5)
        events["HT"] = ak.sum(events["Jet_pt"][mask][:,:4],axis=-1)

    for _,dataset in sig_datasets.items():
        events = dataset["events"]
        mask = (events.Jet_pt > 10.) & (abs(events.Jet_eta) < 2.5)
        events["HT"] = ak.sum(events["Jet_pt"][mask][:,:4],axis=-1)

    nbins, start, stop = 100, 0., 2000.; width = (stop - start) / nbins
    kwargs = {
        "nbins":nbins, "start":start, "stop":stop,
        "xlabel":"$H_{T}$ [GeV] (leading 4 jets)", 
        "ylabel":f"Entries / {width:.0f} GeV",
        "year":None, "com":14, "lumi":lumi}
    sig_histos = histos_1d(sig_datasets,"HT",**kwargs)
    qcd_histos = histos_1d(bkg_datasets,"HT",**kwargs)
    plot_1d(qcd_histos,sig_histos,**kwargs)

#############################################################################################
#
def plot_1d_reco_ht_alljets(bkg_datasets, sig_datasets=None, lumi=300., nevents=None, skip=0, verbosity=0):

    # Calculate HT from the leading 4 jets
    for _,dataset in bkg_datasets.items():
        dataset["events"]["HT"] = ak.sum(dataset["events"]["Jet_pt"][:,:4],axis=-1)
    for _,dataset in sig_datasets.items():
        dataset["events"]["HT"] = ak.sum(dataset["events"]["Jet_pt"][:,:4],axis=-1)

    nbins, start, stop = 100, 0., 2000.; width = (stop - start) / nbins
    kwargs = {
        "nbins":nbins, "start":start, "stop":stop,
        "xlabel":"$H_{T}$ [GeV] (leading 4 jets)", 
        "ylabel":f"Entries / {width:.0f} GeV",
        "year":None, "com":14, "lumi":lumi}
    sig_histos = histos_1d(sig_datasets,"HT",**kwargs)
    qcd_histos = histos_1d(bkg_datasets,"HT",**kwargs)
    plot_1d(qcd_histos,sig_histos,**kwargs)



# ### EXECUTE

# In[ ]:


def selections_bbbb(**kwargs):
    
    # Default settings
    nevents = kwargs["nevents"] if "nevents" in kwargs.keys() else 10000
    skip = kwargs["skip"] if "skip" in kwargs.keys() else 0
    verbosity = kwargs["verbosity"] if "verbosity" in kwargs.keys() else 0
    gen_pt_min = kwargs["gen_pt_min"] if "gen_pt_min" in kwargs.keys() else 10.
    gen_eta_max = kwargs["gen_eta_max"] if "gen_eta_max" in kwargs.keys() else 2.5
    off_pt_min = kwargs["off_pt_min"] if "off_pt_min" in kwargs.keys() else 35.
    off_eta_max = kwargs["off_eta_max"] if "off_eta_max" in kwargs.keys() else 2.5
    off_btag_min = kwargs["off_btag_min"] if "off_btag_min" in kwargs.keys() else 0. #@@ DEFAULT IS ZERO ???
    sct_pt_min = kwargs["sct_pt_min"] if "sct_pt_min" in kwargs.keys() else 20.
    sct_eta_max = kwargs["sct_eta_max"] if "sct_eta_max" in kwargs.keys() else 2.5
    use_matched = kwargs["use_matched"] if "use_matched" in kwargs.keys() else False
    lumi = kwargs["lumi"] if "lumi" in kwargs.keys() else 300.
    
    _, sig_datasets = load_datasets_signal(nevents=nevents, lumi=lumi, skip=skip, verbosity=verbosity)
    bkg_events, bkg_datasets = load_datasets_qcd(nevents=nevents, lumi=lumi, skip=skip, verbosity=verbosity)
    
    if verbosity>=2:
        print()
        print("FULL DEBUG MODE!!!")
        print(" Verbosity: ", verbosity)
        print(" Num evts:  ", len(bkg_events["nGenPart"]))

    plot_1d_gen_pt_hat(bkg_datasets, sig_datasets=sig_datasets, lumi=lumi, nevents=nevents, skip=skip, verbosity=verbosity)
    plot_1d_reco_ht_lead4jets(bkg_datasets, sig_datasets=sig_datasets, lumi=lumi, nevents=nevents, skip=skip, verbosity=verbosity)

settings = settings_.copy()
settings.update({"off_btag_min":0.})
print(settings)
selections_bbbb(**settings)


# In[ ]:





# In[ ]:





# In[ ]:




