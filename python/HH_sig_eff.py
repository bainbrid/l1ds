#!/usr/bin/env python
# coding: utf-8

# # HH 

# ## To do
# 
# - Check GEN,reco matching criteria, dR, dpt
# - L1 jet pt efficiency
# - L1 tau pt efficiency
# - b-tagging efficiency
# - Signal efficiency vs jet rank
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
from plothist import plot_2d_hist, plot_2d_hist_with_projections
import boost_histogram as bh
import mplhep as hep
hep.style.use("CMS")

# from common.py
pwd = os.getcwd()
sys.path.insert(1,"/Users/bainbrid/Repositories/L1DS/HH/l1ds/python/")
from common import build_decay_graph, print_hierarchy, export_decay_hierarchy, draw_decay_graph, process_gen_objects, process_event_data
from common import print_summary, print_matching_base, print_matching
from common import filter_events
from common import objects, base_objects, gen_objects, jet_objects, tau_objects, muon_objects
from common import L1Jet_objects, L1DJet_objects, L1Tau_objects, L1TauP2_objects, HLT_objects, L1Mu_objects
from common import delta_phi, geometric_matching_base, geometric_matching
from common import object_matching_base, object_matching, hlt_matching_base, hlt_matching
from common import L1T_passing, HLT_passing
from common import clopper_pearson_interval, plot_sig_eff_vs_jet_rank, plot_perf_vs_pt, plot_sig_eff_vs_jet_rank_and_btag_score


# ## Common

# ### Settings

# In[ ]:


# DEBUG MODE
debug = False

settings_ = immutable_dict({
    # Misc
    "debug":debug, 
    "nevents":10000 if not debug else 1,
    "skip":0 if not debug else 0,
    "verbosity":0 if not debug else 2,
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
def load_data_bbbb(nevents=None,skip=0,verbosity=0):

    # Open ROOT file with uproot
    example_file = "../data/Run3/HHTo4B/data_0.root"
    file = uproot.open(example_file)
    tree = file["Events"]

    if verbosity>=3:
        keys = tree.keys()
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

    branches = [
        "GenPart_pt", "GenPart_eta", "GenPart_phi", # GEN-level kinematics
        "nGenPart","GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_statusFlags", # GEN-level information
        "L1_HTT280er", # L1 seeds
        "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55", # HLT paths
        "nTrigObj", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id", "TrigObj_filterBits", # Trigger objects
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagPNetB", # Offline jets
        "nL1Jet", "L1Jet_pt", "L1Jet_eta", "L1Jet_phi", # L1 jets 
        "nL1Tau", "L1Tau_pt", "L1Tau_eta", "L1Tau_phi", # L1 taus
    ]

    # Load data into awkward arrays
    events = tree.arrays(branches, library="ak")
    events = events[skip:nevents+skip] if nevents is not None else events[skip:]
    return events


# ### ACC

# In[ ]:


#############################################################################################
#
def GEN_acceptance_bbbb(events,pt_min,eta_max,verbosity=0):

    # Create objects from GEN particle info
    gen = gen_objects(events)
    gen = gen[ak.argsort(gen.pt, ascending=False, axis=-1)]

    # Identify interesting daughters (b-quarks) from Higgs decay
    #gen["daughter"] = (gen.id == 5) & (gen.mother_id == 25)
    is_first_copy = (gen.first_copy == 1) & (gen.is_hard_process == 1)
    is_last_copy = (gen.last_copy == 1) & (gen.from_hard_process == 1)
    gen["daughter"] = (gen.id == 5) & is_last_copy & ~is_first_copy & gen.is_prompt
    daughter_idx = ak.mask(ak.local_index(gen.daughter),gen.daughter)

    # Filter GEN particles to keep only b-quarks from Higgs decay
    gen = gen[gen.daughter]

    # Filter GEN particles to keep only those within acceptance
    gen["in_acc"] = (gen.pt > pt_min) & (abs(gen.eta) < eta_max)
    gen = gen[gen.in_acc]

    # Identify at least 4 GEN particles within acceptance
    num_in_acc = ak.sum(gen.in_acc, axis=-1)
    num_b_quarks = ak.sum(gen.daughter, axis=-1) # Redundant w.r.t. above, but consistent with bbtautau
    passed_GEN = (num_in_acc >= 4) & (num_b_quarks >= 4)
    passed_idx, = ak.where(passed_GEN) # comma dereferences the tupl(ak.array)

    # Only keep events for which all GEN particles are fully in acceptance
    gen = ak.mask(gen,passed_GEN)

    if verbosity>=1:
        print()
        print("[gen_acceptance_bbbb]")
        print(f"GEN acceptance satisfied for the following events:")
        print(", ".join([f"{x}" for x in passed_idx]))
        print_matching(gen,None)

    return passed_GEN, gen


# ### L1T

# In[ ]:


#############################################################################################
#
def L1T_passing_bbbb(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    seed = "L1_HTT280er" #@@ what about L1_QuadJet60er2p5 and L1_Mu6_HTT240er ??
    return L1T_passing(events,seed,verbosity=verbosity)

#############################################################################################
#
def L1T_matching_bbbb(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    L1Jets = L1Jet_objects(events)
    matched_L1T = object_matching(
        gen,
        L1Jets,
        passed=passed,
        gen_id_filter=5,
        n=4,
        #dr_max=0.3,dpt_min=0.2,dpt_max=2.0, # these are the default values
        label="[L1T_matching_bbbb]",
        verbosity=verbosity)
    return matched_L1T


# ### HLT

# In[ ]:


#############################################################################################
#
def HLT_passing_bbbb(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    path = "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55"
    return HLT_passing(events,path,verbosity=verbosity)

#############################################################################################
# Filters: https://cmshltinfo.app.cern.ch/path/HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_v
# Trigger bits: https://github.com/cms-sw/cmssw/blob/CMSSW_13_3_1_patch1/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L187-L217
# Useful bits (all four jets): [3,12,28] == hlt4PFCentralJetTightIDPt30, hltPFCentralJetLooseIDQuad30, hlt2PFCentralJetTightIDPt30
# More useful bits (required of just two jets): [26] == hltPFCentralJetPt30PNet2BTagMean0p55 
def HLT_matching_bbbb(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    trg = HLT_objects(events)

    # All 4 jets
    print()
    matched_HLT_4j = hlt_matching(
        gen,
        trg,
        passed=passed,
        gen_id_filter=5, # b quarks only
        hlt_id_filter=1, # Jets
        hlt_bits_filter=[3,12,28], # See above
        n=4,
        dpt_min=None,dpt_max=None, # No requirements on delta pT
        label="[HLT_matching_bbbb]",
        verbosity=verbosity)

    # Two b-jets
    print("[HLT_matching_bbbb]")
    matched_HLT_2b = hlt_matching(
        gen,
        trg,
        passed=passed,
        gen_id_filter=5, # b quarks only
        hlt_id_filter=1, # Jets
        hlt_bits_filter=[26], # See above
        n=2, # Just two jets
        dpt_min=None,dpt_max=None, # No requirements on delta pT
        label="[HLT_matching_bbbb]",
        verbosity=verbosity)

    matched_HLT = matched_HLT_4j & matched_HLT_2b

    return matched_HLT


# ### OFF

# In[ ]:


#############################################################################################
#
def OFF_matching_bbbb(events,gen,pt_min,eta_max,btag_min=0.,passed=None,verbosity=0):
    events = filter_events(events,passed)

    # Extract jet info and filter to keep only those within acceptance
    jet = jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    matched_OFF_jets,gen_,jet_ = object_matching_base(
        gen,
        jet,
        passed=passed,
        gen_id_filter=5,
        n=4,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        label="[OFF_matching_bbbb]",
        verbosity=verbosity)

    matched_OFF_bjets = object_matching(
        gen,
        jet[jet.btag >= btag_min], #@@ CURRENTLY OFFLINE B-TAGGING THRESHOLD IS ZERO TO BE CONISTENT WITH L1 SCOUTING CAPABILITIES IN RUN 3
        passed=passed,
        gen_id_filter=5,
        n=4, # Require 4 b-tagged jets
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        label="[OFF_matching_bbbb]",
        verbosity=verbosity)
    
    matched_OFF = matched_OFF_jets & matched_OFF_bjets

    return matched_OFF


# ### SCT

# In[ ]:


#############################################################################################
#
def SCT_matching_bbbb(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)
    
    # Extract L1 objects and filter to keep only those within acceptance
    jet = L1Jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    # tau = L1Tau_objects(events)
    # tau["in_acc"] = (tau.pt > pt_min) & (abs(tau.eta) < eta_max)
    # tau = tau[tau.in_acc]

    # Concatenate L1 jets and L1 taus
    # jet = ak.concatenate([jet,tau],axis=-1) #@@ NEED TO REMOVE OVERLAPPING JETS AND TAUS ??!!

    # Match either L1 jets or L1 taus
    matched_SCT = object_matching(
        gen,
        jet,
        passed=passed,
        gen_id_filter=5,
        n=4, 
        label="[SCT_matching_bbbb]",
        verbosity=verbosity)

    return matched_SCT


# ### PLOT

# In[ ]:


#############################################################################################
#
def SCT_plot_sig_eff_vs_jet_rank_bbbb(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    plot_sig_eff_vs_jet_rank(events,"L1Jet",gen,pt_min,eta_max,passed=passed,verbosity=verbosity,year=2023,com=13.6)

#############################################################################################
#
def SCT_plot_eff_vs_jet_pt_bbbb(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":2023, "com":13.6, "nbins":41, "start":0., "stop":205., "xlabel":"GEN b quark p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "eff",events,"L1Jet",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=5,
        n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)

#############################################################################################
#
def SCT_plot_purity_vs_jet_pt_bbbb(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":2023, "com":13.6, "nbins":41, "start":0., "stop":205., "xlabel":"L1 jet p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "purity",events,"L1Jet",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=5,
        n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)

#############################################################################################
#
def SCT_plot_eff_vs_tau_pt_bbbb(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":2023, "com":13.6, "nbins":41, "start":0., "stop":205., "xlabel":"GEN b quark p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "eff",events,"L1Tau",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=5,
        n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)

#############################################################################################
#
def SCT_plot_purity_vs_tau_pt_bbbb(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":2023, "com":13.6, "nbins":41, "start":0., "stop":205., "xlabel":"L1 tau p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "purity",events,"L1Tau",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=5,
        n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)


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
    #option = kwargs["option"] if "option" in kwargs.keys() else "bbtautau"
  
    events = load_data_bbbb(nevents=nevents,skip=skip,verbosity=verbosity)
    
    if verbosity>=2:
        print()
        print("FULL DEBUG MODE!!!")
        print(" Verbosity: ", verbosity)
        print(" Num evts:  ", len(events["nGenPart"]))
    
    passed_GEN, gen = GEN_acceptance_bbbb(events,pt_min=gen_pt_min,eta_max=gen_eta_max,verbosity=verbosity)
    passed_L1T = L1T_passing_bbbb(events,passed=passed_GEN,verbosity=verbosity)
    matched_L1T = L1T_matching_bbbb(events,gen,passed=passed_L1T,verbosity=verbosity) if use_matched else ak.full_like(passed_L1T,True,dtype=bool)
    passed_HLT = HLT_passing_bbbb(events,passed=matched_L1T,verbosity=verbosity)
    matched_HLT = HLT_matching_bbbb(events,gen,passed=passed_HLT,verbosity=verbosity) if use_matched else ak.full_like(passed_HLT,True,dtype=bool)
    matched_OFF = OFF_matching_bbbb(events,gen,pt_min=off_pt_min,eta_max=off_eta_max,btag_min=off_btag_min,passed=matched_HLT,verbosity=verbosity)
    matched_SCT = SCT_matching_bbbb(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    # Plotting (only plot once)
    if use_matched == False:
        SCT_plot_sig_eff_vs_jet_rank_bbbb(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_eff_vs_jet_pt_bbbb(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_purity_vs_jet_pt_bbbb(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_eff_vs_tau_pt_bbbb(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_purity_vs_tau_pt_bbbb(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    print_summary(
        events,
        passed_GEN=passed_GEN,
        passed_L1T=passed_L1T,
        matched_L1T=matched_L1T,
        passed_HLT=passed_HLT,
        matched_HLT=matched_HLT,
        matched_OFF=matched_OFF,
        matched_SCT=matched_SCT,
        use_matched=use_matched, # Use passed or matched for L1T and HLT
        )

settings = settings_.copy()
settings.update({"off_btag_min":0.})
print(settings)
selections_bbbb(**settings)
settings.update({"use_matched":True})
print(settings)
selections_bbbb(**settings)


# ## bbtautau (Run 3)

# ### LOAD

# In[ ]:


#############################################################################################
#
def load_data_bbtautau(nevents=None,skip=0,verbosity=0):

    # Open ROOT file with uproot
    example_file = "../data/Run3/HHTo2B2Tau/data_0.root"
    file = uproot.open(example_file)
    tree = file["Events"]

    if verbosity>=3:
        keys = tree.keys()
        print()
        print("[load_data_bbtautau]")
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

    branches = [
        "event", "run", "luminosityBlock", # Event identification
        "GenPart_pt", "GenPart_eta", "GenPart_phi", # GEN-level kinematics
        "nGenPart","GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_statusFlags", # GEN-level information
        "L1_DoubleIsoTau34er2p1","L1_HTT280er", # L1 seeds
        "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55", # HLT paths
        "nTrigObj", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id", "TrigObj_filterBits", # Trigger objects
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagPNetB", # Offline jets
        "nTau", "Tau_pt", "Tau_eta", "Tau_phi", # Offline taus
        "nL1Jet", "L1Jet_pt", "L1Jet_eta", "L1Jet_phi", # L1 jets 
        "nL1Tau", "L1Tau_pt", "L1Tau_eta", "L1Tau_phi", # L1 taus
    ]

    # Load data into awkward arrays
    events = tree.arrays(branches, library="ak")
    events = events[skip:nevents+skip] if nevents is not None else events[skip:]
    return events


# ### ACC

# In[ ]:


#############################################################################################
#
def GEN_acceptance_bbtautau(events,pt_min,eta_max,verbosity=0):

    # Create objects from GEN particle info
    gen = gen_objects(events)

    # Identify taus from Higgs decays
    tau_from_higgs = (gen.id == 15) & (gen.mother_id == 25)

    # Identify indices of taus from Higgs decays
    idx_tau_from_higgs = ak.mask(ak.local_index(tau_from_higgs),tau_from_higgs)

    # Identify daughters of taus
    x,y = ak.unzip(ak.cartesian([idx_tau_from_higgs, gen.mother_idx], nested=True, axis=-1))
    tau_dau = (x == y)
    tau_dau = ak.fill_none(tau_dau,False)
    idx_tau_dau = ak.firsts(ak.argsort(tau_dau,axis=2,ascending=False),axis=2)

    # Identify hadronic tau decays (i.e. daughters are not electrons nor muons)
    had_tau = (abs(gen.id[idx_tau_dau]) != 11) & (abs(gen.id[idx_tau_dau]) != 13)

    # Identify hadronically decaying taus from Higgs decays
    gen["had_tau_from_higgs"] = had_tau & tau_from_higgs 
    #idx_had_tau_from_higgs = ak.mask(ak.local_index(gen.had_tau_from_higgs),gen.had_tau_from_higgs)

    # Identify GEN b-quarks from Higgs decay
    gen["b_quarks_from_higgs"] = (gen.id == 5) & (gen.mother_id == 25)

    # Identify interesting daughters (had taus or b-quarks) from Higgs decay
    gen["daughter"] = gen.had_tau_from_higgs | gen.b_quarks_from_higgs
    daughter_idx = ak.mask(ak.local_index(gen.daughter),gen.daughter)

    # Filter GEN particles to keep only b-quarks from Higgs decay
    gen = gen[gen.daughter]

    # Filter GEN particles to keep only those within acceptance
    gen["in_acc"] = (gen.pt > pt_min) & (abs(gen.eta) < eta_max)
    gen = gen[gen.in_acc]

    # Identify at least 4 GEN particles within acceptance
    num_in_acc = ak.sum(gen.in_acc, axis=-1)
    num_b_quarks = ak.sum(gen.daughter, axis=-1) # Redundant w.r.t. above, but consistent with bbtautau
    passed_GEN = (num_in_acc >= 4) & (num_b_quarks >= 4)
    passed_idx, = ak.where(passed_GEN) # comma dereferences the tupl(ak.array)

    # Identify at least 4 GEN particles within acceptance, two of which are taus and two are b quarks
    num_in_acc = ak.sum(gen.in_acc, axis=-1)
    num_b_quarks = ak.sum(gen.b_quarks_from_higgs, axis=-1)
    num_tau_had = ak.sum(gen.had_tau_from_higgs, axis=-1)
    passed_GEN = (num_in_acc >= 4) & (num_tau_had >= 2) & (num_b_quarks >= 2)
    passed_idx, = ak.where(passed_GEN) # comma dereferences the tupl(ak.array)

    # Only keep events for which all GEN particles are fully in acceptance
    gen = ak.mask(gen,passed_GEN)

    if verbosity>=1:
        print()
        print("[gen_acceptance_bbtautau]")
        print(f"GEN acceptance satisfied for the following events:")
        print(", ".join([f"{x}" for x in passed_idx]))
        print_matching(gen,None)

    return passed_GEN, gen


# ### L1T

# In[ ]:


#############################################################################################
#
def L1T_passing_tautau(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    seed = "L1_DoubleIsoTau34er2p1"
    return L1T_passing(events,seed,verbosity=verbosity)

#############################################################################################
#
def L1T_passing_bb(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    seed = "L1_HTT280er"
    return L1T_passing(events,seed,verbosity=verbosity)

#############################################################################################
#
def L1T_passing_bbtautau(events,passed=None,option=None,verbosity=0):
    if option == "tautau":
        return L1T_passing_tautau(events,passed=passed,verbosity=verbosity)
    elif option == "bb":
        return L1T_passing_bb(events,passed=passed,verbosity=verbosity)
    elif option == "bbtautau" or option is None:
        passed_L1T_tautau = L1T_passing_tautau(events,passed=passed,verbosity=verbosity)
        passed_L1T_bb = L1T_passing_bb(events,passed=passed,verbosity=verbosity)
        return passed_L1T_tautau | passed_L1T_bb
    else:
        raise ValueError(f"Invalid option: {option}")

#############################################################################################
#
def L1T_matching_tautau(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    L1Taus = L1Tau_objects(events)
    L1Taus = L1Taus[L1Taus.pt>34.] # Two taus matched to GEN, each satisfying pT > 34 GeV 
    matched_L1T = object_matching(gen,L1Taus,gen_id_filter=15,n=2,verbosity=verbosity)
    return matched_L1T

#############################################################################################
#
def L1T_matching_bb(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    L1Jets = L1Jet_objects(events)
    L1Jets = L1Jets[L1Jets.pt>30.] # Four jets matched to GEN, each contributing 30 GeV to HTT sum
    matched_L1T = object_matching(gen,L1Jets,n=4,verbosity=verbosity)
    return matched_L1T

#############################################################################################
#
def L1T_matching_bbtautau(events,gen,passed=None,option=None,verbosity=0):
    if option == "tautau":
        return L1T_matching_tautau(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bb":
        return L1T_matching_bb(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bbtautau" or option is None:
        matched_L1T_tautau = L1T_matching_tautau(events,gen,passed=passed,verbosity=verbosity)
        matched_L1T_bb = L1T_matching_bb(events,gen,passed=passed,verbosity=verbosity)
        return matched_L1T_tautau | matched_L1T_bb
    else:
        raise ValueError(f"Invalid option: {option}")


# ### HLT

# In[ ]:


#############################################################################################
#
def HLT_passing_tautau(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    path = "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1"
    return HLT_passing(events,path,verbosity=verbosity)

#############################################################################################
#
def HLT_passing_bb(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    path = "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55"
    return HLT_passing(events,path,verbosity=verbosity)

#############################################################################################
#
def HLT_passing_bbtautau(events,passed=None,option=None,verbosity=0):
    if option == "tautau":
        return HLT_passing_tautau(events,passed=passed,verbosity=verbosity)
    elif option == "bb":
        return HLT_passing_bb(events,passed=passed,verbosity=verbosity)
    elif option == "bbtautau" or option is None:
        passed_HLT_tautau = HLT_passing_tautau(events,passed=passed,verbosity=verbosity)
        passed_HLT_bb = HLT_passing_bb(events,passed=passed,verbosity=verbosity)
        return passed_HLT_tautau | passed_HLT_bb
    else:
        raise ValueError(f"Invalid option: {option}")
    
#############################################################################################
# Filters: https://cmshltinfo.app.cern.ch/path/HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1_v
# Trigger bits: https://github.com/cms-sw/cmssw/blob/CMSSW_13_3_1_patch1/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L137-L166
# Useful bits (two taus): [1,3,5,10] == *Medium*, *DeepTau*, *Hps*, hlt*DoublePFTau*L1HLTMatched
def HLT_matching_tautau(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    trg = HLT_objects(events)

    # DoubleTau trigger matching
    matched_HLT = hlt_matching(
        gen,
        trg,
        passed=passed,
        gen_id_filter=15, # taus only
        hlt_id_filter=15, # taus only
        hlt_bits_filter=[1,3,5,10], # See above
        n=2, # Both taus
        dpt_min=None,dpt_max=None,
        verbosity=verbosity)
    
    return matched_HLT

#############################################################################################
# Filters: https://cmshltinfo.app.cern.ch/path/HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_v
# Trigger bits: https://github.com/cms-sw/cmssw/blob/CMSSW_13_3_1_patch1/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L187-L217
# Useful bits (all four jets): [3,12,28] == hlt4PFCentralJetTightIDPt30, hltPFCentralJetLooseIDQuad30, hlt2PFCentralJetTightIDPt30
# More useful bits (required of just two jets): [26] == hltPFCentralJetPt30PNet2BTagMean0p55 
def HLT_matching_bb(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    trg = HLT_objects(events)

    # All 4 jets
    matched_HLT_4j = hlt_matching(
        gen,
        trg,
        passed=passed,
        gen_id_filter=5, # b quarks only
        hlt_id_filter=1, # Jets
        hlt_bits_filter=[3,12,28], # See above
        n=2, # All four jets
        dpt_min=None,dpt_max=None, # No requirements on delta pT
        verbosity=verbosity)

    # Two b-jets
    matched_HLT_2b = hlt_matching(
        gen,
        trg,
        passed=passed,
        gen_id_filter=5, # b quarks only
        hlt_id_filter=1, # Jets
        hlt_bits_filter=[26], # See above
        n=2, # Just two jets
        dpt_min=None,dpt_max=None, # No requirements on delta pT
        verbosity=verbosity)

    return matched_HLT_4j & matched_HLT_2b

#############################################################################################
#
def HLT_matching_bbtautau(events,gen,passed=None,option=None,verbosity=0):
    if option == "tautau":
        return HLT_matching_tautau(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bb":
        return HLT_matching_bb(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bbtautau" or option is None:
        passed_HLT_tautau = HLT_matching_tautau(events,gen,passed=passed,verbosity=verbosity)
        passed_HLT_bb = HLT_matching_bb(events,gen,passed=passed,verbosity=verbosity)
        return passed_HLT_tautau | passed_HLT_bb
    else:
        raise ValueError(f"Invalid option: {option}")


# ###  OFF

# In[ ]:


#############################################################################################
#
def OFF_matching_tautau(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)

    # Extract tau info and filter to keep only those within acceptance
    tau = tau_objects(events)
    tau["in_acc"] = (tau.pt > pt_min) & (abs(tau.eta) < eta_max)
    tau = ak.mask(tau,tau.in_acc)

    # Match reco to gen
    matched_OFF = object_matching(
        gen,
        tau,
        passed=passed,
        gen_id_filter=15,
        n=2,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )
    
    return matched_OFF        

#############################################################################################
#
def OFF_matching_bb(events,gen,pt_min,eta_max,btag_min=0.,passed=None,verbosity=0):

    events = filter_events(events,passed)

    # Extract jet info and filter to keep only those within acceptance
    jet = jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = ak.mask(jet,jet.in_acc)

    matched_OFF_jets = object_matching(
        gen,
        jet, # Only consider jets
        passed=passed,
        gen_id_filter=5,
        n=2,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )
    
    matched_OFF_bjets = object_matching(
        gen,
        jet[jet.btag >= btag_min], #@@ CURRENTLY OFFLINE B-TAGGING NOT APPLIED TO BE CONISTENT WITH L1 SCOUTING CAPABILITIES IN RUN 3
        passed=passed,
        gen_id_filter=5,
        n=2,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )

    
    return matched_OFF_jets #@@ & matched_OFF_bjets

#############################################################################################
#
def OFF_matching_bbtautau(events,gen,pt_min=35.,eta_max=2.5,btag_min=0.,passed=None,verbosity=0):
    matched_OFF_tautau = OFF_matching_tautau(events,gen,pt_min=pt_min,eta_max=eta_max,passed=passed,verbosity=verbosity)
    matched_OFF_bb = OFF_matching_bb(events,gen,pt_min=pt_min,eta_max=eta_max,btag_min=btag_min,passed=passed,verbosity=verbosity)
    return matched_OFF_tautau & matched_OFF_bb


# ### SCT

# In[ ]:


#############################################################################################
#
def SCT_matching_tautau(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)
    
    # Extract L1 objects and filter to keep only those within acceptance
    tau = L1Tau_objects(events)
    tau["in_acc"] = (tau.pt > pt_min) & (abs(tau.eta) < eta_max)
    tau = tau[tau.in_acc]

    # Match L1 taus
    matched_SCT = object_matching(
        gen,
        tau,
        passed=passed,
        gen_id_filter=15,
        n=2,
        verbosity=verbosity)

    return matched_SCT

#############################################################################################
#
def SCT_matching_bb(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)
    
    # Extract L1 objects and filter to keep only those within acceptance
    jet = L1Jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    # tau = L1Tau_objects(events)
    # tau["in_acc"] = (tau.pt > pt_min) & (abs(tau.eta) < eta_max)
    # tau = tau[tau.in_acc]

    # L1 jets
    L1_jet_pt = events["L1Jet_pt"]
    L1_jet_eta = events["L1Jet_eta"]
    L1_jet_phi = events["L1Jet_phi"]
    valid_L1_jets = (L1_jet_pt > pt_min) & (abs(L1_jet_eta) < eta_max)
    L1_jet_pt = L1_jet_pt[valid_L1_jets]
    L1_jet_eta = L1_jet_eta[valid_L1_jets]
    L1_jet_phi = L1_jet_phi[valid_L1_jets]
    L1_jets = ak.zip({"pt": L1_jet_pt, "eta": L1_jet_eta, "phi": L1_jet_phi, "id": ak.full_like(L1_jet_pt,1)})

    # Match L1 jets 
    matched_SCT = object_matching(
        gen,
        L1_jets,
        passed=passed,
        gen_id_filter=5,
        n=2,
        verbosity=verbosity)

    return matched_SCT

#############################################################################################
#
def SCT_matching_bbtautau(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    matched_SCT_tautau = SCT_matching_tautau(events,gen,pt_min=pt_min,eta_max=eta_max,passed=passed,verbosity=verbosity)
    matched_SCT_bb = SCT_matching_bb(events,gen,pt_min=pt_min,eta_max=eta_max,passed=passed,verbosity=verbosity)
    return matched_SCT_tautau & matched_SCT_bb


# ### PLOT

# In[ ]:


#############################################################################################
#
def SCT_plot_eff_vs_tau_pt_bbtautau(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":2023, "com":13.6, "nbins":41, "start":0., "stop":205., "xlabel":"GEN tau lepton p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "eff",events,"L1Tau",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=15,
        n=2,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)

#############################################################################################
#
def SCT_plot_purity_vs_tau_pt_bbtautau(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":2023, "com":13.6, "nbins":41, "start":0., "stop":205., "xlabel":"L1 tau p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "purity",events,"L1Tau",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=15,
        n=2,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)


# ### EXECUTE

# In[ ]:


def selections_bbtautau(**kwargs):

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
    option = kwargs["option"] if "option" in kwargs.keys() else "bbtautau"

    events = load_data_bbtautau(nevents=nevents,skip=skip,verbosity=verbosity)

    if verbosity>=2:
        print()
        print("FULL DEBUG MODE!!!")
        print(" Verbosity: ", verbosity)
        print(" Num evts:  ", len(events["nGenPart"]))

    passed_GEN, gen = GEN_acceptance_bbtautau(events,pt_min=gen_pt_min,eta_max=gen_eta_max,verbosity=verbosity)
    passed_L1T = L1T_passing_bbtautau(events,passed=passed_GEN,option=option,verbosity=verbosity)
    matched_L1T = L1T_matching_bbtautau(events,gen,passed=passed_L1T,option=option,verbosity=verbosity) if use_matched else ak.full_like(passed_L1T,True,dtype=bool)
    passed_HLT = HLT_passing_bbtautau(events,passed=matched_L1T,option=option,verbosity=verbosity)
    matched_HLT = HLT_matching_bbtautau(events,gen,passed=passed_HLT,option=option,verbosity=verbosity) if use_matched else ak.full_like(passed_HLT,True,dtype=bool)
    matched_OFF = OFF_matching_bbtautau(events,gen,pt_min=off_pt_min,eta_max=off_eta_max,btag_min=off_btag_min,passed=matched_HLT,verbosity=verbosity)
    matched_SCT = SCT_matching_bbtautau(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    # Plotting (only plot once)
    if use_matched == False and option == "tautau":
        SCT_plot_eff_vs_tau_pt_bbtautau(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_purity_vs_tau_pt_bbtautau(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    print_summary(
        events,
        passed_GEN=passed_GEN,
        passed_L1T=passed_L1T,
        matched_L1T=matched_L1T,
        passed_HLT=passed_HLT,
        matched_HLT=matched_HLT,
        matched_OFF=matched_OFF,
        matched_SCT=matched_SCT,
        use_matched=use_matched, # Use passed or matched for L1T and HLT
        )

# option="tautau"
print()
settings = settings_.copy()
settings.update({"off_btag_min":0.})
settings.update({"option":"tautau"})
print(settings)
selections_bbtautau(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbtautau(**settings) 

# option="bb"
print()
settings = settings_.copy()
settings.update({"off_btag_min":0.})
settings.update({"option":"bb"})
print(settings)
selections_bbtautau(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbtautau(**settings) 

# option="bbtautau"
print()
settings = settings_.copy()
settings.update({"off_btag_min":0.})
settings.update({"option":"bbtautau"})
print(settings)
selections_bbtautau(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbtautau(**settings) 


# ## bbbb (Phase 2)

# ### LOAD

# In[ ]:


#############################################################################################
# 
def load_data_bbbb_phase2(nevents=None,skip=0,verbosity=0):

    # Open ROOT file with uproot
    example_file = "../data/Phase2/HHTo4B/data_0.root"
    file = uproot.open(example_file)
    tree = file["Events"]

    if verbosity>=3:
        keys = tree.keys()
        print()
        print("[load_data_bbbb_phase2]")
        print("All branches:")
        for key in keys:
            print(f"  {key}")
        print()
        print("L1 seeds:")
        for key in keys:
            if key.startswith("L1T_") : print(f"  {key}")
        print()
        print("HLT paths:")
        for key in keys:
            if key.startswith("HLT_") : print(f"  {key}")

    branches = [
        "GenPart_pt", "GenPart_eta", "GenPart_phi", # GEN-level kinematics
        "nGenPart", "GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_statusFlags", # GEN-level information
        "L1T_PFHT400PT30_QuadPFPuppiJet_70_55_40_40_2p4", # L1 seeds
        "HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4", # HLT paths
        "HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4", # HLT paths
        "nTrigObj", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id", "TrigObj_filterBits", # Trigger objects
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagPNetB", "Jet_btagDeepFlavB", # Offline jets
        "nL1Jet", "L1Jet_pt", "L1Jet_eta", "L1Jet_phi", # L1 jets 
        "nL1DisplacedJet", "L1DisplacedJet_pt", "L1DisplacedJet_eta", "L1DisplacedJet_phi", "L1DisplacedJet_btagScore", # L1 displaced jets 
    ]

    # Load data into awkward arrays
    events = tree.arrays(branches, library="ak")
    events = events[skip:nevents+skip] if nevents is not None else events[skip:]
    return events


# ### ACC

# In[ ]:


#############################################################################################
#
def GEN_acceptance_bbbb_phase2(events,pt_min,eta_max,verbosity=0):
    return GEN_acceptance_bbbb(events,pt_min=pt_min,eta_max=eta_max,verbosity=verbosity)


# ### L1T

# In[ ]:


#############################################################################################
#
def L1T_passing_bbbb_phase2(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    seed = "L1T_PFHT400PT30_QuadPFPuppiJet_70_55_40_40_2p4"
    return L1T_passing(events,seed,verbosity=verbosity)

#############################################################################################
#
def L1T_matching_bbbb_phase2(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    L1Jets = L1Jet_objects(events)
    matched_L1T = object_matching(gen,L1Jets,passed=passed,gen_id_filter=5,n=4,verbosity=verbosity)
    return matched_L1T


# ### HLT

# In[ ]:


#############################################################################################
#
def HLT_passing_bbbb_phase2(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    path = "HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4" # Looser
    #path = "HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4" # Tighter
    return HLT_passing(events,path,verbosity=verbosity)

#############################################################################################
# Trigger bits: https://github.com/ic-l1ds/cmssw/blob/14_2_0_pre1_trigobj_ph2/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L190-L193
# Useful bits (all four jets): [1,2,3] == hltPFPuppiCentralJetsQuad*HT*MaxEta2p4, hlt*PFPuppiCentralJet*MaxEta2p4, hltPFPuppiCentralJetQuad*MaxEta2p4
# More useful bits (required of just two jets): [0] == hltBTagPFPuppiDeepFlavour*Eta2p4TripleEta2p4 
def HLT_matching_bbbb_phase2(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    trg = HLT_objects(events)

    # DoubleTau trigger matching
    matched_HLT = hlt_matching(
        gen,
        trg,
        passed=passed,
        gen_id_filter=5, # Only b quarks
        hlt_id_filter=1, # Only jets
        hlt_bits_filter=[0,2,3], #@@ These seeem to be the most useful bits, bit=1 rarely set ...??
        n=4,
        dpt_min=None,dpt_max=None, # No requirements on delta pT
        verbosity=verbosity)
    
    return matched_HLT


# ### OFF

# In[ ]:


#############################################################################################
#
def OFF_matching_bbbb_phase2(events,gen,pt_min,eta_max,btag_min=0.,passed=None,verbosity=0):
    events = filter_events(events,passed)

    # Extract jet info and filter to keep only those within acceptance
    jet = jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    matched_OFF_jets = object_matching(
        gen,
        jet,
        passed=passed,
        gen_id_filter=5,
        n=4,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )
    
    matched_OFF_bjets = object_matching(
        gen,
        jet[jet.btag >= btag_min], # Subset of jets that satisfy b-tagging requirement
        passed=passed,
        gen_id_filter=5,
        n=4, # Require 4 b-tagged jets
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )

    #@@ WHAT B-TAGGING REQ SHOULD WE APPLY HERE????

    return matched_OFF_jets# & matched_OFF_bjets


# ### SCT

# In[ ]:


#############################################################################################
#
def SCT_matching_bbbb_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)
    
    # Extract L1 jets and filter to keep only those within acceptance
    jet = L1Jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    # Extract L1 displaced jets and filter to keep only those within acceptance
    djet = L1DJet_objects(events)
    djet["btag"] = events["L1DisplacedJet_btagScore"]
    djet["in_acc"] = (djet.pt > pt_min) & (abs(djet.eta) < eta_max)
    djet = djet[djet.in_acc]

    # Concatenate L1 jets and L1 displaced jets
    #@@ jet = ak.concatenate([jet,djet],axis=-1) #@@ NEED TO REMOVE OVERLAPPING JETS

    matched_SCT_jets = object_matching(
        gen,
        jet,
        passed=passed,
        gen_id_filter=5,
        n=4,
        verbosity=verbosity)

    matched_SCT_bjets = object_matching(
        gen,
        djet[djet.btag > 0.55], # Subset of jets that satisfy b-tagging requirement
        passed=passed,
        gen_id_filter=5,
        n=4, # Require 4 b-tagged jets
        verbosity=verbosity)

    #@@ WHAT B-TAGGING REQ SHOULD WE APPLY HERE????

    return matched_SCT_jets# & matched_SCT_bjets


# ### PLOT

# In[ ]:


#############################################################################################
#
def SCT_plot_sig_eff_vs_jet_rank_bbbb_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    plot_sig_eff_vs_jet_rank(events,"L1DisplacedJet",gen,pt_min,eta_max,passed=passed,verbosity=verbosity,year="Phase 2",com=14)

#############################################################################################
#
def SCT_plot_sig_eff_vs_jet_rank_and_btag_score_bbbb_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    plot_sig_eff_vs_jet_rank_and_btag_score(events,"L1DisplacedJet",gen,pt_min,eta_max,passed=passed,verbosity=verbosity,year="Phase 2",com=14)

#############################################################################################
#
def SCT_plot_eff_vs_jet_pt_bbbb_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":"Phase 2", "com":14, "nbins":41, "start":0., "stop":205., "xlabel":"GEN b quark p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "eff",events,"L1Jet",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=5,
        n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)

#############################################################################################
#
def SCT_plot_purity_vs_jet_pt_bbbb_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":"Phase 2", "com":14, "nbins":41, "start":0., "stop":205., "xlabel":"L1 jet p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "purity",events,"L1Jet",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=5,
        n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)


# ### EXECUTE

# In[ ]:


def selections_bbbb_phase2(**kwargs):

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
    #option = kwargs["option"] if "option" in kwargs.keys() else "bbtautau"

    events = load_data_bbbb_phase2(nevents=nevents,skip=skip,verbosity=verbosity)

    if verbosity>=2:
        print()
        print("FULL DEBUG MODE!!!")
        print(" Verbosity: ", verbosity)
        print(" Num evts:  ", len(events["nGenPart"]))
    
    passed_GEN, gen = GEN_acceptance_bbbb_phase2(events,pt_min=gen_pt_min,eta_max=gen_eta_max,verbosity=verbosity)
    passed_L1T = L1T_passing_bbbb_phase2(events,passed=passed_GEN,verbosity=verbosity)
    matched_L1T = L1T_matching_bbbb_phase2(events,gen,passed=passed_L1T,verbosity=verbosity) if use_matched else ak.full_like(passed_L1T,True,dtype=bool)
    passed_HLT = HLT_passing_bbbb_phase2(events,passed=matched_L1T,verbosity=verbosity)
    matched_HLT = HLT_matching_bbbb_phase2(events,gen,passed=passed_HLT,verbosity=verbosity) if use_matched else ak.full_like(passed_HLT,True,dtype=bool)
    matched_OFF = OFF_matching_bbbb_phase2(events,gen,pt_min=off_pt_min,eta_max=off_eta_max,btag_min=off_btag_min,passed=matched_HLT,verbosity=verbosity)
    matched_SCT = SCT_matching_bbbb_phase2(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    # Plotting (only plot once)
    if use_matched == False:
        SCT_plot_sig_eff_vs_jet_rank_bbbb_phase2(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_eff_vs_jet_pt_bbbb_phase2(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_purity_vs_jet_pt_bbbb_phase2(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_sig_eff_vs_jet_rank_and_btag_score_bbbb_phase2(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    print_summary(
        events,
        passed_GEN=passed_GEN,
        passed_L1T=passed_L1T,
        matched_L1T=matched_L1T,
        passed_HLT=passed_HLT,
        matched_HLT=matched_HLT,
        matched_OFF=matched_OFF,
        matched_SCT=matched_SCT,
        use_matched=use_matched, # Use passed or matched for L1T and HLT
        )

settings = settings_.copy()
print(settings)
selections_bbbb_phase2(**settings)
settings.update({"use_matched":True})
print(settings)
selections_bbbb_phase2(**settings)


# ## bbtautau (Phase 2)

# ### LOAD

# In[ ]:


#############################################################################################
# 
def load_data_bbtautau_phase2(nevents=None,skip=0,verbosity=0):

    # Open ROOT file with uproot
    example_file = "../data/Phase2/HHTo2B2Tau/data_0.root"
    file = uproot.open(example_file)
    tree = file["Events"]

    if verbosity>=3:
        keys = tree.keys()
        print()
        print("[load_data_bbtautau_phase2]")
        print("All branches:")
        for key in keys:
            print(f"  {key}")
        print()
        print("L1 seeds:")
        for key in keys:
            if key.startswith("L1T_") : print(f"  {key}")
        print()
        print("HLT paths:")
        for key in keys:
            if key.startswith("HLT_") : print(f"  {key}")

    branches = [
        "GenPart_pt", "GenPart_eta", "GenPart_phi", # GEN-level kinematics
        "nGenPart", "GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_statusFlags", # GEN-level information
        "L1_pDoublePuppiTau52_52_final", #@@ L1 seeds
        "L1T_PFHT400PT30_QuadPFPuppiJet_70_55_40_40_2p4", # L1 seeds
        "HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1", # HLT paths
        "HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4", # HLT paths
        "HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4", # HLT paths
        "nTrigObj", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id", "TrigObj_filterBits", # Trigger objects
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagPNetB", # Offline jets
        "nTau", "Tau_pt", "Tau_eta", "Tau_phi", # Offline taus
        "nL1Jet", "L1Jet_pt", "L1Jet_eta", "L1Jet_phi", # L1 displaced jets 
        "nL1DisplacedJet", "L1DisplacedJet_pt", "L1DisplacedJet_eta", "L1DisplacedJet_phi", "L1DisplacedJet_btagScore", # L1 displaced jets 
        "nL1GTnnTau", "L1GTnnTau_pt", "L1GTnnTau_eta", "L1GTnnTau_phi", # L1 taus 
    ]

    # Load data into awkward arrays
    events = tree.arrays(branches, library="ak")
    events = events[skip:nevents+skip] if nevents is not None else events[skip:]
    return events


# ### ACC

# In[ ]:


#############################################################################################
#
def GEN_acceptance_bbtautau_phase2(events,pt_min,eta_max,verbosity=0):
    return GEN_acceptance_bbtautau(events,pt_min=pt_min,eta_max=eta_max,verbosity=verbosity)


# ### L1T

# In[ ]:


#############################################################################################
#
def L1T_passing_tautau_phase2(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    seed = "L1_pDoublePuppiTau52_52_final"
    return L1T_passing(events,seed,verbosity=verbosity)

#############################################################################################
#
def L1T_passing_bb_phase2(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    seed = "L1T_PFHT400PT30_QuadPFPuppiJet_70_55_40_40_2p4"
    return L1T_passing(events,seed,verbosity=verbosity)

#############################################################################################
#
def L1T_passing_bbtautau_phase2(events,passed=None,option=None,verbosity=0):
    if option == "tautau":
        return L1T_passing_tautau_phase2(events,passed=passed,verbosity=verbosity)
    elif option == "bb":
        return L1T_passing_bb_phase2(events,passed=passed,verbosity=verbosity)
    elif option == "bbtautau" or option is None:
        passed_L1T_tautau = L1T_passing_tautau_phase2(events,passed=passed,verbosity=verbosity)
        passed_L1T_bb = L1T_passing_bb_phase2(events,passed=passed,verbosity=verbosity)
        return passed_L1T_tautau | passed_L1T_bb
    else:
        raise ValueError(f"Invalid option: {option}")

#############################################################################################
#
def L1T_matching_tautau_phase2(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    L1Taus = L1TauP2_objects(events)
    L1Taus = L1Taus[L1Taus.pt>52.] # Two taus matched to GEN, each satisfying pT > 52 GeV 
    matched_L1T = object_matching(gen,L1Taus,passed=passed,gen_id_filter=15,n=2,verbosity=verbosity)
    return matched_L1T

#############################################################################################
#
def L1T_matching_bb_phase2(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    L1Jets = L1Jet_objects(events)
    L1Jets = L1Jets[L1Jets.pt>30.] # Four jets matched to GEN, each contributing 30 GeV to HTT sum (conservative, thresholds are actually higher)
    matched_L1T = object_matching(gen,L1Jets,passed=passed,n=4,verbosity=verbosity)
    return matched_L1T

#############################################################################################
#
def L1T_matching_bbtautau_phase2(events,gen,passed=None,option=None,verbosity=0):
    if option == "tautau":
        return L1T_matching_tautau_phase2(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bb":
        return L1T_matching_bb_phase2(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bbtautau" or option is None:
        matched_L1T_tautau = L1T_matching_tautau_phase2(events,gen,passed=passed,verbosity=verbosity)
        matched_L1T_bb = L1T_matching_bb_phase2(events,gen,passed=passed,verbosity=verbosity)
        return matched_L1T_tautau | matched_L1T_bb 
    else:
        raise ValueError(f"Invalid option: {option}")


# ### HLT

# In[ ]:


#############################################################################################
#
def HLT_passing_tautau_phase2(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    path = "HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1"
    return HLT_passing(events,path,verbosity=verbosity)

#############################################################################################
#
def HLT_passing_bb_phase2(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    path = "HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4" # Looser
    #path = "HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4" # Tighter
    return HLT_passing(events,path,verbosity=verbosity)

#############################################################################################
#
def HLT_passing_bbtautau_phase2(events,passed=None,option=None,verbosity=0):
    if option == "tautau":
        return HLT_passing_tautau_phase2(events,passed=passed,verbosity=verbosity)    
    elif option == "bb":
        return HLT_passing_bb_phase2(events,passed=passed,verbosity=verbosity)
    elif option == "bbtautau" or option is None:
        passed_HLT_tautau = HLT_passing_tautau_phase2(events,passed=passed,verbosity=verbosity)
        passed_HLT_bb = HLT_passing_bb_phase2(events,passed=passed,verbosity=verbosity)
        return passed_HLT_tautau | passed_HLT_bb
    else:
        raise ValueError(f"Invalid option: {option}")

#############################################################################################
# Trigger bits: https://github.com/ic-l1ds/cmssw/blob/14_2_0_pre1_trigobj_ph2/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L139-L169
# Useful bits (two taus): [0,1,2,3] == *Medium*, *DeepTau*, *Hps*, hlt*DoublePFTau*, *Hps*
def HLT_matching_tautau_phase2(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    trg = HLT_objects(events)

    # DoubleTau trigger matching
    matched_HLT = hlt_matching(
        gen,
        trg,
        passed=passed,
        gen_id_filter=15, # taus only
        hlt_id_filter=15, # taus only
        hlt_bits_filter=[0,1,2,3], # See above
        n=2, # Both taus
        dpt_min=None,dpt_max=None,
        verbosity=verbosity)
    
    return matched_HLT

#############################################################################################
# Trigger bits: https://github.com/ic-l1ds/cmssw/blob/14_2_0_pre1_trigobj_ph2/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L190-L193
# Useful bits (all four jets): [1,2,3] == hltPFPuppiCentralJetsQuad*HT*MaxEta2p4, hlt*PFPuppiCentralJet*MaxEta2p4, hltPFPuppiCentralJetQuad*MaxEta2p4
# More useful bits (required of just two jets): [0] == hltBTagPFPuppiDeepFlavour*Eta2p4TripleEta2p4 
def HLT_matching_bb_phase2(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    trg = HLT_objects(events)

    # DoubleTau trigger matching
    matched_HLT = hlt_matching(
        gen,
        trg,
        passed=passed,
        gen_id_filter=5, # Only b quarks
        hlt_id_filter=1, # Only jets
        hlt_bits_filter=[0,2,3], #@@ These seeem to be the most useful bits, bit=1 rarely set ...??
        n=4,
        dpt_min=None,dpt_max=None, # No requirements on delta pT
        verbosity=verbosity)
    
    return matched_HLT

#############################################################################################
#
def HLT_matching_bbtautau_phase2(events,gen,passed=None,option=None,verbosity=0):
    if option == "tautau":
        return HLT_matching_tautau_phase2(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bb":
        return HLT_matching_bb_phase2(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bbtautau" or option is None:
        passed_HLT_tautau = HLT_matching_tautau_phase2(events,gen,passed=passed,verbosity=verbosity)
        passed_HLT_bb = HLT_matching_bb_phase2(events,gen,passed=passed,verbosity=verbosity)
        return passed_HLT_tautau | passed_HLT_bb
    else:
        raise ValueError(f"Invalid option: {option}")


# ### OFF

# In[ ]:


#############################################################################################
#
def OFF_matching_tautau_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)

    # Extract tau info and filter to keep only those within acceptance
    tau = tau_objects(events)
    tau["in_acc"] = (tau.pt > pt_min) & (abs(tau.eta) < eta_max)
    tau = ak.mask(tau,tau.in_acc)

    # Match reco to gen
    matched_OFF = object_matching(
        gen,
        tau,
        passed=passed,
        gen_id_filter=15,
        n=2,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )
    
    return matched_OFF        

#############################################################################################
#
def OFF_matching_bb_phase2(events,gen,pt_min,eta_max,btag_min=0.,passed=None,verbosity=0):

    events = filter_events(events,passed)

    # Extract jet info and filter to keep only those within acceptance
    jet = jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = ak.mask(jet,jet.in_acc)

    matched_OFF_jets = object_matching(
        gen,
        jet, # Only consider jets
        passed=passed,
        gen_id_filter=5,
        n=2,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )
    
    matched_OFF_bjets = object_matching(
        gen,
        jet[jet.btag >= btag_min], #@@ CURRENTLY OFFLINE B-TAGGING NOT APPLIED TO BE CONISTENT WITH L1 SCOUTING CAPABILITIES IN RUN 3
        passed=passed,
        gen_id_filter=5,
        n=2,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )

    return matched_OFF_jets #@@ & matched_OFF_bjets

#############################################################################################
#
def OFF_matching_bbtautau_phase2(events,gen,pt_min=35.,eta_max=2.5,btag_min=0.,passed=None,verbosity=0):
    matched_OFF_tautau = OFF_matching_tautau_phase2(events,gen,pt_min=pt_min,eta_max=eta_max,passed=passed,verbosity=verbosity)
    matched_OFF_bb = OFF_matching_bb_phase2(events,gen,pt_min=pt_min,eta_max=eta_max,btag_min=btag_min,passed=passed,verbosity=verbosity)
    return matched_OFF_tautau & matched_OFF_bb


# ### SCT

# In[ ]:


#############################################################################################
#
def SCT_matching_tautau_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)

    # Extract L1 objects and filter to keep only those within acceptance
    tau = L1TauP2_objects(events)
    tau["in_acc"] = (tau.pt > pt_min) & (abs(tau.eta) < eta_max)
    tau = tau[tau.in_acc]

    # Match L1 taus
    matched_SCT = object_matching(
        gen,
        tau,
        passed=passed,
        gen_id_filter=15,
        n=2,
        verbosity=verbosity)

    return matched_SCT

#############################################################################################
#
def SCT_matching_bb_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)
    
    # Extract L1 jets and filter to keep only those within acceptance
    jet = L1Jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    # Extract L1 displaced jets and filter to keep only those within acceptance
    djet = L1DJet_objects(events)
    djet["in_acc"] = (djet.pt > pt_min) & (abs(djet.eta) < eta_max)
    djet = djet[djet.in_acc]

    # Concatenate L1 jets and L1 displaced jets
    #@@ jet = ak.concatenate([jet,djet],axis=-1) #@@ NEED TO REMOVE OVERLAPPING JETS

    matched_SCT_jets = object_matching(
        gen,
        djet,
        passed=passed,
        gen_id_filter=5,
        n=2,
        verbosity=verbosity)

    matched_SCT_bjets = object_matching(
        gen,
        djet[djet.btag > 0.55], # Subset of jets that satisfy b-tagging requirement
        passed=passed,
        gen_id_filter=5,
        n=4, # Require 4 b-tagged jets
        verbosity=verbosity)

    #@@ WHAT B-TAGGING REQ SHOULD WE APPLY HERE????

    return matched_SCT_jets# & matched_SCT_bjets

#############################################################################################
#
def SCT_matching_bbtautau_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    matched_SCT_tautau = SCT_matching_tautau_phase2(events,gen,pt_min=pt_min,eta_max=eta_max,passed=passed,verbosity=verbosity)
    matched_SCT_bb = SCT_matching_bb_phase2(events,gen,pt_min=pt_min,eta_max=eta_max,passed=passed,verbosity=verbosity)
    return matched_SCT_tautau & matched_SCT_bb



# ### PLOT

# In[ ]:


#############################################################################################
#
def SCT_plot_eff_vs_tau_pt_bbtautau_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":"Phase 2", "com":14, "nbins":41, "start":0., "stop":205., "xlabel":"GEN tau lepton p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "eff",events,"L1GTnnTau",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=15,
        n=2,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)

#############################################################################################
#
def SCT_plot_purity_vs_tau_pt_bbtautau_phase2(events,gen,pt_min,eta_max,passed=None,verbosity=0):
    kwargs = {"year":"Phase 2", "com":14, "nbins":41, "start":0., "stop":205., "xlabel":"L1 tau p$_{T}$ [GeV]"}
    plot_perf_vs_pt(
        "purity",events,"L1GTnnTau",gen,
        pt_min,eta_max,
        passed=passed,
        gen_id_filter=15,
        n=2,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity,
        **kwargs)


# ### EXECUTE

# In[ ]:


def selections_bbtautau_phase2(**kwargs):

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
    option = kwargs["option"] if "option" in kwargs.keys() else "bbtautau"

    events = load_data_bbtautau_phase2(nevents=nevents,skip=skip,verbosity=verbosity)

    if verbosity>=2:
        print()
        print("FULL DEBUG MODE!!!")
        print(" Verbosity: ", verbosity)
        print(" Num evts:  ", len(events["nGenPart"]))
    
    passed_GEN, gen = GEN_acceptance_bbtautau_phase2(events,pt_min=gen_pt_min,eta_max=gen_eta_max,verbosity=verbosity)
    passed_L1T = L1T_passing_bbtautau_phase2(events,passed=passed_GEN,option=option,verbosity=verbosity)
    matched_L1T = L1T_matching_bbtautau_phase2(events,gen,passed=passed_L1T,option=option,verbosity=verbosity) if use_matched else ak.full_like(passed_L1T,True,dtype=bool)
    passed_HLT = HLT_passing_bbtautau_phase2(events,passed=matched_L1T,option=option,verbosity=verbosity)
    matched_HLT = HLT_matching_bbtautau_phase2(events,gen,passed=passed_HLT,option=option,verbosity=verbosity) if use_matched else ak.full_like(passed_HLT,True,dtype=bool)
    matched_OFF = OFF_matching_bbtautau_phase2(events,gen,pt_min=off_pt_min,eta_max=off_eta_max,btag_min=off_btag_min,passed=matched_HLT,verbosity=verbosity)
    matched_SCT = SCT_matching_bbtautau_phase2(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    # Plotting (only plot once)
    if use_matched == False and option == "tautau":
        SCT_plot_eff_vs_tau_pt_bbtautau_phase2(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)
        SCT_plot_purity_vs_tau_pt_bbtautau_phase2(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    print_summary(
        events,
        passed_GEN=passed_GEN,
        passed_L1T=passed_L1T,
        matched_L1T=matched_L1T,
        passed_HLT=passed_HLT,
        matched_HLT=matched_HLT,
        matched_OFF=matched_OFF,
        matched_SCT=matched_SCT,
        use_matched=use_matched, # Use passed or matched for L1T and HLT
        )

# option="tautau"
print()
settings = settings_.copy()
settings.update({"option":"tautau"})
print(settings)
selections_bbtautau_phase2(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbtautau_phase2(**settings) 

# option="bb"
print()
settings = settings_.copy()
settings.update({"option":"bb"})
print(settings)
selections_bbtautau_phase2(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbtautau_phase2(**settings) 

# option="bbtautau"
print()
settings = settings_.copy()
settings.update({"option":"bbtautau"})
print(settings)
selections_bbtautau_phase2(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbtautau_phase2(**settings) 



# ## bbbb (Run 3, muon)

# ### LOAD

# In[ ]:


#############################################################################################
#
def load_data_bbbb_muon(nevents=None,skip=0,verbosity=0):

    # Open ROOT file with uproot
    example_file = "../data/Run3/HHTo4B/data_0.root"
    file = uproot.open(example_file)
    tree = file["Events"]

    if verbosity>=3:
        keys = tree.keys()
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

    branches = [
        "GenPart_pt", "GenPart_eta", "GenPart_phi", # GEN-level kinematics
        "nGenPart","GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_statusFlags", # GEN-level information
        "L1_HTT280er", # L1 seeds
        "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55", # HLT paths
        "nTrigObj", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id", "TrigObj_filterBits", # Trigger objects
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagPNetB", # Offline jets
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_dxy", "Muon_dxyErr", # Offline muons
        "nL1Jet", "L1Jet_pt", "L1Jet_eta", "L1Jet_phi", # L1 jets 
        "nL1Mu", "L1Mu_pt", "L1Mu_eta", "L1Mu_phi", "L1Mu_etaAtVtx", "L1Mu_phiAtVtx", "L1Mu_hwQual",  # L1 muons
    ]

    # Load data into awkward arrays
    events = tree.arrays(branches, library="ak")
    events = events[skip:nevents+skip] if nevents is not None else events[skip:]
    return events


# ### ACC

# In[ ]:


#############################################################################################
#
def GEN_acceptance_bbbb_muon(events,pt_min,eta_max,verbosity=0):
    return GEN_acceptance_bbbb(events,pt_min=pt_min,eta_max=eta_max,verbosity=verbosity)


# ### L1T

# In[ ]:


#############################################################################################
#
def L1_SingleMu11_SQ14_BMTF(events):
    L1Mu = L1Mu_objects(events)
    L1Mu = L1Mu[(L1Mu.pt >= 11.) & (abs(L1Mu.eta) <= 0.8) & (L1Mu.qual >= 12)] #@@ HWQual >= 14 doesn't ever occur??
    return L1Mu

#############################################################################################
#
def L1T_passing_muon(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    #seed = "L1_SingleMu11_SQ14_BMTF" #@@ L1_SingleMu7_SQ14_BMTF, L1_SingleMu8_SQ14_BMTF, L1_SingleMu9_SQ14_BMTF, L1_SingleMu10_SQ14_BMTF
    #return L1T_passing(events,seed,verbosity=verbosity)
    emulate_L1T = L1_SingleMu11_SQ14_BMTF(events)
    passed_L1T = ak.num(emulate_L1T) > 0 
    return passed_L1T

#############################################################################################
#
def L1T_passing_bbbb_muon(events,passed=None,option=None,verbosity=0):
    if option == "muon":
        return L1T_passing_muon(events,passed=passed,verbosity=verbosity)
    elif option == "bbbb":
        return L1T_passing_bbbb(events,passed=passed,verbosity=verbosity)
    elif option == "bbbb_muon" or option is None:
        passed_L1T_muon = L1T_passing_muon(events,passed=passed,verbosity=verbosity)
        passed_L1T_bbbb = L1T_passing_bbbb(events,passed=passed,verbosity=verbosity)
        return passed_L1T_muon | passed_L1T_bbbb
        #return passed_L1T_muon & passed_L1T_bbbb
        #return passed_L1T_muon & ~passed_L1T_bbbb
        #return passed_L1T_bbbb & ~passed_L1T_muon 
    else:
        raise ValueError(f"Invalid option: {option}")

#############################################################################################
#
def L1T_matching_muon(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    #L1Mu = L1Mu_objects(events)
    #L1Mu = L1Mu[(L1Mu.pt > 10.) & (abs(L1Mu.eta) < 0.8) & (L1Mu.qual >= 12)] # HWQual > 14?
    L1Mu = L1_SingleMu11_SQ14_BMTF(events)
    matched_L1T = object_matching(
        gen,
        L1Mu,
        passed=passed,
        gen_id_filter=5,
        n=1, # Match muon to just one b-quark
        dr_max=0.3, #@@ Loose requirement
        dpt_min=None, dpt_max=None,
        verbosity=verbosity)
    return matched_L1T

#############################################################################################
#
def L1T_matching_bbbb_muon(events,gen,passed=None,option=None,verbosity=0):
    if option == "muon":
        return L1T_matching_muon(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bbbb":
        return L1T_matching_bbbb(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bbbb_muon" or option is None:
        matched_L1T_muon = L1T_matching_muon(events,gen,passed=passed,verbosity=verbosity)
        matched_L1T_bbbb = L1T_matching_bbbb(events,gen,passed=passed,verbosity=verbosity)
        return matched_L1T_muon | matched_L1T_bbbb
        #return matched_L1T_muon & matched_L1T_bbbb
        #return matched_L1T_muon & ~matched_L1T_bbbb
        #return matched_L1T_bbbb & ~matched_L1T_muon 
    else:
        raise ValueError(f"Invalid option: {option}")


# ### HLT

# In[ ]:


#############################################################################################
#
def HLT_Mu10_Barrel_L1HP11_IP6(events):
    muon = muon_objects(events)
    muon = muon[(muon.pt >= 10.) & (abs(muon.eta) <= 2.5) & (muon.dxysig >= 0.)]
    return muon

#############################################################################################
#
def HLT_passing_muon(events,passed=None,verbosity=0):
    events = filter_events(events,passed)
    #path = "HLT_Mu10_Barrel_L1HP11_IP6" #@@ HLT_Mu6_Barrel_L1HP7_IP6, HLT_Mu7_Barrel_L1HP8_IP6, HLT_Mu8_Barrel_L1HP9_IP6, HLT_Mu9_Barrel_L1HP10_IP6
    #return HLT_passing(events,path,verbosity=verbosity)
    emulate_HLT = HLT_Mu10_Barrel_L1HP11_IP6(events)
    passed_HLT = ak.num(emulate_HLT) > 0 
    return passed_HLT

#############################################################################################
#
def HLT_passing_bbbb_muon(events,passed=None,option=None,verbosity=0):
    if option == "muon":
        return HLT_passing_muon(events,passed=passed,verbosity=verbosity)
    elif option == "bbbb":
        return HLT_passing_bbbb(events,passed=passed,verbosity=verbosity)
    elif option == "bbbb_muon" or option is None:
        passed_HLT_muon = HLT_passing_muon(events,passed=passed,verbosity=verbosity)
        passed_HLT_bbbb = HLT_passing_bbbb(events,passed=passed,verbosity=verbosity)
        return passed_HLT_muon | passed_HLT_bbbb
        #return passed_HLT_muon & passed_HLT_bbbb
        #return passed_HLT_muon & ~passed_HLT_bbbb
        #return passed_HLT_bbbb & ~passed_HLT_muon 
    else:
        raise ValueError(f"Invalid option: {option}")

#############################################################################################
#
def HLT_matching_muon(events,gen,passed=None,verbosity=0):
    events = filter_events(events,passed)
    #muon = muon_objects(events)
    #muon = muon[(muon.pt > 9.) & (abs(muon.eta) < 0.8) & (muon.dxysig > 0.)]
    muon = HLT_Mu10_Barrel_L1HP11_IP6(events)

    # All 4 jets
    matched_HLT = object_matching(
        gen,
        muon,
        passed=passed,
        gen_id_filter=5, # b quarks only
        n=1,
        dr_max=0.3,
        dpt_min=None,dpt_max=None,
        verbosity=verbosity)

    return matched_HLT

#############################################################################################
#
def HLT_matching_bbbb_muon(events,gen,passed=None,option=None,verbosity=0):
    if option == "muon":
        return HLT_matching_muon(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bbbb":
        return HLT_matching_bbbb(events,gen,passed=passed,verbosity=verbosity)
    elif option == "bbbb_muon" or option is None:
        matched_HLT_muon = HLT_matching_muon(events,gen,passed=passed,verbosity=verbosity)
        matched_HLT_bbbb = HLT_matching_bbbb(events,gen,passed=passed,verbosity=verbosity)
        return matched_HLT_muon | matched_HLT_bbbb
        #return matched_HLT_muon & matched_HLT_bbbb
        #return matched_HLT_muon & ~matched_HLT_bbbb
        #return matched_HLT_bbbb & ~matched_HLT_muon 
    else:
        raise ValueError(f"Invalid option: {option}")


# ### OFF

# In[ ]:


#############################################################################################
#
def OFF_matching_muon(events,gen,pt_min,eta_max,btag_min=0.,passed=None,verbosity=0):
    events = filter_events(events,passed)

    # Extract jet info and filter to keep only those within acceptance
    jet = jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    matched_OFF_jets = object_matching(
        gen,
        jet,
        passed=passed,
        gen_id_filter=5,
        n=4,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )
    
    matched_OFF_bjets = object_matching(
        gen,
        jet[jet.btag >= btag_min], # Subset of jets that satisfy b-tagging requirement
        passed=passed,
        gen_id_filter=5,
        n=4,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=verbosity
        )
    
    muon = muon_objects(events)
    muon = muon[(muon.pt >= 10.) & (abs(muon.eta) <= 2.5) & (muon.dxysig >= 0.)]

    matched_OFF_muon = object_matching(
        gen,
        muon, 
        passed=passed,
        gen_id_filter=5,
        n=1, # Require muon to match one b quark
        dr_max=0.3,
        dpt_min=None,dpt_max=None,
        verbosity=verbosity
        )

    matched_OFF = matched_OFF_jets & matched_OFF_bjets & matched_OFF_muon

    return matched_OFF

#############################################################################################
#
def OFF_matching_bbbb_muon(events,gen,passed=None,option=None,verbosity=0):
    if option == "muon":
        pt_min, eta_max, btag_min = 20., 2.5, 0.
        return OFF_matching_muon(events,gen,pt_min,eta_max,btag_min=btag_min,passed=passed,verbosity=verbosity)
    elif option == "bbbb":
        pt_min, eta_max, btag_min = 35., 2.5, 0.
        return OFF_matching_bbbb(events,gen,pt_min,eta_max,btag_min=btag_min,passed=passed,verbosity=verbosity)
    elif option == "bbbb_muon" or option is None:
        pt_min, eta_max, btag_min = 20., 2.5, 0.
        matched_OFF_muon = OFF_matching_muon(events,gen,pt_min,eta_max,btag_min=btag_min,passed=passed,verbosity=verbosity)
        pt_min, eta_max, btag_min = 35., 2.5, 0.
        matched_OFF_bbbb = OFF_matching_bbbb(events,gen,pt_min,eta_max,btag_min=btag_min,passed=passed,verbosity=verbosity)
        return matched_OFF_muon | matched_OFF_bbbb
    else:
        raise ValueError(f"Invalid option: {option}")


# ### SCT

# In[ ]:


#############################################################################################
#
def SCT_matching_bbbb_muon(events,gen,pt_min,eta_max,passed=None,verbosity=0):

    events = filter_events(events,passed)
    
    # Extract L1 objects and filter to keep only those within acceptance
    jet = L1Jet_objects(events)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    # tau = L1Tau_objects(events)
    # tau["in_acc"] = (tau.pt > pt_min) & (abs(tau.eta) < eta_max)
    # tau = tau[tau.in_acc]

    # Concatenate L1 jets and L1 taus
    # jet = ak.concatenate([jet,tau],axis=-1) #@@ NEED TO REMOVE OVERLAPPING JETS AND TAUS ??!!

    # Match either L1 jets or L1 taus
    matched_SCT = object_matching(
        gen,
        jet,
        passed=passed,
        gen_id_filter=5,
        n=4, 
        verbosity=verbosity)

    return matched_SCT


# ### EXECUTE

# In[ ]:


def selections_bbbb_muon(**kwargs):
    
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
    option = kwargs["option"] if "option" in kwargs.keys() else "unknown"

    events = load_data_bbbb_muon(nevents=nevents,skip=skip,verbosity=verbosity)
    
    if verbosity>=2:
        print()
        print("FULL DEBUG MODE!!!")
        print(" Verbosity: ", verbosity)
        print(" Num evts:  ", len(events["nGenPart"]))
    
    passed_GEN, gen = GEN_acceptance_bbbb_muon(events,pt_min=gen_pt_min,eta_max=gen_eta_max,verbosity=verbosity)
    passed_L1T = L1T_passing_bbbb_muon(events,passed=passed_GEN,option=option,verbosity=verbosity)
    matched_L1T = L1T_matching_bbbb_muon(events,gen,passed=passed_L1T,option=option,verbosity=verbosity) if use_matched else ak.full_like(passed_L1T,True,dtype=bool)
    passed_HLT = HLT_passing_bbbb_muon(events,passed=matched_L1T,option=option,verbosity=verbosity)
    matched_HLT = HLT_matching_bbbb_muon(events,gen,passed=passed_HLT,option=option,verbosity=verbosity) if use_matched else ak.full_like(passed_HLT,True,dtype=bool)
    matched_OFF = OFF_matching_bbbb_muon(events,gen,passed=matched_HLT,option=option,verbosity=verbosity) #@@ pt_min, eta_max, and btag_min and set in the method itself for the different options!
    matched_SCT = SCT_matching_bbbb_muon(events,gen,pt_min=sct_pt_min,eta_max=sct_eta_max,passed=passed_GEN,verbosity=verbosity)

    print_summary(
        events,
        passed_GEN=passed_GEN,
        passed_L1T=passed_L1T,
        matched_L1T=matched_L1T,
        passed_HLT=passed_HLT,
        matched_HLT=matched_HLT,
        matched_OFF=matched_OFF,
        matched_SCT=matched_SCT,
        use_matched=use_matched, # Use passed or matched for L1T and HLT
        )


# NOTA BENE: "off_pt_min", "off_eta_max", and "off_btag_min" settings are OVERRIDDEN IN THE METHODS, differently for each option

# option="muon"
print()
settings = settings_.copy()
settings.update({"off_pt_min":None, "off_eta_max":None, "off_btag_min":None, })
settings.update({"option":"muon"})
print(settings)
selections_bbbb_muon(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbbb_muon(**settings) 

# option="bbbb"
print()
settings = settings_.copy()
settings.update({"off_pt_min":None, "off_eta_max":None, "off_btag_min":None, })
settings.update({"option":"bbbb"})
print(settings)
selections_bbbb_muon(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbbb_muon(**settings) 

# option="bbbb_muon"
print()
settings = settings_.copy()
settings.update({"off_pt_min":None, "off_eta_max":None, "off_btag_min":None, })
settings.update({"option":"bbbb_muon"})
print(settings)
selections_bbbb_muon(**settings) 
settings.update({"use_matched":True})
print(settings)
selections_bbbb_muon(**settings) 

