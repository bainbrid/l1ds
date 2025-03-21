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
from plothist import plot_2d_hist, plot_2d_hist_with_projections
import boost_histogram as bh
import mplhep as hep
hep.style.use("CMS")


# ## Common

# ### Print

# In[ ]:


#############################################################################################
#
def print_summary(
    events,
    passed_GEN=None,
    passed_L1T=None,
    matched_L1T=None,
    passed_HLT=None,
    matched_HLT=None,
    matched_OFF=None,
    matched_SCT=None,
    use_matched=False,
    ):

    passed_tot = ak.ones_like(events["nGenPart"])
    if passed_GEN is None: passed_GEN = ak.ones_like(passed_tot)
    if passed_L1T is None: passed_L1T = ak.ones_like(passed_tot)
    if matched_L1T is None: matched_L1T = ak.ones_like(passed_tot)
    if passed_HLT is None: passed_HLT = ak.ones_like(passed_tot)
    if matched_HLT is None: matched_HLT = ak.ones_like(passed_tot)
    if matched_OFF is None: matched_OFF = ak.ones_like(passed_tot)
    if matched_SCT is None: matched_SCT = np.ones_like(passed_tot)

    tot = passed_tot
    acc = tot&passed_GEN
    l1t = acc&passed_L1T&matched_L1T if use_matched else acc&passed_L1T
    hlt = l1t&passed_HLT&matched_HLT if use_matched else l1t&passed_HLT
    off = hlt&matched_OFF
    sct = passed_GEN&matched_SCT

    tot = np.sum(tot)
    acc = np.sum(acc)
    l1t = np.sum(l1t)
    hlt = np.sum(hlt)
    off = np.sum(off)
    sct = np.sum(sct)

    print()
    print("==================================")
    print(f"              Events    Eff   Gain")
    print("==================================")
    print("STANDARD (W/ matching @ L1 & HLT)" if use_matched else "STANDARD (No matching @ L1 & HLT)")
    print(f"Inclusive     {tot:6.0f} ")
    print(f"Acceptance    {acc:6.0f} {acc/tot:6.2f} {sct/acc:6.2f}")
    print(f"L1T           {l1t:6.0f} {l1t/tot:6.2f} {sct/l1t:6.2f}")
    print(f"HLT           {hlt:6.0f} {hlt/tot:6.2f} {sct/hlt:6.2f}")
    print(f"Offline       {off:6.0f} {off/tot:6.2f} {sct/off:6.2f}")
    print("----------------------------------")
    print("SCOUTING")
    print(f"Inclusive     {tot:6.0f}")
    print(f"Acceptance    {acc:6.0f} {acc/tot:6.2f}")
    print(f"Scouting      {sct:6.0f} {sct/tot:6.2f}")
    print("==================================")

#############################################################################################
#
def print_matching_base(x):
    if "match" not in x.fields or x.match is None: x["match"] = False #@@ Needed?
    if "dr" not in x.fields or x.dr is None: x["dr"] = 9.99 #@@ Needed?
    if "idx" not in x.fields or x.idx is None: x["idx"] = -1 #@@ Needed?
    if "id" not in x.fields or x.id is None: x["id"] = -1 #@@ Needed?
    pad = " "*6
    basic = f"pt: {x.pt:5.1f}, eta: {x.eta:5.2f}, phi: {x.phi:5.2f}, id: {x.id:2.0f}"
    match = f", dr: {x.dr:4.2f}, idx: {x.idx:2.0f}, match: {x.match:1.0f}"
    return pad+basic+match

#############################################################################################
#
def print_matching(gen,obj):
    if obj is None: #@@ dirty hack to broadcast to event-level if obj=None
        obj = ak.full_like(ak.num(gen,1),False,dtype=bool)
        obj = ak.mask(obj,obj,valid_when=True)
    for idx,(gen_,obj_) in enumerate(zip(gen,obj)):
        if gen_ is None and obj_ is None: continue
        print(f"  Event {idx}:")
        print("    GEN:")
        if gen_ is None: gen_ = []
        for x in gen_:
            if x is None: continue
            base = print_matching_base(x)
            # if x.id is None: x["id"] = -1 #@@ Needed?
            # print(f"      pt: {x.pt:5.1f}, eta: {x.eta:5.2f}, phi: {x.phi:5.2f}, id: {x.id:2.0f}, status: {x.status:.0f}, last_copy: {x.last_copy:.0f}")
            status = f", status: {x.status:.0f}, last_copy: {x.last_copy:.0f}"
            print(base+status)
        print("    OBJ:")
        if obj_ is None or obj_ is False: obj_ = []
        for x in obj_:
            if x is None: continue
            base = print_matching_base(x)
            # if x.match is None: x["match"] = False #@@ Needed?
            # if x.dr is None: x["dr"] = 9.99 #@@ Needed?
            # if x.idx is None: x["idx"] = -1 #@@ Needed?
            # if x.id is None: x["id"] = -1 #@@ Needed?
            # pad = " "*6
            # basic = f"pt: {x.pt:5.1f}, eta: {x.eta:5.2f}, phi: {x.phi:5.2f}, dr: {x.dr:4.2f}, id: {x.id:2.0f}"
            # match = f", idx: {x.idx:2.0f}, match: {x.match:1.0f}"
            # base=pad+basic+match
            trg = ""
            if "bits_ok" in x.fields: trg += f", bits_ok: {x.bits_ok if x.bits_ok is not None else -1:2.0f}"
            #if "bits"    in x.fields: trg += f", bits: {x.bits if x.bits is not None else 0:>032b}"
            if "bits"    in x.fields: 
                bits_list, = np.where([x.bits >> i & 1 for i in range(32)]) # comma dereferences the tupl(np.array,)
                bits_list = ", ".join([f'{x:.0f}' for x in bits_list])
                trg += f", bits: {bits_list}"
            print(base+trg)


# ### Filters

# In[ ]:


#############################################################################################
#
def filter_events(events,passed):
    if passed is not None:
        events = ak.mask(events,passed)
    return events


# ### Objects

# In[ ]:


#############################################################################################
#
def objects(events,**kwargs):
    obj = {}
    for key,val in kwargs.items():
        if type(val) is str:
            obj[key] = events[val]
        else:
            obj[key] = val
    obj = ak.zip(obj)
    return obj

#############################################################################################
#
def base_objects(events,label,id=None):
    kwargs = {
        "pt":f"{label}_pt", "eta":f"{label}_eta", "phi":f"{label}_phi", # Kine
    }
    if id is not None:  kwargs["id"] = ak.full_like(events[f"{label}_pt"],id)
    return objects(events,**kwargs)

#############################################################################################
#
def gen_objects(events):
    gen = base_objects(events,label="GenPart")
    gen["id"] = abs(events["GenPart_pdgId"])
    gen["mother_idx"] = events["GenPart_genPartIdxMother"]
    gen["mother_id"] = abs(gen.id[gen.mother_idx])
    gen["status"] = events["GenPart_statusFlags"]

    # https://github.com/cms-sw/cmssw/blob/master/DataFormats/HepMCCandidate/interface/GenStatusFlags.h

    # Is particle prompt (not from hadron, muon, or tau decay)
    gen["is_prompt"] = (gen.status & (1 << 0) != 0)
    # This particle is part of the hard process
    gen["is_hard_process"] = (gen.status & (1 << 7) != 0)
    # This particle is the direct descendant of a hard process particle of the same pdg id
    gen["from_hard_process"] = (gen.status & (1 << 8) != 0)
    # This particle is the first copy of the particle in the chain with the same pdg id
    gen["first_copy"] = (gen.status & (1 << 12) != 0)
    # This particle is the last copy of the particle in the chain with the same pdg id
    gen["last_copy"] = (gen.status & (1 << 13) != 0)

    return gen

#############################################################################################
#
def jet_objects(events):
    jet = base_objects(events,label="Jet",id=1)
    jet["btag"] = events["Jet_btagPNetB"]
    return jet

#############################################################################################
#
def tau_objects(events):
    return base_objects(events,label="Tau",id=15)

#############################################################################################
#
def muon_objects(events):
    muon = base_objects(events,label="Muon",id=13)
    muon["dxy"] = events["Muon_dxy"]
    muon["dxyerr"] = events["Muon_dxyErr"]
    muon["dxysig"] = muon.dxy/muon.dxyerr
    return muon

#############################################################################################
#
def L1Jet_objects(events):
    return base_objects(events,label="L1Jet",id=1)

#############################################################################################
#
def L1DJet_objects(events):
    jet = base_objects(events,label="L1DisplacedJet",id=1)
    jet["btag"] = events["L1DisplacedJet_btagScore"]
    return jet

#############################################################################################
#
def L1Tau_objects(events):
    return base_objects(events,label="L1Tau",id=15)

#############################################################################################
#
def L1TauP2_objects(events):
    return base_objects(events,label="L1GTnnTau",id=15)

#############################################################################################
#
def HLT_objects(events):
    hlt = base_objects(events,label="TrigObj")
    hlt["id"] = events["TrigObj_id"]
    hlt["bits"] = events["TrigObj_filterBits"]
    return hlt

#############################################################################################
#
def L1Mu_objects(events):
    muon = base_objects(events,label="L1Mu",id=13)
    muon["qual"] = events["L1Mu_hwQual"]
    muon["etaAtVtx"] = events["L1Mu_etaAtVtx"]
    muon["phiAtVtx"] = events["L1Mu_phiAtVtx"]
    return muon


# ### GEN

# In[ ]:


def build_decay_graph(gen_objects,idx=0):
    """Builds a directed graph representing the particle decay hierarchy."""
    pdgId = gen_objects["id"][idx]  # Extract the first (and only) sublist
    motherIdx = gen_objects["mother_idx"][idx]

    G = nx.DiGraph()

    for i, pid in enumerate(pdgId):
        particle_name = f"PDG: {pid} (idx: {i}, m: {motherIdx[i]})"
        G.add_node(i, label=particle_name)

        mother = motherIdx[i]
        if mother >= 0 and mother < len(pdgId):  # Ensure valid mother index
            G.add_edge(mother, i)
    
    return G

def print_hierarchy(G, node, depth=0):
    """Recursively prints the decay hierarchy in a tree format."""
    if node not in G.nodes:
        return
    label = G.nodes[node]["label"]
    indent = "  " * depth
    print(f"{indent}{label}")
    for child in G.successors(node):
        print_hierarchy(G, child, depth + 1)

def export_decay_hierarchy(G):
    """Finds the root nodes and prints the decay hierarchy."""
    roots = [node for node in G.nodes if G.in_degree(node) == 0]
    print("--- Decay Hierarchy ---")
    for root in roots:
        print_hierarchy(G, root)

def draw_decay_graph(G):
    """Draws a graphical representation of the decay hierarchy."""
    pos = nx.spring_layout(G, seed=42)
    labels = nx.get_node_attributes(G, "label")

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color="lightblue", edge_color="gray", node_size=1000, font_size=8)
    plt.title("Particle Decay Graph")
    plt.show()

def process_gen_objects(gen_objects,idx=0):
    """Processes event data to generate and visualize the decay graph."""
    G = build_decay_graph(gen_objects,idx=idx)
    export_decay_hierarchy(G)
    draw_decay_graph(G)

def process_event_data(event_data,idx=0):
    """Processes event data to generate and visualize the decay graph."""
    gen = gen_objects(event_data)
    process_gen_objects(gen_objects,idx=idx)


# ### Matching

# In[ ]:


#############################################################################################
#
def delta_phi(x):
    """Normalize phi angle to [-π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi

#############################################################################################
#
def geometric_matching_base(gen,obj,direction="reco_to_gen",dr_max=0.3,dpt_min=0.2,dpt_max=2.0,verbosity=0):

    # Pairs: all possible pairs of objects of type x and y (x = outer loop, y = inner loop)
    # e.g. [(x1,y1), (x1,y2), (x2,y1), (x2,y2), ...] 
    # "gen_to_reco" matching is where x = gen and y = obj
    # "reco_to_gen" matching is where x = obj and y = gen

    pairs = None
    if    direction == "reco_to_gen": pairs = ak.cartesian([obj, gen], nested=True) 
    elif  direction == "gen_to_reco": pairs = ak.cartesian([gen, obj], nested=True) 
    else: raise ValueError(f'[geometric_matching_base] Unknown direction "{direction}"')
    # print("pairs",pairs)

    # Unzip the pairs to get the reco and GEN objects separately and calc dPt, dEta, dPhi, dR
    pairs_y, pairs_x = ak.unzip(pairs)
    dpt  = pairs_y.pt / pairs_x.pt if direction == "reco_to_gen" else pairs_x.pt / pairs_y.pt
    deta = pairs_y.eta - pairs_x.eta
    dphi = delta_phi(pairs_y.phi - pairs_x.phi)
    dr   = np.sqrt(deta**2 + dphi**2)

    #@@ Consider both geometric and pT matching ???
    #dr = np.sqrt(dr**2 + (dpt-1.)**2)

    # Defined dr_none, used as a dummy value in case of None, as a large value (dr_none > dr_max)
    dr_none = 9.99 # Better than 10. for formatting purposes in print_matching()

    # Apply pT matching criteria if required
    if dpt_min is not None: dr = ak.mask(dr,dpt > dpt_min)
    if dpt_max is not None: dr = ak.mask(dr,dpt < dpt_max)
    dr = ak.fill_none(dr,dr_none) # Set masked entries to dummy value

    # For each reco object, sort the GEN objects by dR
    dr_sort = ak.sort(dr,axis=-1)
    dr_min = ak.firsts(dr_sort,axis=-1)

    # Get the index of the reco object with the lowest dR
    dr_min_sort = ak.argsort(dr,axis=-1)
    dr_min_idx = ak.firsts(dr_min_sort,axis=-1)

    # Identify matches if smallest dR is less than the maximum allowed value
    matched = (dr_min is not None) & (dr_min < dr_max)

    # Mask out unmatched values for both dR and the index
    dr_min = ak.mask(dr_min,matched)
    dr_min_idx = ak.mask(dr_min_idx,matched)

    # Replace None values with False
    matched = ak.fill_none(matched,False)
    dr_min = ak.fill_none(dr_min,dr_none)
    dr_min_idx = ak.fill_none(dr_min_idx,-1)

    # Store the results in the reco or GEN object
    if  direction == "reco_to_gen":
        obj["match"] = matched
        obj["dr"] = dr_min
        obj["idx"] = dr_min_idx
    else:
        gen["match"] = matched
        gen["dr"] = dr_min
        gen["idx"] = dr_min_idx

    return gen,obj

#############################################################################################
#
def geometric_matching(gen,obj,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,verbosity=0):
    gen,obj = geometric_matching_base(gen,obj,direction="reco_to_gen",dr_max=dr_max,dpt_min=dpt_min,dpt_max=dpt_max,verbosity=verbosity)
    gen,obj = geometric_matching_base(gen,obj,direction="gen_to_reco",dr_max=dr_max,dpt_min=dpt_min,dpt_max=dpt_max,verbosity=verbosity)
    return gen,obj

#############################################################################################
#
def object_matching_base(
        gen,
        obj,
        passed=None,
        gen_id_filter=None,
        n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        label="[object_matching_base]",
        verbosity=0):

    # Filter obects depending if passed
    if passed is not None:
        gen = ak.mask(gen,passed)
        obj = ak.mask(obj,passed)

    # Filter the GEN objects by PDG ID
    if gen_id_filter is not None:
        gen_id_mask = (gen.id == gen_id_filter)
        #gen = gen[gen_id_mask]
        gen = ak.mask(gen,gen_id_mask)

    # Match GEN and reco objects
    gen,obj = geometric_matching(gen,obj,dr_max=dr_max,dpt_min=dpt_min,dpt_max=dpt_max,verbosity=verbosity)
    
    # Reset matching index to -1 for unmatched objects
    #obj["idx"] = ak.fill_none(ak.mask(obj.idx,obj.match),-1) # No longer needed?

    # Count the number of matched objects
    num_matched = ak.count_nonzero(obj.match, axis=-1)

    # Identify events which are fully matched
    all_matched = (num_matched >= n)

    # Indices of fully matched events, comma dereferences the tupl(ak.array)
    idx_matched, = ak.where(all_matched)

    if verbosity>=1:
        print()
        print(label)
        print(f"The following events are fully matched:")
        print(", ".join([f"{x}" for x in idx_matched]))
    if verbosity>=2:
        print(f"Matching between GEN and RECO objects:")
        tmp = obj # Print all objects
        #tmp = obj[obj.match] # Print only matched objects
        print_matching(gen,tmp)

    return all_matched,gen,obj

#############################################################################################
#
def object_matching(
        gen,
        obj,
        passed=None,
        gen_id_filter=None,
        n=4,
        dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        label=None,
        verbosity=0):
    all_matched,_,_ = object_matching_base(
        gen,
        obj,
        passed=passed,
        gen_id_filter=gen_id_filter,
        n=n,
        dr_max=dr_max,dpt_min=dpt_min,dpt_max=dpt_max,
        label=label,
        verbosity=verbosity)
    return all_matched

#############################################################################################
#
def hlt_matching_base(
    gen, 
    hlt,
    passed=None,
    gen_id_filter=None,
    hlt_id_filter=None,
    hlt_bits_filter=None,
    n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
    label="[hlt_matching_base]",
    verbosity=0):

    # Filter objects depending if passed
    if passed is not None:
        gen = ak.mask(gen,passed)
        hlt = ak.mask(hlt,passed)

    # Filter the GEN objects by PDG ID
    if gen_id_filter is not None:
        gen_id_mask = (gen.id == gen_id_filter)
        gen = gen[gen_id_mask]

    # Filter the trigger objects by ID (1=Jet, 2=MET, 3=HT, 4=MHT, 11=ele, 13=muon, 15=tau, 22=photon)
    # Defined here: search "id = cms.int32(?)"
    # https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py
    if hlt_id_filter is not None:
        hlt_id_mask = (hlt.id == hlt_id_filter)
        hlt = hlt[hlt_id_mask]

    if verbosity>=1: 
        print()
        if hlt_bits_filter is not None:
            print(f"Trigger bits filter: {', '.join([f'{x:.0f}' for x in hlt_bits_filter])}")
        else:
            print(f"Trigger bits filter: no filter applied!")

    # Check if all required trigger bits are set
    if "bits" not in hlt.fields: hlt["bits"] = ak.full_like(hlt.pt,0,dtype=int)
    hlt["bits_ok"] = ak.full_like(hlt.pt,True,dtype=bool)
    if hlt_bits_filter is not None:
        hlt_bits_values = [2**i for i in hlt_bits_filter]
        hlt_bits_filter = np.sum(hlt_bits_values)
        hlt_bits_filter = ak.full_like(hlt.bits,hlt_bits_filter)
        hlt["bits_ok"] = (hlt.bits & hlt_bits_filter) == hlt_bits_filter
        
    # Match GEN and trigger objects
    gen,obj = geometric_matching(gen,hlt,dr_max=dr_max,dpt_min=dpt_min,dpt_max=dpt_max,verbosity=verbosity)
    
    # Require objects to be geometrically matched and have the correct trigger bits set
    obj["match"] = obj.match & obj.bits_ok
     
    # Reset gen index to -1 for unmatched objects (given the above mask is used!)
    obj["idx"] = ak.fill_none(ak.mask(obj.idx,obj.match),-1)

    # Count the number of matched objects
    num_matched = ak.count_nonzero(obj.match, axis=-1)

    # Identify events which are fully matched
    all_matched = (num_matched >= n)

    # Indices of fully matched events, comma dereferences the tupl(ak.array)
    idx_matched, = ak.where(all_matched) 
    
    if verbosity>=1:
        print()
        print(f"The following events are fully matched:")
        print(", ".join([f"{x}" for x in idx_matched]))
    if verbosity>=2:
        print()
        print(f"Matching between GEN and TRIGGER objects:")
        #tmp = obj              # Print all objects
        tmp = obj[obj.bits_ok] # Print objects only if bits_ok
        #tmp = obj[obj.match]   # Print objects only if matched
        print_matching(gen,tmp)

    return all_matched,gen,obj

#############################################################################################
#
def hlt_matching(
    gen, 
    trg,
    passed=None,
    gen_id_filter=None,
    hlt_id_filter=None,
    hlt_bits_filter=None,
    n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
    label="[hlt_matching]",
    verbosity=0):
    all_matched,_,_ = hlt_matching_base(
        gen,trg,
        passed=passed,gen_id_filter=gen_id_filter,hlt_id_filter=hlt_id_filter,hlt_bits_filter=hlt_bits_filter,
        n=n,dr_max=dr_max,dpt_min=dpt_min,dpt_max=dpt_max,label="[hlt_matching]",verbosity=verbosity)
    return all_matched


# ### Trigger

# In[ ]:


#############################################################################################
#
def L1T_passing(events,seed,verbosity=0):

    # Identify if L1 trigger fires
    L1_seed = events[seed]
    passed_L1T = (L1_seed == 1)
    passed_idx, = ak.where(passed_L1T) # comma dereferences the tupl(ak.array)

    if verbosity>=1:
        print()
        print("[L1T_passing_bbbb]")
        print(f"{seed} fired for the following events:")
        print(", ".join([f"{x}" for x in passed_idx]))

    return passed_L1T

#############################################################################################
#
def HLT_passing(events,path,verbosity=0):

    # Identify if HLT path fires
    HLT_trigger = events[path]
    passed_HLT = (HLT_trigger == 1)
    fired, = ak.where(passed_HLT) # comma dereferences the tupl(ak.array)

    if verbosity>=1:
        print()
        print(f"{path} fired for the following events:")
        print(", ".join([f"{x}" for x in fired]))

    return passed_HLT


# ### Plotting

# In[ ]:


#############################################################################################
# https://github.com/scikit-hep/hist/blob/ce6e5996cb6de4d1e331dff4a911aaf5d4c8e114/src/hist/intervals.py#L78C1-L110
# http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
def clopper_pearson_interval(numer,denom,coverage=None,relative=False):
    if coverage is None:
        coverage = stats.norm.cdf(1) - stats.norm.cdf(-1)
    efficiency = numer / denom
    interval_min = stats.beta.ppf((1 - coverage) / 2, numer, denom - numer + 1)
    interval_max = stats.beta.ppf((1 + coverage) / 2, numer + 1, denom - numer)
    if relative:
        interval_min = efficiency - interval_min
        interval_max = interval_max - efficiency
    interval = np.stack((interval_min, interval_max))
    if relative:
        interval[0, numer == 0.0] = 0.0
        interval[1, numer == denom] = 0.0
    else:
        interval[0, numer == 0.0] = 0.0
        interval[1, numer == denom] = 1.0
    return interval

#############################################################################################
#
def plot_sig_eff_vs_jet_rank(
        events,label,
        gen,        
        pt_min,eta_max,
        passed=None,
        verbosity=0,
        year=2023,com=13.6):

    events = filter_events(events,passed)
    jet = base_objects(events,label=label)
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    # Match L1 jets
    _,_,obj = object_matching_base(
        gen,
        jet,
        passed=passed,
        gen_id_filter=5,
        n=4,
        label="[plot_sig_eff_vs_jet_rank]", 
        verbosity=verbosity)

    assert ak.all(obj.pt == ak.sort(obj.pt, ascending=False, axis=-1)), "ak.array not sorted by pT!"

    counts = []
    njets_max = ak.max( ak.num(obj,axis=-1) )
    for N in range(njets_max+1):
        first_njets = obj[:,:N] # Consider only the first 'N' jets
        mask = (ak.count_nonzero(first_njets.match,axis=-1) >=4) # Count at least 4 matches found in first N jets 
        counts.append( ak.count_nonzero(ak.drop_none(mask)) ) # Count number of events satisfying above condition
    total = counts[-1]

    bins = np.linspace(-0.5,njets_max+0.5,num=njets_max+2,endpoint=True)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.style.use([hep.style.CMS, hep.style.firamath])
    fig,ax = plt.subplots(figsize=(10,9))
    yerrs = clopper_pearson_interval(np.array(counts), np.array([total]*len(counts)),relative=True)
    ax.errorbar(centers,counts/total,xerr=None,yerr=yerrs,linestyle="None",color="red",marker="o",label="Data",markersize=8)
    plt.xlabel("Number of L1 jets considered")
    plt.ylabel("Cumulative efficiency")
    ax.set_xlim([bins[0],bins[-1]])
    ax.set_ylim([0.,1.])
    hep.cms.label("Preliminary",data=False, year=year, com=com)

#############################################################################################
#
def plot_sig_eff_vs_jet_rank_and_btag_score(
        events,label,
        gen,        
        pt_min,eta_max,
        passed=None,
        verbosity=0,
        year=2023,com=13.6):

    events = filter_events(events,passed)
    jet = base_objects(events,label=label)
    jet["btag"] = events["L1DisplacedJet_btagScore"]
    jet["in_acc"] = (jet.pt > pt_min) & (abs(jet.eta) < eta_max)
    jet = jet[jet.in_acc]

    # Match L1 jets
    _,_,obj = object_matching_base(
        gen,
        jet,
        passed=passed,
        gen_id_filter=5,
        n=4,
        label="[plot_sig_eff_vs_jet_rank]", 
        verbosity=verbosity)

    assert ak.all(obj.pt == ak.sort(obj.pt, ascending=False, axis=-1)), "ak.array not sorted by pT!"

    njets_max = ak.max( ak.num(obj,axis=-1) ) + 1
    xvals = np.linspace(-0.5,njets_max-0.5,njets_max+1,endpoint=True)
    yvals = np.linspace(-1.2,1.,12,endpoint=True)
    xbins = len(xvals)-1
    ybins = len(yvals)-1
    # print("njets_max",njets_max)
    # print("len(xvals)",len(xvals))
    # print("len(yvals)",len(yvals))
    # print("xbins:",xbins)
    # print("ybins:",ybins)
    # print("xvals",xvals)
    # print("yvals",yvals)
    counts = np.zeros((xbins,ybins))
    for ybin in range(ybins):
        thr = yvals[ybin]
        for xbin in range(xbins):
            N = xbin
            mask = (obj.btag > thr)
            first_njets = ak.drop_none(ak.mask(obj,mask))[:,:N] # Consider only the first 'N' jets that satisfy the b-tagging requirement
            mask = (ak.count_nonzero(first_njets.match,axis=-1) >=4) # Count at least 4 matches found in first N jets 
            counts[xbin,ybin] = ak.count_nonzero(ak.drop_none(mask)) # Count number of events satisfying above condition
    total = counts[-1,0]
    # print ("total",total)

    plt.style.use([hep.style.CMS, hep.style.firamath])

    histo = bh.Histogram(
        bh.axis.Regular(xbins,xvals[0],xvals[-1]),
        bh.axis.Regular(ybins,yvals[0],yvals[-1]),
        storage=bh.storage.Double())
    # histo.view()[:] = counts
    # print(histo.view())
    histo.view()[:] = counts/total

    projections = False
    if projections == False:
        fig, ax, ax_colorbar = plot_2d_hist(
            histo, 
            colorbar_kwargs={"label": "Cumulative efficiency"},
            )
        ax.set_xlabel("# L1 jets considered")
        ax.set_ylabel("b-tag threshold")
    else:
        (fig,ax,ax_x_projection,ax_y_projection,ax_colorbar) = plot_2d_hist_with_projections(
            histo,
            xlabel="L1 jets considered",
            ylabel="b-tag threshold",
            ylabel_x_projection="Cumu eff",
            xlabel_y_projection="Cumu eff",
            offset_x_labels=False,
            colorbar_kwargs={"label": "Cumu eff"},)

    xcenters = (xvals[:-1] + xvals[1:]) / 2
    ycenters = (yvals[:-1] + yvals[1:]) / 2
    for xbin in range(xbins):
        xval = xcenters[xbin]
        for ybin in range(ybins):
            yval = ycenters[ybin]
            ax.text(xval, yval, f"{histo.view()[xbin, ybin]:.2f}", ha='center', va='center', color='red', fontsize=6)

    #hep.cms.label("Preliminary",data=False, year=year, com=com)
    
#############################################################################################
# https://hsf-training.github.io/hsf-training-matplotlib/05-mplhep/index.html#solution
# https://hist.readthedocs.io/en/latest/user-guide/quickstart.html
# https://indico.cern.ch/event/1375573/contributions/6090507/attachments/2916973/5119176/plothist_PyHEPdev.pdf
def plot_perf_vs_pt(
        perf, # "eff" or "purity"
        events,
        label, # Label for the objects, e.g. "L1Jet"
        gen,
        pt_min,eta_max,
        passed=None,
        gen_id_filter=None,
        hlt_id_filter=None,
        hlt_bits_filter=None,
        n=4,dr_max=0.3,dpt_min=0.2,dpt_max=2.0,
        verbosity=0,
        **kwargs
        ):

    year = 2023 if "year" not in kwargs else kwargs["year"]
    com = 13.6 if "com" not in kwargs else kwargs["com"]
    nbins = 50 if "nbins" not in kwargs else kwargs["nbins"]
    start = 0. if "start" not in kwargs else kwargs["start"]
    stop = 100. if "stop" not in kwargs else kwargs["stop"]
    xlabel = f"{label} p$_T$ [GeV]" if "xlabel" not in kwargs else kwargs["xlabel"]
    ylabel = kwargs["ylabel"] if "ylabel" in kwargs else "Efficiency" if perf == "eff" else "Purity"

    events = filter_events(events,passed)
    obj = base_objects(events,label=label)
    obj["in_acc"] = (obj.pt > pt_min) & (abs(obj.eta) < eta_max)
    obj = obj[obj.in_acc]

    # _,gen,obj = object_matching_base(
    #     gen,
    #     obj,
    #     passed=passed,
    #     gen_id_filter=gen_id_filter,
    #     n=4,
    #     label="[plot_perf_vs_pt]", 
    #     verbosity=verbosity)

    # Match objects to GEN (HLT matching if hlt_bits given; otherwise just normal geometric matching)
    _,gen,obj = hlt_matching_base(
        gen, 
        obj,
        passed=None,
        gen_id_filter=gen_id_filter,
        hlt_id_filter=hlt_id_filter,
        hlt_bits_filter=hlt_bits_filter,
        n=4,dr_max=dr_max,dpt_min=dpt_min,dpt_max=dpt_max,
        label="[plot_perf_vs_pt]",
        verbosity=verbosity)

    if perf == "eff":
        denom = ak.ravel(gen.pt)
        numer = ak.ravel(ak.drop_none(ak.mask(gen.pt,gen.match)))
    elif perf == "purity":
        denom = ak.ravel(obj.pt)
        numer = ak.ravel(ak.drop_none(ak.mask(obj.pt,obj.match)))
    else:
        raise ValueError(f"Unknown performance metric '{perf}'")        

    h_denom,bins = np.histogram(denom, bins=nbins, range=(start, stop))
    h_numer,_    = np.histogram(numer, bins=nbins, range=(start, stop))
    centers = (bins[:-1] + bins[1:]) / 2
    
    plt.style.use([hep.style.CMS, hep.style.firamath])
    fig,ax1 = plt.subplots(figsize=(10,9))
    ax2 = ax1.twinx()

    effs = h_numer / h_denom
    yerrs = clopper_pearson_interval(np.array(h_numer), np.array(h_denom),relative=True)
    
    ax1.errorbar(
        centers,
        effs,
        xerr=None,
        yerr=yerrs,
        linestyle="None",
        color="black",
        marker="o",
        label="Eff" if perf == "eff" else "Purity",
        )

    hep.histplot(
        [h_denom,h_numer],
        stack=False,
        bins=bins,
        histtype="fill",
        color=["blue","red"],
        alpha=0.5,
        edgecolor="black",
        label=["Denom","Numer"],
        ax=ax2,
        )

    width = (stop - start) / nbins
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(f"Entries / {width:.0f} GeV")
    ax1.set_xlim(start,stop)
    ax2.set_xlim(start,stop)
    ax1.set_ylim(0.,1.)
    ax2.set_ylim(0.,max(max(h_denom),max(h_numer))*1.05)
    hep.cms.label("Preliminary",data=False, year=year, com=com)

# #############################################################################################
# #
# def plot_efficiency(
#         numer,
#         denom,
#         xbins:int,xmin,xmax,
#         xlabel="x axis label",
#         filename="eff.pdf"):

#     print("[plot_efficiency]")
#     print("binning:",xbins,xmin,xmax)

#     h_denom = Hist(hist.axis.Regular(int(xbins),xmin,xmax,name=xlabel))
#     h_numer = Hist(hist.axis.Regular(int(xbins),xmin,xmax,name=xlabel))
#     h_denom.fill(denom)
#     h_numer.fill(numer)

#     h_denom_bin_edges = h_denom.axes[0].edges
#     h_denom_bin_centers = h_denom.axes[0].centers
#     print("h_denom_bin_edges",h_denom_bin_edges)
#     print("h_denom_bin_centers",h_denom_bin_centers)

#     h_denom_diff = h_denom.view()
#     h_numer_diff = h_numer.view()
#     ymax = max(max(h_numer_diff.view()),max(h_denom_diff.view()))
#     eff_cumu = h_numer_diff / h_denom_diff
    
#     h_denom_cumu = np.cumsum(h_denom_diff[::-1])[::-1]
#     h_numer_cumu = np.cumsum(h_numer_diff[::-1])[::-1]
#     #ymax = max(max(h_numer_cumu.view()),max(h_denom_cumu.view()))
#     #eff_cumu = h_numer_cumu / h_denom_cumu

#     print("h_denom_diff:", ak.sum(h_denom_diff), [f"{x:.1f}" for x in h_denom_diff])
#     print("h_numer_diff:", ak.sum(h_numer_diff), [f"{x:.1f}" for x in h_numer_diff])
#     #print("h_denom_cumu:", ak.sum(h_denom_cumu), [f"{x:.1f}" for x in h_denom_cumu])
#     #print("h_numer_cumu:", ak.sum(h_numer_cumu), [f"{x:.1f}" for x in h_numer_cumu])
#     print("eff_cumu:", [f"{x:.2f}" for x in eff_cumu])

#     # Create plot
#     fig, ax1 = plt.subplots(figsize=(8, 6))
#     ax2 = ax1.twinx()
#     ax1.set_ylim(0.,ymax*1.2)
#     ax2.set_ylim(0.,1.1)

#     # Plot cumulative distributions

#     ax1.fill_between(h_denom_bin_edges[:-1], 0, h_denom_diff, step="post", color="blue", alpha=0.5, label="Denominator (diff)")
#     ax1.fill_between(h_denom_bin_edges[:-1], 0, h_numer_diff, step="post", color="red", alpha=0.5, label="Numerator (diff)")
#     #ax1.plot(h_denom_bin_edges[:-1], h_numer_cumu, drawstyle='steps-post', label='Numerator (cumu)', color='red')
#     #ax1.plot(h_denom_bin_edges[:-1], h_denom_cumu, drawstyle='steps-post', label='Denominator (cumu)', color='blue')

#     # Plot cumulative efficiency
#     ax2.plot(h_denom_bin_centers, eff_cumu, marker='o', linestyle='-', color='black', label='Cumulative efficiency')
#     eff_err = clopper_pearson_interval(h_numer_diff, h_denom_diff)
#     ax2.fill_between(h_denom_bin_edges[:-1], eff_err[0], eff_err[1], color='gray', alpha=0.3, step='post', label='Efficiency uncertainty')

#     # Labels and legends
#     ax1.set_xlabel(xlabel)
#     ax1.set_ylabel("Differential and cumulative counts")
#     ax2.set_ylabel("Cumulative efficiency")
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')

#     #plt.title("Cumulative Efficiency Plot")
#     plt.show()

#     # Save to pdf
#     fig.savefig(filename)


# ## SCRIPT

# In[ ]:


get_ipython().system('jupyter nbconvert HH_common.ipynb --to script --output common')


# In[ ]:




