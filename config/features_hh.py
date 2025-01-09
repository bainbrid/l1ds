from analysis_tools import Feature
from plotting_tools import Label

features = [
    Feature("L1Jet_genb_deltaR", "L1Jet_genb_deltaR",
        binning=(50, 0, 1),
        x_title=Label("#Delta R (L1Jet, gen b)"),
    ),
    Feature("L1Jet_genb_deltaR_longrange", "L1Jet_genb_deltaR",
        binning=(50, 0, 5),
        x_title=Label("#Delta R (L1Jet, gen b)"),
    ),
    Feature("nL1Jet_genbmatched", "L1Jet_genb_deltaR[L1Jet_genb_deltaR < 0.5].size()",
        binning=(8, -0.5, 7.5),
        x_title=Label("Number of gen matched L1 jets"),
    ),
    Feature("L1Jet_L1Muon_deltaR", "L1Jet_L1Muon_deltaR",
        binning=(50, 0, 10),
        x_title=Label("#Delta R (L1Jet, L1Muon)"),
    ),
    Feature("L1Jet_L1Muon_deltaR_genbmatched", "L1Jet_L1Muon_deltaR[L1Jet_genb_deltaR < 0.5]",
        binning=(50, 0, 1),
        x_title=Label("#Delta R (L1Jet, L1Muon) (L1 Jet matched to b)"),
    ),
    Feature("nL1Jet_genbmatched_l1mumatched", "L1Jet_pt[(L1Jet_genb_deltaR < 0.5) && "
            "(L1Jet_L1Muon_deltaR < 0.1)].size()",
        binning=(8, -0.5, 7.5),
        x_title=Label("Number of gen matched L1 jets matched to L1 muons"),
    ),
    Feature("nMuonDiff", "nL1Muon - GenPart_pdgId[abs(GenPart_pdgId) == 13]",
        binning=(11, -5.5, 5.5),
        x_title=Label("Number of L1 jets - Number of gen muons"),
    ),
]
