from analysis_tools import Feature
from plotting_tools import Label

features = [

    ##########
    # TOWERS #
    ##########

    Feature("gens_pt", "GenPart_pt[GenPart_pdgId == 999999]",
        binning=(100, -0.5, 99.5),
        x_title=Label("SUEP PT"),
    ),
    Feature("gena_pt", "GenPart_pt[GenPart_pdgId == 999998]",
        binning=(100, -0.5, 99.5),
        x_title=Label("SUEP decay PT"),
    ),

    Feature("toweriet", "L1EmulCaloTower_iet",
        binning=(1000, -0.5, 999.5),
        x_title=Label("L1EmulCaloTower_iet"),
    ),

    Feature("towerieta", "L1EmulCaloTower_ieta",
        binning=(100, -50, 50),
        x_title=Label("L1EmulCaloTower_ieta"),
    ),

    Feature("toweriphi", "L1EmulCaloTower_iphi",
        binning=(100, 0, 80),
        x_title=Label("L1EmulCaloTower_iphi"),
    ),
    
    Feature("towerieta_iet5", "L1EmulCaloTower_ieta[L1EmulCaloTower_iet >= 5]",
        binning=(100, -50, 50),
        x_title=Label("L1EmulCaloTower_ieta (et >= 5)"),
    ),

    Feature("toweriphi_iet5", "L1EmulCaloTower_iphi[L1EmulCaloTower_iet >= 5]",
        binning=(100, 0, 80),
        x_title=Label("L1EmulCaloTower_iphi (et >= 5)"),
    ),
    
    Feature("towerieta_iet10", "L1EmulCaloTower_ieta[L1EmulCaloTower_iet >= 10]",
        binning=(100, -50, 50),
        x_title=Label("L1EmulCaloTower_ieta (et >= 10)"),
    ),

    Feature("toweriphi_iet10", "L1EmulCaloTower_iphi[L1EmulCaloTower_iet >= 10]",
        binning=(100, 0, 80),
        x_title=Label("L1EmulCaloTower_iphi (et >= 10)"),
    ),

    ##################
    # TOWERS FROM HW #
    ##################

    Feature("toweret", "calotower_et",
        binning=(1000, -0.5, 999.5),
        x_title=Label("L1EmulCaloTower_et"),
    ),
    
    Feature("towereta", "calotower_eta",
        binning=(100, -5, 5),
        x_title=Label("L1EmulCaloTower_eta"),
    ),

    Feature("towerphi", "calotower_phi",
        binning=(100, -3.5, 3.5),
        x_title=Label("L1EmulCaloTower_phi"),
    ),

    Feature("towereta_iet5", "calotower_eta[L1EmulCaloTower_iet >= 5]",
        binning=(100, -5, 5),
        x_title=Label("L1EmulCaloTower_eta (iet >= 5)"),
    ),

    Feature("towerphi_iet5", "calotower_phi[L1EmulCaloTower_iet >= 5]",
        binning=(100, -3.5, 3.5),
        x_title=Label("L1EmulCaloTower_phi (iet >= 5)"),
    ),

    Feature("towereta_iet10", "calotower_eta[L1EmulCaloTower_iet >= 10]",
        binning=(100, -5, 5),
        x_title=Label("L1EmulCaloTower_eta (iet >= 10)"),
    ),

    Feature("towerphi_iet10", "calotower_phi[L1EmulCaloTower_iet >= 10]",
        binning=(100, -3.5, 3.5),
        x_title=Label("L1EmulCaloTower_phi (iet >= 10)"),
    ),

    ###########
    # L1 JETS #
    ###########

    Feature("l1jet_eta", "L1Jet_eta",
        binning=(100, -5, 5),
        x_title=Label("L1Jet_eta"),
    ),

    Feature("l1jet_phi", "L1Jet_phi",
        binning=(100, -3.5, 3.5),
        x_title=Label("L1Jet_phi"),
    ),

    Feature("s_eta", "GenPart_eta[abs(GenPart_pdgId) == 999999]",
        binning=(100, -5, 5),
        x_title=Label("gen s_eta"),
    ),

    Feature("s_phi", "GenPart_phi[abs(GenPart_pdgId) == 999999]",
        binning=(100, -3.5, 3.5),
        x_title=Label("gen s_phi"),
    ),

    Feature("ap_eta", "GenPart_eta[abs(GenPart_pdgId) == 999998]",
        binning=(100, -5, 5),
        x_title=Label("gen A' eta"),
    ),

    Feature("ap_phi", "GenPart_phi[abs(GenPart_pdgId) == 999998]",
        binning=(100, -3.5, 3.5),
        x_title=Label("gen A' phi"),
    ),

    Feature("tower_sphericity", "tower_sphericity",
        binning=(20, 0, 1),
        x_title=Label("Sphericity computed from TT"),
    ),

    Feature("tower_sphericity_et1", "tower_sphericity_et1",
        binning=(20, 0, 1),
        x_title=Label("Sphericity computed from TT with et > 1 GeV"),
    ),

    Feature("tower_sphericity_et2", "tower_sphericity_et2",
        binning=(20, 0, 1),
        x_title=Label("Sphericity computed from TT  with et > 2 GeV"),
    ),

    Feature("tower_sphericity_et4", "tower_sphericity_et4",
        binning=(20, 0, 1),
        x_title=Label("Sphericity computed from TT  with et > 4 GeV"),
    ),

    Feature("tower_sphericity_et6", "tower_sphericity_et6",
        binning=(20, 0, 1),
        x_title=Label("Sphericity computed from TT  with et > 6 GeV"),
    ),

    Feature("l1jet_sphericity", "l1jet_sphericity",
        binning=(20, 0, 1),
        x_title=Label("Sphericity computed from L1 jets"),
    ),

    Feature("ap_sphericity", "ap_sphericity",
        binning=(20, 0, 1),
        x_title=Label("Sphericity computed from Gen Dark photons"),
    ),

    Feature("boosted_ap_sphericity", "boosted_ap_sphericity",
        binning=(20, 0, 1),
        x_title=Label("Sphericity computed from Gen Dark photons in the Higgs rest frame"),
    ),

    Feature("nap", "GenPart_pdgId[GenPart_pdgId == 999998].size()",
        binning=(40, -0.5, 39.5),
        x_title=Label("Multiplicity of Gen Dark photons"),
    ),

    Feature("boostedhiggs_eta", "BoostedHiggs_eta",
        binning=(100, -5, 5),
        x_title=Label("BoostedHiggs_eta"),
    ),

    Feature("boostedhiggs_phi", "BoostedHiggs_phi",
        binning=(100, -3.5, 3.5),
        x_title=Label("BoostedHiggs_phi"),
    ),

    Feature("boostedhiggs_px", "BoostedHiggs_px",
        binning=(100, -0.5, 99.5),
        x_title=Label("Boosted Higgs p_{x}"),
    ),

    Feature("boostedhiggs_py", "BoostedHiggs_py",
        binning=(100, -0.5, 99.5),
        x_title=Label("Boosted Higgs p_{y}"),
    ),

    Feature("boostedhiggs_pz", "BoostedHiggs_pz",
        binning=(100, -0.5, 99.5),
        x_title=Label("Boosted Higgs p_{z}"),
    ),
    
    Feature("boostedhiggs_e", "BoostedHiggs_e",
        binning=(100, -0.5, 199.5),
        x_title=Label("Boosted Higgs E"),
    ),
]
