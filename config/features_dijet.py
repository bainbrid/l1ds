from analysis_tools import Feature
from plotting_tools import Label

features = [

    ###########
    # L1 JETS #
    ###########

    Feature("jet_pt", "Jet_pt",
        binning=(60, 0, 300),
        x_title=Label("Jet_pt"),
    ),
    Feature("jet_eta", "Jet_eta",
        binning=(100, -5, 5),
        x_title=Label("Jet_eta"),
    ),
    Feature("jet_phi", "Jet_phi",
        binning=(100, -3.5, 3.5),
        x_title=Label("Jet_phi"),
    ),
    Feature("mjj", "mjj",
        binning=(110, 150, 700),
        x_title=Label("mjj"),
    ),
    Feature("deta", "deta",
        binning=(30, 0, 3.0),
        x_title=Label("deta"),
    ),

]