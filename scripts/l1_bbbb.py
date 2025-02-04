import ROOT
ROOT.gROOT.SetBatch(True)

example_file = "/vols/cms/jleonhol/l1scouting/samples/hh4b/nano_4b.root"

def apply_l1(df):
    ROOT.gInterpreter.Declare("""
        #include "TMath.h"
        using fRVec = const ROOT::RVec<float>&;
        using iRVec = const ROOT::RVec<int>&;

        bool use_l1taus = false;

        double Phi_mpi_pi(double x) {
            while (x >= 3.14159) x -= (2 * 3.14159);
            while (x < -3.14159) x += (2 * 3.14159);
            return x;
        }
        
        bool pass_l1jet_sel(
            int nL1Jet, fRVec L1Jet_pt, fRVec L1Jet_eta, fRVec L1Jet_phi,
            int nL1Tau, fRVec L1Tau_pt, fRVec L1Tau_eta, fRVec L1Tau_phi,
            int nGenPart, fRVec GenPart_eta, fRVec GenPart_phi,
            fRVec GenPart_pdgId, fRVec GenPart_genPartIdxMother
        ) {
            int jet1_index = -1, jet2_index = -1, jet3_index = -1, jet4_index = -1;
            for (size_t igen = 0; igen < nGenPart; igen++) {
                if (abs(GenPart_pdgId[igen]) != 5)
                    continue;
                if (abs(GenPart_pdgId[GenPart_genPartIdxMother[igen]]) != 25)
                    continue;
                bool found_jet = false;   
                for (size_t ijet = 0; ijet < nL1Jet; ijet++) {
                    auto jet_eta = L1Jet_eta[ijet];
                    auto jet_phi = L1Jet_phi[ijet];
                    auto deta = (GenPart_eta[igen] - jet_eta);
                    auto dphi = Phi_mpi_pi(GenPart_phi[igen] - jet_phi);
                    auto deltaR = TMath::Sqrt(deta * deta + dphi * dphi);
                    // std::cout << igen << " " <<  ijet << " " << deltaR << std::endl;
                    if (deltaR < 0.2) {
                        found_jet = true;
                        // std::cout << "<--" << std::endl;
                        if (jet1_index == -1) {
                            jet1_index = ijet;
                        } else if (jet2_index == -1) {
                            jet2_index = ijet;
                        } else if (jet3_index == -1) {
                            jet3_index = ijet;
                        } else {
                            jet4_index = ijet;
                        }
                        break;
                    }
                }
                if (use_l1taus && !found_jet) {
                    for (size_t itau = 0; itau < nL1Tau; itau++) {
                        auto jet_eta = L1Tau_eta[itau];
                        auto jet_phi = L1Tau_phi[itau];
                        auto deta = (GenPart_eta[igen] - jet_eta);
                        auto dphi = Phi_mpi_pi(GenPart_phi[igen] - jet_phi);
                        auto deltaR = TMath::Sqrt(deta * deta + dphi * dphi);
                        // std::cout << itau << " " << igen << " " << deltaR << std::endl;
                        if (deltaR < 0.2) {
                            // std::cout << "<--" << std::endl;
                            if (jet1_index == -1) {
                                jet1_index = itau;
                            } else if (jet2_index == -1) {
                                jet2_index = itau;
                            } else if (jet3_index == -1) {
                                jet3_index = itau;
                            } else {
                                jet4_index = itau;
                            }
                            break;
                        }
                    }
                }
            }
            // std::cout << jet1_index << " ";
            // std::cout << jet2_index << " ";
            // std::cout << jet3_index << " ";
            // std::cout << jet4_index << std::endl;
            if (jet4_index == -1)
                return false;
            
            return true;
        }
    """
    )
    
    return df.Filter("""pass_l1jet_sel(
            nL1Jet, L1Jet_pt, L1Jet_eta, L1Jet_phi,
            nL1Tau, L1Tau_pt, L1Tau_eta, L1Tau_phi,
            nGenPart, GenPart_eta, GenPart_phi, GenPart_pdgId, GenPart_genPartIdxMother
        )""")


if __name__ == "__main__":
    df = ROOT.RDataFrame("Events", example_file)
    #df = df.Filter("event == 434")
    # from offline import apply_offline
    # df = apply_offline(df)
    df_filt = apply_l1(df)
    # df_filt = df_filt.Snapshot("Events", "l1_tree.root")
    print(df.Count().GetValue(), df_filt.Count().GetValue())

