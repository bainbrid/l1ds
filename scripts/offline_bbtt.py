import ROOT
ROOT.gROOT.SetBatch(True)

trigger_name = "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1" 

example_file = "${ICREDIR}/store/user/jleonhol/samples/l1nanotest/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/v1/241014_110019/0000/TSG-Run3Winter24NanoAOD-00227_1.root"

def apply_gen(df):
    ROOT.gInterpreter.Declare("""
        #include "TMath.h"
        using fRVec = const ROOT::RVec<float>&;
        using iRVec = const ROOT::RVec<int>&;
        
        int get_ntauh(int nGenPart, fRVec GenPart_pdgId, fRVec GenPart_genPartIdxMother) {
            int ntauh = 0;
            for (size_t igen = 0; igen < nGenPart; igen++) {
                if (abs(GenPart_pdgId[igen]) != 15)
                    continue;
                if (abs(GenPart_pdgId[GenPart_genPartIdxMother[igen]]) != 25)
                    continue;
                
                bool istauh = true;
                for (size_t igenlep = 0; igenlep < nGenPart; igenlep++) {
                    if (GenPart_genPartIdxMother[igenlep] == igen) {
                        if (abs(GenPart_pdgId[igenlep]) == 11 || abs(GenPart_pdgId[igenlep]) == 13)
                            istauh = false;
                    }
                }
                if (istauh)
                    ntauh++;
            }
            return ntauh;
        }
    """)
    return df.Define("ntauh", "get_ntauh(nGenPart, GenPart_pdgId, GenPart_genPartIdxMother)")

def apply_offline(df):
    ROOT.gInterpreter.Declare("""
        #include "TMath.h"
        using fRVec = const ROOT::RVec<float>&;
        using iRVec = const ROOT::RVec<int>&;

        double Phi_mpi_pi(double x) {
            while (x >= 3.14159) x -= (2 * 3.14159);
            while (x < -3.14159) x += (2 * 3.14159);
            return x;
        }

        bool match_trigger_object(float off_eta, float off_phi, int obj_id,
            iRVec TrigObj_id, iRVec TrigObj_filterBits, fRVec TrigObj_eta, fRVec TrigObj_phi,
            std::vector<int> bits)
        {
          for (size_t iobj = 0; iobj < TrigObj_id.size(); iobj++) {
            if (TrigObj_id[iobj] != obj_id) continue;
            auto const dPhi(std::abs(Phi_mpi_pi(off_phi - TrigObj_phi[iobj])));
            auto const dEta(std::abs(off_eta - TrigObj_eta[iobj]));
            auto const delR2(dPhi * dPhi + dEta * dEta);
            if (delR2 > 0.5 * 0.5)
              continue;
            bool matched_bits = true;
            for (auto & bit : bits) {
              if ((TrigObj_filterBits[iobj] & bit) == 0) {
                matched_bits = false;
                break;
              }
            }
            if (!matched_bits)
              continue;
            return true;
          }
          return false;
        }
        
        bool pass_tau_sel(
            int nTau, fRVec Tau_pt, fRVec Tau_eta, fRVec Tau_phi,
            int nTrigObj, iRVec TrigObj_id, iRVec TrigObj_filterBits, fRVec TrigObj_eta, fRVec TrigObj_phi,
            int nGenPart, fRVec GenPart_eta, fRVec GenPart_phi,
            fRVec GenPart_pdgId, fRVec GenPart_genPartIdxMother,
            int HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
        ) {
            int tau1_index = -1; int tau2_index = -1;
            if (HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 != 1)
                return false;
            for (size_t itau = 0; itau < nTau; itau++) {
                if (Tau_pt[itau] < 35)
                   continue;
                auto tau_eta = Tau_eta[itau];
                auto tau_phi = Tau_phi[itau];
                for (size_t igen = 0; igen < nGenPart; igen++) {
                    if (abs(GenPart_pdgId[igen]) != 15)
                        continue;
                    if (abs(GenPart_pdgId[GenPart_genPartIdxMother[igen]]) != 25)
                        continue;
                    auto deta = (GenPart_eta[igen] - tau_eta);
                    auto dphi = Phi_mpi_pi(GenPart_phi[igen] - tau_phi);
                    auto deltaR = TMath::Sqrt(deta * deta + dphi * dphi);
                    if (deltaR < 0.2) {
                        if (tau1_index == -1) {
                            tau1_index = itau;
                        } else {
                            tau2_index = itau;
                        }
                        break;
                    }
                }
            }
            if (tau2_index == -1)
                return false;
            
            if (!(match_trigger_object(Tau_eta[tau1_index], Tau_phi[tau1_index], 15,
                     TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi,
                     {2, 8, 32, 1024})))
                 return false;
            if (!(match_trigger_object(Tau_eta[tau2_index], Tau_phi[tau2_index], 15,
                    TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi,
                    {2, 8, 32, 1024})))
                return false;    
            return true;
        }
    """
    )
    
    return df.Filter("""pass_tau_sel(
            nTau, Tau_pt, Tau_eta, Tau_phi,
            nTrigObj, TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi,
            nGenPart, GenPart_eta, GenPart_phi, GenPart_pdgId, GenPart_genPartIdxMother,
            HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
        )""")


if __name__ == "__main__":
    ROOT.gStyle.SetOptStat(0)
    
    df = ROOT.RDataFrame("Events", example_file)

    df = apply_gen(df)
    # histo = df.Histo1D("ntauh")
    # histo.Scale(1./3000.)
    # c = ROOT.TCanvas()
    # histo.Draw()
    # c.SaveAs("histo.pdf")
    df = df.Filter("ntauh == 2")

    df_filt = apply_offline(df)
    print(df.Count().GetValue(), df_filt.Count().GetValue())

