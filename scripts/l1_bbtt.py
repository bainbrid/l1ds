import ROOT
ROOT.gROOT.SetBatch(True)

example_file = "${ICREDIR}/store/user/jleonhol/samples/l1nanotest/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/v1/241014_110019/0000/TSG-Run3Winter24NanoAOD-00227_1.root"

def apply_l1(df):
    ROOT.gInterpreter.Declare("""
        #include "TMath.h"
        using fRVec = const ROOT::RVec<float>&;
        using iRVec = const ROOT::RVec<int>&;

        double Phi_mpi_pi(double x) {
            while (x >= 3.14159) x -= (2 * 3.14159);
            while (x < -3.14159) x += (2 * 3.14159);
            return x;
        }
        
        bool pass_l1tau_sel(
            int nL1Tau, fRVec L1Tau_pt, fRVec L1Tau_eta, fRVec L1Tau_phi,
            int nGenPart, fRVec GenPart_eta, fRVec GenPart_phi,
            fRVec GenPart_pdgId, fRVec GenPart_genPartIdxMother
        ) {
            int tau1_index = -1; int tau2_index = -1;
            for (size_t itau = 0; itau < nL1Tau; itau++) {
                auto tau_eta = L1Tau_eta[itau];
                auto tau_phi = L1Tau_phi[itau];
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
            
            return true;
        }
    """
    )
    
    return df.Filter("""pass_l1tau_sel(
            nL1Tau, L1Tau_pt, L1Tau_eta, L1Tau_phi,
            nGenPart, GenPart_eta, GenPart_phi, GenPart_pdgId, GenPart_genPartIdxMother
        )""")


if __name__ == "__main__":
    df = ROOT.RDataFrame("Events", example_file)
    from offline import apply_gen
    df = apply_gen(df)
    df = df.Filter("ntauh == 2")
    df_filt = apply_l1(df)
    print(df.Count().GetValue(), df_filt.Count().GetValue())

