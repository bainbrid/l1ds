import ROOT
ROOT.gROOT.SetBatch(True)

example_file = "/vols/cms/jleonhol/l1scouting/samples/hh4b/nano_4b.root"

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

        bool match_trigger_object(float off_eta, float off_phi,
            iRVec TrigObj_id, iRVec TrigObj_filterBits, fRVec TrigObj_eta, fRVec TrigObj_phi,
            std::vector<int> bits)
        {
          for (size_t iobj = 0; iobj < TrigObj_id.size(); iobj++) {
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

        bool pass_jet_sel(
            int nJet, fRVec Jet_pt, fRVec Jet_eta, fRVec Jet_phi,
            int nTrigObj, iRVec TrigObj_id, iRVec TrigObj_filterBits, fRVec TrigObj_eta, fRVec TrigObj_phi,
            int nGenPart, fRVec GenPart_eta, fRVec GenPart_phi,
            fRVec GenPart_pdgId, fRVec GenPart_genPartIdxMother,
            int HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55
        ) {
            int jet1_index = -1, jet2_index = -1, jet3_index = -1, jet4_index = -1;
            if (HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55 != 1)
                return false;
            for (size_t igen = 0; igen < nGenPart; igen++) {
                if (abs(GenPart_pdgId[igen]) != 5)
                    continue;
                if (abs(GenPart_pdgId[GenPart_genPartIdxMother[igen]]) != 25)
                    continue;
                for (size_t ijet = 0; ijet < nJet; ijet++) {
                    if (Jet_pt[ijet] < 30)
                       continue;
                    auto jet_eta = Jet_eta[ijet];
                    auto jet_phi = Jet_phi[ijet];

                    auto deta = (GenPart_eta[igen] - jet_eta);
                    auto dphi = Phi_mpi_pi(GenPart_phi[igen] - jet_phi);
                    auto deltaR = TMath::Sqrt(deta * deta + dphi * dphi);
                    // std::cout << ijet << " " << deltaR << std::endl;
                    if (deltaR < 0.2) {
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
            }
            if (jet4_index == -1)
                return false;
            
            // Filters
            // - hlt4PixelOnlyPFCentralJetTightIDPt20 0
            // - hlt4PFCentralJetTightIDPt30 3

            // if (!(match_trigger_object(Jet_eta[jet1_index], Jet_phi[jet1_index],
            //        TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi,
            //        {1, 8})))
            //    return false;  
            return true;
        }
    """
    )
    
    return df.Filter("""pass_jet_sel(
            nJet, Jet_pt, Jet_eta, Jet_phi,
            nTrigObj, TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi,
            nGenPart, GenPart_eta, GenPart_phi, GenPart_pdgId, GenPart_genPartIdxMother,
            HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55
        )""")


if __name__ == "__main__":
    ROOT.gStyle.SetOptStat(0)

    df = ROOT.RDataFrame("Events", example_file)

    # df = df.Filter("event == 435")

    df_filt = apply_offline(df)
    print(df.Count().GetValue(), df_filt.Count().GetValue())

