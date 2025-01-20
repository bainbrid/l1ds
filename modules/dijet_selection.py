from analysis_tools.utils import import_root
ROOT = import_root()

class DijetSelectionProducer():
    def __init__(self, *args, **kwargs):
        ROOT.gInterpreter.Declare("""
            using Vbool = ROOT::RVec<bool>;
            using Vint = ROOT::RVec<int>;
            using Vfloat = ROOT::RVec<float>;
            using Vdouble = ROOT::RVec<double>;

            auto getDijetMass(Vfloat Jet_pt, Vfloat Jet_eta, Vfloat Jet_phi){
                float mjj = -1;
                if(Jet_pt.size() > 1){
                    ROOT::Math::PtEtaPhiMVector jet1(Jet_pt[0], Jet_eta[0], Jet_phi[0], 0);
                    ROOT::Math::PtEtaPhiMVector jet2(Jet_pt[1], Jet_eta[1], Jet_phi[1], 0);
                    mjj = (jet1 + jet2).M();
                }

                return mjj;
            }

            auto getDijetPt(Vfloat Jet_pt, Vfloat Jet_eta, Vfloat Jet_phi){
                float ptjj = -1;
                if(Jet_pt.size() > 1){
                    ROOT::Math::PtEtaPhiMVector jet1(Jet_pt[0], Jet_eta[0], Jet_phi[0], 0);
                    ROOT::Math::PtEtaPhiMVector jet2(Jet_pt[1], Jet_eta[1], Jet_phi[1], 0);
                    ptjj = (jet1 + jet2).Pt();
                }

                return ptjj;
            }

            auto getDijetDPhi(Vfloat Jet_phi){
                float dijet_dphi = -1;
                if(Jet_phi.size() > 1){
                float dphi = std::abs(TVector2::Phi_mpi_pi(Jet_phi[0] - Jet_phi[1]));  
                dijet_dphi = dphi;             
                }
                return dijet_dphi;
            }
        """)

    def run(self, df):
        df = df.Filter("nJet > 1").Filter("(Jet_pt[0] > 30)").Filter("(Jet_pt[1] > 30)")
        df = df.Filter("std::abs(Jet_eta[0]) < 2.5").Filter("std::abs(Jet_eta[1]) < 2.5")
        #Event vetoes
        df = df.Filter("Sum(Jet_pt == 1023.5)==0", "Saturated jet veto")

        #Define quantities
        df = df.Define("mjj", "getDijetMass(Jet_pt, Jet_eta, Jet_phi)")
        df = df.Define("dijet_deta", "std::abs(Jet_eta[0] - Jet_eta[1])").Define("dijet_dphi", "getDijetDPhi(Jet_phi)").Define("dijet_pt", "getDijetPt(Jet_pt, Jet_eta, Jet_phi)")
        #Get a scaled pT
        df = df.Define("Jet_scaled_pt", "Jet_pt/mjj")

        #Further event selections
        df = df.Filter("dijet_dphi > 1.047")

        return df, ["mjj", "dijet_deta", "dijet_dphi", "dijet_pt", "Jet_scaled_pt"]

def DijetSelection(*args, **kwargs):
    return lambda: DijetSelectionProducer()