from analysis_tools.utils import import_root
ROOT = import_root()

class EtaPhiEtFromHardwareProducer():
    def __init__(self, *args, **kwargs):
        ROOT.gInterpreter.Declare("""
            using Vd = const ROOT::RVec<double>&;
            ROOT::RVec<double> setPhiRange(Vd phi) {
                ROOT::RVec<double> new_phi(phi.size(), 999);
                for (size_t i = 0; i < phi.size(); i++) {
                    new_phi[i] = phi[i] >= 3.14159 ? phi[i] - 2. * 3.14159 : phi[i];
                }
                return new_phi;
            }
            
        """)
    
    def run(self, df):
        df = df.Define("calotower_eta", "L1EmulCaloTower_ieta * 0.087")
        df = df.Define("tmp", "L1EmulCaloTower_iphi * 3.14159 / 36.")
        df = df.Define("calotower_phi", "setPhiRange(tmp)")
        df = df.Define("calotower_et", "L1EmulCaloTower_iet / 2.")
    
        return df, ["calotower_eta", "calotower_phi", "calotower_et"]


def EtaPhiEtFromHardware(*args, **kwargs):
    return lambda: EtaPhiEtFromHardwareProducer()
