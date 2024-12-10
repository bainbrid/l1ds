import os
from analysis_tools.utils import import_root, randomize
ROOT = import_root()

class SphericityProducer():
    def __init__(self, *args, **kwargs):
        ROOT.gInterpreter.Declare("""
            #include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
            #include <vector>
            #include <math.h>
            #include <cmath>

            using Vd = const ROOT::RVec<double>&;
            float get_sphericity(Vd calotower_eta, Vd calotower_phi, Vd calotower_et) {
                std::vector<math::XYZVector> event_tracks;
                for (size_t i = 0; i < calotower_eta.size(); i++) {
                    auto trk = math::XYZVector(0, 0, 0);
                    trk.SetXYZ(
                        calotower_et[i] * cos(calotower_phi[i]),
                        calotower_et[i] * sin(calotower_phi[i]),
                        calotower_et[i] * sinh(calotower_eta[i])
                    );
                    event_tracks.push_back(trk);
                }
                EventShapeVariables event_algo(event_tracks);
                return event_algo.sphericity();
            }
        
        """)


    def run(self, df):
        # df = df.Define("tower_sphericity", "get_sphericity_towers("
            # "calotower_eta, calotower_phi, calotower_et)")
        df = df.Define("tower_sphericity_et1", "get_sphericity("
            "calotower_eta[calotower_et >= 1], calotower_phi[calotower_et >= 1], calotower_et[calotower_et >= 1])")
        df = df.Define("tower_sphericity_et2", "get_sphericity("
            "calotower_eta[calotower_et >= 2], calotower_phi[calotower_et >= 2], calotower_et[calotower_et >= 2])")
        df = df.Define("tower_sphericity_et4", "get_sphericity("
            "calotower_eta[calotower_et >= 4], calotower_phi[calotower_et >= 4], calotower_et[calotower_et >= 4])")
        df = df.Define("tower_sphericity_et6", "get_sphericity("
            "calotower_eta[calotower_et >= 6], calotower_phi[calotower_et >= 6], calotower_et[calotower_et >= 6])")
        df = df.Define("l1jet_sphericity", "get_sphericity("
            "L1Jet_eta, L1Jet_phi, L1Jet_pt)")
        df = df.Define("ap_sphericity", "get_sphericity("
            "GenPart_eta[GenPart_pdgId == 999998], GenPart_phi[GenPart_pdgId == 999998], GenPart_pt[GenPart_pdgId == 999998])")
        df = df.Define("boosted_ap_sphericity", "get_sphericity("
            "BoostedDarkPhoton_eta, BoostedDarkPhoton_phi, BoostedDarkPhoton_pt)")
        return df, ["tower_sphericity",
            "tower_sphericity_et1", "tower_sphericity_et2", "tower_sphericity_et4", "tower_sphericity_et6",
            "l1jet_sphericity", "ap_sphericity", "boosted_ap_sphericity"]


def Sphericity(*args, **kwargs):
    return lambda: SphericityProducer(*args, **kwargs)
    

class BoostProducer():
    def __init__(*args, **kwargs):
        ROOT.gInterpreter.Declare("""
            #include <TLorentzVector.h>
            #include <TVector3.h>
            #include <vector>

            using Vd = const ROOT::RVec<double>&;
            using Vint = const ROOT::RVec<int>&;
            std::vector<double> get_boost_matrix(int nGenPart,
                    Vd GenPart_pt, Vd GenPart_eta, Vd GenPart_phi, Vd GenPart_mass,
                    Vint GenPart_genPartIdxMother, Vint GenPart_pdgId) {
                int higgs_i;
                for (size_t i = 0; i < nGenPart; i++) {
                    if (GenPart_pdgId[i] != 999999)
                        continue;
                    if (GenPart_pdgId[GenPart_genPartIdxMother[i]] == 25) {
                        higgs_i = GenPart_genPartIdxMother[i];
                        break;
                    }
                }
                TLorentzVector v;
                v.SetPtEtaPhiM(
                    GenPart_pt[higgs_i],
                    GenPart_eta[higgs_i],
                    GenPart_phi[higgs_i],
                    GenPart_mass[higgs_i]
                );
                auto boostvector = v.BoostVector();
                return {boostvector.X(), boostvector.Y(), boostvector.Z()};
            }
        """)

    def run(self, df):
        df = df.Define("boost_vector", "get_boost_matrix(nGenPart, "
            "GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass,"
            "GenPart_genPartIdxMother, GenPart_pdgId)")
        return df, []


def Boost(*args, **kwargs):
    return lambda: BoostProducer(*args, **kwargs)


class HiggsSelectorProducer():
    def __init__(*args, **kwargs):
        ROOT.gInterpreter.Declare("""
            #include <TLorentzVector.h>
            #include <TVector3.h>
            #include <vector>

            using Vd = const ROOT::RVec<double>&;
            using Vint = const ROOT::RVec<int>&;
            int get_higgs_index(int nGenPart, Vint GenPart_genPartIdxMother, Vint GenPart_pdgId) {
                for (size_t i = 0; i < nGenPart; i++) {
                    if (GenPart_pdgId[i] != 999999)
                        continue;
                    if (GenPart_pdgId[GenPart_genPartIdxMother[i]] == 25) {
                        return GenPart_genPartIdxMother[i];
                    }
                }
            }
        """)

    def run(self, df):
        df = df.Define("higgs_index", "get_higgs_index("
            "nGenPart, GenPart_genPartIdxMother, GenPart_pdgId)")
        return df, ["higgs_index"]


def HiggsSelector(*args, **kwargs):
    return lambda: HiggsSelectorProducer(*args, **kwargs)


class ParticleBoosterProducer():
    def __init__(self, *args, **kwargs):
        self.pt = kwargs.pop("pt")
        self.eta = kwargs.pop("eta")
        self.phi = kwargs.pop("phi")
        self.mass = kwargs.pop("mass")
        self.output_prefix = kwargs.pop("output_prefix")

        if not os.getenv("_boost"):
            os.environ["_boost"] = "_boost"

            ROOT.gInterpreter.Declare("""
                #include <TLorentzVector.h>
                #include <vector>

                using Vd = const ROOT::RVec<double>&;
                std::vector<ROOT::RVec<double>> boost_particle (
                    Vd pt, Vd eta, Vd phi, Vd mass,
                    std::vector<double> boost_vector
                ) {
                    TVector3 v1;
                    v1.SetX(boost_vector[0]);
                    v1.SetY(boost_vector[1]);
                    v1.SetZ(boost_vector[2]);

                    ROOT::RVec<double> boosted_pt(pt.size(), -1);
                    ROOT::RVec<double> boosted_eta(pt.size(), -1);
                    ROOT::RVec<double> boosted_phi(pt.size(), -1);

                    for (size_t i = 0; i < pt.size(); i++) {
                        TLorentzVector v;
                        v.SetPtEtaPhiM(pt[i], eta[i], phi[i], mass[i]);
                        v.Boost(-v1);
                        boosted_pt[i] = v.Pt();
                        boosted_eta[i] = v.Eta();
                        boosted_phi[i] = v.Phi();
                    }
                    return {boosted_pt, boosted_eta, boosted_phi};
                }
            """)

    def run(self, df):
        tmp = randomize("tmp")
        df = df.Define(tmp, f"boost_particle({self.pt}, {self.eta}, {self.phi}, {self.mass}, "
            "boost_vector)"
            ).Define(f"{self.output_prefix}_pt", f"{tmp}[0]"
            ).Define(f"{self.output_prefix}_eta", f"{tmp}[1]"
            ).Define(f"{self.output_prefix}_phi", f"{tmp}[2]"
        )
        return df, [f"{self.output_prefix}_pt", f"{self.output_prefix}_eta", f"{self.output_prefix}_phi"]
             
            
def ParticleBooster(*args, **kwargs):
    return lambda: ParticleBoosterProducer(*args, **kwargs)


class SingleParticleBoosterProducer():
    def __init__(self, *args, **kwargs):
        self.pt = kwargs.pop("pt")
        self.eta = kwargs.pop("eta")
        self.phi = kwargs.pop("phi")
        self.mass = kwargs.pop("mass")
        self.output_prefix = kwargs.pop("output_prefix")

        if not os.getenv("_singleboost"):
            os.environ["_singleboost"] = "_singleboost"

            ROOT.gInterpreter.Declare("""
                #include <TLorentzVector.h>
                #include <vector>

                using Vd = const ROOT::RVec<double>&;
                std::vector<double> single_boost_particle (
                    double pt, double eta, double phi, double mass,
                    std::vector<double> boost_vector
                ) {
                    TVector3 v1;
                    v1.SetX(boost_vector[0]);
                    v1.SetY(boost_vector[1]);
                    v1.SetZ(boost_vector[2]);

                    TLorentzVector v;
                    v.SetPtEtaPhiM(pt, eta, phi, mass);
                    v.Boost(-v1);
                    return {v.Pt(), v.Eta(),  v.Phi()};
                }
            """)

    def run(self, df):
        tmp = randomize("tmp")
        df = df.Define(tmp, f"single_boost_particle({self.pt}, {self.eta}, {self.phi}, {self.mass}, "
            "boost_vector)"
            ).Define(f"{self.output_prefix}_pt", f"{tmp}[0]"
            ).Define(f"{self.output_prefix}_eta", f"{tmp}[1]"
            ).Define(f"{self.output_prefix}_phi", f"{tmp}[2]"
        )
        return df, [f"{self.output_prefix}_pt", f"{self.output_prefix}_eta", f"{self.output_prefix}_phi"]
             
            
def SingleParticleBooster(*args, **kwargs):
    return lambda: SingleParticleBoosterProducer(*args, **kwargs)


class MomentumMakerProducer():
    def __init__(self, *args, **kwargs):
        self.pt = kwargs.pop("pt")
        self.eta = kwargs.pop("eta")
        self.phi = kwargs.pop("phi")
        self.mass = kwargs.pop("mass")
        self.output_prefix = kwargs.pop("output_prefix")

        if not os.getenv("_momentum"):
            os.environ["_momentum"] = "_momentum"
            ROOT.gInterpreter.Declare("""
                #include <TLorentzVector.h>
                #include <vector>
                std::vector<double> get_particle_momentum (
                    double pt, double eta, double phi, double mass
                ) {
                    TLorentzVector v;
                    v.SetPtEtaPhiM(pt, eta, phi, mass);
                    return {v.Px(), v.Py(), v.Pz(), v.E()};
                }
            """)

    def run(self, df):
        branches = [f"{self.output_prefix}_{elem}" for elem in ["px", "py", "pz", "e"]]
        tmp = randomize("tmp")
        df = df.Define(tmp, f"get_particle_momentum({self.pt}, {self.eta}, {self.phi}, {self.mass})"
            ).Define(f"{self.output_prefix}_px", f"{tmp}[0]"
            ).Define(f"{self.output_prefix}_py", f"{tmp}[1]"
            ).Define(f"{self.output_prefix}_pz", f"{tmp}[2]"
            ).Define(f"{self.output_prefix}_e", f"{tmp}[3]"
        )
        return df, branches


def MomentumMaker(*args, **kwargs):
    return lambda: MomentumMakerProducer(*args, **kwargs)
