from analysis_tools import ObjectCollection, Category, Process, Dataset, Feature, Systematic
from analysis_tools.utils import DotDict
from analysis_tools.utils import join_root_selection as jrs
from plotting_tools import Label
from collections import OrderedDict

from cmt.config.base_config import Config as cmt_config
from cmt.base_tasks.base import Task


class Config(cmt_config):
    def add_categories(self, **kwargs):
        categories = [
            Category("base", "base", selection="event >= 0"),
            Category("dum", "Event 1", selection="event == 1"),
        ]
        return ObjectCollection(categories)

    def add_processes(self):
        processes = [
            Process("suep", Label("SUEP"), color=(0, 0, 0), isSignal=True),
            Process("qcd", Label("QCD (p_{T} 15-7000)"), color=(255, 0, 0)),
        ]

        process_group_names = {
            "default": [
                "qcd",
                "suep",
                # "data_tau",
                # "dy_high",
                # "tt_dl",
                # "data",
                # "background",
            ],
        }

        process_training_names = {}

        # adding reweighed processes
        processes = ObjectCollection(processes)

        return processes, process_group_names, process_training_names


    def add_datasets(self):

        datasets = [
            Dataset("suep",
                folder="/vols/cms/jleonhol/l1scouting/samples/2023/suep/",
                process=self.processes.get("suep"),
                xs=1.,
                tags=["ul"]),
            Dataset("suep_unique",
                folder="/vols/cms/jleonhol/l1scouting/samples/2023/suep_unique/",
                process=self.processes.get("suep"),
                xs=1.,
                tags=["ul"]),
            Dataset("qcd",
                folder="/vols/cms/jleonhol/l1scouting/samples/2023/QCD_PT-15to7000/",
                process=self.processes.get("qcd"),
                xs=1.,
                tags=["ul"]),
        ]
        return ObjectCollection(datasets)

    def add_features(self):
        from config.features import features            
        return ObjectCollection(features)

    def add_weights(self):
        weights = DotDict()
        weights.default = "1"

        weights.total_events_weights = ["genWeight"]
        # weights.total_events_weights = ["genWeight"]
        # weights.total_events_weights = ["1"]

        weights.base = ["genWeight"]  # others needed
        # weights.base = ["1"]  # others needed

        for category in self.categories:
            weights[category.name] = weights.base

        return weights

    def add_systematics(self):
        systematics = [

        ]
        return ObjectCollection(systematics)

    def add_default_module_files(self):
        defaults = {}
        # defaults["PreprocessRDF"] = "modules"
        # defaults["PreCounter"] = "weights"
        return defaults

    # other methods
    def get_norm_systematics(self, processes_datasets, region):
        """
        Method to extract all normalization systematics from the KLUB files.
        It considers the processes given by the process_group_name and their parents.
        """
        # systematics
        systematics = {}
        all_signal_names = []
        all_background_names = []
        for p in self.processes:
            if p.isSignal:
                all_signal_names.append(p.get_aux("llr_name")
                    if p.get_aux("llr_name", None) else p.name)
            elif not p.isData:
                all_background_names.append(p.get_aux("llr_name")
                    if p.get_aux("llr_name", None) else p.name)

        from cmt.analysis.systReader import systReader
        syst_folder = "config/systematics/"
        filename = f"systematics_{self.year}.cfg"
        if self.get_aux("isUL", False):
            filename = f"systematics_UL{str(self.year)[2:]}.cfg"
        syst = systReader(Task.retrieve_file(self, syst_folder + filename),
            all_signal_names, all_background_names, None)
        syst.writeOutput(False)
        syst.verbose(False)
        syst.writeSystematics()
        for isy, syst_name in enumerate(syst.SystNames):
            if "CMS_scale_t" in syst.SystNames[isy] or "CMS_scale_j" in syst.SystNames[isy]:
                continue
            for process in processes_datasets:
                original_process = process
                found = False
                while True:
                    process_name = (process.get_aux("llr_name")
                        if process.get_aux("llr_name", None) else process.name)
                    if process_name in syst.SystProcesses[isy]:
                        iproc = syst.SystProcesses[isy].index(process_name)
                        systVal = syst.SystValues[isy][iproc]
                        if syst_name not in systematics:
                            systematics[syst_name] = {}
                        systematics[syst_name][original_process.name] = eval(systVal)
                        found = True
                        break
                    elif process.parent_process:
                        process=self.processes.get(process.parent_process)
                    else:
                        break
                if not found:
                    for children_process in self.get_children_from_process(original_process.name):
                        if children_process.name in syst.SystProcesses[isy]:
                            if syst_name not in systematics:
                                systematics[syst_name] = {}
                            iproc = syst.SystProcesses[isy].index(children_process.name)
                            systVal = syst.SystValues[isy][iproc]
                            systematics[syst_name][original_process.name] = eval(systVal)
                            break
        return systematics


# config = Config("base", year=2018, ecm=13, lumi_pb=59741)
config = Config("suep_2023", year=2023, ecm=13.6, lumi_pb=1000)
