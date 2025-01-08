# l1ds
This is a user code for the configuration of analyses using L1 scouting used with the [main code](https://gitlab.cern.ch/cms-phys-ciemat/nanoaod_base_analysis.git) of the NanoAOD-base-analysis, which aims to process NanoAOD datasets, allowing to generate different root files, histograms and plots with the desired selection of events, variables and branches.

## User guide:

Information about the code, how to install it, setting a configuration to use it and more useful information about this framework cand be found [here](https://nanoaod-base-analysis.readthedocs.io).

## Installation

```
git clone https://gitlab.cern.ch/ic-l1ds/l1ds.git
cd l1ds
git clone https://gitlab.cern.ch/cms-phys-ciemat/nanoaod_base_analysis.git --branch py3 nanoaod_base_analysis/
source setup.sh
law index --verbose #to do only after installation or including a new task
```
