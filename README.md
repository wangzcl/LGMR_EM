# Eigen Microstates of LGMR Mid- to late Holocene Global SST

## Usage

1. Clone the project repo
2. `cd` the project root directory
3. Create a Python virtual environment
4. install project with `pip install -e .`
5. download [LGMR data](data/LGMR/README.md)
6. Run scripts or notebooks in the virtual environment.

## Structure

`src` defined some source code for data processing, analysis and visulization,
including two packages which can be installed by `pip insatll -e .`: `lgmr_em` for some data processing and analysis utils,
and `eigen_microstates` for implementaion of eigen microstates.

`notebooks/em_decomposition` is the main notebook for data analysis and visualization.

`scripts` are essentially some extra scripts for plotting some figures in supplementary info.