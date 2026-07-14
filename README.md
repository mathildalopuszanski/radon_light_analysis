# radon_light_analysis

Files that were used for the analysis of radon injection in the 2x2 Demonstrator to find BiPo coincidences in the light data. The BiPoSelection.py file has functions that are frequently used. The process_charge_and_light.py can be used to find BiPo coincidences and the corresponding cluster and store them as a file. 

Everything should be run on NERSC to access the flowed files. Instruction to setup NERSC can be found here https://docs.nersc.gov/getting-started/. To run the notebooks and python files, some (basic) python packages are needed (numpy, matplotlib etc.) which you can install on nersc in a virtual environment. 

I learned how to access flowed files from these tutorials https://gitlab.nikhef.nl/mnuland/dune/-/blob/tsonius-main-patch-78889/Learn.ipynb?ref_type=heads.
These tutorials are also an overview of how to use the light files https://github.com/jvmead/lrs_tutorials. 

ndlar-flow is not necessary to run these files if you use already flowed files. Otherwise instructions on how to setup ndlar-flow can be found here https://github.com/DUNE/ndlar_flow . 
