# PI-CAI_reader
PI-CAI_reader is a simple tool that creates dataloaders for the T2W MRI series and the corresponding gland masks of the PI-CAI prostate dataset


Required libraries:
-------------------

    os, cv2, numpy, matplotlib, torch, medpy.io
    
    
Files and description:
----------------------
```
PI-CAI_reader
│    README.md
│
└─── pi_cai_reader_core
│    │   pre_processing.py  # File containing the functions for the dataset creation
│    │   dataloader.py # Custom torch dataset class
│   
│    PI-CAI_reader.ipynb # Colab/Jupyter notebook testbench
```

## Installation

The INBreast_XML_parser can be cloned from here or it can be installed using Python pip tool

- Option 1: Clone the repository and see the demo files in python and notebook folders
- Option 2:
  - Install tool using ```pip3 install git+https://github.com/itzortzis/PI-CAI_reader.git```
  - Import the needed components ```import pi_cai_reader_core.pre_processing``` and ```import pi_cai_reader_core.dataloader```
  

## The PI-CAI Challenge dataset:
--------------------------------

A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge", DOI: 10.5281/zenodo.6522364
