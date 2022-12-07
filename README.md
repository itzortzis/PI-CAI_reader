# PI-CAI_reader
PI-CAI_reader is a simple tool that creates dataloaders for the T2W MRI series and the corresponding gland masks of the PI-CAI prostate dataset


Files and description:
----------------------
```
PI-CAI_reader
│    README.md
│
└─── utils
│    │   pre_processing.py  # File containing the functions for the dataset creation
│    │   dataloader.py # Custom torch dataset class
│   
│   
└─── notebook
     │   PI-CAI_reader.ipynb # Colab/Jupyter notebook testbench
```


The PI-CAI Challenge dataset:
-----------------------------

A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge", DOI: 10.5281/zenodo.6522364
