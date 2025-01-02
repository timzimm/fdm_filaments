### Summary & Attribution
Code to reproduce the plots in [arXiv:2412.10829](https://arxiv.org/abs/2412.10829).
We kindly ask you to cite this work as:
```
@article{Zimmermann:fdm-filaments,
        author = {{Zimmermann}, Tim and {Marsh}, David J.~E. and {Winther}, Hans A. and {Shen}, Sijing},
        title = "{Interference in Fuzzy Dark Matter Filaments: Idealised Models and Statistics}",
         year = 2024,
        month = dec,
          doi = {10.48550/arXiv.2412.10829},
       eprint = {2412.10829},
 primaryClass = {astro-ph.CO},
}
```

The analysis relies on our reconstruction tool 
[fuzzylli](https://github.com/timzimm/fuzzylli).

### What's in the Box
```bash
.
├── pyproject.toml
├── setup.cfg
├── README.md
├── LICENSE
├── populate_box.sh
├── render_box.sh
├── rotate_box.sh
├── powerspectrum_from_box.sh                           
├── parameters                                 
│   ├── single_cylinder.yaml                           # 8 GByte on disk
│   ├── L_2.00_N_2048_Ncyl_8_m22_1.00_aLASSO_seed_20.yaml                           # 192 GByte on disk
├── notebooks
│   ├── paper_plots.ipynb
│   ├── ellipsoidal_collapse.ipynb
└── fdmfilaments
    ├── compute_powerspectrum_from_box.py
    ├── correlation_functions_from_box.py
    ├── cosmology.py
    └── ellipsoidal_collapse.py
    └── generate_cylinder_gas.py
    └── generate_density.py
    └── io_utils.py
    └── parameters_from_ellipsoidal_collapse.py
    └── render_box.py
    └── rotate_box.py
```

### How to Install

```bash
$ git clone git@github.com:timzimm/fdmfilaments.git
$ cd fdmfilaments
$ pip install -e .
```

### How to Reproduce the Plots
Execute the jupyter notebooks and follow the instructions therein

**NOTE:** Be aware that depending on the runtime (hyper)parameters, most notably
`NPROC` and `m22`, peak memory consumption can be O(TB).

### Contributors
Tim Zimmermann  
David J.E. Marsh  
Keir Rogers  
Hans  A. Winther  
Sijing Shen  

### Acknowledgement
![eu](https://github.com/timzimm/boson_dsph/blob/94c8984fca269edb8b5a47ca43b346f07e80e1cc/images/eu_acknowledgement_compsci_3.png#gh-light-mode-only)
![eu](https://github.com/timzimm/boson_dsph/blob/94c8984fca269edb8b5a47ca43b346f07e80e1cc/images/eu_acknowledgement_compsci_3_white.png#gh-dark-mode-only)
