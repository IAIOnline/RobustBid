# Robust autobidding for noisy conversion prediction models.
Code for anonimyzed submission on NeurIPS.

This repository contains code for running experiments comparing robust and non-robust autobidding approaches across three different setups: synthetic data, IPinYou dataset, and BAT.
The BAT experiment implementation builds upon the original [BAT autobidding benchmark](https://github.com/avito-tech/bat-autobidding-benchmark/)

## Repository Structure
```
.
├── Synthetic_and_IPinYou/
│   ├── BidderModel/        # Core bidding strategy implementations
│   ├── Utils/              # Helper functions and utilities
│   └── IPinYouData/        # Directory for IPinYou dataset files
│
└── BAT/
│   ├── simulator/
│   ├── model/             # Bidding models implementation
│   ├── simulation/        # BAT environment simulation
│   └── validation/        # Model validation tools
└── example_notebooks/     # Usage examples and experiments
└── data/                  # Directory for BAT data
```
Please, place the data files from [source](https://www.kaggle.com/datasets/lastsummer/ipinyou) into `Synthetic_and_IPinYou/IPinYouData/` for IPinYou experiment. And data files from [source](https://github.com/avito-tech/bat-autobidding-benchmark/) into BAT/data/ for the BAT experiment.

For reproducibility, please refer to the notebooks in the corresponding folders.
