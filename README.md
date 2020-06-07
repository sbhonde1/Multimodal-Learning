# Installation
The Multimodal Learning code requires
* Python 3.6 or higher
* PyTorch 1.0 or higher

and the requirements highlighted in [requirements.txt](requirements.txt) (for Anaconda)

This code was executed on a single GPU. Therefore, I strongly recommend to adapt this code according to the configuration of your cluster.


# References
- [model source: kinetics i3d pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/src/i3dpt.py)
- [Senz3D dataset](https://lttm.dei.unipd.it/downloads/gesture/)
- BibTeX reference to cite, if you use it:
```bibtex
@misc{abavisani2018improving,
    title={Improving the Performance of Unimodal Dynamic Hand-Gesture Recognition with Multimodal Training},
    author={Mahdi Abavisani and Hamid Reza Vaezi Joze and Vishal M. Patel},
    year={2018},
    eprint={1812.06145},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
- Bibtex reference for Senz3D dataset
```bibtex
@inproceedings {stag.20151288,
booktitle = {Smart Tools and Apps for Graphics - Eurographics Italian Chapter Conference},
editor = {Andrea Giachetti and Silvia Biasotti and Marco Tarini},
title = {{Exploiting Silhouette Descriptors and Synthetic Data for Hand Gesture Recognition}},
author = {Memo, Alvise and Minto, Ludovico and Zanuttigh, Pietro},
year = {2015},
publisher = {The Eurographics Association},
ISBN = {978-3-905674-97-2},
DOI = {10.2312/stag.20151288}
}

@article{Memo_2016,
	doi = {10.1007/s11042-016-4223-3},
	url = {https://doi.org/10.1007%2Fs11042-016-4223-3},
	year = 2016,
	month = {dec},
	publisher = {Springer Science and Business Media {LLC}},
	volume = {77},
	number = {1},
	pages = {27--53},
	author = {Alvise Memo and Pietro Zanuttigh},
	title = {Head-mounted gesture controlled interface for human-computer interaction},
	journal = {Multimedia Tools and Applications}
}
```
