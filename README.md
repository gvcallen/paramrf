# ParamRF: Parametric Radio Frequency Modelling, Fitting and Sampling

**ParamRF**, or ``pmrf``, is an open-source radio frequency (RF) modelling framework. It provides an object-orientated means for frequency-domain modelling, fitting and sampling of RF models, with focus on circuit models.

| **ParamRF** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/paramrf/paramrf](https://github.com/paramrf/paramrf) |
| **Docs** | [paramrf.github.io/paramrf](https://paramrf.github.io/paramrf) |
| **Paper** | [ParamRF: A JAX-Native Framework for Declarative Circuit Modelling](https://doi.org/10.48550/arXiv.2510.15881) |

## Installation
ParamRF can be installed using pip directly from the GitHub page:

``
pip install git+https://github.com/paramrf/paramrf@main
``

### Optional dependencies
Several additional dependency packs can also be installed instead of manually installing the packages.

For Polychord fitting:

``
pip install 'paramrf[polychord] @ git+https://github.com/paramrf/paramrf@main'
``

For blackjax fitting:

``
pip install 'paramrf[blackjax] @ git+https://github.com/paramrf/paramrf@main'
``

## Citation

If you have used ParamRF for academic work, please cite the original [paper](https://doi.org/10.48550/arXiv.2510.15881):

> G.V.C. Allen, D.I.L. de Villiers, (2025). ParamRF: A JAX-native Framework for Declarative Circuit Modelling. arXiv, https://doi.org/10.48550/arXiv.2510.15881.

or with BibTeX:

```bibtex
@article{paramrf,
    doi = {10.48550/arXiv.2510.15881},
    url = {https://doi.org/10.48550/arXiv.2510.15881}, 
    year = {2025},
    month = {Oct},
    title = {ParamRF: A JAX-native Framework for Declarative Circuit Modelling}, 
    author = {Gary V. C. Allen and Dirk I. L. de Villiers},
    eprint = {2510.15881},
    archivePrefix = {arXiv},
    primaryClass = {cs.OH},
}
```

## Contributions

Since the library is in very early stages and likely contains several bugs, contributions are more than welcome by submitting a PR (especially for the docs!). However, before making any major changes, please open an issue so we can discuss the general intention of the contribution.