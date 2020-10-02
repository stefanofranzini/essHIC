# essHIC
---

Here we introduce essHi-C, a method to isolate the specific, or essential component of Hi-C matrices from the aspecific portion of the spectrum that is compatible with random matrices. Systematic comparisons shows that essHi-C improves the clarity of the interaction patterns, enhances the robustness against sequencing depth, allows the unsupervised clustering of experiments in different cell lines and recovers the cell-cycle phasing of single-cells based on Hi-C data. Thus, essHi-C provides means for isolating significant biological and physical features from Hi-C matrices.

---

#### essHIC contains many useful modules to guide users throughout the analysis:

- [essHIC.makehic](https://github.com/stefanofranzini/essHIC/wiki/essHIC.make_hic): enables the user to process raw matrix data and their metadata to merge multiple sources into a single well ordered data repository, split chromosomes and obtain Observed over Expected matrices if needed.

- [essHIC.hic](https://github.com/stefanofranzini/essHIC/wiki/essHIC.hic): wrapper class for Hi-C matrices containing metadata along with the matrix of contact probabilities for the chosen experiment. It contains useful tools
to normalize and clean the matrix, compute its spectral properties, and process it to obtain its essential component. It also provides plotting functions to
obtain high quality pictures with minimal knowledge of the matplotlib library.

- [essHIC.ess](https://github.com/stefanofranzini/essHIC/wiki/essHIC.ess): tool to compute a dataset-wide distance matrix between all couples of experiments using the essential metric distance.

- [essHIC.dist](https://github.com/stefanofranzini/essHIC/wiki/essHIC.dist): a useful explorative tool to analyze distance matrices. It allows to perform hierarchical and spectral clustering on the dataset, to compute the ROC curves according to the cell-types provided in the metadata, and to perform multiscaling dimensional reduction (MDS). It also contains plotting tools to visualize
the results of all mentioned analyses.

More information is available on the package [wiki](https://github.com/stefanofranzini/essHIC/wiki).

---

## INSTALLING

essHIC is written in python, it has been tested in python2.7. Both the python3 language and the required packages need to be installed. To install the package through the python package manager, copy and paste the snippet below in your terminal:

```bash
pip install --upgrade essHIC 
```

Please notice that you may need to have administrator priviliges in order to be able to install the package. Using this method will take care of the dependencies.

Otherwise, you may simply clone this repository to your computer. To use the package in a python script you will need to link its local path; to do so write the snippet below in your python code:

```python
import sys

sys.path.append('path/to/essHIC')

import essHIC
```

For *essHIC* to function correctly you will need to install the required dependencies:

```bash
numpy
scipy
sklearn
matplotlib
```

---

## USAGE

To use *essHIC* in one of your python scripts import the package with

```python
import essHIC
```

for more information on how to use essHIC, please refer to the [tutorial](https://github.com/stefanofranzini/essHIC/wiki/tutorial) on the package [wiki](https://github.com/stefanofranzini/essHIC/wiki)


