# essHIC
---

### A python3 package to analyze Hi-C matrices, enhance their specific patterns through spectral filtering and compute metric distances between couples of experiments.

---

#### essHIC contains many useful modules to guide users throughout the analysis:

- **essHIC.makehic** : enables the user to process raw matrix data and their metadata to merge multiple sources into a single well ordered data repository, split chromosomes and obtain Observed over Expected matrices if needed.

- **essHIC.hic** : wrapper class for Hi-C matrices containing metadata along with the matrix of contact probabilities for the chosen experiment. It contains useful tools
to normalize and clean the matrix, compute its spectral properties, and process it to obtain its essential component. It also provides plotting functions to
obtain high quality pictures with minimal knowledge of the matplotlib library.

- **essHIC.ess** : tool to compute a dataset-wide distance matrix between all couples of experiments using the essential metric distance.

- **essHIC.dist**: a useful explorative tool to analyze distance matrices. It allows to perform hierarchical and spectral clustering on the dataset, to compute the ROC curves according to the cell-types provided in the metadata, and to perform multiscaling dimensional reduction (MDS). It also contains plotting tools to visualize
the results of all mentioned analyses.

---

## INSTALLING

essHIC is written in python3, it has been tested in python3.5.2. Both the python3 language and the required packages need to be installed. To install the package through the python3 package manager, copy and paste the snippet below in your terminal:

```bash
pip install --upgrade essHIC 
```

Please notice that you may need to have administrator priviliges in order to be able to install the package. Using this method will take care of the dependencies.

Otherwise, you may simply clone this repository to your computer. To use the package in a python3 script you will need to link its local path; to do so write the snippet below in your python3 code:

```python3
import sys

sys.path.append('path/to/essHIC')

import essHIC
```

For the package to function correctly you will need to install the required dependencies:

```bash
numpy
scipy
sklearn
matplotlib
```

---

## USAGE

