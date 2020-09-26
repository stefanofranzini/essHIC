# essHIC
---

### A python3 package to analyze Hi-C matrices, enhance their specific patterns through spectral filtering and compute metric distances between couples of experiments.

---

#### essHIC contains many useful modules to guide users throughout the analysis:

- **makehic** : enables the user to process raw matrix data and their metadata to merge multiple sources into a single well ordered data repository, split chromosomes and obtain Observed over Expected matrices if needed.

- **hic** : wrapper class for hic matrices containing metadata along with the matrix of contact probabilities for the chosen experiment. It contains useful tools
to normalize and clean the matrix, compute its spectral properties, and process it to obtain its essential component. It also provides plotting functions to
obtain high quality pictures with minimal knowledge of the matplotlib library.

- **ess** : tool to compute a dataset-wide distance matrix between all couples of experiments using the essential metric distance.

- **dist**: a useful explorative tool to analyze distance matrices. It allows to perform hierarchical and spectral clustering on the dataset, to compute the ROC curves according to the cell-types provided in the metadata, and to perform multiscaling dimensional reduction (MDS). It also contains plotting tools to visualize
the results of all mentioned analyses.

---

## INSTALLING

to install the package, please 
