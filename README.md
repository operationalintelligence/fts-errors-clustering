# Clustering of FTS error messages
This project contains attempts for clustering FTS data transfer errors. The idea is to leverage an *unsupervised* approach to avoid prior expectations about what the error categories are and, hence, in the hope of discovering new failure patterns.

## Structure
The repository is mainly organised into 3 folders:

 - `code/`: contains python modules useful to run the analysis
 - `notebooks/`: contains jupyter notebooks used both actually to perform the analysis and as tutorials of different bits of the pipeline
 - `references/`: contains references that support the techniques adopted

## Installation
To use the code, simply open a terminal and run:

```r
#clone the repository
git clone https://github.com/operationalintelligence/rucio-log-clustering.git

#enter the folder
cd rucio-log-clustering

#create a new conda environment with requirements (Anaconda pre-installed)
conda create --name <ENV_NAME> --file requirements.txt

#setup jupyter notebook
conda activate <ENV_NAME>
conda install -c conda-forge jupyter-notebook
```

**Note:** You may have to add some channels to conda (`conda config --append channels new_channel`) in order to get all the packages in *requirements.txt* directly.
For completeness check the list of channels required:

```r
conda config --get channels
--add channels 'intel'   # lowest priority
--add channels 'defaults'
--add channels 'conda-forge'   # highest priority
```

## Usage
Currently, 4 notebooks are available to perform and end-to-end analysis:
 - 010_FTS_data_extraction to extract FTS data directly from Hadoop
 - 020_NLP_on_FTS to apply word2vec language model on the selected data
 - 030_Clustering_on_FTS-unique_messages to run K-Means algorithm of the word2vec representation of messages
 - 040_Visualisation_clusters to explore results, both at first glance and more in depth

## Sample data
The sample data can be obtained directly from Panos' framework running the python code *fetch_issues.py* from the root of the repository:
```r
python code/fetch_issues.py
```

Also, it is now possible to fetch data directly from Hadoop as shown in the *010* notebook.

##### Maintainers:
<sub>Luca Clissa</sub>
