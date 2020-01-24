# Why not buy 2? data folder

## Introduction

This folder stores the data used for the implementation of *Why not buy 2?* targeted at the Montreal library system.

## Directory structure

 * cleaned: Contains by_titles.csv, a CSV containing columns of interest for the analysis, for rows with complete entries (i.e. no entries missing), and entries formatted (e.g. removal of text from numeric columns such as year).
 * processed:
  * cluster_labels_for_each_title.csv: Labels for each title in the dataset based on the results of the clustering algorithm, as well as numeric output for non-clustered variables.
  * cluster_labels_for_each_variable.csv: Mapping of numeric cluster labels onto string values for each clustered variable.
  * OHE_parameters.jobib: Saved parameters for OneHotEncoding of categorical variables.
  * SGD_model_parameters.jobib: Saved parameters for SGD regression on training set.
 * raw: Library catalogue data from Montréal's open data portal.
 * test:

## Conventions used

To be added.

## For more information

To be added.
