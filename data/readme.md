# Why not buy 2? data folder

## Introduction

This folder stores the data used for the implementation of *Why not buy 2?* targeted at the Montreal library system.

## Directory structure

 * cleaned: Contains by_titles.csv, a CSV containing columns of interest for the analysis, for rows with complete entries (i.e. no entries missing), and entries formatted (e.g. removal of text from numeric columns such as year).
 * processed:
  * features_labels_for_each_title.csv: Labels for each title in the dataset based on the lifetime number of borrows attributed to the author, publisher, or country,language, and book-type, as well as years since publication for each item.
  * expanded_regression_tree_parameters.jobib: Saved parameters from the random forest regressor used to estimate demand for items.
  * expanded_variable_dictionaries.jobib: Dictionaries linking feature level names to their values which are passed to the prediction function.
 * raw: Library catalogue data from Montréal's open data portal.

