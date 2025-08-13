# How to use pymovements in R

Install the R-package **reticulate** for interoperability between Python and R.
```r
 install.packages("reticulate")
```

Load the package.
```r
library(reticulate)
```

Install pymovements, if you haven't yet.
```r
py_install("pymovements")
```

Import pymovements as `pm`.
```r
pm <- import("pymovements")
```

Now pymovements should appear as `pm` under values in your environment

Access functions and data within python modules and classes via the `$` operator

To test, you can proceed with the "pymovements in 10 minutes" tutorial,
for example this is how you download the ToyDataset:
```r
dataset = pm$Dataset('ToyDataset', path='data/ToyDataset')
dataset$download()
```

Now let's load in the dataset into R and display the found files:
```r
dataset$load()
dataset$fileinfo
```


#### Related handy functions:

Load a python shell in R.
```r
repl_python()
```


Information about the version of Python currently being used by reticulate
```r
py_config()
```
