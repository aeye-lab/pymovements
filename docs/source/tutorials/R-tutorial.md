How to use pymovements in R:

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

Access functions and data within python modules and classes via the $ operator

To test, you can proceed with the "Working with Datasets" tutorial, for example download the ToyDataset
```r
dataset = pm$datasets$ToyDataset(root='testdata', download=TRUE)
```



### Related handy functions:

Load a python shell in R.
```r
repl_python()
```


Information about the version of Python currently being used by reticulate
```r
py_config()
```
