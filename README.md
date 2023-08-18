```
              %++   +%*+
            +#@@&  +#@@@%
           +#@&@@ +#@&@@#                                              **   *
          +#@% #@%#@&+@@%                 %%******+                   +@&  &*       +++++++++
         +##*  %###& +@@+     +*%%%*     &@%%&#@#&%+      *%%%%%&&+   *@+ +#       *@@##&%%%%+      %*   *%+
        +@@*   +&@#  *@#    *##&**##+   &@&@@@#%+        *@#%%%%*+    ##%%#&      %##@###&%        &@@+ &@&
       +@#*     +%   *@&   *@%    *@%  &#+  +**%#%      %#*          *@+ *@+    *#@#***%%%*       &@##&&@%
      +%              %+   *#&%%%##%   *+              *#+           &*  &%    *@@###&&%**      *#@@+#@@*
     ##                      +***+                    +*             +           +            +#@@@% +*+
     *+                                                                                     +%@@@#*
                      ++*%%&&##@@@@@@@@####&&%%*+++**%%%%%*                               +&@@@#%
                +*%&#@@@@@#&%%**++++++****%&&##@@@@@@@@@@@@#*                          +%#@@@&+
            *%#@@@@@#&&%*                     +#@@@@#&&&@@@@#                      +*&#@@#&*
        +&#@@@@#&**+                          #@@@@*    &@@@@%*+            ++**%&#@@@#%+
      +%@@@@#%+                               #@@@@@##&#@@@@##@@@#########@@@@@@#&&*+
    +%@@@@&*                                  +&@@@@@@@@@@#*   +****%%%%%%***++
   *#@@@&+                                       +**%%%%*+
  +@@@@*
  +&#+
```


# morphen
Collection of functions for astronomical image analysis and processing. 
These tools involves pure python, but also are integrated with CASA 
(https://casa.nrao.edu/) in order to use common `casatasks` as well `wsclean` -- a 
code for fast interferometric imaging (https://wsclean.readthedocs.io/en/latest/).

*NOTES: This readme file is under development. I am also currently adding basic usages 
to Jupyter notebooks guides. A major update and core functionalities will be provided 
September/2023 with the release of the paper.

# Image deconvolution Using `wsclean`
Directory [```imaging/```](imaging/) contains a python script called 
[```imaging/imaging_with_wsclean_v3.py```](imaging/imaging_with_wsclean_v3.py) which is just a support code for easy use to call wsclean on the command line. 
See the intructions file of how to use it: [```imaging/wsclean_imaging.md```](imaging/wsclean_imaging.md)


# Image Decomposition
Folder [```image_decomposition/```](image_decomposition/) contains sets of functions to perform multi-purpose image decomposition. It uses the LMFIT package with object-oriented implementation, easy to use and manageable number of n-components. 

# Image Analysis (focused to radio interferometric images)
## Jupyter Notebook
For all analysis tools, we provide a library file called `libs.py` which contains all relevant functions to be used easily with a Jupyter Notebook called `radio_morphen.ipynb`.
## Comandline application.
*COMING OUT SOON* 

We are porting the iterative Jupyter Notebook to a command line interface, 
in which the code can be used easily in a single run, 
performing the most common tasks for image processing and providing 
outputs plots and measurements.


## Basic Image Analysis
### `level_statistics()`
Function to compute basic image statistics, such as total flux density and
uncertain flux. The flux uncertainty is computed as being the flux from 
 `5*rms` to `3*rms`.   





 







