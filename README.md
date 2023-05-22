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

*NOTE: This readme file is under development.*

# Image deconvolution Using `wsclean`
Directory [```imaging/```](imaging/) contains a python script called 
[```imaging/imaging_with_wsclean_v3.py```](imaging/imaging_with_wsclean_v3.py) which is just a support code for easy use to call wsclean on the command line. 
See the intructions file of how to use it: [```imaging/wsclean_imaging.md```](imaging/wsclean_imaging.md)


## Image Decomposition
Folder [```image_decomposition/```](image_decomposition/) contains sets of functions to perform multi-purpose image decomposition. It uses the LMFIT package with object-oriented implementation, easy to use and manageable number of n-components. 

## Image Analysis (focused to radio interferometric images)



