CompSensePack Package
=====================

Main Modules
------------

Modules in the `CompSensePack` **main directory**: *SL0* recovery method module, *measurement matrix* generation module, utilities to run the program and to visualize the results of the various tests are offered in *utils* and *eval*. The *compressedSensing* class is a high level object that it's used to call the various functions and methods that components of the library, it guarantees modularity, felxibility, readability.

.. toctree::
   :maxdepth: 1

   SL0
   utils
   eval
   measurement_matrix
   comp_sense_class

Dictionaries Subpackage
-----------------------

The subpackage contains all the routine and utilities to generate both the *DCT fixed dictionary* and to use the *KSVD* and *MOD* algorithms.

.. toctree::
   :maxdepth: 1

   dictionaries/index
