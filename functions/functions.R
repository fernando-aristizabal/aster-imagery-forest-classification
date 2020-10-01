###################################################################################################
######################### Contains Additional Functions Required ##################################
###################################################################################################

###################################################################################################
## install and load all packages

install_load <- function(pkgList) {
  
  for (i in pkgList) {
    if ((i %in% installed.packages()[,1]) == FALSE) {install.packages(i)}
    if (paste0("package:",i) %in% search() == FALSE) {library(i,character.only = T)}
  }  
  
  message("All necessary packages have been installed and loaded")
}

###################################################################################################


