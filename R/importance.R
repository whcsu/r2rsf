# -------------------------------------------------------------------------------
#   This file is part of Ranger.
#
# Ranger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ranger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ranger. If not, see <http://www.gnu.org/licenses/>.
#
# Written by:
#
#   Marvin N. Wright
# Institut fuer Medizinische Biometrie und Statistik
# Universitaet zu Luebeck
# Ratzeburger Allee 160
# 23562 Luebeck
# Germany
#
# http://www.imbs-luebeck.de
# wright@imbs.uni-luebeck.de
# -------------------------------------------------------------------------------

##' @export
importance <- function(x, ...)  UseMethod("importance")

##' Extract variable importance of ranger object.
##'
##'
##' @title ranger variable importance
##' @param x ranger object.
##' @param ... Further arguments passed to or from other methods.
##' @return Variable importance measures.
##' @seealso \code{\link{ranger}}
##' @author Marvin N. Wright
##' @aliases importance
##' @export 
importance.rangernew <- function(x, ...) {
  if (class(x) != "rangernew") {
    stop("Object ist no ranger object.")
  }
  if (is.null(x$variable.importance) | length(x$variable.importance) < 1) {
    stop("No variable importance found. Please use 'importance' option when growing the forest.")
  }
  return(x$variable.importance)
}

##' Compute variable importance with p-values.
##'
##'
##' @title ranger variable importance p-values
##' @param x ranger or holdoutRF object.
##' @param method Method to compute p-values. Use "janitza" for the method by Janitza et al. (2015) or "altmann" for the non-parametric method by Altmann et al. (2010).
##' @param num.permutations Number of permutations. Used in the "altmann" method only.
##' @param formula Object of class formula or character describing the model to fit. Used in the "altmann" method only.
##' @param data Training data of class data.frame or matrix. Used in the "altmann" method only.
##' @param ... Further arguments passed to ranger(). Used in the "altmann" method only.
##' @return Variable importance and p-values.
##' @seealso \code{\link{ranger}}
##' @author Marvin N. Wright
##' @references
##'   Janitza, S., Celik, E. & Boulesteix, A.-L., (2015). A computationally fast variable importance test for random forest for high dimensional data, Technical Report 185, University of Munich, \url{https://epub.ub.uni-muenchen.de/25587}. \cr
##'   Altmann, A., Tolosi, L., Sander, O. & Lengauer, T. (2010). Permutation importance: a corrected feature importance measure, Bioinformatics 26(10):1340-1347.
##' @export 
importance_pvalues <- function(x, method = c("janitza", "altmann"), num.permutations = 100, formula = NULL, data = NULL, ...) {
  if (class(x) != "ranger" & class(x) != "holdoutRF") {
    stop("Object is no ranger or holdoutRF object.")
  }
  if (x$importance.mode == "none" | is.null(x$variable.importance) | length(x$variable.importance) < 1) {
    stop("No variable importance found. Please use 'importance' option when growing the forest.")
  }

  if (method == "janitza") {
    if (x$importance.mode == "impurity") {
      stop("Impurity variable importance found. Please use (hold-out) permutation importance to use this method.")
    }
    if (class(x) != "holdoutRF" & x$importance.mode == "permutation") {
      warning("Permutation variable importance found, inaccurate p-values. Please use hold-out permutation importance to use this method.")
    }
    if (x$treetype != "Classification") {
      warning("This method is tested for classification only, use with care.")
    }
    
    ## Mirrored VIMP
    m1 <- x$variable.importance[x$variable.importance < 0]
    m2 <- x$variable.importance[x$variable.importance == 0]
    vimp <- c(m1, -m1, m2)
    
    ## Compute p-value
    pval <- 1 - ecdf(vimp)(x$variable.importance)
    
    ## TODO: 100 ok? increase? 
    if (length(m1) == 0) {
      stop("No negative importance values found. Consider the 'altmann' approach.")
    }
    if (length(m1) < 100) {
      warning("Only few negative importance values found, inaccurate p-values. Consider the 'altmann' approach.")
    }
  } else if (method == "altmann") {
    if (class(x) != "ranger") {
      stop("Altmann method not available for holdoutRF objects.")
    }
    if (is.null(formula) | is.null(data)) {
      stop("Formula and data required for the 'altmann' method.")
    }
    
    ## Permute and compute importance again
    dependent.variable.name <- all.vars(formula)[1]
    vimp <- replicate(num.permutations, {
      dat <- data
      dat[, dependent.variable.name] <- sample(dat[, dependent.variable.name])
      ranger(formula, dat, num.trees = x$num.trees, mtry = x$mtry, min.node.size = x$min.node.size, 
             importance = x$importance.mode, ...)$variable.importance
    })
    
    ## Compute p-value
    pval <- sapply(1:nrow(vimp), function(i) {
      1 - ecdf(vimp[i, ])(x$variable.importance[i])
    })
    
  } else {
    stop("Unknown p-value method. Available methods are: 'janitza' and 'altmann'.")
  }
  
  ## Return VIMP and p-values
  res <- cbind(x$variable.importance, pval)
  colnames(res) <- c("importance", "pvalue")
  return(res)
}
