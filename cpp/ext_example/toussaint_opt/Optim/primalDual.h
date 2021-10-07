/*  ------------------------------------------------------------------
    Copyright (c) 2019 Marc Toussaint
    email: marc.toussaint@informatik.uni-stuttgart.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

#include "optimization.h"
#include "lagrangian.h"

struct PrimalDualProblem : ScalarFunction {
  LagrangianProblem L;

  //duality gap parameter (log barrier parameter) of the primal dual equation system
  double mu;

  uint n_eq=0, n_ineq=0;
  arr x_lambda; //last evaluation
  double dualityMeasure=1.;
  bool primalFeasible=false;

  PrimalDualProblem(const arr& x, ConstrainedProblem& P, OptOptions opt=NOOPT, arr& lambdaInit=NoArr);

  double primalDual(arr& r, arr& R, const arr& x); ///< CORE METHOD: the unconstrained scalar function F

  void updateMu();
};

//==============================================================================
//
// Solvers
//

struct OptPrimalDual {
  arr& x;
  PrimalDualProblem PD;
  OptNewton newton;
  OptOptions opt;
  uint its=0;
  ofstream* fil=nullptr;

  OptPrimalDual(arr& x, arr& dual, ConstrainedProblem& P, int verbose=-1, OptOptions opt=NOOPT);
  ~OptPrimalDual();
  uint run(uint maxIt=1000);
};
