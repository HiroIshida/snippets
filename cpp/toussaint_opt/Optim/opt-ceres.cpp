#include "opt-ceres.h"

#ifdef RAI_CERES

#undef LOG
#undef CHECK
#undef CHECK_EQ
#undef CHECK_LE
#undef CHECK_GE
#include <ceres/problem.h>
#include <ceres/cost_function.h>
#include <ceres/solver.h>
#include <ceres/loss_function.h>

class Conv_CostFunction : public ceres::CostFunction {
  std::shared_ptr<MathematicalProgram> MP;
  uint feature_id;
  uint featureDim;
  intA varIds;
  uintA varDims;
  uint varTotalDim;

public:
  Conv_CostFunction(const ptr<MathematicalProgram>& _MP,
                    uint _feature_id,
                    const uintA& variableDimensions,
                    const uintA& featureDimensions,
                    const intAA& featureVariables);

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const;
};

Conv_CostFunction::Conv_CostFunction(const ptr<MathematicalProgram>& _MP, uint _feature_id, const uintA& variableDimensions, const uintA& featureDimensions, const intAA& featureVariables)
  : MP(_MP),
    feature_id(_feature_id) {
  featureDim = featureDimensions(feature_id);
  varIds = featureVariables(feature_id);
  varDims.resize(varIds.N);
  for(uint i=0;i<varIds.N;i++){
    varDims(i) = variableDimensions(varIds(i));
  }
  varTotalDim = sum(varDims);
  //ceres internal:
  mutable_parameter_block_sizes()->resize(varDims.N);
  for(uint i=0;i<varDims.N;i++) (*mutable_parameter_block_sizes())[i] = varDims(i);
  set_num_residuals(featureDim);
}

bool Conv_CostFunction::Evaluate(const double* const * parameters, double* residuals, double** jacobians) const{
  //set variables individually
  {
    arr x;
    for(uint i=0;i<varIds.N;i++){
      x.referTo(parameters[i], varDims(i));
      MP->setSingleVariable(varIds(i), x);
    }
  }
  {
    arr phi, J;
    phi.referTo(residuals, featureDim);
    if(jacobians){
      J.referTo(jacobians[0], featureDim*varTotalDim);
      J.reshape(featureDim, varTotalDim);
      MP->evaluateSingleFeature(feature_id, phi, J, NoArr);
    }else{
      MP->evaluateSingleFeature(feature_id, phi, NoArr, NoArr);
    }
  }
  return true;
}

Conv_MatematicalProgram_CeresProblem::Conv_MatematicalProgram_CeresProblem(const ptr<MathematicalProgram>& _MP) : MP(_MP) {

  //you must never ever resize these arrays, as ceres takes pointers directly into these fixed memory buffers!
  x_base.resize(MP->getDimension());
  MP->getBounds(bounds_lo, bounds_up);
  MP->getFeatureTypes(featureTypes);
  MP->getStructure(variableDimensions, featureDimensions, featureVariables);
  variableDimIntegral = integral(variableDimensions);

  x.resize(variableDimensions.N);
  uint n=0;
  for(uint i=0;i<x.N;i++){
    uint d = variableDimensions(i);
    x(i).referTo(x_base.p+n, d);
    n += d;
  }
  //CHECK_EQ(n, x_base.N, "");

  phi.resize(featureDimensions.N);
  for(uint i=0;i<phi.N;i++) phi(i).resize(featureDimensions(i));

  J.resize(phi.N);
  for(uint i=0;i<J.N;i++){
    uint n=0;
    for(uint j : featureVariables(i)) n += variableDimensions(j);
    J(i).resize(n);
  }

  ceresProblem = make_shared<ceres::Problem>();
  
  for(arr& xi: x){
    if(xi.N){
      ceresProblem->AddParameterBlock(xi.p, xi.N);
#if 0 //bounds
      for(uint i=0;i<xi.N;i++){
        uint i_all = i + xi.p-x_base.p;
        if(bounds_lo(i_all)<bounds_up(i_all)){
          ceresProblem->SetParameterLowerBound(xi.p, i, bounds_lo(i_all));
          ceresProblem->SetParameterUpperBound(xi.p, i, bounds_up(i_all));
        }else{
          ceresProblem->SetParameterLowerBound(xi.p, i, -10.);
          ceresProblem->SetParameterUpperBound(xi.p, i, 10.);
        }
      }
#endif
    }
  }

  for(uint i=0;i<phi.N;i++){
    if(featureDimensions(i)){
      rai::Array<double*> parameter_blocks(featureVariables(i).N);
      for(uint k=0;k<featureVariables(i).N;k++){
        parameter_blocks(k) = x(featureVariables(i)(k)).p;
      }
      auto fct = new Conv_CostFunction(MP, i, variableDimensions, featureDimensions, featureVariables);
      ceresProblem->AddResidualBlock(fct, nullptr, parameter_blocks);
    }
  }
}

#else //RAI_CERES

Conv_MatematicalProgram_CeresProblem::Conv_MatematicalProgram_CeresProblem(const ptr<MathematicalProgram>& _MP) {
  NICO
}

#endif
