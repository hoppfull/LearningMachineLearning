namespace NeuralNetwork

module NNModel =
    open MathNet.Numerics.LinearAlgebra

    type LayerWeights      = float Matrix
    type LayerBiases       = float Vector
    type LayerInput        = float Matrix
    type LayerOutput       = LayerInput
    type LayerParameters   = { w: LayerWeights
                               b: LayerBiases }
    type LayerGradients    = LayerParameters
    type LossValues        = LayerInput
    type LayerDimensions   = int list    
    type ActivationFnVectorized = { f_v:  LayerInput -> LayerOutput
                                    f'_v: LayerInput -> LayerOutput }
    type ActivationFn = { f:  float -> float
                          f': float -> float }
    type Normalization = { mu: float Vector
                           sigma: float Vector }
    type NNConfig = { activations: ActivationFnVectorized list
                      normalization: Normalization
                      layer_dims: LayerDimensions }
