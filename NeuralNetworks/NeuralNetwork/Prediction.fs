namespace NeuralNetwork

module Prediction =
    open NNModel
    open Optimization
    open Preprocessing

    let private layer_eval (x: LayerInput) (p: LayerParameters) (f: ActivationFnVectorized): LayerOutput =
        layer_linear x p.w p.b |> f.f_v

    let predict (config: NNConfig) (parameters: LayerParameters list) (x: LayerInput): LayerOutput =
        let x = normalize config.normalization x
        List.fold2 layer_eval x parameters config.activations
