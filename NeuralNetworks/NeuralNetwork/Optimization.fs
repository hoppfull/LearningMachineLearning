namespace NeuralNetwork

module Optimization =
    open MathNet.Numerics.LinearAlgebra

    open ActivationFunctions
    open NNModel
    open Cost

    let layer_linear (x: LayerInput) (w: LayerWeights) (b: LayerBiases): LayerOutput =
        Matrix.mapRows (fun _ -> (+) b) (x * w)

    type ForwardPropagationCache = { a: LayerOutput
                                     z: LayerOutput }

    let propagate_forward (x: LayerInput, cache: ForwardPropagationCache list) (activation: ActivationFnVectorized, ps: LayerParameters): LayerOutput * ForwardPropagationCache list =
        let z = layer_linear x ps.w ps.b
        let a = activation.f_v z
        (a, {a=x; z=z}::cache)

    let forward_propagation (input: LayerInput) (activation: ActivationFnVectorized list) (parameters: LayerParameters list): LayerOutput * ForwardPropagationCache list =
        List.fold propagate_forward (input, []) (List.zip activation parameters)

    let propagate_backward (da: LossValues, gradients: LayerGradients list) (activation: ActivationFnVectorized, cache: ForwardPropagationCache, parameters: LayerParameters): LossValues * LayerGradients list =
        let dz = da.PointwiseMultiply <| activation.f'_v cache.z

        let m = float cache.a.RowCount

        let dw = cache.a.TransposeThisAndMultiply dz / m
        let db = dz.ColumnSums() / m
        let da = dz.TransposeAndMultiply parameters.w

        da, {w=dw; b=db}::gradients

    let backward_propagation (da: LossValues) (activation: ActivationFnVectorized list) (cache: ForwardPropagationCache list) (parameters: LayerParameters list): LayerGradients list =
        List.fold propagate_backward (da, []) (List.zip3 activation cache parameters) |> snd

    type GradientResult = { cost: float
                            gradients: LayerGradients list }

    let gradients (activation: ActivationFnVectorized list) (x: LayerInput) (y: LayerOutput) (parameters: LayerParameters list): GradientResult =
        let (a, cache) = forward_propagation x activation parameters
        { cost = mean_squared_cost a y
          gradients = backward_propagation (mean_squared_loss a y) (List.rev activation) cache (List.rev parameters) }

    let private L2_regularize (l: float) (m: float) (p: LayerParameters) (g: LayerGradients): LayerGradients =
        let c = (l / m)
        { w = g.w + c * p.w
          b = g.b + c * p.b }

    let regularize (lambda: float) (training_examples: int): LayerParameters list -> LayerGradients list -> LayerGradients list =
        List.map2 (L2_regularize lambda (float training_examples))

    let gradient_step (learning_rate: float): LayerParameters list -> LayerGradients list -> LayerParameters list =
        List.map2 (fun p g -> { w = p.w - learning_rate * g.w
                                b = p.b - learning_rate * g.b })

    /// beta -> iteration -> previous -> gradients -> gradients with momentum
    let private momentum_ (beta: float) (i: int): LayerGradients list -> LayerGradients list -> LayerGradients list =
        let ``1-beta^2`` = if i < 100 // TODO: check if this optimization is unecessary
                           then 1. - beta ** float i else 1.
        List.map2 (fun vg g -> { w = (beta * vg.w + (1. - beta) * g.w) / ``1-beta^2``
                                 b = (beta * vg.b + (1. - beta) * g.b) / ``1-beta^2`` })

    /// beta -> iteration -> previous -> gradients -> gradients with momentum
    let private rmsprop_ (beta: float) (i: int): LayerGradients list -> LayerGradients list -> LayerGradients list =
        let ``1-beta^2`` = if i < 100 // TODO: check if this optimization is unecessary
                           then 1. - beta ** float i else 1.
        List.map2 (fun sg g -> { w = (beta * sg.w + (1. - beta) * g.w.PointwisePower 2.) / ``1-beta^2``
                                 b = (beta * sg.b + (1. - beta) * g.b.PointwisePower 2.) / ``1-beta^2`` })

    type AdamData = { sgrads: LayerGradients list
                      vgrads: LayerGradients list }

    let adam_gradient_step (momentum: float) (rmsprop: float) (iteration: int) (grads: LayerGradients list) (prev: AdamData): LayerGradients list * AdamData =
        let vgrads = momentum_ momentum iteration prev.vgrads grads
        let sgrads = rmsprop_ rmsprop iteration prev.sgrads grads

        let epsilon: float = 1.e-8
        List.map2 (fun vg sg -> { w = vg.w.PointwiseDivide(epsilon + Matrix.Sqrt sg.w)
                                  b = vg.b.PointwiseDivide(epsilon + Vector.Sqrt sg.b) }) vgrads sgrads
        ,
        { sgrads = sgrads
          vgrads = vgrads }
