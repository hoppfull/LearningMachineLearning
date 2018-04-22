namespace NeuralNetwork

module Initialization =
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Distributions

    open NNModel
    open ActivationFunctions

    let initialize_weights (n_num: int) (x_num: int): float Matrix =
        CreateMatrix.Random(x_num, n_num, ContinuousUniform()) * sqrt (2. / float x_num)

    let initialize_biases (n_num: int): float Vector =
        vector [for _ in [1..n_num] do yield 0.]

    let initialize_layer_parameters (x_num: int) (n_num: int): LayerParameters =
        { w = initialize_weights n_num x_num
          b = initialize_biases n_num }

    let initialize_parameters (layer_dimensions: LayerDimensions): LayerParameters list =
        #if DEBUG
        do if not (2 <= layer_dimensions.Length)
           then raise <| System.ArgumentException("Value constraint: 2 <= layer_dimensions.Length")
        #endif

        let init_params = initialize_layer_parameters
        let (h, t) = let xs = List.rev layer_dimensions in List.head xs, List.tail xs

        List.fold (fun (n, acc) x -> x, (init_params x n)::acc) (h, []) t |> snd

    let initialize_vectorized_activations (hidden_layers_activationfn: ActivationFnVectorized) (hidden_layers: int) (output_functions: ActivationFn list): ActivationFnVectorized list =
        [for _ in [1 .. hidden_layers] do yield hidden_layers_activationfn
         yield vectorize_output_functions output_functions]
