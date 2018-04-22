namespace NeuralNetwork

open MathNet.Numerics.LinearAlgebra

open NNModel
open ActivationFunctions
open Initialization
open Optimization
open Preprocessing

[<AbstractClass; Sealed>]
type Learning() =
    static member Config(output_functions: ActivationFn list, layer_dims: LayerDimensions, normalization: Normalization, ?hidden_layers_activationfn: ActivationFnVectorized): NNConfig =
        let hidden_layers_activationfn = defaultArg hidden_layers_activationfn ReLU_Leaky

        let hidden_layers = (layer_dims.Length - 2)

        { activations = initialize_vectorized_activations hidden_layers_activationfn hidden_layers output_functions
          normalization = normalization
          layer_dims = layer_dims }

    static member Train(batch_size: int, x: LayerInput, y: LayerOutput, config: NNConfig, ?parameters: LayerParameters list, ?learning_rate: float, ?regularization: float, ?threshold: float, ?epochs: int): float * LayerParameters list =
        let parameters      = defaultArg parameters (initialize_parameters config.layer_dims)
        let learning_rate   = defaultArg learning_rate 1.
        let epochs          = defaultArg epochs 1

        let training_examples = x.RowCount
        let x = normalize config.normalization x
        let x = slice_training_set batch_size x
        let y = slice_training_set batch_size y
        let t = List.length x |> float

        let rec loop (i: int) (cost: float) (parameters: LayerParameters list) (data: AdamData) (learning_rate: float): float * LayerParameters list =
            if i > 0 && match threshold with | Some threshold -> cost > threshold | None -> true
            then
                let tasks = seq { for x, y in Seq.zip x y do yield async { return gradients config.activations x y parameters } }
                            |> Async.Parallel

                let tasks = Async.RunSynchronously tasks

                let cost' = Array.fold (fun acc c -> c.cost + acc) 0. tasks
                let cost' = cost' / t

                let gradients = (Array.tail tasks) |> Array.fold (fun g c -> List.map2 (fun g c -> { w = g.w + c.w; b = g.b + c.b }) g c.gradients) (Array.head tasks).gradients
                let gradients = List.map (fun g -> { w = g.w / t; b = g.b / t }) gradients

                let (gradients, data) = adam_gradient_step 0.9 0.999 i gradients data

                let gradients = match regularization with
                                | None -> gradients
                                | Some regularization -> regularize regularization training_examples parameters gradients

                let parameters = gradient_step (max learning_rate (learning_rate * (cost' ** 1.5))) parameters gradients

                printfn "cost = %.4f, learning_rate = %.4f, epoch = %i" cost' (learning_rate * cost') (epochs - i + 1)

                loop (i - 1) cost' parameters data learning_rate
            else (cost, parameters)

        let data = { vgrads = (List.map (fun p -> { w = p.w * 0.; b = p.b * 0. }) parameters)
                     sgrads = (List.map (fun p -> { w = p.w * 0.; b = p.b * 0. }) parameters) }
        loop epochs System.Double.PositiveInfinity parameters data learning_rate
