module LogicalCircuit
open MathNet.Numerics.LinearAlgebra

open NeuralNetwork
open NeuralNetwork.Initialization
open NeuralNetwork.Prediction
open NeuralNetwork.Optimization
open NeuralNetwork.ActivationFunctions
open Preprocessing

let x = matrix [[0.; 0.]
                [1.; 0.]
                [0.; 1.]
                [1.; 1.]]

let y_or   = matrix [[0.]; [1.]; [1.]; [1.]]
let y_and  = matrix [[0.]; [0.]; [0.]; [1.]]
let y_xor  = matrix [[0.]; [1.]; [1.]; [0.]]
let y_nand = matrix [[1.]; [1.]; [1.]; [0.]]

//let run_logic_training (): unit =
//    for i in [1..20] do
//        let layer_dims = [2; 3; 3; 1]
//        let config = Learning.Config([Sigmoid], layer_dims, normalization x)
//        let (cost, parameters) = Learning.Train(x, y_xor, config, learning_rate = 0.2)

//        let logical_circuit (a: bool) (b: bool) =
//            let r = matrix [[(if a then 1. else 0.); (if b then 1. else 0.)]]
//                    |> predict config parameters
//            r.Item(0, 0) > 0.5

//        printfn "cost: %A" cost

//        printfn "run: %i" i
//        printfn "false xor false = %b" <| logical_circuit false false
//        printfn "true  xor false = %b" <| logical_circuit true false
//        printfn "false xor true  = %b" <| logical_circuit false true
//        printfn "true  xor true  = %b" <| logical_circuit true true
