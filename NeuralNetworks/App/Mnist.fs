module Mnist

// https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/

open System
open System.IO
open System.Diagnostics

open MathNet.Numerics.LinearAlgebra

open NeuralNetwork
open NeuralNetwork.Initialization
open NeuralNetwork.Prediction
open NeuralNetwork.Optimization
open NeuralNetwork.ActivationFunctions
open NeuralNetwork.Preprocessing

let read_x (m: int) (file: string): float Matrix =
    use fs = new FileStream(file, FileMode.Open)
    use x_raw = new BinaryReader(fs)

    let (_, _, _, _) = x_raw.ReadInt32()
                     , x_raw.ReadInt32()
                     , x_raw.ReadInt32()
                     , x_raw.ReadInt32()

    let n = 28 * 28
    let a: float [,] = Array2D.create m n 0.
    let mutable i = -1
    while (i <- i + 1; i < m) do
        let mutable j = -1
        while (j <- j + 1; j < n) do
            a.[i, j] <- float (x_raw.ReadByte())
    DenseMatrix.ofArray2 a

let read_y (m: int) (file: string): float Matrix =
    use fs = new FileStream(file, FileMode.Open)
    use y_raw = new BinaryReader(fs)
    
    let (_, _) = y_raw.ReadInt32()
               , y_raw.ReadInt32()

    let n = 10
    let a: float [,] = Array2D.create m n 0.
    let mutable i = -1
    while (i <- i + 1; i < m) do
        a.[i, int <| y_raw.ReadByte()] <- 1.
    DenseMatrix.ofArray2 a

let run_mnist_training (): unit =
    printfn "Running MNIST training"

    let m = 30_000
    let x_train = read_x m "res/mnist/train-images.idx3-ubyte"
    let y_train = read_y m "res/mnist/train-labels.idx1-ubyte"
    let n = x_train.ColumnCount

    let m = 10_000
    let x_test = read_x m "res/mnist/t10k-images.idx3-ubyte"
    let y_test = read_y m "res/mnist/t10k-labels.idx1-ubyte"

    //for label, row, _ in (Seq.zip3 (Matrix.toRowSeq y_train) (Matrix.toRowSeq x_train) [1..10]) do
    //    printfn "label: %A" label
    //    printfn "image: "
    //    let mutable i = -1
    //    while (i <- i + 1; i < 28) do
    //        printfn ""
    //        let mutable j = -1
    //        while (j <- j + 1; j < 28) do
    //            printf "%s" <| if row.Item(j + (i * 28)) > 128. then "█" else " "
    //    printfn ""

    let config = Learning.Config([for _ in [1..y_train.ColumnCount] do yield Sigmoid], [n; 100; 100; 50; y_train.ColumnCount], normalization x_train)

    let timer = Stopwatch()

    printfn "Training... "
    timer.Start()
    let (cost, parameters) = Learning.Train(1024, x_train, y_train, config, learning_rate = 0.01, epochs = 500, regularization = 0.5)
    timer.Stop()
    printfn "done!"

    let t = timer.Elapsed
    printfn "Time = %02im:%02is" t.Minutes t.Seconds
    printfn "Cost = %f" cost

    let y_hat_train =
        predict config parameters x_train
        |> Matrix.map (fun y_hat -> if y_hat > 0.5 then 1. else 0.)

    let y_hat_test =
        predict config parameters x_test
        |> Matrix.map (fun y_hat -> if y_hat > 0.5 then 1. else 0.)

    printfn "train correct: %.2f%%" <| 100. * (Matrix.sum (y_hat_train.PointwiseMultiply y_train)) / float y_hat_train.RowCount
    printfn "test  correct: %.2f%%" <| 100. * (Matrix.sum (y_hat_test.PointwiseMultiply y_test)) / float y_hat_test.RowCount

(*
regularization = 0.5
iterations = 500
time = 22m:23s
cost = 0.03
train correct = 95.76%
test  correct = 90.55%
*)
