namespace NeuralNetwork

module Preprocessing =
    open MathNet.Numerics.LinearAlgebra

    open NNModel

    let normalization (x: LayerInput): Normalization =
        let m = float x.RowCount
        { mu = x.ColumnSums() / m
          sigma = Vector.map (function 0. -> 1. | s -> s) <| x.PointwisePower(2.).ColumnSums() / m }

    let normalize (n: Normalization): LayerInput -> LayerInput =
        Matrix.mapRows (fun _ row -> (row - n.mu) / n.sigma)

    let slice_training_set (batch_size: int): float Matrix -> float Matrix list =
        let rec slice (acc: float Matrix list) (m: float Matrix): float Matrix list =
            match m.RowCount with
            | 0 -> acc
            | n when n > batch_size -> slice (m.[0..batch_size - 1, *]::acc) (m.[batch_size.., *])
            | _ -> m.[0.., *]::acc
        slice []

