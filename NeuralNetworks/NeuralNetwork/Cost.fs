namespace NeuralNetwork

module Cost =
    open MathNet.Numerics.LinearAlgebra

    open NNModel

    //let cross_entropy_cost (a: LayerOutput) (y: LayerOutput): float =
    //    (Matrix.sum <| -y.PointwiseMultiply (Matrix.Log a) - (1. - y).PointwiseMultiply (Matrix.Log <| 1. - a)) / float (Matrix.rowCount y)
        
    //let cross_entropy_loss (a: LayerOutput) (y: LayerOutput): LossValues =
    //    (1. - y).PointwiseDivide(1. - a) - (y).PointwiseDivide(a)

    let mean_squared_cost (a: LayerOutput) (y: LayerOutput): float =
        let m = y - a in (m.PointwisePower(2.) |> Matrix.sum) / float (2 * Matrix.rowCount y)

    let mean_squared_loss (a: LayerOutput) (y: LayerOutput): LossValues =
        a - y
