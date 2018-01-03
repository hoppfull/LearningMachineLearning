namespace LogisticRegression

module Optimization =
    open DataTypes
    open Utilities

    let sigmoid x = 1. / (1. + exp(-x))

    let zeroes: FeatureType seq = seq { while true do yield 0. }

    let ``wsT * xs + b`` (ws: FeatureType seq): FeatureType seq -> FeatureType =
        Seq.fold2 (fun acc w x -> acc + w * x) (Seq.head ws) (Seq. tail ws)

    let private cost (y: FeatureType) (y_hat: FeatureType): FeatureType =
        if y = 1. then log y_hat
        else if y = 0. then log (1. - y_hat)
        else y * (log y_hat) + (1. - y) * (log (1. - y_hat))

    let private logisticOutput (ws: FeatureType seq): FeatureType seq -> FeatureType =
        sigmoid << (``wsT * xs + b`` ws)

    let totalCost (xss: DataSet) (ys: FeatureType seq) (ws: FeatureType seq): FeatureType =
        let y_hats = Seq.map (logisticOutput ws) xss
        -Seq.average (Seq.map2 cost ys y_hats)

    let weightGradient (xss: DataSet) (ys: FeatureType seq) (ws: FeatureType seq) : FeatureType seq =
        let y_hats = Seq.map (logisticOutput ws) xss
        let ``y_hats - ys`` = Seq.map2 (-) y_hats ys
        seq { yield Seq.average ``y_hats - ys``
              yield! let ``1/m`` = xss |> Seq.length |> float |> (/) 1. in
                     Seq.map2 (Seq.map << (*)) ``y_hats - ys`` xss
                     |> columnReduce (+)
                     |> Seq.map ((*) ``1/m``) }

    // TODO: Remove 'propagate' and distribute its tests to 'weightGradient' and 'totalCost':
    let propagate (ws: FeatureType seq) (xss: DataSet) (ys: FeatureType seq): FeatureType seq * FeatureType =
        weightGradient xss ys ws, totalCost xss ys ws

    let optimizeWithGivenWeights (ws: FeatureType seq) (xss: DataSet) (ys: FeatureType seq) (learningRate: float): int -> FeatureType seq =
        let ``learning rate * weight gradient`` = Seq.map ((*) learningRate) << weightGradient xss ys
        let rec loop ws = function
            | i when i > 0 -> let dws = ``learning rate * weight gradient`` ws
                              loop (Seq.map2 (-) ws dws) (i - 1)
            | _ -> ws
        loop ws

    let optimize (xss: DataSet): FeatureType seq -> float -> int -> FeatureType seq =
       let n = Seq.length (Seq.head xss)
       optimizeWithGivenWeights (Seq.take n zeroes) xss
