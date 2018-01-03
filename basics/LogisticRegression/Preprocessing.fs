namespace LogisticRegression

module Preprocessing =
    open System
    open DataTypes
    open Utilities

    // column wise means:
    let columnMeans (dataSet: DataSet): FeatureType seq =
        let length = Seq.length dataSet
        let columnSums = columnReduce (+) dataSet
        Seq.map (fun a -> a / float length) columnSums

    let private ``standard deviation with precalculated means`` (means: FeatureType seq) (dataSet: DataSet): FeatureType seq =
        let distanceSquared (avg: FeatureType) (x: FeatureType): FeatureType = (x - avg)**2.
        dataSet
        |> Seq.map (Seq.map2 distanceSquared means)
        |> columnReduce (+)
        |> Seq.map (fun x -> sqrt (x / 3.))

    // column wise standard deviations:
    let columnStandardDeviation (dataSet: DataSet): FeatureType seq =
        let means = columnMeans dataSet
        ``standard deviation with precalculated means`` means dataSet

    // feature scaling for faster gradient descent:
    let featureScaling (dataSet: DataSet): DataSet =
        let means = columnMeans dataSet
        let sds = ``standard deviation with precalculated means`` means dataSet
        let ``subtract the mean and divide by standard deviation`` m sd x = (x - m) / sd
        Seq.map (Seq.map3 ``subtract the mean and divide by standard deviation`` means sds) dataSet
