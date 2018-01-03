namespace LogisticRegression_Tests

module Utilities_Tests =
    open Xunit
    open FsCheck
    open FsCheck.Xunit

    open LogisticRegression.DataTypes
    open LogisticRegression.Utilities

    [<Theory>]
    [<InlineData(0., 0., 0., 0., 0., 0.)>]
    [<InlineData(1., 0., 2., 1., 0., 0.)>]
    [<InlineData(2., 4., 5., 0., 2., 4.)>]
    [<InlineData(0., 8., 9., 4., 5., 0.)>]
    let ``columnReduce tests`` (x: float, y: float, z: float, a: float, b: float, c: float): unit =
        let actual = columnReduce (+) [[x; y; z]
                                       [a; b; c]]

        let expected = [x + a
                        y + b
                        z + c]

        Assert.Equal<FeatureType list>(expected, List.ofSeq actual)
