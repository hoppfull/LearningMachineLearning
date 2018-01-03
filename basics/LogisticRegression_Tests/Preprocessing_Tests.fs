namespace LogisticRegression_Tests

module Preprocessing_Tests =
    open Xunit
    open FsCheck
    open FsCheck.Xunit

    open LogisticRegression.DataTypes
    open LogisticRegression.Preprocessing

    [<Theory>]
    [<InlineData(0., 0., 0., 0., 0., 0.)>]
    [<InlineData(1., 0., 2., 1., 0., 0.)>]
    [<InlineData(2., 4., 5., 0., 2., 4.)>]
    [<InlineData(0., 8., 9., 4., 5., 0.)>]
    let ``columnMeans tests`` (x: float, y: float, a: float, b: float, u: float, v: float): unit =
        let actual = columnMeans [[x; y]
                                  [a; b]
                                  [u; v]]

        let expected = [(x + a + u) / 3.
                        (y + b + v) / 3.]

        Assert.Equal<FeatureType list>(expected, List.ofSeq actual)

    [<Theory>]
    [<InlineData(0., 0., 0., 0., 0., 0.)>]
    [<InlineData(1., 0., 2., 1., 0., 0.)>]
    [<InlineData(2., 4., 5., 0., 2., 4.)>]
    [<InlineData(0., 8., 9., 4., 5., 0.)>]
    let ``columnStandardDeviation tests`` (x: float, y: float, a: float, b: float, u: float, v: float): unit =
        let col0m = List.average [x; a; u]
        let col1m = List.average [y; b; v]

        let actual = columnStandardDeviation [[x; y]
                                              [a; b]
                                              [u; v]]

        let expected = [sqrt (((x - col0m)**2. + (a - col0m)**2. + (u - col0m)**2.) / 3.)
                        sqrt (((y - col1m)**2. + (b - col1m)**2. + (v - col1m)**2.) / 3.)]

        Assert.Equal<FeatureType list>(expected, List.ofSeq actual)

    [<Theory>]
    [<InlineData(0., 0., 0., 0., 0., 0., 0., 0., 0.)>]
    [<InlineData(1., 2., 3., 4., 5., 6., 7., 8., 9.)>]
    [<InlineData(9., 8., 7., 6., 5., 4., 3., 2., 1.)>]
    let ``featureScaling tests`` (x: float, y: float, z: float, a: float, b: float, c: float, u: float, v: float, w: float): unit =
        let col0m = List.average [x; a; u]
        let col1m = List.average [y; b; v]
        let col2m = List.average [z; c; w]

        let col0sd = sqrt(((x - col0m)**2. + (a - col0m)**2. + (u - col0m)**2.) / 3.)
        let col1sd = sqrt(((y - col1m)**2. + (b - col1m)**2. + (v - col1m)**2.) / 3.)
        let col2sd = sqrt(((z - col2m)**2. + (c - col2m)**2. + (w - col2m)**2.) / 3.)

        let actual = featureScaling [[x; y; z]
                                     [a; b; c]
                                     [u; v; w]]

        let expected = [[(x - col0m) / col0sd; (y - col1m) / col1sd; (z - col2m) / col2sd]
                        [(a - col0m) / col0sd; (b - col1m) / col1sd; (c - col2m) / col2sd]
                        [(u - col0m) / col0sd; (v - col1m) / col1sd; (w - col2m) / col2sd]]

        Assert.Equal<FeatureType list>(expected, List.ofSeq <| Seq.map List.ofSeq actual)
