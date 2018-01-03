namespace LogisticRegression_Tests

module Optimization_Tests =
    open Xunit
    open FsCheck
    open FsCheck.Xunit

    open LogisticRegression.DataTypes
    open LogisticRegression.Optimization

    [<Theory>]
    [<InlineData(0., 0.5)>]
    [<InlineData(+4., 0.98)>]
    [<InlineData(-4., 0.02)>]
    [<InlineData(+5., 0.99)>]
    [<InlineData(-5., 0.01)>]
    [<InlineData(+7., 1.)>]
    [<InlineData(-7., 0.)>]
    [<InlineData(+10., 1.)>]
    [<InlineData(-10., 0.)>]
    let ``sigmoid tests`` (x: float, y: float): unit =
        let actual = sigmoid x
        let expected = y
        Assert.Equal (expected, System.Math.Round (actual, 2))

    [<Theory>]
    [<InlineData(0., 0., 0., 0., 0., 0., 0.)>]
    [<InlineData(1., 2., 3., 4., 5., 6., 7.)>]
    [<InlineData(7., 6., 5., 4., 3., 2., 1.)>]
    let ``wsT * xs + b tests`` (wb: float, w0: float, w1: float, w2: float, x0: float, x1: float, x2: float): unit =
        let ws = [wb; w0; w1; w2]
        let xs = [x0; x1; x2]

        let actual = ``wsT * xs + b`` ws xs
        let expected = wb + w0 * x0 + w1 * x1 + w2 * x2

        Assert.Equal (expected, actual)

    [<Fact>]
    let ``propagate test`` (): unit =
        let ws = [2.; 1.; 2.]

        let xss = [[ 1.; 3.]
                   [ 2.; 4.]
                   [-1.; -3.2]]
                  |> List.map (Seq.ofList)

        let ys = [1.; 0.; 1.]

        let dw_actual, cost_actual = propagate ws xss ys 

        let round (x: FeatureType) = System.Math.Round(x, 5)

        let cost_actual = round cost_actual

        let cost_expected = round 5.80154531939

        let dw_expected = [0.00145557813678
                           0.99845601
                           2.39507239]
                          |> List.map round

        let dw_actual = Seq.map round dw_actual |> List.ofSeq

        Assert.Equal<FeatureType> (cost_expected, cost_actual)
        Assert.Equal<FeatureType> (dw_expected, dw_actual)
