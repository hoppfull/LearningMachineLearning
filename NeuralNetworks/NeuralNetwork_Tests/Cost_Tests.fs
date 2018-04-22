namespace NeuralNetwork_Tests

module Cost_Tests =
    open FsCheck
    open FsCheck.Xunit

    type CF = CustomTestFunctionality

    open MathNet.Numerics.LinearAlgebra

    open NeuralNetwork.Cost

    //[<Property>]
    //let ``cross_entropy_cost tests`` (): Property =
    //    let data = Gen.elements [0.01..0.01..0.99]
    //               |> Gen.listOfLength 3
    //               |> Gen.listOfLength 4
    //               |> Gen.two |> Arb.fromGen

    //    let test (a: float list list, y: float list list): unit =
    //        let A = matrix a
    //        let Y = matrix y

    //        let expected = -y.[0].[0] * log a.[0].[0] - (1. - y.[0].[0]) * log (1. - a.[0].[0])
    //                     + -y.[0].[1] * log a.[0].[1] - (1. - y.[0].[1]) * log (1. - a.[0].[1])
    //                     + -y.[0].[2] * log a.[0].[2] - (1. - y.[0].[2]) * log (1. - a.[0].[2])
    //                     + -y.[1].[0] * log a.[1].[0] - (1. - y.[1].[0]) * log (1. - a.[1].[0])
    //                     + -y.[1].[1] * log a.[1].[1] - (1. - y.[1].[1]) * log (1. - a.[1].[1])
    //                     + -y.[1].[2] * log a.[1].[2] - (1. - y.[1].[2]) * log (1. - a.[1].[2])
    //                     + -y.[2].[0] * log a.[2].[0] - (1. - y.[2].[0]) * log (1. - a.[2].[0])
    //                     + -y.[2].[1] * log a.[2].[1] - (1. - y.[2].[1]) * log (1. - a.[2].[1])
    //                     + -y.[2].[2] * log a.[2].[2] - (1. - y.[2].[2]) * log (1. - a.[2].[2])
    //                     + -y.[3].[0] * log a.[3].[0] - (1. - y.[3].[0]) * log (1. - a.[3].[0])
    //                     + -y.[3].[1] * log a.[3].[1] - (1. - y.[3].[1]) * log (1. - a.[3].[1])
    //                     + -y.[3].[2] * log a.[3].[2] - (1. - y.[3].[2]) * log (1. - a.[3].[2])
    //        let expected = expected / 4.

    //        let actual = cross_entropy_cost A Y

    //        let expected = CF.Round(8, expected)
    //        let actual = CF.Round(8, actual)

    //        CF.AssertEqual(actual, expected)


    //    test |> Prop.forAll data

    [<Property>]
    let ``mean_squared_cost tests`` (): Property =
        let data = Gen.elements [-20.0..0.01..20.0]
                   |> Gen.listOfLength 3
                   |> Gen.listOfLength 4
                   |> Gen.two |> Arb.fromGen

        let test (a: float list list, y: float list list): unit =
            let A = matrix a
            let Y = matrix y

            let expected = (y.[0].[0] - a.[0].[0]) ** 2.
                         + (y.[0].[1] - a.[0].[1]) ** 2.
                         + (y.[0].[2] - a.[0].[2]) ** 2.
                         + (y.[1].[0] - a.[1].[0]) ** 2.
                         + (y.[1].[1] - a.[1].[1]) ** 2.
                         + (y.[1].[2] - a.[1].[2]) ** 2.
                         + (y.[2].[0] - a.[2].[0]) ** 2.
                         + (y.[2].[1] - a.[2].[1]) ** 2.
                         + (y.[2].[2] - a.[2].[2]) ** 2.
                         + (y.[3].[0] - a.[3].[0]) ** 2.
                         + (y.[3].[1] - a.[3].[1]) ** 2.
                         + (y.[3].[2] - a.[3].[2]) ** 2.
            let expected = 0.5 * expected / 4.

            let actual = mean_squared_cost A Y

            let expected = CF.Round(8, expected)
            let actual = CF.Round(8, actual)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data
