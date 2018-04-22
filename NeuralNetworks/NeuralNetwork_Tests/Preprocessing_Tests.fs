namespace NeuralNetwork_Tests

module Preprocessing_Tests =
    open Xunit
    open FsCheck
    open FsCheck.Xunit

    type CF = CustomTestFunctionality

    open MathNet.Numerics.LinearAlgebra

    open NeuralNetwork.NNModel
    open NeuralNetwork.Preprocessing


    [<Property>]
    let ``normalization tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 20
                   |> Arb.fromGen

        let test (ls: float list) =
            /// 5x4-matrix
            let x: LayerInput = matrix [ls.[00..03]
                                        ls.[04..07]
                                        ls.[08..11]
                                        ls.[12..15]
                                        ls.[16..19]]

            let actual = normalization x

            let expected = { mu = x.ColumnSums() / 5.
                             sigma = (x.PointwiseMultiply x).ColumnSums() / 5. }
            
            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``normalize tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 24
                   |> Arb.fromGen

        let test (ls: float list) =
            /// 4x4-matrix
            let x: LayerInput = matrix [ls.[00..03]
                                        ls.[04..07]
                                        ls.[08..11]
                                        ls.[12..15]]

            let n: Normalization = { mu = vector ls.[16..19]
                                     sigma = vector ls.[20..23] }

            let actual = normalize n x

            let expected = [ls.[00..03]
                            ls.[04..07]
                            ls.[08..11]
                            ls.[12..15]]
                           |> List.map (fun r -> List.map2 (-) r ls.[16..19])
                           |> List.map (fun r -> List.map2 (/) r ls.[20..23])
                           |> matrix

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``slice_training_set tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 66 |> Arb.fromGen

        let test (ls: float list) =
            let m: float Matrix = matrix [ls.[00..05]
                                          ls.[06..11]
                                          ls.[12..17]
                                          ls.[18..23]
                                          ls.[24..29]
                                          ls.[30..35]
                                          ls.[36..41]
                                          ls.[42..47]
                                          ls.[48..53]
                                          ls.[54..59]
                                          ls.[60..65]]


            let actual = slice_training_set 3 m
            let expected = [matrix [ls.[00..05]
                                    ls.[06..11]
                                    ls.[12..17]]
                            matrix [ls.[18..23]
                                    ls.[24..29]
                                    ls.[30..35]]
                            matrix [ls.[36..41]
                                    ls.[42..47]
                                    ls.[48..53]]
                            matrix [ls.[54..59]
                                    ls.[60..65]]]
            CF.AssertEqual(actual, List.rev expected)


        test |> Prop.forAll data

