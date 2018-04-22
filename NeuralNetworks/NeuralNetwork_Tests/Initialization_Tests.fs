namespace NeuralNetwork_Tests

module Initialization_Tests =
    open Xunit
    open FsCheck
    open FsCheck.Xunit

    open MathNet.Numerics.LinearAlgebra

    open NeuralNetwork.NNModel
    open NeuralNetwork.Initialization
    open NeuralNetwork.ActivationFunctions

    type CF = CustomTestFunctionality

    [<Property>]
    let ``initialize_weights tests`` (): Property =
        let data = Gen.choose(1, 20) |> Gen.two |> Arb.fromGen

        let test (n_num: int, x_num: int): unit =
            let expected = (x_num, n_num)

            let actual = let result = initialize_weights n_num x_num
                         (result.RowCount, result.ColumnCount)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``initialize_biases tests`` (): Property =
        let data = Gen.choose(1, 20) |> Arb.fromGen

        let test (n_num: int): unit =
            let expected = n_num

            let actual = (initialize_biases n_num).Count

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``initialize_layer_parameters tests`` (): Property =
        let data = Gen.choose(1, 20) |> Gen.two |> Arb.fromGen

        let test (n_num: int, x_num: int): unit =
            let expected = (x_num, n_num, n_num)

            let actual = let {w=w; b=b} = initialize_layer_parameters x_num n_num
                         (w.RowCount, w.ColumnCount, b.Count)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Fact>]
    let ``initialize_parameters test`` (): unit =
        let expected = [1, 2, 2
                        2, 5, 5]

        let actual = let r = initialize_parameters [1; 2; 5]
                     [r.[0].w.RowCount, r.[0].w.ColumnCount, r.[0].b.Count
                      r.[1].w.RowCount, r.[1].w.ColumnCount, r.[1].b.Count]

        CF.AssertEqual(actual, expected)

    [<Property>]
    let ``initialize_vectorized_activations tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 20
                   |> Arb.fromGen

        let test (ls: float list) =
            let hidden_layers_activationfn: ActivationFnVectorized =
                { f_v  = (*) ls.[0]
                  f'_v = (*) ls.[1] }

            let hidden_layers: int = 2

            let output_functions: ActivationFn list =
                [{f=(*) ls.[2]; f'=(*) ls.[3]}
                 {f=(*) ls.[4]; f'=(*) ls.[5]}
                 {f=(*) ls.[6]; f'=(*) ls.[7]}]

            let actual = initialize_vectorized_activations hidden_layers_activationfn hidden_layers output_functions

            let expected = [hidden_layers_activationfn
                            hidden_layers_activationfn
                            vectorize_output_functions output_functions]

            let m = matrix [ls.[08..10]
                            ls.[11..13]
                            ls.[14..16]
                            ls.[17..19]]

            let actual    = actual   |> List.map (fun act -> act.f_v  m, act.f'_v m)
            let expected  = expected |> List.map (fun act -> act.f_v  m, act.f'_v m)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data
