namespace NeuralNetwork_Tests

module Optimization_Tests =
    open Xunit
    open FsCheck
    open FsCheck.Xunit

    type CF = CustomTestFunctionality

    open MathNet.Numerics.LinearAlgebra

    open NeuralNetwork.NNModel
    open NeuralNetwork.ActivationFunctions
    open NeuralNetwork.Optimization
    open NeuralNetwork.Cost

    [<Property>]
    let ``layer_linear tests`` (): Property =
        let data = Gen.elements [-20. .. 0.01 .. 20.] |> Gen.listOfLength 17 |> Arb.fromGen

        let test (ls: float List) =
            let x = matrix [[ls.[0]; ls.[1]]
                            [ls.[2]; ls.[3]]
                            [ls.[4]; ls.[5]]
                            [ls.[6]; ls.[7]]]

            let w = matrix [[ls.[08]; ls.[09]; ls.[10]]
                            [ls.[11]; ls.[12]; ls.[13]]]

            let b = vector [ls.[14]; ls.[15]; ls.[16]]

            let expected = matrix [[ls.[0] * ls.[08] + ls.[1] * ls.[11] + ls.[14]
                                    ls.[0] * ls.[09] + ls.[1] * ls.[12] + ls.[15]
                                    ls.[0] * ls.[10] + ls.[1] * ls.[13] + ls.[16]]
                                   [ls.[2] * ls.[08] + ls.[3] * ls.[11] + ls.[14]
                                    ls.[2] * ls.[09] + ls.[3] * ls.[12] + ls.[15]
                                    ls.[2] * ls.[10] + ls.[3] * ls.[13] + ls.[16]]
                                   [ls.[4] * ls.[08] + ls.[5] * ls.[11] + ls.[14]
                                    ls.[4] * ls.[09] + ls.[5] * ls.[12] + ls.[15]
                                    ls.[4] * ls.[10] + ls.[5] * ls.[13] + ls.[16]]
                                   [ls.[6] * ls.[08] + ls.[7] * ls.[11] + ls.[14]
                                    ls.[6] * ls.[09] + ls.[7] * ls.[12] + ls.[15]
                                    ls.[6] * ls.[10] + ls.[7] * ls.[13] + ls.[16]]]

            let actual = layer_linear x w b

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``propagate_forward tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 32
                   |> Arb.fromGen

        let test (ls: float list) =
            let x: float Matrix = matrix [ls.[0..5]
                                          ls.[6..11]
                                          ls.[12..17]]

            let parameters: LayerParameters = { w = matrix [ls.[18..19]
                                                            ls.[20..21]
                                                            ls.[22..23]
                                                            ls.[24..25]
                                                            ls.[26..27]
                                                            ls.[28..29]]
                                                b = vector ls.[30..31] }

            let z = Matrix.mapRows (fun _ -> (+) parameters.b) (x * parameters.w)
            
            let triple = {f=(*) 3.; f'=(*) 3.}
            let activation = vectorize_output_functions [triple; triple]

            let a = activation.f_v z

            let expected = (a, {a=x; z=z}::[])

            let actual = propagate_forward (x, []) (activation, parameters)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``forward_propagation tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 17
                   |> Arb.fromGen
        
        let test (ls: float list) =
            let x: LayerInput = matrix [ls.[0..2]
                                        ls.[3..5]]

            let triple = {f=(*) 3.; f'=(*) 3.}
            let activation: ActivationFnVectorized list =
                [vectorize_output_functions [triple; triple]
                 vectorize_output_functions [triple]]

            let f0 = activation.[0].f_v
            let w0 = matrix [ls.[06..07]
                             ls.[08..09]
                             ls.[10..11]]
            let b0 = vector ls.[12..13]

            let f1 = activation.[1].f_v
            let w1 = matrix [[ls.[14]]
                             [ls.[15]]]
            let b1 = vector [ls.[16]]

            let parameters: LayerParameters list =
                [{ w = w0; b = b0 }
                 { w = w1; b = b1 }]

            let actual = forward_propagation x activation parameters

            let z0: float Matrix = layer_linear x w0 b0
            let a0: float Matrix = f0 z0

            let z1: float Matrix = layer_linear a0 w1 b1
            let a1: float Matrix = f1 z1

            let expected = a1, [{a=a0; z=z1} 
                                {a=x;  z=z0}]

            CF.AssertEqual((z0.ColumnCount, z0.RowCount), (a0.ColumnCount, a0.RowCount))
            CF.AssertEqual((z0.ColumnCount, z0.RowCount), (a0.ColumnCount, a0.RowCount))
            CF.AssertEqual((z1.ColumnCount, z1.RowCount), (a1.ColumnCount, a1.RowCount))

            CF.AssertEqual(w1.ColumnCount, a1.ColumnCount)
            CF.AssertEqual(x.RowCount, a0.RowCount)
            CF.AssertEqual(x.RowCount, a1.RowCount)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``propagate_backward tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 18
                   |> Arb.fromGen

        let test (ls: float list) =
            let da: LossValues = matrix [ls.[0..1]
                                         ls.[2..3]
                                         ls.[4..5]]

            let w = matrix [ls.[6..7]
                            ls.[8..9]]
            let b = vector ls.[10..11]

            let z = matrix [ls.[12..13]
                            ls.[14..15]
                            ls.[16..17]]

            let activation: ActivationFnVectorized =
                vectorize_output_functions [{f=(*) 3.; f'=(*) 3.}
                                            {f=(*) 3.; f'=(*) 3.}]

            let a = activation.f_v z

            let cache: ForwardPropagationCache = { a=a; z=z }

            let parameters: LayerParameters = { w=w; b=b }

            let actual = propagate_backward (da, []) (activation, cache, parameters)

            let dz = da.PointwiseMultiply (activation.f'_v z)

            let m = float a.RowCount

            let dw = (a.Transpose() * dz) / m
            let db = dz.ColumnSums() / m
            let da = dz * w.Transpose()

            let expected = (da, [{w=dw; b=db}])

            CF.AssertEqual((w.RowCount, w.ColumnCount), (dw.RowCount, dw.ColumnCount))

            CF.AssertEqual(b.Count, db.Count)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``backward_propagation tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 65
                   |> Arb.fromGen

        let test (ls: float list) =
            let da2: LossValues = matrix [ls.[0..1]
                                          ls.[2..3]
                                          ls.[4..5]]

            let f = { f=(*) 3.; f'=(*) 2. }
            let activation: ActivationFnVectorized list =
                [vectorize_output_functions [f; f; f]
                 vectorize_output_functions [f]
                 vectorize_output_functions [f; f]]

            let a2 = matrix [ls.[06..07]
                             ls.[08..09]
                             ls.[10..11]]
            let z2 = matrix [ls.[12..13]
                             ls.[14..15]
                             ls.[16..17]]

            let a1 = matrix [[ls.[18]]
                             [ls.[19]]
                             [ls.[20]]]
            let z1 = matrix [[ls.[21]]
                             [ls.[22]]
                             [ls.[23]]]

            let a0 = matrix [ls.[24..26]
                             ls.[27..29]
                             ls.[30..32]]
            let z0 = matrix [ls.[33..35]
                             ls.[36..38]
                             ls.[39..41]]

            let x = matrix [ls.[42..43]
                            ls.[44..45]
                            ls.[46..47]]

            let cache: ForwardPropagationCache list = [{a=a1; z=z2}
                                                       {a=a0; z=z1}
                                                       {a=x;  z=z0}]

            let w2 = matrix [ls.[48..49]]
            let b2 = vector ls.[50..51]

            let w1 = matrix [[ls.[52]]
                             [ls.[53]]
                             [ls.[54]]]
            let b1 = vector [ls.[55]]

            let w0 = matrix [ls.[56..58]
                             ls.[59..61]]
            let b0 = vector ls.[62..64]

            let parameters: LayerParameters list = [{w=w0; b=b0}
                                                    {w=w1; b=b1}
                                                    {w=w2; b=b2}]

            let actual = backward_propagation da2 (List.rev activation) cache (List.rev parameters)

            let m_training_examples = float a2.RowCount

            let dz2 = da2.PointwiseMultiply (activation.[2].f'_v z2)
            let dw2 = (a1.Transpose() * dz2) / m_training_examples
            let db2 = dz2.ColumnSums() / m_training_examples
            let da1 = dz2 * w2.Transpose()

            let dz1 = da1.PointwiseMultiply (activation.[1].f'_v z1)
            let dw1 = (a0.Transpose() * dz1) / m_training_examples
            let db1 = dz1.ColumnSums() / m_training_examples
            let da0 = dz1 * w1.Transpose()

            let dz0 = da0.PointwiseMultiply (activation.[0].f'_v z0)
            let dw0 = (x.Transpose() * dz0) / m_training_examples
            let db0 = dz0.ColumnSums() / m_training_examples

            let expected = [{w=dw0; b=db0}
                            {w=dw1; b=db1}
                            {w=dw2; b=db2}]

            CF.AssertEqual((w0.RowCount, w0.ColumnCount), (dw0.RowCount, dw0.ColumnCount))
            CF.AssertEqual((w1.RowCount, w1.ColumnCount), (dw1.RowCount, dw1.ColumnCount))
            CF.AssertEqual((w2.RowCount, w2.ColumnCount), (dw2.RowCount, dw2.ColumnCount))

            CF.AssertEqual(b0.Count, db0.Count)
            CF.AssertEqual(b1.Count, db1.Count)
            CF.AssertEqual(b2.Count, db2.Count)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``gradients tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 72
                   |> Arb.fromGen

        let test (ls: float list) =
            let activation: ActivationFnVectorized list =
                [{f_v=(*) ls.[0]; f'_v=(*) ls.[1]}
                 {f_v=(*) ls.[2]; f'_v=(*) ls.[3]}
                 {f_v=(*) ls.[4]; f'_v=(*) ls.[5]}
                 {f_v=(*) ls.[6]; f'_v=(*) ls.[7]}]

            /// 5x3-matrix
            let x: LayerInput = matrix [ls.[08..10]
                                        ls.[11..13]
                                        ls.[14..16]
                                        ls.[17..19]
                                        ls.[20..22]]

            /// 5x2-matrix
            let y: LayerOutput = matrix [ls.[23..24]
                                         ls.[25..26]
                                         ls.[27..28]
                                         ls.[29..30]
                                         ls.[31..32]]

            /// 3x4-matrix
            let w0 = matrix [ls.[33..36]
                             ls.[37..40]
                             ls.[41..44]]
            /// 4-vector
            let b0 = vector ls.[45..48]

            /// 4x3
            let w1 = matrix [ls.[49..51]
                             ls.[52..54]
                             ls.[55..57]
                             ls.[58..60]]
            /// 3-vector
            let b1 = vector ls.[61..63]

            /// 3x1-matrix
            let w2 = matrix [[ls.[64]]
                             [ls.[66]]
                             [ls.[66]]]
            /// 1-vector
            let b2 = vector [ls.[67]]

            /// 1x2-matrix
            let w3 = matrix [ls.[68..69]]
            /// 2-vector
            let b3 = vector ls.[70..71]

            let parameters: LayerParameters list =
                [{ w=w0; b=b0 }
                 { w=w1; b=b1 }
                 { w=w2; b=b2 }
                 { w=w3; b=b3 }]

            let z0 = layer_linear x w0 b0
            let a0 = activation.[0].f_v z0

            let z1 = layer_linear a0 w1 b1
            let a1 = activation.[1].f_v z1

            let z2 = layer_linear a1 w2 b2
            let a2 = activation.[2].f_v z2

            let z3 = layer_linear a2 w3 b3
            let a3 = activation.[3].f_v z3

            let cost = mean_squared_cost a3 y
            let da3 = mean_squared_loss a3 y

            let m_training_examples = float x.RowCount

            let dz3 = da3.PointwiseMultiply (activation.[3].f'_v z3)
            let dw3 = a2.TransposeThisAndMultiply dz3 / m_training_examples
            let db3 = dz3.ColumnSums() / m_training_examples
            let da2 = dz3.TransposeAndMultiply w3

            let dz2 = da2.PointwiseMultiply (activation.[2].f'_v z2)
            let dw2 = a1.TransposeThisAndMultiply dz2 / m_training_examples
            let db2 = dz2.ColumnSums() / m_training_examples
            let da1 = dz2.TransposeAndMultiply w2

            let dz1 = da1.PointwiseMultiply (activation.[1].f'_v z1)
            let dw1 = a0.TransposeThisAndMultiply dz1 / m_training_examples
            let db1 = dz1.ColumnSums() / m_training_examples
            let da0 = dz1.TransposeAndMultiply w1

            let dz0 = da0.PointwiseMultiply (activation.[0].f'_v z0)
            let dw0 = x.TransposeThisAndMultiply dz0 / m_training_examples
            let db0 = dz0.ColumnSums() / m_training_examples

            let actual = gradients activation x y parameters

            let expected = { cost = cost
                             gradients = [{ w=dw0; b=db0 }
                                          { w=dw1; b=db1 }
                                          { w=dw2; b=db2 }
                                          { w=dw3; b=db3 }] }

            CF.AssertEqual((w0.RowCount, w0.ColumnCount), (dw0.RowCount, dw0.ColumnCount))
            CF.AssertEqual((w1.RowCount, w1.ColumnCount), (dw1.RowCount, dw1.ColumnCount))
            CF.AssertEqual((w2.RowCount, w2.ColumnCount), (dw2.RowCount, dw2.ColumnCount))
            CF.AssertEqual((w3.RowCount, w3.ColumnCount), (dw3.RowCount, dw3.ColumnCount))

            CF.AssertEqual(b0.Count, db0.Count)
            CF.AssertEqual(b1.Count, db1.Count)
            CF.AssertEqual(b2.Count, db2.Count)
            CF.AssertEqual(b3.Count, db3.Count)

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``regularize tests`` (): Property =
        let data =
            let gen_ls = Gen.elements [-20. .. 0.1 .. 20.] |> Gen.listOfLength 89
            let gen_m = Gen.choose(-20, 20)
            Gen.zip gen_ls gen_m |> Arb.fromGen

        let test (ls: float list, m: int) =
            let lambda: float = ls.[0]

            let training_examples: int = m

            /// 5x4-matrix
            let w0 = matrix [ls.[01..04]
                             ls.[05..08]
                             ls.[09..12]
                             ls.[13..16]
                             ls.[17..20]]
            /// 4-vector
            let b0 = vector ls.[21..24]

            /// 3x5-matrix
            let w1 = matrix [ls.[25..29]
                             ls.[30..34]
                             ls.[35..39]]
            /// 5-vector
            let b1 = vector ls.[40..44]

            let parameters: LayerParameters list = [{ w=w0; b=b0 }
                                                    { w=w1; b=b1 }]

            /// 5x4-matrix
            let dw0 = matrix [ls.[45..48]
                              ls.[49..52]
                              ls.[53..56]
                              ls.[57..60]
                              ls.[61..64]]
            /// 4-vector
            let db0 = vector ls.[65..68]

            /// 3x5-matrix
            let dw1 = matrix [ls.[69..73]
                              ls.[74..78]
                              ls.[79..83]]
            /// 5-vector
            let db1 = vector ls.[84..88]

            let gradients: LayerParameters list = [{ w=dw0; b=db0 }
                                                   { w=dw1; b=db1 }]

            let actual = regularize lambda training_examples parameters gradients

            let m = float m
            let expected = let c = (lambda / m)
                           [{ w = dw0 + c * w0; b = db0 + c * b0 }
                            { w = dw1 + c * w1; b = db1 + c * b1 }]

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``gradient_step tests`` (): Property =
        let data = Gen.elements [-20. .. 0.1 .. 20.]
                   |> Gen.listOfLength 87
                   |> Arb.fromGen

        let test (ls: float list) =
            let l: float = ls.[86]

            /// 5x4-matrix
            let w0 = matrix [ls.[00..03]
                             ls.[04..07]
                             ls.[08..11]
                             ls.[12..15]
                             ls.[16..19]]
            // 4-vector
            let b0 = vector ls.[20..23]

            /// 4x2-matrix
            let w1 = matrix [ls.[24..25]
                             ls.[26..27]
                             ls.[28..29]
                             ls.[30..31]]
            /// 2-vector
            let b1 = vector ls.[32..33]

            /// 2x3-matrix
            let w2 = matrix [ls.[34..36]
                             ls.[37..39]]
            /// 3-vector
            let b2 = vector ls.[40..42]

            let parameters = [{ w=w0; b=b0 }
                              { w=w1; b=b1 }
                              { w=w2; b=b2 }]

            /// 5x4-matrix
            let dw0 = matrix [ls.[43..46]
                              ls.[47..50]
                              ls.[51..54]
                              ls.[55..58]
                              ls.[59..62]]
            // 4-vector
            let db0 = vector ls.[63..66]

            /// 4x2-matrix
            let dw1 = matrix [ls.[67..68]
                              ls.[69..70]
                              ls.[71..72]
                              ls.[73..74]]
            /// 2-vector
            let db1 = vector ls.[75..76]

            /// 2x3-matrix
            let dw2 = matrix [ls.[77..79]
                              ls.[80..82]]
            /// 3-vector
            let db2 = vector ls.[83..85]

            let gradients = [{ w=dw0; b=db0 }
                             { w=dw1; b=db1 }
                             { w=dw2; b=db2 }]

            let actual = gradient_step l parameters gradients

            let expected = [{ w = w0 - l * dw0; b = b0 - l * db0 }
                            { w = w1 - l * dw1; b = b1 - l * db1 }
                            { w = w2 - l * dw2; b = b2 - l * db2 }]

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data

    [<Property>]
    let ``gradient checking`` (): Property =
        let data = Gen.elements [-5. .. 0.01 .. 5.]
                   |> Gen.listOfLength 105
                   |> Arb.fromGen

        let test (ls: float list) =
            let epsilon: float = 1e-6

            /// requires a list of length = 70
            let genParameters (ls: float list): LayerParameters list =
                /// 6x5-matrix
                let w0 = matrix [ls.[00..04]
                                 ls.[05..09]
                                 ls.[10..14]
                                 ls.[15..19]
                                 ls.[20..24]
                                 ls.[25..29]]
                /// 5-vector
                let b0 = vector ls.[30..34]

                /// 5x3-matrix
                let w1 = matrix [ls.[35..37]
                                 ls.[38..40]
                                 ls.[41..43]
                                 ls.[44..46]
                                 ls.[47..49]]
                /// 3-vector
                let b1 = vector ls.[50..52]

                /// 3x2-matrix
                let w2 = matrix [ls.[53..54]
                                 ls.[55..56]
                                 ls.[57..58]]
                /// 2-vector
                let b2 = vector ls.[59..60]

                /// 2x3-matrix
                let w3 = matrix [ls.[61..63]
                                 ls.[64..66]]
                /// 3-vector
                let b3 = vector ls.[67..69]

                [{ w=w0; b=b0 }
                 { w=w1; b=b1 }
                 { w=w2; b=b2 }
                 { w=w3; b=b3 }]

            /// 3x6-matrix
            let x = matrix [ls.[00..05]
                            ls.[06..11]
                            ls.[12..17]]

            /// 3x3-matrix
            let y = matrix [ls.[18..20]
                            ls.[21..23]
                            ls.[24..26]]

            let activations: ActivationFnVectorized list = [Tanh; Tanh; Tanh; ReLU_Leaky]

            let ls = ls.[35..104]

            let f (i: int) _: float =
                let ``ls+`` = List.mapi (fun j l -> if i = j then l + epsilon else l) ls
                let ``ls-`` = List.mapi (fun j l -> if i = j then l - epsilon else l) ls

                let ``theta+`` = genParameters ``ls+``
                let ``theta-`` = genParameters ``ls-``
                let ``J+`` = (forward_propagation x activations ``theta+`` |> fst |> mean_squared_cost) y
                let ``J-`` = (forward_propagation x activations ``theta-`` |> fst |> mean_squared_cost) y

                (``J+`` - ``J-``) / (2. * epsilon)

            let v_g_approx: float Vector = vector (List.mapi f ls)

            let parameters = genParameters ls
            let { gradients=gradients } = gradients activations x y parameters

            let v_g: float Vector = vector [ for p in gradients do yield! [for w in (Matrix.toRowSeq p.w) do yield! w
                                                                           yield! p.b]]
            
            let diff = (v_g - v_g_approx).L2Norm() / (v_g.L2Norm() + v_g_approx.L2Norm())

            //sprintf "DEBUG %A" (diff, v_g.L2Norm(), v_g_approx.L2Norm()) |> System.Exception |> raise
            Assert.True(diff < (2. * epsilon), sprintf "%A" (diff, v_g.L2Norm(), v_g_approx.L2Norm()))

        test |> Prop.forAll data

    [<Property>]
    let ``momentum tests`` (): Property =
        let data =
            let lists = Gen.elements [-20. .. 0.1 .. 20.] |> Gen.listOfLength 32
            let betas = Gen.elements [0. .. 0.01 .. 1.]
            Gen.zip lists betas |> Arb.fromGen

        let test (ls: float list, beta: float) =
            /// 3x4-matrix
            let w0: float Matrix = matrix [ls.[00..03]
                                           ls.[04..07]
                                           ls.[08..11]]
            /// 4-vector
            let b0: float Vector = vector ls.[12..15]

            /// 3x4-matrix
            let w1: float Matrix = matrix [ls.[16..19]
                                           ls.[20..23]
                                           ls.[24..27]]
            /// 4-vector
            let b1: float Vector = vector ls.[28..31]

            let p0: LayerGradients list = [{ w = w0; b = b0 }]
            let p1: LayerGradients list = [{ w = w1; b = b1 }]

            let actual = momentum beta p0 p1

            let expected: LayerGradients list = [{ w = beta * p0.[0].w + (1. - beta) * p1.[0].w
                                                   b = beta * p0.[0].b + (1. - beta) * p1.[0].b }]

            CF.AssertEqual(actual, expected)

        test |> Prop.forAll data
