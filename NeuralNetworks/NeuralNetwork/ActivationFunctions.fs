namespace NeuralNetwork

module ActivationFunctions =
    open MathNet.Numerics.LinearAlgebra

    open NNModel

    let private tanh_v: float Matrix -> float Matrix =
        Matrix.map tanh

    let private square (x: float): float = x * x

    let private tanh'_v (xs: float Matrix): float Matrix =
        1. - Matrix.map (tanh >> square) xs

    //let private relu_v: float Matrix -> float Matrix =
    //    Matrix.map (max 0.)

    //let private relu'_v (xs: float Matrix): float Matrix =
    //    Matrix.map (fun x -> if x > 0. then 1. else 0.) xs

    let private relu_v_leaky: float Matrix -> float Matrix =
        Matrix.map (fun x -> if x >= 0. then x else 0.01 * x)

    let private relu'_v_leaky: float Matrix -> float Matrix =
        Matrix.map (fun x -> if x >= 0. then 1. else 0.01)

    let private sigmoid (x: float): float =
        1. / (1. + (exp -x))

    let private sigmoid' (x: float): float =
        let s = sigmoid x in s * (1. - s)

    /// Output layer activation function
    let Sigmoid: ActivationFn = { f = sigmoid; f' = sigmoid' }
    /// Output layer activation function
    let Linear: ActivationFn = { f = id; f' = fun _ -> 1. }

    /// Hidden layer activation function
    let Tanh: ActivationFnVectorized = { f_v  = tanh_v
                                         f'_v = tanh'_v }

    //let ReLU: ActivationFnVectorized = { f_v  = relu_v
    //                                     f'_v = relu'_v }

    /// Hidden layer activation function
    let ReLU_Leaky: ActivationFnVectorized = { f_v  = relu_v_leaky
                                               f'_v = relu'_v_leaky }

    let vectorize_output_functions (fs: ActivationFn list): ActivationFnVectorized =
        { f_v  = Matrix.mapCols (fun i -> Vector.map fs.[i].f)
          f'_v = Matrix.mapCols (fun i -> Vector.map fs.[i].f') }
