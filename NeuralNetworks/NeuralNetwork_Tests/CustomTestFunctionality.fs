namespace NeuralNetwork_Tests

open Xunit.Sdk
open MathNet.Numerics.LinearAlgebra

type CustomTestFunctionality() =
    static member AssertEqual(msg: string, actual: 'a, expected: 'a): unit =
        if actual = expected then ()
        else raise <| AssertActualExpectedException(expected, actual, msg)

    static member AssertEqual(actual: 'a, expected: 'a): unit =
        CustomTestFunctionality.AssertEqual("", actual, expected)

    static member Round(decimals: int, c: float): float =
        let mag = 10.0 ** (float decimals)
        round (c * mag) / mag

    static member Round(decimals: int, v: float Vector): float Vector =
        Vector.map (fun c -> CustomTestFunctionality.Round(decimals, c)) v

    static member Round(decimals: int, m: float Matrix): float Matrix =
        Matrix.map (fun c -> CustomTestFunctionality.Round(decimals, c)) m
