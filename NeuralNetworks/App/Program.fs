//open LogicalCircuit
open Mnist
open MathNet.Numerics.LinearAlgebra

[<EntryPoint>]
let main _ =
    //run_logic_training ()
    run_mnist_training ()

    //let a: float [,] = Array2D.create 6 3 0.
    //let m = DenseMatrix.ofArray2 a

    //printfn "%A" m

    0 // return an integer exit code

(* TODO
    * verification module (?)
    * test on data
        Generated test data

        Iris
        MNIST
        Titanic
*)
