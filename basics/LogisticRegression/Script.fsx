#load "DataTypes.fs"
#load "Utilities.fs"
#load "Optimization.fs"

open LogisticRegression.Optimization

let ws = [2.; 1.; 2.]

let xss = [[ 1.; 3.]
           [ 2.; 4.]
           [-1.; -3.2]]
          |> List.map (Seq.ofList)

let ys = [1.; 0.; 1.]

let dw, cost = propagate ws xss ys
