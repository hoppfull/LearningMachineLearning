namespace LogisticRegression

module Utilities =
    // column wise sums:
    let columnReduce<'T> =
        Seq.reduce<'T seq> << Seq.map2
