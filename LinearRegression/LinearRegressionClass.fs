namespace LinearRegression

open Microsoft.ML

type LinearRegression<'TInput when 'TInput : not struct> (model: ITransformer) =
    let model = model
    
    new(path: string) =
        LinearRegression(LinearRegression.LoadModel path)

    new(path: string, nonencodedColumns: string[], encodedColumns: string[]) =
        LinearRegression(LinearRegression.TrainModel<'TInput> path nonencodedColumns encodedColumns)

    member this.TestModel (dataPath: string) =
        LinearRegression.TestModel<'TInput> model dataPath    
    
    member this.SaveModel (path: string) =
        LinearRegression.SaveModel model path

    member this.SingleSamplePrediction<'TOutput
                                        when 'TOutput : not struct
                                        and 'TOutput : (new: unit -> 'TOutput)>
                                        (sample: 'TInput) =
        LinearRegression.SingleSamplePrediction<'TInput, 'TOutput> model sample