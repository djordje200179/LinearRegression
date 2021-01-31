namespace LinearRegressor

open Microsoft.ML

type LinearRegressor<'TInput when 'TInput : not struct> (model: ITransformer) =
    let model = model
    
    new(path: string) =
        LinearRegressor(LinearRegressor.LoadModel path)

    new(path: string, nonencodedColumns: string[], encodedColumns: string[]) =
        LinearRegressor(LinearRegressor.TrainModel<'TInput> path nonencodedColumns encodedColumns)

    member this.TestModel (dataPath: string) =
        LinearRegressor.TestModel<'TInput> model dataPath    
    
    member this.SaveModel (path: string) =
        LinearRegressor.SaveModel model path

    member this.SingleSamplePrediction<'TOutput when 'TOutput : not struct and 'TOutput : (new: unit -> 'TOutput)> (sample: 'TInput) =
        LinearRegressor.SingleSamplePrediction<'TInput, 'TOutput> model sample