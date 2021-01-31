namespace LinearRegressor

module LinearRegressor =
    open Microsoft.ML
    open Microsoft.ML.Data

    let context = MLContext 0
    let schema = ref null

    let private LoadData<'T> (context: MLContext) path =
        context.Data.LoadFromTextFile<'T>(path, ',', true)

    let private CreateEstimator (context: MLContext) (nonencodedColumns: string[]) (encodedColumns: string[]) = 
        let AddEncodedColumn (estimator: EstimatorChain<Transforms.OneHotEncodingTransformer>) (column: string) =
            estimator.Append(context.Transforms.Categorical.OneHotEncoding column)

        let CompleteEstimator (estimator: EstimatorChain<Transforms.OneHotEncodingTransformer>) =
            let totalList = Array.append encodedColumns nonencodedColumns

            estimator
                .Append(context.Transforms.Concatenate("Features", totalList))
                .AppendCacheCheckpoint(context)
                .Append(context.Regression.Trainers.FastTree())

        encodedColumns
        |> Array.fold AddEncodedColumn (EstimatorChain())
        |> CompleteEstimator 

    let TrainModel<'T> (path: string) (nonencodedColumns: string[]) (encodedColumns: string[]) =
        let SaveSchema (schema: DataViewSchema ref) (data: IDataView) = 
            schema := data.Schema
            data

        path
        |> LoadData<'T> context 
        |> SaveSchema schema 
        |> (CreateEstimator context nonencodedColumns encodedColumns).Fit 
        :> ITransformer

    let TestModel<'T> (model: ITransformer) (dataPath: string) =
        dataPath 
        |> LoadData<'T> context 
        |> model.Transform 
        |> context.Regression.Evaluate

    let LoadModel (path: string) =
        let ExtractModel (schemaReference: DataViewSchema ref) ((model: ITransformer), (schema: DataViewSchema)) = 
            schemaReference := schema
            model

        path 
        |> context.Model.Load
        |> ExtractModel schema

    let SaveModel (model: ITransformer) (path: string) = 
        context.Model.Save(model, !schema, path)

    let SingleSamplePrediction<'TInput, 'TOutput when 'TInput : not struct and 'TOutput : not struct and 'TOutput : (new: unit -> 'TOutput)> 
            (model: ITransformer) (sample: 'TInput) =
        let engine = context.Model.CreatePredictionEngine<'TInput, 'TOutput> model
        let prediction = engine.Predict sample
        prediction