open LinearRegressor.LinearRegressor
open System.IO
open System
open Models

[<EntryPoint>]
let main argv =
    let trainDataPath = Path.Combine(__SOURCE_DIRECTORY__, "Data", "train.csv")
    let testDataPath = Path.Combine(__SOURCE_DIRECTORY__ , "Data", "test.csv")
    let nonencodedColumns = [|"PassengerCount"; "TripDistance"|]
    let encodedColumns = [|"VendorId"; "RateCode"; "PaymentType"|]

    try
        let model =
            if not (File.Exists "model.lrm") then
                printfn "DEBUG: Training model..." 
                TrainModel<TaxiTrip> trainDataPath nonencodedColumns encodedColumns
            else
                printfn "Trained model is found"
                printf "Do you want to load the existing model or train a new model? [L/T] "
            
                match Console.ReadLine() with
                | "L" | "l" -> 
                    printfn "DEBUG: Loading model..."
                    LoadModel "model.lrm"
                | "T" | "t" ->
                    printfn "DEBUG: Training model..." 
                    TrainModel<TaxiTrip> trainDataPath nonencodedColumns encodedColumns
                | _ -> failwith "Invalid input"
                
        printfn "DEBUG: Testing model..."
        let metrics = TestModel<TaxiTrip> model testDataPath

        printfn ""
        printfn "*************************************************"
        printfn "*       Model quality metrics evaluation         "
        printfn "*------------------------------------------------"
        printfn $"*  R2 Score:       %.2f{metrics.RSquared}"
        printfn $"*  RMS Error:      %.2f{metrics.RootMeanSquaredError}" 
        printfn "*************************************************"

        printfn "DEBUG: Saving model..."
        SaveModel model "model.lrm"

        let sample = {
            VendorId = "VTS"
            RateCode = int RateCode.Standard
            PassengerCount = 1f
            TripTime = 1140f
            TripDistance = 3.75f
            PaymentType = "CRD"
            FareAmount = 15.5f
        }

        let sampleResult = SingleSamplePrediction<TaxiTrip, TaxiTripFarePrediction> model sample

        printfn ""
        printfn "*************************************************"
        printfn "*       Testing model with a sample              "
        printfn "*------------------------------------------------"
        printfn $"*  Predicted fare: %.2f{sampleResult.FareAmount}" 
        printfn $"*  Actual fare:    %.2f{sample.FareAmount}" 
        printfn "*************************************************"
    with
    | Failure(msg) -> printfn "ERROR: %s" msg
    | _ as ex -> printfn "ERROR: %s" ex.Message
    0