module Models

open Microsoft.ML.Data

type RateCode = 
    | Standard = 1
    | JFK = 2 
    | Newark = 3
    | Nassau = 4 
    | Negotiated = 5 
    | Group = 6

type TaxiTrip = {
    [<LoadColumn(0)>] VendorId: string
    [<LoadColumn(1)>] RateCode: int
    [<LoadColumn(2)>] PassengerCount: single
    [<LoadColumn(3)>] TripTime: single
    [<LoadColumn(4)>] TripDistance: single
    [<LoadColumn(5)>] PaymentType: string
    [<LoadColumn(6); ColumnName("Label")>] FareAmount: single
}

[<CLIMutable>]
type TaxiTripFarePrediction = {
    [<ColumnName("Score")>] FareAmount: single
}