using Microsoft.ML.Data;

namespace TestOOP.Models {
    public enum RateCode {
        Standard = 1,
        JFK,
        Newark,
        Nassau,
        Negotiated,
        Group,
    }

    public record TaxiTrip {
        [LoadColumn(0)]
        public string VendorId { get; init; }

        [LoadColumn(1)]
        public int RateCode { get; init; }

        [LoadColumn(2)]
        public float PassengerCount { get; init; }

        [LoadColumn(3)]
        public float TripTime { get; init; }

        [LoadColumn(4)]
        public float TripDistance { get; init; }

        [LoadColumn(5)]
        public string PaymentType { get; init; }

        [LoadColumn(6), ColumnName("Label")]
        public float FareAmount { get; init; }
    }

    public class TaxiTripFarePrediction {
        [ColumnName("Score")]
        public float FareAmount;
    }
}