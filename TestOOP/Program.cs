using TestOOP.Models;
using System;
using System.IO;
using LinearRegressor;

namespace TestOOP {
	internal static class Program {
		private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");
		private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test.csv");

		private static readonly string[] _nonencodedColumns = {
			nameof(TaxiTrip.PassengerCount),
			nameof(TaxiTrip.TripDistance)
		};

		private static readonly string[] _encodedColumns = {
			nameof(TaxiTrip.VendorId),
			nameof(TaxiTrip.RateCode),
			nameof(TaxiTrip.PaymentType)
		};

		private static void Main(string[] args) {
			try {
				LinearRegressor<TaxiTrip> regressor;

				#region Creating model
				if (!File.Exists("model.lrm")) {
					Console.WriteLine("DEBUG: Training model...");

					regressor = new LinearRegressor<TaxiTrip>(_trainDataPath, _nonencodedColumns, _encodedColumns);
				} else {
					Console.WriteLine("Trained model is found");
					Console.Write("Do you want to load the existing model or train a new model? [L/T] ");

					var input = Console.ReadLine();
					if (input is "L" or "l") {
						Console.WriteLine("DEBUG: Loading model...");

						regressor = new LinearRegressor<TaxiTrip>( "model.lrm");
					} else if (input is "T" or "t") {
						Console.WriteLine("DEBUG: Training model...");

						regressor = new LinearRegressor<TaxiTrip>(_trainDataPath, _nonencodedColumns, _encodedColumns);
					} else
						throw new ArgumentException("Invalid input");
				}
				#endregion

				#region Testing model
				Console.WriteLine("DEBUG: Testing model...");
				var metrics = regressor.TestModel(_testDataPath);

				Console.WriteLine();
				Console.WriteLine("*************************************************");
				Console.WriteLine("*       Model quality metrics evaluation         ");
				Console.WriteLine("*------------------------------------------------");
				Console.WriteLine($"*  R2 Score:       {metrics.RSquared, 0:F2}");
				Console.WriteLine($"*  RMS Error:      {metrics.RootMeanSquaredError, 0:F2}");
				Console.WriteLine("*************************************************");
				#endregion

				#region Saving model
				Console.WriteLine("DEBUG: Saving model...");
				regressor.SaveModel("model.lrm");
				#endregion

				#region Testing model with a single sample
				var sample = new TaxiTrip() {
					VendorId = "VTS",
					RateCode = (int)RateCode.Standard,
					PassengerCount = 1,
					TripTime = 1140,
					TripDistance = 3.75f,
					PaymentType = "CRD",
					FareAmount = 15.5f
				};

				var sampleResult = regressor.SingleSamplePrediction<TaxiTripFarePrediction>(sample);

				Console.WriteLine();
				Console.WriteLine("*************************************************");
				Console.WriteLine("*       Testing model with a sample              ");
				Console.WriteLine("*------------------------------------------------");
				Console.WriteLine($"*  Predicted fare: {sampleResult.FareAmount, 0:F2}");
				Console.WriteLine($"*  Actual fare:    {sample.FareAmount, 0:F2}");
				Console.WriteLine("*************************************************");
				#endregion
			} catch (Exception e) {
				Console.WriteLine($"ERROR: {e.Message}");
			}
		}
	}
}
