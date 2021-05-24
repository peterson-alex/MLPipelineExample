using System;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLPipelineExample.Models;


namespace MLPipelineExample
{
    /// <summary>
    /// This project will demonstrate how to create a logistic
    /// regression model from read in json data using ML.NET. 
    /// </summary>
    public class Program
    {
        // path to default json data
        private static readonly string _defaultDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "mockexampledata.json");

        public static void Main(string[] args)
        {
            // read the json data into the view model
            var imageResults = ConvertJsonToImageResultViewModel();

            // create ml context
            var context = new MLContext();

            // load data into context
            IDataView trainingData = context.Data.LoadFromEnumerable(imageResults);

            // get estimators for categorical and feature variables
            var userIdEstimator = GetOneHotEncodingEstimator(context, "UserID");
            var features = new[] { "UserID", "Value" };
            var featureEstimator = GetConcatenatedFeaturesEstimator(context, features);

            // chain estimators together
            var dataPipe = userIdEstimator.Append(featureEstimator);

            // define options for logistic regression trainer
            var options = new LbfgsLogisticRegressionBinaryTrainer.Options()
            {
                LabelColumnName = "ReadingSuccess",
                FeatureColumnName = "Features",
                MaximumNumberOfIterations = 100,
                OptimizationTolerance = 1e-8f
            };

            // train the model
            var model = TrainModel(context, trainingData, dataPipe, options);

            // get model performance metrics
            var metrics = GetModelMetrics(context, model, trainingData, "ReadingSuccess");

            // evaluate the model
            Console.Write("Model accuracy on training data = ");
            Console.WriteLine(metrics.Accuracy.ToString("F4"));
            Console.Write("F1 Score on training data = ");
            Console.WriteLine(metrics.F1Score.ToString("F4"));
        }

        /// <summary>
        /// Produces an ImageResultIntputViewModel from the provided json data.
        /// </summary>
        /// <param name="JsonFilePath"></param>
        /// <returns></returns>
        public static List<ImageResultInputModel> ConvertJsonToImageResultViewModel(string JsonFilePath = null)
        {
            // if provided file path is null, use default
            if (JsonFilePath == null)
            {
                // convert json to ImageResultInputViewModel
                var DefaultImageResultViewModel = JsonConvert.DeserializeObject<ImageResultInputViewModel>(File.ReadAllText(_defaultDataPath));

                // pull out the Image Results and return
                return DefaultImageResultViewModel.ImageResults;
            }

            // convert json to ImageResultViewModel
            var ImageResultInputViewModel = JsonConvert.DeserializeObject<ImageResultInputViewModel>(File.ReadAllText(JsonFilePath));

            // pull out the image results and return
            return ImageResultInputViewModel.ImageResults;
        }

        /// <summary>
        /// Prepares a categorical variable defined by 'key' to be 
        /// transformed via One Hot Encoding.
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public static OneHotEncodingEstimator GetOneHotEncodingEstimator(MLContext context, string key)
        {
            return context.Transforms.Categorical.OneHotEncoding(new[]
            {
                new InputOutputColumnPair(key)
            });
        }

        /// <summary>
        /// Prepares feature variables as a concatenated 
        /// column estimator.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="features"></param>
        /// <returns></returns>
        public static ColumnConcatenatingEstimator GetConcatenatedFeaturesEstimator(MLContext context, string[] features)
        {
            // concatenate features into one output column
            return context.Transforms.Concatenate("Features", features);
        }

        /// <summary>
        /// Trains the logistic regression model on the provided training 
        /// data. 
        /// </summary>
        /// <param name="options"></param>
        /// <returns></returns>
        public static ITransformer TrainModel(MLContext context, IDataView trainingData, 
            EstimatorChain<ColumnConcatenatingTransformer> dataPipe,
            LbfgsLogisticRegressionBinaryTrainer.Options options)
        {
            // define training pipe
            var trainer = context.BinaryClassification.Trainers.LbfgsLogisticRegression(options);
            var trainPipe = dataPipe.Append(trainer);

            return trainPipe.Fit(trainingData);
        }

        /// <summary>
        /// Returns the metrics for training data of 
        /// the binary classification model.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="model"></param>
        /// <param name="trainingData"></param>
        /// <param name="predictedLabel"></param>
        /// <returns></returns>
        public static BinaryClassificationMetrics GetModelMetrics(MLContext context, ITransformer model, IDataView trainingData, string predictedLabel)
        {
            // get model metrics
            IDataView predictions = model.Transform(trainingData);
            return context.BinaryClassification.EvaluateNonCalibrated(predictions, "ReadingSuccess", "Score");
        }
    }
}
