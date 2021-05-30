using System;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLPipelineExample.Models;
using MLPipelineExample.Builders;


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
            var imageResults = ConvertJsonToImageResultInputModels(_defaultDataPath);

            var builder = new LogisticRegressionModelBuilder();
            builder.LoadTrainingData(imageResults);
            builder.SetCategoricalVariables(new string[] { "UserID" });
            builder.SetFeatureVariables(new string[] { "UserID", "Value" });
            builder.SetLabel("ReadingSuccess");
            builder.TrainModel();
            var metrics = builder.EvaluateModel();

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
        public static List<ImageResultInputModel> ConvertJsonToImageResultInputModels(string JsonFilePath)
        {
            // convert json to ImageResultViewModel
            var ImageResultInputViewModel = JsonConvert.DeserializeObject<ImageResultInputViewModel>(File.ReadAllText(JsonFilePath));

            // pull out the image results and return
            return ImageResultInputViewModel.ImageResults;
        }
    }
}
