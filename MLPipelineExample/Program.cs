using System;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json;
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

            // create new logistic regression model builder
            var builder = new LogisticRegressionModelBuilder();

            // load training data and set variables
            // note that these steps can be performed in any order
            builder.LoadTrainingData(imageResults);
            builder.SetCategoricalVariables(new string[] { "UserID" });
            builder.SetFeatureVariables(new string[] { "UserID", "Value" });
            builder.SetLabel("ReadingSuccess");

            // train the model and get performance metrics
            builder.TrainModel();
            var metrics = builder.EvaluateModel();

            // print metrics
            Console.Write(LogisticRegressionModelBuilder.GetPrintableMetrics(metrics));

            // save the model
            builder.SaveModel(); 
        }

        /// <summary>
        /// Produces an ImageResultIntputViewModel from the provided json data.
        /// </summary>
        /// <param name="JsonFilePath"></param>
        /// <returns></returns>
        public static List<ImageResultBasicViewModel> ConvertJsonToImageResultInputModels(string JsonFilePath)
        {
            // convert json to ImageResultViewModel
            return JsonConvert.DeserializeObject<List<ImageResultBasicViewModel>>(File.ReadAllText(JsonFilePath));
        }
    }
}
