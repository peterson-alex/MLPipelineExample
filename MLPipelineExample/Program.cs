using System;
using System.IO;
using Newtonsoft.Json;
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
            var ImageResultInputViewModel = ConvertJsonToImageResultViewModel();

            // pull out the ImageResultInputModels
            var ImageResultInputModels = ImageResultInputViewModel.ImageResults;
        }

        /// <summary>
        /// Produces an ImageResultIntputViewModel from the provided json data.
        /// </summary>
        /// <param name="JsonFilePath"></param>
        /// <returns></returns>
        public static ImageResultInputViewModel ConvertJsonToImageResultViewModel(string JsonFilePath = null)
        {
            // if provided file path is null, use default
            if (JsonFilePath == null)
            {
                // convert json to ImageResultInputViewModel
                return JsonConvert.DeserializeObject<ImageResultInputViewModel>(File.ReadAllText(_defaultDataPath));
            }

            // convert json to ImageResultViewModel
            return JsonConvert.DeserializeObject<ImageResultInputViewModel>(File.ReadAllText(JsonFilePath));
        }
    }
}
