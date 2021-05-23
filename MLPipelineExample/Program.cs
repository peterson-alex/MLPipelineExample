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
        public static void Main(string[] args)
        {
            // read the json data into the view model
            var ImageResultInputViewModel = ConvertJsonToImageResultViewModel();
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
                // get path to default json file
                var workingDirectory = Environment.CurrentDirectory;
                var projectDirectory = Directory.GetParent(workingDirectory).Parent.Parent.FullName;
                var DefaultJsonFilePath = Path.Combine(projectDirectory, @"Data\mockexampledata.json");

                // convert json to ImageResultInputViewModel
                return JsonConvert.DeserializeObject<ImageResultInputViewModel>(File.ReadAllText(DefaultJsonFilePath));
            }

            return JsonConvert.DeserializeObject<ImageResultInputViewModel>(File.ReadAllText(JsonFilePath));
        }
    }
}
