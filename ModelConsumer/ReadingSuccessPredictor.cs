using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;

namespace ModelConsumer
{
    public class ReadingSuccessPredictor
    {
        /// <summary>
        /// Prediction engine to make predictions using model. 
        /// </summary>
        private static Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictionEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(CreatePredictionEngine);

        /// <summary>
        /// Makes a reading success prediction given the 
        /// input. 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static ModelOutput Predict(ModelInput input)
        {
            ModelOutput result = PredictionEngine.Value.Predict(input);
            return result;
        }

        /// <summary>
        /// Creates prediction engine that can be used to predict
        /// reading success on the given input.
        /// </summary>
        /// <returns></returns>
        public static PredictionEngine<ModelInput, ModelOutput> CreatePredictionEngine()
        {
            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Get the absolute path to the model
            string modelPath = AppDomain.CurrentDomain.BaseDirectory + @"\model_latest.zip";

            // create the prediction engine
            ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            return predEngine;
        }
    }
}
