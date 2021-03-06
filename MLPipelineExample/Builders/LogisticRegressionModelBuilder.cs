using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;
using MLPipelineExample.Models;
using Microsoft.ML.Data;
using System.Collections;


namespace MLPipelineExample.Builders
{

    /// <summary>
    /// Builds a logistic regression model in ML.NET.
    /// </summary>
    public class LogisticRegressionModelBuilder
    {
        /// <summary>
        /// Date time format used to timestamp saved model and meta data files.
        /// For example, June, 5 2021 4:12:23pm translates to "20210605162123".
        /// </summary>
        private const string _dateTimeFormat = "yyyyMMddHHmmss";

        /// <summary>
        /// ML context used for all model training, building, and evaluation.
        /// </summary>
        public MLContext Context { get; private set; }
        
        /// <summary>
        /// Converts categorical variables using on-hot encoding.
        /// </summary>
        public OneHotEncodingEstimator CategoricalVariableConverter { get; private set; }

        /// <summary>
        /// Condenses all feature variables into one column.
        /// </summary>
        public ColumnConcatenatingEstimator FeatureVariableConcatenator { get; private set; }

        /// <summary>
        /// The dataset used for all training and testing 
        /// of the model.
        /// </summary>
        public IDataView DataSet { get; private set; }

        /// <summary>
        /// The variable to be predicted on.
        /// </summary>
        public string Label { get; private set; }

        // Options that the model uses to train on
        public LbfgsLogisticRegressionBinaryTrainer.Options TrainingOptions { get; private set; } = new LbfgsLogisticRegressionBinaryTrainer.Options()
        {
            FeatureColumnName = "Features",
            MaximumNumberOfIterations = 100, 
            OptimizationTolerance = 1e-8f
        };

        /// <summary>
        /// The trained model.
        /// </summary>
        public ITransformer TrainedModel { get; private set; }

        /// <summary>
        /// The classification metrics of the trained model.
        /// </summary>
        public BinaryClassificationMetrics BinaryClassificationMetrics { get; private set; }
        
        /// <summary>
        /// Default constructor. 
        /// </summary>
        public LogisticRegressionModelBuilder()
        {
            Context = new MLContext();
        }

        /// <summary>
        /// Loads the training data into the machine learning context.
        /// </summary>
        /// <param name="imageResults"></param>
        /// <returns></returns>
        public IDataView LoadTrainingData<T>(IEnumerable<T> dataSet) where T : class
        {
            DataSet = Context.Data.LoadFromEnumerable(dataSet);
            return DataSet; 
        }

        /// <summary>
        /// Sets the variables that will be interpreted as categorical 
        /// variables by the model trainer. The model trainer will use 
        /// one hot encoding to transform categorical variables.
        /// </summary>
        /// <param name="categoricalVariables"></param>
        public OneHotEncodingEstimator SetCategoricalVariables(string[] categoricalVariables)
        {
            var categoricalVariableList = new List<InputOutputColumnPair>(); 
            foreach (var key in categoricalVariables)
            {
                categoricalVariableList.Add(new InputOutputColumnPair(key));
            }

            CategoricalVariableConverter = Context.Transforms.Categorical.OneHotEncoding(categoricalVariableList.ToArray());

            return CategoricalVariableConverter; 
        }

        /// <summary>
        /// Set the feature variables of the model. The label variable 
        /// (the variable to be predicted) cannot be included here or the 
        /// training will fail.
        /// </summary>
        /// <param name="featureVariables"></param>
        /// <returns></returns>
        public ColumnConcatenatingEstimator SetFeatureVariables(string[] featureVariables)
        {
            FeatureVariableConcatenator = Context.Transforms.Concatenate(TrainingOptions.FeatureColumnName, featureVariables);
            return FeatureVariableConcatenator;
        }

        /// <summary>
        /// Sets the label variable (the variable to be predicted on)
        /// for the model.
        /// </summary>
        /// <param name="label"></param>
        /// <returns></returns>
        public string SetLabel(string label)
        {
            Label = label;
            return Label;
        }

        /// <summary>
        /// Trains the model. 
        /// </summary>
        /// <param name="options"></param>
        /// <returns></returns>
        public ITransformer TrainModel()
        {
            // set feature and label columns on options
            TrainingOptions.LabelColumnName = Label;

            // define the trainer
            var trainer = Context.BinaryClassification.Trainers.LbfgsLogisticRegression(TrainingOptions);

            // instantiate data pipe
            var dataPipe = new EstimatorChain<ColumnConcatenatingTransformer>();

            // no feature variables defined so can't train model
            if (FeatureVariableConcatenator == null)
            {
                return null; 
            }

            // no categorical variables set 
            if (CategoricalVariableConverter == null)
            {
                dataPipe = dataPipe.Append(FeatureVariableConcatenator);
            }
            else // categorical variables set 
            {
                dataPipe = dataPipe.Append(CategoricalVariableConverter).Append(FeatureVariableConcatenator);
            }

            // define the training pipe
            var trainingPipe = dataPipe.Append(trainer);

            // fit the model and return
            TrainedModel = trainingPipe.Fit(DataSet);
            return TrainedModel;
        }

        /// <summary>
        /// Evaluates the model on the training data provided.
        /// </summary>
        /// <returns></returns>
        public BinaryClassificationMetrics EvaluateModel()
        {
            IDataView predictions = TrainedModel.Transform(DataSet);
            var metrics = Context.BinaryClassification.EvaluateNonCalibrated(predictions, Label);
            BinaryClassificationMetrics = metrics;
            return metrics; 
        }

        /// <summary>
        /// Saves the trained model as a zip file and the model 
        /// meta data + metrics to a json file.
        /// <param name="filePath"></param>
        public SavedModelPathModel SaveModel(string directory = null)
        {

            // get binary classification metrics model
            var metrics = GetBinaryClassificationMetricsModel(BinaryClassificationMetrics);

            // get current utc timestamp
            var dateTime = DateTime.UtcNow;

            // create meta data model
            var modelMetaData = new BinaryClassifierMetaDataModel()
            {
                ModelGeneratedDateTime = dateTime,
                Metrics = metrics,
            };

            // create json serializer options
            var options = new JsonSerializerOptions()
            {
                WriteIndented = true, // so json is pretty
            };

            // convert meta data to json string
            var metaDataJson = JsonSerializer.Serialize(modelMetaData, options);

            // get a formatted date time string to append to file names
            // example 20210605134923 -> June 05, 2021 at 13:49:23.
            var dateTimeString = dateTime.ToString(_dateTimeFormat);

            // define directory path builders for model and meta data
            var modelPathBuilder = new StringBuilder();
            var metaDataPathBuilder = new StringBuilder();

            // save model and meta data to disk
            if (Directory.Exists(directory)) // save to provided directory if it exists
            {
                // append relative paths of model and meta data to provided directory
                modelPathBuilder.Append(directory + @"\model_" + dateTimeString + ".zip");
                metaDataPathBuilder.Append(directory + @"\modelmetadata_" + dateTimeString + ".json");
            }
            else
            {
                // get current working directory 
                var currentDirectory = Directory.GetCurrentDirectory();

                // append relative paths of model and meta data to current directory
                modelPathBuilder.Append(currentDirectory + @"\model_" + dateTimeString + ".zip");
                metaDataPathBuilder.Append(currentDirectory + @"\modelmetadata_" + dateTimeString + ".json");
            }
            
            // save meta data to json file 
            File.WriteAllText(metaDataPathBuilder.ToString(), metaDataJson);

            // save model to zip file
            Context.Model.Save(TrainedModel, DataSet.Schema, modelPathBuilder.ToString());

            // return paths of model and meta data
            return new SavedModelPathModel()
            {
                ModelPath = modelPathBuilder.ToString(),
                MetaDataPath = metaDataPathBuilder.ToString()
            };
        }

        /// <summary>
        /// Returns a human readable string of the evaluation 
        /// metrics for the model.
        /// </summary>
        /// <returns></returns>
        public static string GetPrintableMetrics(BinaryClassificationMetrics metrics)
        {
            if (metrics != null)
            {
                // build the string of model metrics
                StringBuilder builder = new StringBuilder();
                builder.Append("Model accuracy = " + metrics.Accuracy.ToString("F4") + "\n");
                builder.Append("F1 Score = " + metrics.F1Score.ToString("F4") + "\n\n");
                builder.Append(metrics.ConfusionMatrix.GetFormattedConfusionTable());

                // return the printable metrics
                return builder.ToString();
            }

            return null;
        }

        /// <summary>
        /// Takes a BinaryClassificationMetrics object and turns it into 
        /// a human readable BinaryClassificationMetricsModel. 
        /// </summary>
        /// <param name="metrics"></param>
        /// <returns></returns>
        public static BinaryClassificationMetricsModel GetBinaryClassificationMetricsModel(BinaryClassificationMetrics metrics)
        {
            // get counts for confusion matrix
            var counts = metrics.ConfusionMatrix.Counts;

            // create human readable confusion matrix model
            var confusionMatrixModel = new ConfusionMatrixModel()
            {
                TruePositiveCount = (int)counts[0][0],
                FalseNegativeCount = (int)counts[0][1],
                FalsePositiveCount = (int)counts[1][0],
                TrueNegativeCount = (int)counts[1][1]
            };

            // create and return BinaryClassificationMetricsModel
            return new BinaryClassificationMetricsModel()
            {
                Accuracy = metrics.Accuracy,
                AreaUnderPrecisionRecallCurve = metrics.AreaUnderPrecisionRecallCurve,
                AreaUnderRocCurve = metrics.AreaUnderRocCurve,
                F1Score = metrics.F1Score,
                PositivePrecision = metrics.PositivePrecision,
                NegativePrecision = metrics.NegativePrecision,
                PositiveRecall = metrics.PositiveRecall,
                NegativeRecall = metrics.NegativeRecall,
                ConfusionMatrix = confusionMatrixModel
            };
        }
    }
}
