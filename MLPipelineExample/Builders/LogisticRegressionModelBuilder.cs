using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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

        private MLContext _context; // ML context used to train and build model
        private OneHotEncodingEstimator _categoricalVariables; // variables that will be interpreted as categorical variables by the trainer
        private ColumnConcatenatingEstimator _featureVariables; // feature variables of the model
        private const string _featureVariablesName = "Features"; // the column name of the feature variables

        /// <summary>
        /// The dataset used for all training and testing 
        /// of the model.
        /// </summary>
        public IDataView DataSet { get; private set; }

        /// <summary>
        /// The variable to be predicted on.
        /// </summary>
        public string Label { get; private set; }

        /// default options to train model on
        private LbfgsLogisticRegressionBinaryTrainer.Options _trainingOptions = new LbfgsLogisticRegressionBinaryTrainer.Options()
        {
            MaximumNumberOfIterations = 100, 
            OptimizationTolerance = 1e-8f
        }; 
        // ITransformer _trainedModel; // the model after it has been transformed

        /// <summary>
        /// The trained model.
        /// </summary>
        public ITransformer TrainedModel { get; private set; }
        
        /// <summary>
        /// Default constructor. 
        /// </summary>
        public LogisticRegressionModelBuilder()
        {
            _context = new MLContext();
        }

        /// <summary>
        /// Loads the training data into the machine learning context.
        /// </summary>
        /// <param name="imageResults"></param>
        /// <returns></returns>
        public IDataView LoadTrainingData<T>(IEnumerable<T> dataSet) where T : class
        {
            DataSet = _context.Data.LoadFromEnumerable(dataSet);
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

            _categoricalVariables = _context.Transforms.Categorical.OneHotEncoding(categoricalVariableList.ToArray());

            return _categoricalVariables; 
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
            _featureVariables = _context.Transforms.Concatenate("Features", featureVariables);
            return _featureVariables;
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
            _trainingOptions.LabelColumnName = Label;
            _trainingOptions.FeatureColumnName = _featureVariablesName;

            // define the trainer
            var trainer = _context.BinaryClassification.Trainers.LbfgsLogisticRegression(_trainingOptions);

            // instantiate data pipe
            var dataPipe = new EstimatorChain<ColumnConcatenatingTransformer>();

            // no feature variables defined so can't train model
            if (_featureVariables == null)
            {
                return null; 
            }

            // no categorical variables set 
            if (_categoricalVariables == null)
            {
                dataPipe = dataPipe.Append(_featureVariables);
            }
            else // categorical variables set 
            {
                dataPipe = dataPipe.Append(_categoricalVariables).Append(_featureVariables);
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
            return _context.BinaryClassification.EvaluateNonCalibrated(predictions, Label);
        }

        /// <summary>
        /// Saves the trained model as a zip file. If no file path is 
        /// specified, saves in working directory as 'model.zip'.
        /// </summary>
        /// <param name="filePath"></param>
        public void SaveModel(string filePath = null)
        {
            if (filePath != null)
            {
                _context.Model.Save(TrainedModel, DataSet.Schema, filePath);
            }
            else
            {
                _context.Model.Save(TrainedModel, DataSet.Schema, "model.zip");
            }
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
    }
}
