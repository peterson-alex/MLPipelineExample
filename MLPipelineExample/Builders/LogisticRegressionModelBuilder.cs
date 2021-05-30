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

namespace MLPipelineExample.Builders
{

    /// <summary>
    /// Builds a logistic regression model in ML.NET.
    /// </summary>
    public class LogisticRegressionModelBuilder
    {

        private MLContext _context; // ML context used to train and build model
        private IDataView _trainingData; // training data used to train model
        private OneHotEncodingEstimator _categoricalVariables; // variables that will be interpreted as categorical variables by the trainer
        private ColumnConcatenatingEstimator _featureVariables; // feature variables of the model
        private string _featureVariablesName = "Features"; // the column name of the feature variables
        private string _label; // the variable to be predicted on
        private LbfgsLogisticRegressionBinaryTrainer.Options _trainingOptions = new LbfgsLogisticRegressionBinaryTrainer.Options()
        {
            MaximumNumberOfIterations = 100, 
            OptimizationTolerance = 1e-8f
        }; // default options to train model on
        ITransformer _model; // the model after it has been transformed
        
        /// <summary>
        /// Default constructor. 
        /// </summary>
        public LogisticRegressionModelBuilder()
        {
            _context = new MLContext();
        }

        /// <summary>
        /// Constructor. 
        /// </summary>
        /// <param name="context"></param>
        public LogisticRegressionModelBuilder(MLContext context)
        {
            _context = context;
        }

        /// <summary>
        /// Loads the training data into the 
        /// </summary>
        /// <param name="imageResults"></param>
        /// <returns></returns>
        public IDataView LoadTrainingData(IEnumerable<ImageResultInputModel> imageResults)
        {
            _trainingData = _context.Data.LoadFromEnumerable(imageResults);
            return _trainingData; 
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
            _label = label;
            return _label;
        }

        /// <summary>
        /// Trains the model. 
        /// </summary>
        /// <param name="options"></param>
        /// <returns></returns>
        public ITransformer TrainModel()
        {
            // set feature and label columns on options
            _trainingOptions.LabelColumnName = _label;
            _trainingOptions.FeatureColumnName = _featureVariablesName;

            // define the trainer
            var trainer = _context.BinaryClassification.Trainers.LbfgsLogisticRegression(_trainingOptions);

            // instantiate data pipe
            var dataPipe = new EstimatorChain<ColumnConcatenatingTransformer>();

            // append categorical variables to data pipe
            if (_categoricalVariables != null)
            {
                dataPipe.Append(_categoricalVariables);
            }
            
            // append feature variables to data pipe
            if (_featureVariables != null)
            {
                dataPipe.Append(_featureVariables);
            }

            // define the training pipe
            var trainingPipe = dataPipe.Append(trainer);

            // fit the model and return
            _model = trainingPipe.Fit(_trainingData);
            return _model;
        }

        /// <summary>
        /// Evaluates the model on the training data provided.
        /// </summary>
        /// <returns></returns>
        public BinaryClassificationMetrics EvalauteModel()
        {
            IDataView predictions = _model.Transform(_trainingData);
            return _context.BinaryClassification.EvaluateNonCalibrated(predictions, _label, "Score");
        }
    }
}
