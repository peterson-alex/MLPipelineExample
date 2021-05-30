using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using MLPipelineExample.Models;

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
            return _context.Transforms.Concatenate("Features", featureVariables);
        }
    }
}
