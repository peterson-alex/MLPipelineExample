using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
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
    }
}
