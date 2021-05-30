using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;

namespace MLPipelineExample.Builders
{

    /// <summary>
    /// Builds a logistic regression model in ML.NET.
    /// </summary>
    public class LogisticRegressionModelBuilder
    {

        private MLContext _context; // ML context used to train and build model
        
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
    }
}
