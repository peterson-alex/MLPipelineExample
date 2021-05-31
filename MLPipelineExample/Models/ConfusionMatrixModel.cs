using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPipelineExample.Models
{
    /// <summary>
    /// Holds the four parameters of a confusion matrix. 
    /// </summary>
    public class ConfusionMatrixModel
    {
        /// <summary>
        /// The number of true positives in the test data.
        /// </summary>
        public int TruePositiveCount { get; set; }

        /// <summary>
        /// The number of false positives in the test data.
        /// </summary>
        public int FalsePositiveCount { get; set; }

        /// <summary>
        /// The number of true negatives in the test data.
        /// </summary>
        public int TrueNegativeCount { get; set; }

        /// <summary>
        /// The number of false negatives in the test data.
        /// </summary>
        public int FalseNegativeCount { get; set; }
    }
}
