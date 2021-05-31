using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPipelineExample.Models
{
    /// <summary>
    /// Holds meta data for a binary classification model, 
    /// including the model metrics.
    /// </summary>
    public class BinaryClassifierMetaDataModel
    {
        /// <summary>
        /// The Date + time that the model was generated.
        /// </summary>
        public DateTime ModelGeneratedDateTime { get; set; } = DateTime.Now;

        // other metadata goes here

        /// <summary>
        /// The metrics used to evaluate the model. 
        /// </summary>
        public BinaryClassificationMetricsModel Metrics { get; set; }
    }
}
