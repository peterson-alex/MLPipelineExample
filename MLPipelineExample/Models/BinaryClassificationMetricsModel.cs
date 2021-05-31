using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPipelineExample.Models
{
    /// <summary>
    /// Holds all of the evaluation metrics for a 
    /// binary classification model. 
    /// </summary>
    public class BinaryClassificationMetricsModel
    {
        /// <summary>
        /// Accuracy of the classifier. The closer to 
        /// 1.00, the better, provided the data sample is 
        /// balanced between the two possible classes.
        /// </summary>
        public double Accuracy { get; set; }

        /// <summary>
        /// The area under the precision recall curve. Values closer 
        /// to 1.00 indicate high precision and high recall.
        /// </summary>
        public double AreaUnderPrecisionRecallCurve { get; set; }

        /// <summary>
        /// Area under the Roc (receiving operating characeristic)
        /// curve. The closer to 1.00, the better. If under .50, the 
        /// model is worthless.
        /// </summary>
        public double AreaUnderRocCurve { get; set; }

        /// <summary>
        /// F1 score. The F1 score represents a balance between precision
        /// and recall. The closer to 1.00, the better.
        /// </summary>
        public double F1Score { get; set; }

        /// <summary>
        /// The ratio of true positives divided by total 
        /// predicted positives.
        /// </summary>
        public double PositivePrecision { get; set; }

        /// <summary>
        /// The ratio of true negatives divided by total 
        /// predicted negatives.
        /// </summary>
        public double NegativePrecision { get; set; }
        
        /// <summary>
        /// The ratio of true positives divided by total 
        /// actual positives.
        /// </summary>
        public double PositiveRecall { get; set; }

        /// <summary>
        /// The ratio of true negatives divided by total
        /// actual negatives.
        /// </summary>
        public double NegativeRecall { get; set; }

        /// <summary>
        /// The full confusion matrix of the model test.
        /// </summary>
        public ConfusionMatrixModel ConfusionMatrix { get; set; }
    }
}
