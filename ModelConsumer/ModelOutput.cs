using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ModelConsumer
{
    public class ModelOutput
    {
        /// <summary>
        /// The predicted value for the given 
        /// input. 
        /// </summary>
        [ColumnName("ReadingSuccess")]
        public bool Prediction { get; set; }

        /// <summary>
        /// The f1 score associated with this prediction.
        /// </summary>
        public float Score { get; set; }
    }
}
