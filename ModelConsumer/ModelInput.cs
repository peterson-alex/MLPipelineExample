using System;
using System.Collections.Generic;
using System.Text;

namespace ModelConsumer
{
    public class ModelInput
    {
        /// <summary>
        /// The bilirubin value of the reading.
        /// </summary>
        public float Value { get; set; }

        /// <summary>
        /// The user that took the reading.
        /// </summary>
        public string UserID { get; set; }

        /// <summary>
        /// Whether the reading was declared successful
        /// or not.
        /// </summary>
        public bool ReadingSuccess { get; set; }
    }
}
