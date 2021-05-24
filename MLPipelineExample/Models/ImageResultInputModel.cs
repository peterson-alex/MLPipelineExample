﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace MLPipelineExample.Models
{
    /// <summary>
    /// An ImageResultInputModel represents an instance
    /// of the data used to train the model.
    /// </summary>
    public class ImageResultInputModel
    {
        /// <summary>
        /// The ID of the user who took the reading.
        /// </summary>
        public string UserID { get; set; }

        /// <summary>
        /// The value of the Bilirubin reading.
        /// </summary>
        public float Value { get; set; }

        /// <summary>
        /// Indicates whether the reading was successful 
        /// or not (true if successful, false otherwise).
        /// </summary>
        public bool ReadingSuccess { get; set; }
    }
}
