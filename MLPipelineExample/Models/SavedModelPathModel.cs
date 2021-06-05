using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPipelineExample.Models
{
    /// <summary>
    /// Stores the full paths of the saved model and the 
    /// model meta data file.
    /// </summary>
    public class SavedModelPathModel
    {
        /// <summary>
        /// The full path to the model (.zip file).
        /// </summary>
        public string ModelPath { get; set; }

        /// <summary>
        /// The full path to the meta data file (.json file).
        /// </summary>
        public string MetaDataPath { get; set; }
    }
}
