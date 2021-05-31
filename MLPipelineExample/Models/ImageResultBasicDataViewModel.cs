using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPipelineExample.Models
{
    /// <summary>
    /// View model that maps onto the shape of the json 
    /// data that will be consumed by the pipeline.
    /// </summary>
    public class ImageResultBasicDataViewModel
    {
        /// <summary>
        /// The list of ImageResultBasicViewModels
        /// </summary>
        public List<ImageResultBasicViewModel> ImageResultBasicViewModels { get; set; }
    }
}
