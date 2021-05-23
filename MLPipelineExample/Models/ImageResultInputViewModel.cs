using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPipelineExample.Models
{
    /// <summary>
    /// Wrapper class to facilitate the conversion of json 
    /// data to ImageResultInputModels.
    /// </summary>
    public class ImageResultInputViewModel
    {
        public List<ImageResultInputModel> ImageResults { get; set; }
    }
}
