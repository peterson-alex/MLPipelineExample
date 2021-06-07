using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLPipelineExample.Models;
using NUnit.Framework;

namespace UnitTests.MLPipelineExample.Models
{
    /// <summary>
    /// Unit tests for ImageResultBasicDataViewModel.
    /// </summary>
    public class ImageResultBasicDataViewModelTests
    {
        [Test]
        public void ImageResultBasicDataViewModel_Default_Not_Null()
        {
            // Act
            var item = new ImageResultBasicDataViewModel();

            // Assert
            Assert.NotNull(item);
        }

        [Test]
        public void ImageResultBasicDataViewModel_Default_Properties_Null()
        {
            // Act
            var item = new ImageResultBasicDataViewModel();

            // Assert
            Assert.AreEqual(item.ImageResultBasicViewModels, null);
        }

        [Test]
        public void ImageResultBasicDataViewModel_Set_Properties_Not_Null()
        {
            // Arrange
            var imageResults = new List<ImageResultBasicViewModel>()
            { 
                new ImageResultBasicViewModel() 
                { 
                    Value = 1.0f, 
                    UserID = "bogus",
                    ReadingSuccess = true
                }
            };

            // Act
            var item = new ImageResultBasicDataViewModel()
            {
                ImageResultBasicViewModels = imageResults
            };

            // Assert
            Assert.NotNull(item.ImageResultBasicViewModels);
        }
    }
}
