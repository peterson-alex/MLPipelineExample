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
    /// Unit Tests for ImageResultBasicViewModel
    /// </summary>
    public class ImageResultBasicViewModelTests
    {
        [Test]
        public void ImageResultBasicViewModel_Default_Not_Null()
        {
            // Act
            var item = new ImageResultBasicViewModel();

            // Assert
            Assert.NotNull(item);
        }

        [Test]
        public void ImageResultBasicViewModel_Default_Properties_Valid()
        {
            // Act
            var item = new ImageResultBasicViewModel();

            // Assert
            Assert.AreEqual(item.Value, 0.0f);
            Assert.AreEqual(item.UserID, null);
            Assert.AreEqual(item.ReadingSuccess, false);
        }

        [Test]
        public void ImageResultBasicViewModel_Set_Properties_Valid()
        {
            // Act
            var item = new ImageResultBasicViewModel()
            {
                Value = 1.0f,
                UserID = "bogus",
                ReadingSuccess = true
            };

            // Assert
            Assert.AreEqual(item.Value, 1.0f);
            Assert.AreEqual(item.UserID, "bogus");
            Assert.AreEqual(item.ReadingSuccess, true);
        }
    }
}
