using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using MLPipelineExample.Models;

namespace UnitTests.MLPipelineExample.Models
{
    /// <summary>
    /// Unit tests for SavedModelPathModel.
    /// </summary>
    public class SavedModelPathModelTests
    {
        /// <summary>
        /// Unit Tests for ImageResultBasicViewModel
        /// </summary>
        public class ImageResultBasicViewModelTests
        {
            [Test]
            public void SavedModelPathModel_Default_Not_Null()
            {
                // Act
                var item = new SavedModelPathModel();

                // Assert
                Assert.NotNull(item);
            }

            [Test]
            public void SavedModelPathModel_Default_Properties_Valid()
            {
                // Act
                var item = new SavedModelPathModel();

                // Assert
                Assert.AreEqual(item.ModelPath, null);
                Assert.AreEqual(item.MetaDataPath, null);
            }

            [Test]
            public void SavedModelPathModel_Set_Properties_Valid()
            {
                // Act
                var item = new SavedModelPathModel()
                {
                    ModelPath = "bogus1",
                    MetaDataPath = "bogus2"
                };

                // Assert
                Assert.AreEqual(item.ModelPath, "bogus1");
                Assert.AreEqual(item.MetaDataPath, "bogus2");
            }
        }
    }
}
