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
    /// Unit tests for Confusion Matrix model.
    /// </summary>
    public class ConfusionMatrixModelTests
    {
        [Test]
        public void ConfusionMatrixModel_Default_Not_Null()
        {
            // Act
            var item = new ConfusionMatrixModel();

            // Assert
            Assert.NotNull(item);
        }

        [Test]
        public void ConfusionMatrixModel_Default_Properties_Valid()
        {
            // Act
            var item = new ConfusionMatrixModel();

            // Assert
            Assert.AreEqual(item.TruePositiveCount, 0);
            Assert.AreEqual(item.TrueNegativeCount, 0);
            Assert.AreEqual(item.FalsePositiveCount, 0);
            Assert.AreEqual(item.FalseNegativeCount, 0);
        }

        [Test]
        public void ConfusionMatrixModel_Set_Properties_Valid()
        {
            // Act
            var item = new ConfusionMatrixModel()
            {
                TruePositiveCount = 1, 
                TrueNegativeCount = 2, 
                FalsePositiveCount = 3, 
                FalseNegativeCount = 4
            };

            // Assert
            Assert.AreEqual(item.TruePositiveCount, 1);
            Assert.AreEqual(item.TrueNegativeCount, 2);
            Assert.AreEqual(item.FalsePositiveCount, 3);
            Assert.AreEqual(item.FalseNegativeCount, 4);
        }
    }
}
