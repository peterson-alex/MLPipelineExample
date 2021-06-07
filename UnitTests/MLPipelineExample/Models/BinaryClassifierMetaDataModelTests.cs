using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLPipelineExample.Models;
using NUnit.Framework;

namespace UnitTests.MLPipelineExample.Models
{
    public class BinaryClassifierMetaDataModelTests
    {
        [Test]
        public void BinaryClassifierMetaDataModel_Default_Not_Null()
        {
            // Act
            var item = new BinaryClassifierMetaDataModel();

            // Assert
            Assert.NotNull(item);
        }

        [Test]
        public void BinaryClassifierMetaDataModel_Default_Properties_Valid()
        {
            // Act
            var item = new BinaryClassifierMetaDataModel();

            // Assert
            Assert.NotNull(item.ModelGeneratedDateTime);
            Assert.AreEqual(item.Metrics, null);
        }

        [Test]
        public void ImageResultBasicDataViewModel_Set_Properties_Valid()
        {
            // Act
            var metrics = new BinaryClassificationMetricsModel()
            {
                Accuracy = 1.0,
                AreaUnderPrecisionRecallCurve = 2.0,
                AreaUnderRocCurve = 3.0,
                ConfusionMatrix = new ConfusionMatrixModel
                {
                    TruePositiveCount = 1,
                    FalsePositiveCount = 2,
                    TrueNegativeCount = 3,
                    FalseNegativeCount = 4
                },
                F1Score = 4.0,
                NegativePrecision = 5.0,
                NegativeRecall = 6.0,
                PositivePrecision = 7.0,
                PositiveRecall = 8.0
            };

            // Act
            var item = new BinaryClassifierMetaDataModel()
            {
                ModelGeneratedDateTime = DateTime.MinValue,
                Metrics = metrics
            };

            // Assert
            Assert.AreEqual(item.ModelGeneratedDateTime, DateTime.MinValue);
            Assert.AreEqual(item.Metrics, metrics);
        }
    }
}
