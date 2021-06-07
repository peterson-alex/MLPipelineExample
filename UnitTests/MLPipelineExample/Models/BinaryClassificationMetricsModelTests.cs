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
    /// Unit tests for BinaryClassificationMetricsModel.
    /// </summary>
    public class BinaryClassificationMetricsModelTests
    {
        [Test]
        public void BinaryClassificationMetricsModel_Default_Not_Null()
        {
            // Act
            var item = new BinaryClassificationMetricsModel();

            // Assert
            Assert.NotNull(item);
        }

        [Test]
        public void BinaryClassificationMetricsModel_Default_Properties_Valid()
        {
            // Act
            var item = new BinaryClassificationMetricsModel();

            // Assert
            Assert.AreEqual(item.Accuracy, 0.0);
            Assert.AreEqual(item.AreaUnderPrecisionRecallCurve, 0.0);
            Assert.AreEqual(item.AreaUnderRocCurve, 0.0);
            Assert.AreEqual(item.ConfusionMatrix, null);
            Assert.AreEqual(item.F1Score, 0.0);
            Assert.AreEqual(item.NegativePrecision, 0.0);
            Assert.AreEqual(item.NegativeRecall, 0.0);
            Assert.AreEqual(item.PositivePrecision, 0.0);
            Assert.AreEqual(item.PositiveRecall, 0.0);
        }

        [Test]
        public void BinaryClassificationMetricsModel_Set_Properties_Valid()
        {
            // Act
            var item = new BinaryClassificationMetricsModel()
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

            // Assert
            Assert.AreEqual(item.Accuracy, 1.0);
            Assert.AreEqual(item.AreaUnderPrecisionRecallCurve, 2.0);
            Assert.AreEqual(item.AreaUnderRocCurve, 3.0);
            Assert.AreEqual(item.ConfusionMatrix.TruePositiveCount, 1);
            Assert.AreEqual(item.ConfusionMatrix.FalsePositiveCount, 2);
            Assert.AreEqual(item.ConfusionMatrix.TrueNegativeCount, 3);
            Assert.AreEqual(item.ConfusionMatrix.FalseNegativeCount, 4);
            Assert.AreEqual(item.F1Score, 4.0);
            Assert.AreEqual(item.NegativePrecision, 5.0);
            Assert.AreEqual(item.NegativeRecall, 6.0);
            Assert.AreEqual(item.PositivePrecision, 7.0);
            Assert.AreEqual(item.PositiveRecall, 8.0);
        }
    }
}
