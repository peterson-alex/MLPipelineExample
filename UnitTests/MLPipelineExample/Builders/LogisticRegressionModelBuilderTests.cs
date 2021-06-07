using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using MLPipelineExample.Builders;

namespace UnitTests.MLPipelineExample.Builders
{
    /// <summary>
    /// Unit tests for LogisticRegressionModelBuilder.
    /// </summary>
    public class LogisticRegressionModelBuilderTests
    {
        /// <summary>
        /// Test the constructor. 
        /// </summary>
        [Test]
        public void Constructor_Default_Not_Null()
        {
            // Act
            var builder = new LogisticRegressionModelBuilder();

            // Assert
            Assert.NotNull(builder);
        }

        /// <summary>
        /// Tests each public property to ensure that they hold 
        /// expected values after object initialization.
        /// </summary>
        [Test]
        public void Constructor_Default_Properties_Valid()
        {
            // Act
            var builder = new LogisticRegressionModelBuilder();

            // Assert
            Assert.NotNull(builder.Context);
            Assert.AreEqual(builder.CategoricalVariableConverter, null);
            Assert.AreEqual(builder.FeatureVariableConcatenator, null);
            Assert.AreEqual(builder.DataSet, null);
            Assert.AreEqual(builder.Label, null);
            Assert.NotNull(builder.TrainingOptions);
            Assert.AreEqual(builder.TrainingOptions.FeatureColumnName, "Features");
            Assert.AreEqual(builder.TrainingOptions.MaximumNumberOfIterations, 100);
            Assert.AreEqual(builder.TrainingOptions.OptimizationTolerance, 1e-8f);
            Assert.AreEqual(builder.TrainedModel, null);
            Assert.AreEqual(builder.BinaryClassificationMetrics, null);
        }
    }
}
