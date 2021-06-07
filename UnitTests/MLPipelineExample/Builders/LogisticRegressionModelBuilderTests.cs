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
    }
}
