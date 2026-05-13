import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')

print(iris.head())
print(iris.info())
iris.hist(figsize=(10,8))

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(data=iris)

plt.show()
print(iris.describe())
sns.boxplot(x=iris['sepal_width'])

plt.show()
Inference
Most features are normally distributed.
Petal length and petal width show clear variation among species.
Sepal width contains some outliers visible in the box plot.
Median values are approximately centered in most distributions.
Petal measurements have less spread compared to sepal measurements.

import org.apache.spark.sql.SparkSession

object SimpleSparkApp {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession.builder()
            .appName("Simple Spark App")
            .master("local[*]")
            .getOrCreate()

        val data = Seq(
            ("Amit", 85),
            ("Rahul", 90),
            ("Sneha", 88)
        )

        import spark.implicits._

        val df = data.toDF("Name", "Marks")

        df.show()

        spark.stop()
    }
}
