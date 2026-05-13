import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

sns.boxplot(x='sex', y='age', hue='survived', data=titanic)

plt.xlabel('Gender')

plt.ylabel('Age')

plt.title('Age Distribution by Gender and Survival')

plt.show()

2. Observations and Inference
Female passengers had a higher survival rate compared to male passengers.
Most surviving females were between age group 20 to 40.
Male passengers show more spread in age distribution.
Many younger passengers survived compared to older passengers.
The box plot shows presence of some outliers in age data.
Median age of passengers is around 30 years.
  
import org.apache.spark.sql.SparkSession

object SparkExample {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession.builder()
            .appName("Spark Example")
            .master("local[*]")
            .getOrCreate()

        val numbers = Seq(10, 20, 30, 40, 50)

        val rdd = spark.sparkContext.parallelize(numbers)

        val sum = rdd.reduce(_ + _)

        println("Sum = " + sum)

        spark.stop()
    }
}
