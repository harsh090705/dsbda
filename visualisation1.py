import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

print(titanic.head())

sns.countplot(x='survived', data=titanic)
plt.show()

sns.countplot(x='sex', hue='survived', data=titanic)
plt.show()

sns.countplot(x='pclass', hue='survived', data=titanic)
plt.show()

sns.heatmap(titanic.isnull(), yticklabels=False, cbar=False)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

plt.hist(titanic['fare'], bins=30)

plt.xlabel('Fare')

plt.ylabel('Number of Passengers')

plt.title('Distribution of Ticket Fare')

plt.show()

import org.apache.spark.sql.SparkSession

object SimpleSparkProgram {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession.builder()
            .appName("Simple Spark Program")
            .master("local[*]")
            .getOrCreate()

        val data = Seq(
            ("Harsh", 21),
            ("Rahul", 22),
            ("Sneha", 20)
        )

        import spark.implicits._

        val df = data.toDF("Name", "Age")

        df.show()

        spark.stop()
    }
}
