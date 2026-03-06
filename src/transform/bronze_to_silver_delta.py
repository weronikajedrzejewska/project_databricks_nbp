from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)
from delta import configure_spark_with_delta_pip

BRONZE_PATH = "data/bronze/nbp_raw.jsonl"
SILVER_DELTA_PATH = "data/delta/silver/nbp_rates"


def get_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("bronze_to_silver_nbp")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()

def main() -> None:
    spark = get_spark()

    bronze_schema = StructType(
        [
            StructField("ingestion_ts", StringType(), True),
            StructField("source_url", StringType(), True),
            StructField("table_type", StringType(), True),
            StructField("effective_date", StringType(), True),
            StructField("raw_payload", StringType(), True),
        ]
    )

    rate_schema = StructType(
        [
            StructField("currency", StringType(), True),
            StructField("code", StringType(), True),
            StructField("mid", DoubleType(), True),
        ]
    )

    payload_schema = StructType(
        [
            StructField("table", StringType(), True),
            StructField("effectiveDate", StringType(), True),
            StructField("rates", ArrayType(rate_schema), True),
        ]
    )

    bronze = spark.read.schema(bronze_schema).json(BRONZE_PATH)

    parsed = (
        bronze.withColumn("payload", F.from_json(F.col("raw_payload"), payload_schema))
        .withColumn("ingestion_ts", F.to_timestamp("ingestion_ts"))
    )

    exploded = (
        parsed.withColumn("rate", F.explode(F.col("payload.rates")))
        .select(
            F.col("table_type"),
            F.to_date(F.coalesce(F.col("effective_date"), F.col("payload.effectiveDate"))).alias("rate_date"),
            F.col("rate.code").alias("currency_code"),
            F.col("rate.currency").alias("currency_name"),
            F.col("rate.mid").alias("mid_rate"),
            F.col("ingestion_ts"),
        )
        .filter(F.col("rate_date").isNotNull())
        .filter(F.col("currency_code").isNotNull())
    )

    w = Window.partitionBy("table_type", "rate_date", "currency_code").orderBy(F.col("ingestion_ts").desc())

    silver = (
        exploded.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    silver.write.format("delta").mode("overwrite").save(SILVER_DELTA_PATH)

    print(f"OK: Silver Delta written to {SILVER_DELTA_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
