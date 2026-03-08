import argparse
import logging
from pathlib import Path

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger("bronze_to_silver")


def get_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("bronze_to_silver_nbp")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform NBP bronze JSONL into silver Delta table.")
    parser.add_argument("--bronze-path", default="data/bronze/nbp_raw.jsonl")
    parser.add_argument("--silver-path", default="data/delta/silver/nbp_rates")
    return parser.parse_args()


def main() -> None:
    logger = setup_logger()
    args = parse_args()

    bronze_schema = StructType(
        [
            StructField("ingestion_ts", StringType(), True),
            StructField("source_url", StringType(), True),
            StructField("table_type", StringType(), True),  # lineage/traceability
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

    spark = get_spark()

    try:
        logger.info("Reading bronze data from %s", args.bronze_path)
        bronze = spark.read.schema(bronze_schema).json(args.bronze_path)

        parsed = (
            bronze.withColumn("payload", F.from_json(F.col("raw_payload"), payload_schema))
            .withColumn("ingestion_ts", F.to_timestamp("ingestion_ts"))
            .select("table_type", "effective_date", "ingestion_ts", "payload")
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
            .filter(F.length(F.col("currency_code")) == 3)
            .filter(F.col("mid_rate").isNotNull())
            .filter(F.col("mid_rate") > 0)
        )

        # Keep only latest row per business key inside current batch
        w = Window.partitionBy("table_type", "rate_date", "currency_code").orderBy(F.col("ingestion_ts").desc())
        silver_batch = exploded.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")

        silver_path = Path(args.silver_path)
        delta_exists = (silver_path / "_delta_log").exists()

        if not delta_exists:
            logger.info("Creating silver Delta table at %s", args.silver_path)
            (
                silver_batch.write.format("delta")
                .mode("overwrite")
                .partitionBy("rate_date")
                .save(args.silver_path)
            )
        else:
            logger.info("Merging batch into existing silver Delta table at %s", args.silver_path)
            target = DeltaTable.forPath(spark, args.silver_path)
            (
                target.alias("t")
                .merge(
                    silver_batch.alias("s"),
                    """
                    t.table_type = s.table_type
                    AND t.rate_date = s.rate_date
                    AND t.currency_code = s.currency_code
                    """,
                )
                .whenMatchedUpdate(
                    set={
                        "currency_name": "s.currency_name",
                        "mid_rate": "s.mid_rate",
                        "ingestion_ts": "s.ingestion_ts",
                    }
                )
                .whenNotMatchedInsert(
                    values={
                        "table_type": "s.table_type",
                        "rate_date": "s.rate_date",
                        "currency_code": "s.currency_code",
                        "currency_name": "s.currency_name",
                        "mid_rate": "s.mid_rate",
                        "ingestion_ts": "s.ingestion_ts",
                    }
                )
                .execute()
            )

        logger.info("Silver transformation completed successfully")
    except Exception:
        logger.exception("Silver transformation failed")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
