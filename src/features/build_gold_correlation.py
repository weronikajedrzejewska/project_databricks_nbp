import argparse
import logging
from typing import Set

from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession, Window, functions as F

SILVER_PATH_DEFAULT = "data/delta/silver/nbp_rates"
GOLD_CORR_PATH_DEFAULT = "data/delta/gold/fx_correlation_30d"


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return logging.getLogger("gold_correlation")


def get_spark() -> SparkSession:
    builder = (
        SparkSession.builder.appName("build_gold_correlation")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 30-observation FX correlation snapshot.")
    parser.add_argument("--silver-path", default=SILVER_PATH_DEFAULT)
    parser.add_argument("--gold-path", default=GOLD_CORR_PATH_DEFAULT)
    parser.add_argument("--silver-table", default=None, help="UC table, e.g. fx_lakehouse.nbp.silver_nbp_rates")
    parser.add_argument("--gold-table", default=None, help="UC table, e.g. fx_lakehouse.nbp.gold_fx_correlation_30d")
    return parser.parse_args()


def validate_input(df: DataFrame, required: Set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def read_silver(spark: SparkSession, path: str, table: str | None) -> DataFrame:
    return spark.read.table(table) if table else spark.read.format("delta").load(path)


def write_gold(df: DataFrame, path: str, table: str | None) -> None:
    if table:
        df.write.format("delta").mode("overwrite").partitionBy("as_of_date").saveAsTable(table)
    else:
        df.write.format("delta").mode("overwrite").partitionBy("as_of_date").save(path)


def build_correlation_snapshot(df: DataFrame) -> DataFrame:
    """
    Build correlation snapshot for all FX pairs using last 30 observations per currency.

    Output schema:
        as_of_date
        currency_a
        currency_b
        corr_30d
        obs_cnt
    """
    validate_input(df, {"rate_date", "currency_code", "mid_rate"})

    # returns_1d
    w_order = Window.partitionBy("currency_code").orderBy("rate_date")
    with_ret = (
        df.select("rate_date", "currency_code", "mid_rate")
        .withColumn("return_1d", (F.col("mid_rate") / F.lag("mid_rate", 1).over(w_order)) - 1)
        .filter(F.col("return_1d").isNotNull())
    )

    # last 30 observations per currency
    w_desc = Window.partitionBy("currency_code").orderBy(F.col("rate_date").desc())
    last30 = with_ret.withColumn("rn", F.row_number().over(w_desc)).filter(F.col("rn") <= 30).drop("rn")

    a = last30.alias("a")
    b = last30.alias("b")

    joined = (
        a.join(b, on=[F.col("a.rate_date") == F.col("b.rate_date")], how="inner")
        .filter(F.col("a.currency_code") < F.col("b.currency_code"))
    )

    corr = (
        joined.groupBy(
            F.max(F.col("a.rate_date")).alias("as_of_date"),
            F.col("a.currency_code").alias("currency_a"),
            F.col("b.currency_code").alias("currency_b"),
        )
        .agg(
            F.corr(F.col("a.return_1d"), F.col("b.return_1d")).alias("corr_30d"),
            F.count("*").alias("obs_cnt"),
        )
    )

    return corr


def main() -> None:
    logger = setup_logger()
    args = parse_args()
    spark = get_spark()

    try:
        logger.info("Reading silver")
        silver = read_silver(spark, args.silver_path, args.silver_table)

        logger.info("Building correlation snapshot")
        corr = build_correlation_snapshot(silver)

        logger.info("Writing gold correlation")
        write_gold(corr, args.gold_path, args.gold_table)

        logger.info("Gold correlation completed")
    except Exception:
        logger.exception("Gold correlation failed")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
