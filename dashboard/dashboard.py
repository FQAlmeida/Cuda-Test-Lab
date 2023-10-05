import streamlit as st
import polars as pl

import plotly.express as px

st.set_page_config(page_title="Convolution Comparison GPU vs CPU Par", layout="wide")

df = pl.read_csv("data/bench.csv")
df = df.select(
    pl.col("name").str.split("/").list.get(0).alias("name"),
    pl.col("name").str.split("/").list.get(1).cast(pl.UInt64).alias("size"),
    pl.col("name").str.split("/").list.get(2).cast(pl.UInt64).alias("iters"),
    (pl.col("name").str.split("/").list.get(3).cast(pl.UInt32) * 2 + 1).alias(
        "kernel_size"
    ),
    pl.col("real_time").cast(pl.UInt64).alias("real_time_raw"),
)

df_speedup = (
    df.group_by(["size", "iters", "kernel_size"])
    .agg(
        pl.col("name"),
        pl.col("real_time_raw"),
        pl.col("real_time_raw").alias("speedup"),
    )
    .explode("name", "real_time_raw")
    .with_columns(
        pl.when(pl.col("name") == "BM_convolution_cpu_par")
        .then(
            pl.col("speedup").list.reverse().list.first()
            / pl.col("speedup").list.reverse().list.last()
        )
        .otherwise(pl.col("speedup").list.first() / pl.col("speedup").list.last())
        .alias("speedup")
    )
).sort(by=["size", "iters", "kernel_size"])

df = df.join(
    other=df_speedup,
    how="inner",
    on=["name", "size", "iters", "kernel_size", "real_time_raw"],
).with_columns(
    pl.duration(nanoseconds=pl.col("real_time_raw").cast(pl.UInt64)).alias("real_time")
)

# st.dataframe(df_speedup.to_pandas())

for size, iters in (
    df.select(pl.col("size"), pl.col("iters"))
    .unique()
    .sort(
        by=[
            pl.col("iters"),
            pl.col("size"),
        ]
    )
    .iter_rows()
):
    plot = (
        px.line(
            data_frame=df.filter((pl.col("size") == size) & (pl.col("iters") == iters)),
            y="real_time",
            x="kernel_size",
            color="name",
            text="kernel_size",
            title=f"Benchmark size({32*int(size)}^2) iters({iters})",
            hover_name="name",
            hover_data=["speedup"],
        )
        # .update_xaxes(tickformat="%M:%S.%f")
        .update_traces(textposition="top center")
    )
    st.plotly_chart(plot, use_container_width=True)
