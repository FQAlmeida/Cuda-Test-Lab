import streamlit as st
import polars as pl

import plotly.express as px

st.set_page_config(layout="wide")

st.title("Hello")

df = pl.read_csv("data/bench.csv")
df = df.select(
    pl.col("name").str.split("/").list.get(0).alias("name"),
    pl.col("name").str.split("/").list.get(1).cast(pl.UInt64).alias("size"),
    pl.col("name").str.split("/").list.get(2).cast(pl.UInt64).alias("iters"),
    (pl.col("name").str.split("/").list.get(3).cast(pl.UInt32) * 2 + 1).alias(
        "kernel_size"
    ),
    (
        pl.duration(nanoseconds=pl.col("real_time").cast(pl.UInt64))
        + pl.datetime(year=1970, month=1, day=1)
    ).alias("real_time"),
)

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
    plot = px.line(
        data_frame=df.filter((pl.col("size") == size) & (pl.col("iters") == iters)),
        y="real_time",
        x="kernel_size",
        color="name",
        title=f"Benchmark size({32*int(size)}^2) iters({iters})",
    ).update_xaxes(tickformat="%M:%S.%f")
    st.plotly_chart(plot, use_container_width=True)
