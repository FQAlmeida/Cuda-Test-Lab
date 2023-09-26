import streamlit as st
import polars as pl

import plotly.express as px

st.title("Hello")

df = pl.read_csv("data/bench.csv")
df = df.select(
    pl.col("name").str.split("/").list.get(0).alias("name"),
    pl.col("name").str.split("/").list.get(1).alias("size"),
    pl.col("name").str.split("/").list.get(2).alias("iters"),
    (pl.col("name").str.split("/").list.get(3).cast(pl.UInt32) * 2 + 1).alias(
        "kernel_size"
    ),
    pl.col("real_time").keep_name(),
).to_pandas()

plot = px.line(
    data_frame=df, y="real_time", x="kernel_size", color=["name", "size", "iters"]
)
st.plotly_chart(plot)
