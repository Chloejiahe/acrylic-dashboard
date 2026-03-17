from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import Dash, html, dcc, Input, Output, State, callback


# =========================================================
# 路径
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

review_path = PROCESSED_DIR / "reviews_segmented.parquet"
quote_path = PROCESSED_DIR / "feature_quotes.parquet"
bundle_path = PROCESSED_DIR / "bundle_quotes.parquet"
attr_meta_path = PROCESSED_DIR / "attribute_label_meta.parquet"

required_files = [review_path, quote_path]
missing = [str(p.name) for p in required_files if not p.exists()]
if missing:
    raise FileNotFoundError(f"缺少这些处理后文件：{missing}。请先运行 segmentation_pipeline.py")


# =========================================================
# 读取数据
# =========================================================
review_df = pd.read_parquet(review_path)
quote_df = pd.read_parquet(quote_path)
bundle_df = pd.read_parquet(bundle_path) if bundle_path.exists() else pd.DataFrame()
attr_meta_df = pd.read_parquet(attr_meta_path) if attr_meta_path.exists() else pd.DataFrame()

# 只抓原始主标签列，不抓 score/hint/display
attr_top_cols = [
    c for c in review_df.columns
    if c.startswith("ATTR_TOP__")
    and not c.startswith("ATTR_TOP_DISPLAY__")
    and not c.startswith("ATTR_TOP_HINT__")
    and not c.startswith("ATTR_TOP_MATCHED__")
    and not c.startswith("ATTR_TOP_MATCHED_HINT__")
]
neg_cols = [c for c in review_df.columns if c.startswith("NEG__")]
pos_cols = [c for c in review_df.columns if c.startswith("POS__")]

COL_INK = "出墨方式" if "出墨方式" in review_df.columns else None
COL_PRICE = "Price Level" if "Price Level" in review_df.columns else None

segment_list = (
    sorted(review_df["primary_segment"].dropna().astype(str).unique().tolist())
    if "primary_segment" in review_df.columns else []
)
default_segment = segment_list[0] if segment_list else None


# =========================================================
# 工具函数
# =========================================================
def options_with_all(values):
    vals = sorted(pd.Series(values).dropna().astype(str).unique().tolist())
    return [{"label": "全部", "value": "ALL"}] + [{"label": v, "value": v} for v in vals]


def pct_text(value):
    value = 0 if pd.isna(value) else float(value)
    return f"{value * 100:.0f}%"


def star_text(value):
    if pd.isna(value):
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    full = max(0, min(5, int(round(v))))
    return "★" * full + "☆" * (5 - full)


def apply_global_filters(df, ink_mode, brand, price_level, asin):
    dff = df.copy()

    if COL_INK and ink_mode != "ALL":
        dff = dff[dff[COL_INK].astype(str) == str(ink_mode)]

    if brand != "ALL" and "Brand" in dff.columns:
        dff = dff[dff["Brand"].astype(str) == str(brand)]

    if COL_PRICE and price_level != "ALL":
        dff = dff[dff[COL_PRICE].astype(str) == str(price_level)]

    if asin != "ALL" and "Asin" in dff.columns:
        dff = dff[dff["Asin"].astype(str) == str(asin)]

    return dff


def get_attr_display_column(dim_name):
    return f"ATTR_TOP_DISPLAY__{dim_name}"


def get_attr_raw_column(dim_name):
    return f"ATTR_TOP__{dim_name}"


def get_attr_hint_column(dim_name):
    return f"ATTR_TOP_HINT__{dim_name}"


def resolve_attribute_display(temp_df, dim_name):
    """
    优先使用 pipeline 里写好的 ATTR_TOP_DISPLAY__维度
    若不存在，则回退到 attribute_label_meta.parquet
    再回退原 label
    """
    raw_col = get_attr_raw_column(dim_name)
    display_col = get_attr_display_column(dim_name)

    out = temp_df.copy()
    out["attribute_label_raw"] = out[raw_col].astype(str)

    if display_col in out.columns:
        out["attribute_label"] = out[display_col].fillna(out["attribute_label_raw"]).astype(str)
        return out

    if not attr_meta_df.empty:
        meta = attr_meta_df.copy()
        meta["attribute_dimension"] = meta["attribute_dimension"].astype(str)
        meta["attribute_label"] = meta["attribute_label"].astype(str)

        out["attribute_dimension"] = str(dim_name)
        out = out.merge(
            meta[["attribute_dimension", "attribute_label", "attribute_label_display"]],
            left_on=["attribute_dimension", "attribute_label_raw"],
            right_on=["attribute_dimension", "attribute_label"],
            how="left"
        )
        out["attribute_label"] = out["attribute_label_display"].fillna(out["attribute_label_raw"]).astype(str)
        return out

    out["attribute_label"] = out["attribute_label_raw"]
    return out


def build_attribute_long(dff):
    if dff.empty or not attr_top_cols:
        return pd.DataFrame(columns=[
            "primary_segment", "attribute_dimension", "attribute_label",
            "attribute_label_raw", "label_count", "dimension_mentioned_count", "pct"
        ])

    rows = []

    for col in attr_top_cols:
        dim_name = col.replace("ATTR_TOP__", "")

        temp = dff[["primary_segment", col] + ([get_attr_display_column(dim_name)] if get_attr_display_column(dim_name) in dff.columns else [])].copy()
        temp = temp.dropna(subset=["primary_segment", col])
        temp = temp[temp[col].astype(str) != "未提及"].copy()

        if temp.empty:
            continue

        temp = resolve_attribute_display(temp, dim_name)

        cnt = (
            temp.groupby(["primary_segment", "attribute_label", "attribute_label_raw"], dropna=False)
            .size()
            .reset_index(name="label_count")
        )
        den = (
            temp.groupby("primary_segment")
            .size()
            .reset_index(name="dimension_mentioned_count")
        )

        out = cnt.merge(den, on="primary_segment", how="left")
        out["attribute_dimension"] = dim_name
        out["pct"] = out["label_count"] / out["dimension_mentioned_count"]

        rows.append(out[[
            "primary_segment", "attribute_dimension", "attribute_label",
            "attribute_label_raw", "label_count", "dimension_mentioned_count", "pct"
        ]])

    if not rows:
        return pd.DataFrame(columns=[
            "primary_segment", "attribute_dimension", "attribute_label",
            "attribute_label_raw", "label_count", "dimension_mentioned_count", "pct"
        ])

    return pd.concat(rows, ignore_index=True)


def build_attribute_matrix_tables(attr_long, segment_order):
    if attr_long.empty:
        return dbc.Alert("当前筛选条件下暂无人群画像数据。", color="light")

    children = []
    dim_order = attr_long["attribute_dimension"].dropna().astype(str).unique().tolist()

    for dim in dim_order:
        dim_df = attr_long[attr_long["attribute_dimension"].astype(str) == str(dim)].copy()
        if dim_df.empty:
            continue

        pivot = dim_df.pivot_table(
            index="attribute_label",
            columns="primary_segment",
            values="pct",
            aggfunc="max"
        ).fillna(0)

        ordered_segments = [s for s in segment_order if s in pivot.columns]
        if not ordered_segments:
            ordered_segments = pivot.columns.tolist()

        ordered_labels = (
            dim_df.groupby("attribute_label")["pct"]
            .max()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        pivot = pivot.reindex(index=ordered_labels, columns=ordered_segments, fill_value=0)

        header = html.Thead(
            html.Tr([html.Th("标签")] + [html.Th(seg) for seg in ordered_segments])
        )

        body_rows = []
        for label, row in pivot.iterrows():
            row_max = float(row.max()) if len(row) else 0
            cells = [html.Td(label, className="matrix-row-title")]

            for seg in ordered_segments:
                value = float(row.get(seg, 0))
                class_name = "matrix-best-cell" if row_max > 0 and value == row_max else "matrix-cell"
                cells.append(html.Td(pct_text(value), className=class_name))

            body_rows.append(html.Tr(cells))

        table = dbc.Table(
            [header, html.Tbody(body_rows)],
            bordered=True,
            hover=True,
            responsive=True,
            class_name="profile-matrix-table mb-0"
        )

        children.append(
            dbc.Card(
                dbc.CardBody([
                    html.H6(dim, className="matrix-dimension-title"),
                    table
                ]),
                className="mb-3"
            )
        )

    return children if children else dbc.Alert("当前筛选条件下暂无人群画像数据。", color="light")


def build_segment_attribute_cards(attr_long, selected_segment):
    if attr_long.empty or not selected_segment:
        return dbc.Alert("请选择一个人群查看 attribute 细分占比。", color="light")

    seg_df = attr_long[attr_long["primary_segment"].astype(str) == str(selected_segment)].copy()
    if seg_df.empty:
        return dbc.Alert("当前筛选条件下，该人群暂无 attribute 细分数据。", color="light")

    dim_order = seg_df["attribute_dimension"].dropna().astype(str).unique().tolist()
    cards = []

    for dim in dim_order:
        dim_df = (
            seg_df[seg_df["attribute_dimension"].astype(str) == str(dim)]
            .sort_values(["pct", "label_count"], ascending=[False, False])
            .copy()
        )

        rows = []
        for _, row in dim_df.iterrows():
            value = 0 if pd.isna(row["pct"]) else float(row["pct"])
            label_text = str(row["attribute_label"])

            rows.append(
                html.Div([
                    html.Div(label_text, className="attribute-progress-label"),
                    html.Div([
                        dbc.Progress(
                            value=round(value * 100, 1),
                            class_name="attribute-progress-bar",
                            style={"height": "12px"}
                        ),
                        html.Span(pct_text(value), className="attribute-progress-value")
                    ], className="attribute-progress-right")
                ], className="attribute-progress-row")
            )

        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6(dim, className="matrix-dimension-title"),
                        html.Div(rows)
                    ]),
                    className="segment-breakdown-card h-100"
                ),
                xs=12, lg=6, xxl=4,
                className="mb-3"
            )
        )

    return dbc.Row(cards, className="g-3") if cards else dbc.Alert("当前筛选条件下，该人群暂无 attribute 细分数据。", color="light")


def get_analysis_frames(ink_mode, brand, price_level, asin, selected_segment, analysis_scope):
    base_reviews = apply_global_filters(review_df, ink_mode, brand, price_level, asin)

    if analysis_scope == "segment" and selected_segment and "primary_segment" in base_reviews.columns:
        analysis_reviews = base_reviews[base_reviews["primary_segment"].astype(str) == str(selected_segment)].copy()
    else:
        analysis_reviews = base_reviews.copy()

    qdf = quote_df.copy()
    if "review_id" in qdf.columns and "review_id" in analysis_reviews.columns:
        valid_ids = set(analysis_reviews["review_id"].tolist())
        qdf = qdf[qdf["review_id"].isin(valid_ids)]

    if analysis_scope == "segment" and selected_segment and "primary_segment" in qdf.columns:
        qdf = qdf[qdf["primary_segment"].astype(str) == str(selected_segment)]

    bdf = bundle_df.copy()
    if not bdf.empty and "review_id" in bdf.columns and "review_id" in analysis_reviews.columns:
        valid_ids = set(analysis_reviews["review_id"].tolist())
        bdf = bdf[bdf["review_id"].isin(valid_ids)]

    return analysis_reviews, qdf, bdf


def build_overall_feature_overview(qdf):
    cols = [
        "feature_dimension", "positive_mentions", "negative_mentions", "total_mentions",
        "satisfaction_pct", "avg_rating"
    ]
    if qdf.empty or "feature_dimension" not in qdf.columns:
        return pd.DataFrame(columns=cols)

    temp = qdf.copy()
    temp["pos"] = (temp["sentiment_type"].astype(str) == "positive").astype(int) if "sentiment_type" in temp.columns else 0
    temp["neg"] = (temp["sentiment_type"].astype(str) == "negative").astype(int) if "sentiment_type" in temp.columns else 0

    grp = temp.groupby("feature_dimension", dropna=False).agg(
        positive_mentions=("pos", "sum"),
        negative_mentions=("neg", "sum")
    ).reset_index()

    grp["total_mentions"] = grp["positive_mentions"] + grp["negative_mentions"]
    grp = grp[grp["total_mentions"] > 0].copy()

    if "Rating" in temp.columns:
        rating_df = temp.groupby("feature_dimension", dropna=False)["Rating"].mean().reset_index(name="avg_rating")
        grp = grp.merge(rating_df, on="feature_dimension", how="left")
    else:
        grp["avg_rating"] = 0

    grp["satisfaction_pct"] = grp["positive_mentions"] / grp["total_mentions"]
    grp = grp.sort_values("total_mentions", ascending=False)

    return grp[cols]


def build_feature_overview_chart(feature_overview, title_text):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if feature_overview.empty:
        fig.update_layout(title=title_text, template="plotly_white", height=540)
        return fig

    chart_df = feature_overview.head(15).copy()

    fig.add_trace(
        go.Bar(name="亮点", x=chart_df["feature_dimension"], y=chart_df["positive_mentions"]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(name="痛点", x=chart_df["feature_dimension"], y=chart_df["negative_mentions"]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="满意度 (%)",
            x=chart_df["feature_dimension"],
            y=(chart_df["satisfaction_pct"] * 100).round(1),
            mode="lines+markers+text",
            text=[f"{x:.1f}%" for x in (chart_df["satisfaction_pct"] * 100)],
            textposition="top center"
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            name="维度评分 (1-5)",
            x=chart_df["feature_dimension"],
            y=chart_df["avg_rating"].round(2),
            mode="lines+markers"
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        barmode="group",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    fig.update_xaxes(title_text="", tickangle=-35)
    fig.update_yaxes(title_text="句子提及次数", secondary_y=False)
    fig.update_yaxes(title_text="满意度 / 评分", range=[0, 100], secondary_y=True)
    return fig


def build_root_cause_card(qdf, selected_dim, sentiment_type):
    if qdf.empty:
        return dbc.Alert("当前筛选条件下暂无原声数据。", color="light")

    temp = qdf.copy()
    if selected_dim and "feature_dimension" in temp.columns:
        temp = temp[temp["feature_dimension"].astype(str) == str(selected_dim)]

    if sentiment_type != "ALL" and "sentiment_type" in temp.columns:
        temp = temp[temp["sentiment_type"].astype(str) == str(sentiment_type)]

    if temp.empty:
        return dbc.Alert("当前维度下暂无对应原声。", color="light")

    title = "核心投诉根因" if sentiment_type == "negative" else "核心亮点来源" if sentiment_type == "positive" else "核心标签分布"
    color = "danger" if sentiment_type == "negative" else "success" if sentiment_type == "positive" else "secondary"

    if "feature_tag" in temp.columns and temp["feature_tag"].notna().any():
        tag_df = temp["feature_tag"].fillna("未标注").astype(str).value_counts().head(8).reset_index()
        tag_df.columns = ["feature_tag", "cnt"]
        text_block = " / ".join([f"{r.feature_tag}({int(r.cnt)}次)" for r in tag_df.itertuples()])
    elif "matched_keywords" in temp.columns and temp["matched_keywords"].notna().any():
        kw_df = temp["matched_keywords"].fillna("未标注").astype(str).value_counts().head(8).reset_index()
        kw_df.columns = ["matched_keywords", "cnt"]
        text_block = " / ".join([f"{r.matched_keywords}({int(r.cnt)}次)" for r in kw_df.itertuples()])
    else:
        text_block = f"当前共提取到 {len(temp)} 条原声。"

    return dbc.Card(
        dbc.CardBody([
            html.H4(title, className="fw-bold mb-3"),
            html.Div(text_block, style={"fontSize": "20px", "lineHeight": "1.8"})
        ]),
        color=color,
        outline=True,
        className="mb-3"
    )


def build_quote_cards(qdf, selected_dim, sentiment_type):
    if qdf.empty:
        return dbc.Alert("当前筛选条件下暂无原声数据。", color="light")

    temp = qdf.copy()
    if selected_dim and "feature_dimension" in temp.columns:
        temp = temp[temp["feature_dimension"].astype(str) == str(selected_dim)]

    if sentiment_type != "ALL" and "sentiment_type" in temp.columns:
        temp = temp[temp["sentiment_type"].astype(str) == str(sentiment_type)]

    if temp.empty:
        return dbc.Alert("当前维度下暂无对应原声。", color="light")

    sort_cols = [c for c in ["Rating", "Brand", "Asin", "sentence_index"] if c in temp.columns]
    if sort_cols:
        temp = temp.sort_values(sort_cols)

    temp = temp.head(120).copy()

    items = []
    for _, row in temp.iterrows():
        badges = []

        if "Asin" in row and pd.notna(row["Asin"]):
            badges.append(dbc.Badge(str(row["Asin"]), color="info", className="me-2"))

        if "Brand" in row and pd.notna(row["Brand"]):
            badges.append(dbc.Badge(str(row["Brand"]), color="light", text_color="dark", className="me-2"))

        if "feature_tag" in row and pd.notna(row["feature_tag"]):
            badges.append(dbc.Badge(str(row["feature_tag"]), color="warning", className="me-2"))

        if "matched_keywords" in row and pd.notna(row["matched_keywords"]):
            badges.append(dbc.Badge(str(row["matched_keywords"]), color="secondary", className="me-2"))

        rating_block = (
            html.Span(star_text(row["Rating"]), style={"color": "#f4b400", "marginLeft": "8px"})
            if "Rating" in row and pd.notna(row["Rating"]) else ""
        )

        sentence_text = ""
        if "Sentence" in row and pd.notna(row["Sentence"]):
            sentence_text = str(row["Sentence"])
        elif "Content" in row and pd.notna(row["Content"]):
            sentence_text = str(row["Content"])

        items.append(
            dbc.Card(
                dbc.CardBody([
                    html.Div(badges + [rating_block], className="mb-2"),
                    html.Div(sentence_text, style={"fontSize": "16px", "lineHeight": "1.8"})
                ]),
                className="mb-2"
            )
        )

    return html.Div(
        items,
        style={"maxHeight": "520px", "overflowY": "auto", "paddingRight": "8px"}
    )


def build_bundle_summary(bdf):
    cols = ["bundle_category", "mention_count", "avg_rating", "detail_text"]
    if bdf.empty or "bundle_category" not in bdf.columns:
        return pd.DataFrame(columns=cols)

    grp = bdf.groupby("bundle_category", dropna=False).agg(
        mention_count=("bundle_category", "size"),
        avg_rating=("Rating", "mean")
    ).reset_index()

    if "bundle_sub_item" in bdf.columns:
        detail_rows = []
        for cat, sub in bdf.groupby("bundle_category", dropna=False):
            vc = sub["bundle_sub_item"].fillna("未标注").astype(str).value_counts().head(6)
            detail_rows.append({
                "bundle_category": cat,
                "detail_text": " / ".join([f"{k}({int(v)}次)" for k, v in vc.items()])
            })
        detail_df = pd.DataFrame(detail_rows)
        grp = grp.merge(detail_df, on="bundle_category", how="left")
    else:
        grp["detail_text"] = ""

    grp["avg_rating"] = grp["avg_rating"].round(2)
    grp = grp.sort_values("mention_count", ascending=False)
    return grp[cols]


def build_bundle_chart(bundle_summary):
    if bundle_summary.empty:
        fig = px.bar(title="配件类别提及频次排名")
        fig.update_layout(template="plotly_white", height=360)
        return fig

    chart_df = bundle_summary.head(10).copy()
    fig = px.bar(
        chart_df,
        x="mention_count",
        y="bundle_category",
        orientation="h",
        title="配件类别提及频次排名"
    )
    fig.update_layout(template="plotly_white", height=360, yaxis_title="", xaxis_title="提及次数")
    fig.update_traces(marker_color="#ff9800")
    return fig


def build_bundle_cards(bundle_summary, bdf):
    if bundle_summary.empty or bdf.empty:
        return dbc.Alert("当前评论样本中暂未提取到明显的配件搭配需求。", color="light")

    cards = []
    for _, row in bundle_summary.head(8).iterrows():
        cat = row["bundle_category"]
        cat_df = bdf[bdf["bundle_category"].astype(str) == str(cat)].copy()

        quote_items = []
        sort_cols = [c for c in ["Rating", "Asin"] if c in cat_df.columns]
        if sort_cols:
            cat_df = cat_df.sort_values(sort_cols)
        cat_df = cat_df.head(20)

        for _, q in cat_df.iterrows():
            quote_text = ""
            if "Sentence" in q and pd.notna(q["Sentence"]):
                quote_text = str(q["Sentence"])
            elif "Content" in q and pd.notna(q["Content"]):
                quote_text = str(q["Content"])

            badge_list = []
            if "Asin" in q and pd.notna(q["Asin"]):
                badge_list.append(dbc.Badge(str(q["Asin"]), color="warning", className="me-2"))
            if "bundle_sub_item" in q and pd.notna(q["bundle_sub_item"]):
                badge_list.append(dbc.Badge(str(q["bundle_sub_item"]), color="secondary", className="me-2"))
            if "matched_keywords" in q and pd.notna(q["matched_keywords"]):
                badge_list.append(dbc.Badge(str(q["matched_keywords"]), color="light", text_color="dark", className="me-2"))

            rating_block = (
                html.Span(star_text(q["Rating"]), style={"color": "#f4b400", "marginLeft": "8px"})
                if "Rating" in q and pd.notna(q["Rating"]) else ""
            )

            quote_items.append(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(badge_list + [rating_block], className="mb-2"),
                        html.Div(quote_text, style={"fontSize": "15px", "lineHeight": "1.7"})
                    ]),
                    className="mb-2"
                )
            )

        cards.append(
            dbc.AccordionItem(
                [
                    html.Div([
                        html.P([html.Strong("高频需求："), row["detail_text"]], className="mb-2"),
                        html.P([html.Strong("用户满意度："), f'{row["avg_rating"]} ⭐'], className="mb-3"),
                        html.Div(
                            quote_items if quote_items else dbc.Alert("暂无原声。", color="light"),
                            style={"maxHeight": "360px", "overflowY": "auto", "paddingRight": "8px"}
                        )
                    ])
                ],
                title=f"查看 {cat} 的具体需求"
            )
        )

    return dbc.Accordion(cards, start_collapsed=True, always_open=False)


# =========================================================
# App
# =========================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server
app.title = "丙烯笔分群看板"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("丙烯笔评论分群看板", className="fw-bold"),
            html.P("上半部分保留人群分群；下半部分支持综合评论与分群后评论的句子级分析。", className="text-muted")
        ], width=12)
    ], className="my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("出墨方式"),
            dcc.Dropdown(
                id="ink-filter",
                options=options_with_all(review_df[COL_INK]) if COL_INK else [{"label": "全部", "value": "ALL"}],
                value="ALL",
                clearable=False
            )
        ], width=3),

        dbc.Col([
            html.Label("Brand"),
            dcc.Dropdown(
                id="brand-filter",
                options=options_with_all(review_df["Brand"]) if "Brand" in review_df.columns else [{"label": "全部", "value": "ALL"}],
                value="ALL",
                clearable=False
            )
        ], width=3),

        dbc.Col([
            html.Label("Price Level"),
            dcc.Dropdown(
                id="price-filter",
                options=options_with_all(review_df[COL_PRICE]) if COL_PRICE else [{"label": "全部", "value": "ALL"}],
                value="ALL",
                clearable=False
            )
        ], width=3),

        dbc.Col([
            html.Label("Asin"),
            dcc.Dropdown(
                id="asin-filter",
                options=options_with_all(review_df["Asin"]) if "Asin" in review_df.columns else [{"label": "全部", "value": "ALL"}],
                value="ALL",
                clearable=False
            )
        ], width=3),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(id="kpi-review-count"),
            html.P("当前评论数", className="text-muted mb-0")
        ])), width=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(id="kpi-avg-rating"),
            html.P("平均评分", className="text-muted mb-0")
        ])), width=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(id="kpi-segment-count"),
            html.P("当前主群体数", className="text-muted mb-0")
        ])), width=3),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(id="kpi-brand-count"),
            html.P("当前品牌数", className="text-muted mb-0")
        ])), width=3),
    ], className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H4("① 主分群结果", className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="segment-bar"), width=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("查看某个人群的画像细节"),
                    dcc.Dropdown(
                        id="segment-detail-dropdown",
                        options=[{"label": x, "value": x} for x in segment_list],
                        value=default_segment,
                        clearable=False
                    )
                ], width=4)
            ], className="mb-3"),

            html.H5("人群画像矩阵（按 attribute 分表）", className="section-title"),
            html.Div(id="attribute-matrix-container", className="mb-4"),

            html.H5("选中人群画像拆解（每个 attribute 单独展示）", className="section-title"),
            html.Div(id="segment-attribute-breakdown", className="mb-4"),

            html.H5("分群明细"),
            dag.AgGrid(
                id="segmentation-detail-table",
                columnDefs=[
                    {"field": "review_id", "headerName": "review_id"},
                    {"field": "Asin", "headerName": "Asin"},
                    {"field": "Brand", "headerName": "Brand"},
                    {"field": "Price Level", "headerName": "Price Level"},
                    {"field": "出墨方式", "headerName": "出墨方式"},
                    {"field": "primary_segment", "headerName": "primary_segment"},
                    {"field": "segment_confidence", "headerName": "confidence"},
                    {"field": "segment_score", "headerName": "segment_score"},
                    {"field": "segment_score_gap", "headerName": "score_gap"},
                    {"field": "segment_core_evidence", "headerName": "core evidence"},
                    {"field": "segment_aux_evidence", "headerName": "aux evidence"},
                    {"field": "Content", "headerName": "Content", "wrapText": True, "autoHeight": True},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": True,
                    "filter": True,
                    "resizable": True,
                    "floatingFilter": True,
                },
                dashGridOptions={"pagination": True, "paginationPageSize": 12},
                style={"height": "520px", "width": "100%"}
            )
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H4("② 综合评论分析 / 分群后分析", className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("分析范围"),
                    dcc.Dropdown(
                        id="analysis-scope-dropdown",
                        options=[
                            {"label": "综合全部评论", "value": "overall"},
                            {"label": "选中人群评论", "value": "segment"},
                        ],
                        value="overall",
                        clearable=False
                    )
                ], width=3),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="feature-summary-bar"), width=12),
            ], className="mb-4"),
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H4("③ 深度探查：真实用户评价回溯", className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("原声类型"),
                    dcc.Dropdown(
                        id="sentiment-filter",
                        options=[
                            {"label": "全部", "value": "ALL"},
                            {"label": "亮点", "value": "positive"},
                            {"label": "痛点", "value": "negative"},
                        ],
                        value="negative",
                        clearable=False
                    )
                ], width=3),

                dbc.Col([
                    html.Label("维度"),
                    dcc.Dropdown(
                        id="feature-dimension-dropdown",
                        options=[],
                        value=None,
                        clearable=False
                    )
                ], width=9),
            ], className="mb-3"),

            html.Div(id="root-cause-card", className="mb-3"),
            html.H5(id="quote-list-title", className="mb-3 fw-bold"),
            html.Div(id="feature-quote-list")
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H4("④ 捆绑销售与关联购买洞察 (Bundle Insight)", className="mb-3"),
            html.P("分析用户评论中自发提到的搭配产品，锁定关联销售机会。", className="text-muted"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="bundle-chart"), width=5),
                dbc.Col(html.Div(id="bundle-cards"), width=7),
            ])
        ])
    ], className="mb-4"),
], fluid=True)


# =========================================================
# 联动筛选器：Brand
# =========================================================
@callback(
    Output("brand-filter", "options"),
    Output("brand-filter", "value"),
    Input("ink-filter", "value"),
    State("brand-filter", "value")
)
def update_brand_options(ink_mode, current_brand):
    dff = review_df.copy()
    if COL_INK and ink_mode != "ALL":
        dff = dff[dff[COL_INK].astype(str) == str(ink_mode)]

    options = options_with_all(dff["Brand"]) if "Brand" in dff.columns else [{"label": "全部", "value": "ALL"}]
    valid_values = [x["value"] for x in options]
    value = current_brand if current_brand in valid_values else "ALL"
    return options, value


# =========================================================
# 联动筛选器：Price Level
# =========================================================
@callback(
    Output("price-filter", "options"),
    Output("price-filter", "value"),
    Input("ink-filter", "value"),
    Input("brand-filter", "value"),
    State("price-filter", "value")
)
def update_price_options(ink_mode, brand, current_price):
    if not COL_PRICE:
        return [{"label": "全部", "value": "ALL"}], "ALL"

    dff = review_df.copy()
    if COL_INK and ink_mode != "ALL":
        dff = dff[dff[COL_INK].astype(str) == str(ink_mode)]
    if brand != "ALL" and "Brand" in dff.columns:
        dff = dff[dff["Brand"].astype(str) == str(brand)]

    options = options_with_all(dff[COL_PRICE])
    valid_values = [x["value"] for x in options]
    value = current_price if current_price in valid_values else "ALL"
    return options, value


# =========================================================
# 联动筛选器：ASIN
# =========================================================
@callback(
    Output("asin-filter", "options"),
    Output("asin-filter", "value"),
    Input("ink-filter", "value"),
    Input("brand-filter", "value"),
    Input("price-filter", "value"),
    State("asin-filter", "value")
)
def update_asin_options(ink_mode, brand, price_level, current_asin):
    dff = review_df.copy()

    if COL_INK and ink_mode != "ALL":
        dff = dff[dff[COL_INK].astype(str) == str(ink_mode)]
    if brand != "ALL" and "Brand" in dff.columns:
        dff = dff[dff["Brand"].astype(str) == str(brand)]
    if COL_PRICE and price_level != "ALL":
        dff = dff[dff[COL_PRICE].astype(str) == str(price_level)]

    options = options_with_all(dff["Asin"]) if "Asin" in dff.columns else [{"label": "全部", "value": "ALL"}]
    valid_values = [x["value"] for x in options]
    value = current_asin if current_asin in valid_values else "ALL"
    return options, value


# =========================================================
# 当前过滤条件下，可选 segment
# =========================================================
@callback(
    Output("segment-detail-dropdown", "options"),
    Output("segment-detail-dropdown", "value"),
    Input("ink-filter", "value"),
    Input("brand-filter", "value"),
    Input("price-filter", "value"),
    Input("asin-filter", "value"),
    State("segment-detail-dropdown", "value")
)
def update_segment_options(ink_mode, brand, price_level, asin, current_segment):
    dff = apply_global_filters(review_df, ink_mode, brand, price_level, asin)
    segs = sorted(dff["primary_segment"].dropna().astype(str).unique().tolist()) if "primary_segment" in dff.columns else []
    options = [{"label": x, "value": x} for x in segs]

    if not options:
        return [], None

    valid_values = [x["value"] for x in options]
    value = current_segment if current_segment in valid_values else valid_values[0]
    return options, value


# =========================================================
# 主分群视图
# =========================================================
@callback(
    Output("kpi-review-count", "children"),
    Output("kpi-avg-rating", "children"),
    Output("kpi-segment-count", "children"),
    Output("kpi-brand-count", "children"),
    Output("segment-bar", "figure"),
    Output("attribute-matrix-container", "children"),
    Output("segment-attribute-breakdown", "children"),
    Output("segmentation-detail-table", "rowData"),
    Input("ink-filter", "value"),
    Input("brand-filter", "value"),
    Input("price-filter", "value"),
    Input("asin-filter", "value"),
    Input("segment-detail-dropdown", "value"),
)
def update_segmentation_section(ink_mode, brand, price_level, asin, selected_segment):
    dff = apply_global_filters(review_df, ink_mode, brand, price_level, asin)

    review_count = len(dff)
    avg_rating = round(dff["Rating"].mean(), 2) if review_count > 0 and "Rating" in dff.columns else 0
    segment_count = dff["primary_segment"].nunique() if review_count > 0 and "primary_segment" in dff.columns else 0
    brand_count = dff["Brand"].nunique() if review_count > 0 and "Brand" in dff.columns else 0

    if "primary_segment" in dff.columns:
        seg_df = dff["primary_segment"].dropna().value_counts().reset_index()
        seg_df.columns = ["primary_segment", "count"]
    else:
        seg_df = pd.DataFrame(columns=["primary_segment", "count"])

    if seg_df.empty:
        segment_fig = px.bar(title="主人群分布")
    else:
        segment_fig = px.bar(seg_df, x="primary_segment", y="count", title="主人群分布")
        segment_fig.update_layout(xaxis_title="", yaxis_title="评论数")

    attr_long = build_attribute_long(dff)
    segment_order = seg_df["primary_segment"].tolist() if not seg_df.empty else []

    matrix_children = build_attribute_matrix_tables(attr_long, segment_order)
    segment_breakdown_children = build_segment_attribute_cards(attr_long, selected_segment)

    detail_df = dff[dff["primary_segment"].astype(str) == str(selected_segment)].copy() if selected_segment else dff.copy()

    show_cols = [
        "review_id", "Asin", "Brand", "primary_segment",
        "segment_confidence", "segment_score", "segment_score_gap",
        "segment_core_evidence", "segment_aux_evidence", "Content"
    ]
    if COL_PRICE:
        show_cols.insert(3, COL_PRICE)
    if COL_INK:
        show_cols.insert(4, COL_INK)

    show_cols = [c for c in show_cols if c in detail_df.columns]
    detail_df = detail_df[show_cols].head(300)

    return (
        str(review_count),
        str(avg_rating),
        str(segment_count),
        str(brand_count),
        segment_fig,
        matrix_children,
        segment_breakdown_children,
        detail_df.to_dict("records")
    )


# =========================================================
# feature 维度下拉
# =========================================================
@callback(
    Output("feature-dimension-dropdown", "options"),
    Output("feature-dimension-dropdown", "value"),
    Input("ink-filter", "value"),
    Input("brand-filter", "value"),
    Input("price-filter", "value"),
    Input("asin-filter", "value"),
    Input("segment-detail-dropdown", "value"),
    Input("analysis-scope-dropdown", "value"),
    State("feature-dimension-dropdown", "value")
)
def update_feature_dimension_options(ink_mode, brand, price_level, asin, selected_segment, analysis_scope, current_dim):
    _, qdf, _ = get_analysis_frames(ink_mode, brand, price_level, asin, selected_segment, analysis_scope)
    feature_overview = build_overall_feature_overview(qdf)
    dims = feature_overview["feature_dimension"].tolist() if not feature_overview.empty else []

    options = [{"label": x, "value": x} for x in dims]
    if not options:
        return [], None

    valid_values = [x["value"] for x in options]
    value = current_dim if current_dim in valid_values else valid_values[0]
    return options, value


# =========================================================
# 亮点/痛点分析
# =========================================================
@callback(
    Output("feature-summary-bar", "figure"),
    Output("root-cause-card", "children"),
    Output("quote-list-title", "children"),
    Output("feature-quote-list", "children"),
    Input("ink-filter", "value"),
    Input("brand-filter", "value"),
    Input("price-filter", "value"),
    Input("asin-filter", "value"),
    Input("segment-detail-dropdown", "value"),
    Input("analysis-scope-dropdown", "value"),
    Input("sentiment-filter", "value"),
    Input("feature-dimension-dropdown", "value"),
)
def update_feature_section(ink_mode, brand, price_level, asin, selected_segment, analysis_scope, sentiment_type, selected_dim):
    _, qdf, _ = get_analysis_frames(ink_mode, brand, price_level, asin, selected_segment, analysis_scope)
    feature_overview = build_overall_feature_overview(qdf)

    scope_text = "综合全部评论" if analysis_scope == "overall" else f"选中人群：{selected_segment}"
    fig_title = f"各维度情感倾向分布与满意度趋势（{scope_text}）"
    feature_fig = build_feature_overview_chart(feature_overview, fig_title)

    root_card = build_root_cause_card(qdf, selected_dim, sentiment_type)
    quote_list = build_quote_cards(qdf, selected_dim, sentiment_type)

    temp = qdf.copy()
    if selected_dim and "feature_dimension" in temp.columns:
        temp = temp[temp["feature_dimension"].astype(str) == str(selected_dim)]
    if sentiment_type != "ALL" and "sentiment_type" in temp.columns:
        temp = temp[temp["sentiment_type"].astype(str) == str(sentiment_type)]

    label_text = "原声" if sentiment_type == "ALL" else "亮点原声" if sentiment_type == "positive" else "痛点原声"
    dim_text = selected_dim if selected_dim else "全部维度"
    quote_title = f"用户评价原声回溯（{dim_text}｜{label_text}｜{len(temp)}条）"

    return feature_fig, root_card, quote_title, quote_list


# =========================================================
# Bundle Insight
# =========================================================
@callback(
    Output("bundle-chart", "figure"),
    Output("bundle-cards", "children"),
    Input("ink-filter", "value"),
    Input("brand-filter", "value"),
    Input("price-filter", "value"),
    Input("asin-filter", "value"),
    Input("segment-detail-dropdown", "value"),
    Input("analysis-scope-dropdown", "value"),
)
def update_bundle_section(ink_mode, brand, price_level, asin, selected_segment, analysis_scope):
    _, _, bdf = get_analysis_frames(ink_mode, brand, price_level, asin, selected_segment, analysis_scope)
    bundle_summary = build_bundle_summary(bdf)

    fig = build_bundle_chart(bundle_summary)
    cards = build_bundle_cards(bundle_summary, bdf)

    return fig, cards


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)