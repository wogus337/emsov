
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


STRATEGY_COLUMNS = {
    "bm_return": "BM",
    "top1_return": "Top-1",
    "bot1_return": "Bot-1",
    "top3_return": "Top-3",
    "bot3_return": "Bot-3",
    "top5_return": "Top-5",
    "bot5_return": "Bot-5",
    "top8_return": "Top-8",
    "bot8_return": "Bot-8",
    "top10_return": "Top-10",
    "bot10_return": "Bot-10",
}

REQUIRED_COLUMNS = [
    "eval_period",
    "bm_return",
    "top1_return",
    "bot1_return",
    "top3_return",
    "bot3_return",
    "top5_return",
    "bot5_return",
    "top8_return",
    "bot8_return",
    "top10_return",
    "bot10_return",
]


def infer_model_name(filename: str) -> str:
    stem = Path(filename).stem
    lowered = stem.lower()
    pattern = r"^topbot_cumulative_results_?"
    cleaned = re.sub(pattern, "", lowered)
    if not cleaned:
        return stem
    return cleaned


def load_csv_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} 필수 컬럼 누락: {missing}")

    df = df.copy()
    df["eval_date"] = pd.to_datetime(df["eval_period"] + "-01", errors="coerce")
    if df["eval_date"].isna().all():
        raise ValueError(f"{file_path.name} eval_period를 날짜로 변환할 수 없습니다.")
    df = df.sort_values("eval_date").reset_index(drop=True)
    return df


def collect_model_frames(data_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    model_frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    pattern = "TopBot_Cumulative_Results_*.csv"
    csv_files = sorted(data_dir.glob(pattern))
    for csv_file in csv_files:
        try:
            model_name = infer_model_name(csv_file.name)
            model_frames[model_name] = load_csv_file(csv_file)
        except Exception as exc:
            errors.append(str(exc))

    return model_frames, errors


def extract_yyyymmdd_from_stem(file_stem: str) -> str | None:
    match = re.search(r"(\d{8})$", file_stem)
    if not match:
        return None
    return match.group(1)


def find_latest_prediction_combined(data_dir: Path) -> tuple[list[Path], Path | None, str | None]:
    files = sorted(data_dir.glob("Prediction_Combined_*.csv"))
    if not files:
        return files, None, None

    if len(files) == 1:
        single = files[0]
        return files, single, extract_yyyymmdd_from_stem(single.stem)

    dated_files: list[tuple[pd.Timestamp, Path, str]] = []
    for f in files:
        yyyymmdd = extract_yyyymmdd_from_stem(f.stem)
        if not yyyymmdd:
            continue
        dt = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
        if pd.notna(dt):
            dated_files.append((dt, f, yyyymmdd))

    if not dated_files:
        fallback = files[-1]
        return files, fallback, extract_yyyymmdd_from_stem(fallback.stem)

    dated_files.sort(key=lambda x: x[0])
    _, latest_file, latest_yyyymmdd = dated_files[-1]
    return files, latest_file, latest_yyyymmdd


def build_prediction_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    rank_cols = [c for c in pred_df.columns if c.startswith("순위_")]
    if len(rank_cols) < 5:
        raise ValueError("평균 계산을 위한 순위 칼럼이 5개 미만입니다.")

    use_rank_cols = rank_cols[:5]
    out = pred_df.copy()
    out["평균순위"] = out[use_rank_cols].mean(axis=1)

    if "국가이름" in out.columns:
        nation_loc = out.columns.get_loc("국가이름")
        avg_series = out.pop("평균순위")
        out.insert(nation_loc + 1, "평균순위", avg_series)
    return out


def to_long_chart_df(
    model_frames: dict[str, pd.DataFrame], selected_model: str, selected_columns: list[str], metric_mode: str
) -> pd.DataFrame:
    rows = []
    model_name = selected_model
    df = model_frames[model_name].copy()

    if metric_mode == "누적 수익률(%)":
        for col in selected_columns:
            cum = ((1 + df[col] / 100).cumprod() - 1) * 100
            rows.append(
                pd.DataFrame(
                    {
                        "eval_date": df["eval_date"],
                        "eval_period": df["eval_period"],
                        "model": model_name,
                        "strategy": STRATEGY_COLUMNS[col],
                        "value": cum,
                    }
                )
            )

    else:
        bm_growth = (1 + df["bm_return"] / 100).cumprod()
        for col in selected_columns:
            if col == "bm_return":
                excess_cum = np.zeros(len(df))
            else:
                strategy_growth = (1 + df[col] / 100).cumprod()
                excess_cum = (strategy_growth / bm_growth - 1) * 100
            rows.append(
                pd.DataFrame(
                    {
                        "eval_date": df["eval_date"],
                        "eval_period": df["eval_period"],
                        "model": model_name,
                        "strategy": STRATEGY_COLUMNS[col],
                        "value": excess_cum,
                    }
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=["eval_date", "eval_period", "model", "strategy", "value"]
        )
    return pd.concat(rows, ignore_index=True)


def main():
    st.set_page_config(page_title="[EM Sov] Country Ranking Model", layout="wide")
    st.title("[EM Sov] Country Ranking Model")
    st.markdown(
        """
<div style="line-height:1.35;">

1. 5개의 AI방법론 적용<br>
2. 연단위 재학습 가정하여 결과 누적(연단위로 재학습 후 새로운 모델 적용)<br>
&nbsp;&nbsp;&nbsp;예. 2023년 결과는 2022년까지 데이터로 학습-검증 후 2023년 1년 적용한 결과. 다음은 ~2023년 으로 학습, 2024년 적용<br>
3. 아래 결과는 학습한 모델을 실제 적용한 테스팅 결과<br>
4. Top/Bottom을 선택했을 때의 단순 성과 누적으로 회전율/거래비용 등을 감안한 트레이딩 테스트는 아님<br>

</div>
""",
        unsafe_allow_html=True,
    )
    data_dir = Path(__file__).resolve().parent

    _, latest_pred_file, latest_yyyymmdd = find_latest_prediction_combined(data_dir)
    if latest_pred_file is None:
        st.warning("Prediction_Combined_*.csv 파일이 없어 예측 테이블을 표시하지 않습니다.")
    else:
        try:
            pred_df = pd.read_csv(latest_pred_file)
            pred_table = build_prediction_table(pred_df)
        except Exception as exc:
            st.error(f"예측 테이블 생성 실패: {exc}")
            pred_table = None

        if pred_table is not None and latest_yyyymmdd is not None:
            latest_dt = pd.to_datetime(latest_yyyymmdd, format="%Y%m%d", errors="coerce")
            if pd.notna(latest_dt):
                next_month_dt = latest_dt + pd.DateOffset(months=1)
                pred_title = f"{next_month_dt.year}년 {next_month_dt.month}월 Top/Bottom 국가 예측"
            else:
                pred_title = "Top/Bottom 국가 예측"
        else:
            pred_title = "Top/Bottom 국가 예측"

    model_frames, errors = collect_model_frames(data_dir)
    if errors:
        for msg in errors:
            st.error(msg)

    if not model_frames:
        st.warning("패턴에 맞는 파일이 없습니다: TopBot_Cumulative_Results_*.csv")
        st.code(str(data_dir / "TopBot_Cumulative_Results_*.csv"))
        return

    all_models = sorted(model_frames.keys())
    default_model = "cc_transformer" if "cc_transformer" in all_models else all_models[0]

    strategy_bundles = {
        "BM-Top1-Bot1": ["bm_return", "top1_return", "bot1_return"],
        "BM-Top3-Bot3": ["bm_return", "top3_return", "bot3_return"],
        "BM-Top5-Bot5": ["bm_return", "top5_return", "bot5_return"],
        "BM-Top10-Bot10": ["bm_return", "top10_return", "bot10_return"],
    }

    sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 2])
    with sel_col1:
        metric_mode = st.selectbox(
            "차트 기준",
            options=["누적 수익률(%)", "BM 대비 누적 초과수익(%)"],
            index=0,
        )
    with sel_col2:
        selected_model = st.selectbox("모델 선택", options=all_models, index=all_models.index(default_model))
    with sel_col3:
        selected_bundle_name = st.selectbox(
            "전략 선택",
            options=list(strategy_bundles.keys()),
            index=2,
        )

    chart_df = to_long_chart_df(
        model_frames=model_frames,
        selected_model=selected_model,
        selected_columns=strategy_bundles[selected_bundle_name],
        metric_mode=metric_mode,
    )
    chart_df["series"] = chart_df["strategy"]

    color_map = {
        "BM": "rgb(206, 206, 206)",
        "Top-1": "rgb(245, 130, 32)",
        "Top-3": "rgb(245, 130, 32)",
        "Top-5": "rgb(245, 130, 32)",
        "Top-10": "rgb(245, 130, 32)",
        "Bot-1": "rgb(4, 59, 114)",
        "Bot-3": "rgb(4, 59, 114)",
        "Bot-5": "rgb(4, 59, 114)",
        "Bot-10": "rgb(4, 59, 114)",
    }

    color_domain = [STRATEGY_COLUMNS[c] for c in strategy_bundles[selected_bundle_name]]
    active_color_map = {s: color_map[s] for s in color_domain}

    st.subheader("Performance Chart")
    fig = px.line(
        chart_df,
        x="eval_date",
        y="value",
        color="series",
        color_discrete_map=active_color_map,
        markers=True,
        hover_data={"eval_period": True, "value": ":.3f"},
    )
    fig.update_traces(line={"width": 3})
    fig.update_layout(
        yaxis_title=metric_mode,
        xaxis_title="Date",
        hovermode="x unified",
        legend_title_text="Series",
    )
    st.plotly_chart(fig, use_container_width=True)

    if latest_pred_file is not None and pred_table is not None:
        st.subheader(pred_title)
        st.caption(f"기준 파일: `{latest_pred_file.name}`")
        st.dataframe(pred_table, height=420, use_container_width=False)

    st.subheader("과거 결과 확인(월별 Top/Bottom 국가)")
    all_years = sorted({str(int(y)) for df in model_frames.values() for y in df["eval_year"].unique()})
    scope_options = ["전체"] + all_years
    raw_col1, raw_col2 = st.columns([2, 2])
    with raw_col1:
        inspect_model = st.selectbox("원본 조회 모델", options=all_models, index=all_models.index(selected_model))
    with raw_col2:
        raw_scope = st.selectbox("원본 조회 구분", options=scope_options, index=0)

    raw_df = model_frames[inspect_model].copy()
    if raw_scope != "전체":
        raw_df = raw_df[raw_df["eval_year"] == int(raw_scope)].reset_index(drop=True)

    st.dataframe(raw_df, height=500, use_container_width=False)
    st.download_button(
        label="원본 데이터 CSV 다운로드",
        data=raw_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name=f"raw_{inspect_model}_{raw_scope}.csv",
        mime="text/csv",
    )

    return


if __name__ == "__main__":
    main()
