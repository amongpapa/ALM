import os
import math
import time
from io import BytesIO
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from scipy.optimize import minimize


# =========================================================
# 0) Streamlit 기본 설정 (임원 보고용: 넓은 폭 + 기본 UI 숨김)
# =========================================================
st.set_page_config(
    page_title="ALM One-Page Visualizer PRO",
    layout="wide",
    initial_sidebar_state="expanded",  # collapsed -> expanded
)

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
html, body, [class*="css"]  {
    font-family: "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.0rem;
    max-width: 1680px;
}

/* 사이드바 스타일링 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #073763;
    font-weight: 800;
    padding: 10px 0;
}
[data-testid="stSidebar"] .stMarkdown strong {
    color: #2563eb;
    font-size: 13px;
}
[data-testid="stSidebar"] hr {
    margin: 8px 0;
    border-color: rgba(127,182,255,0.3);
}

.card {
    background: #ffffff;
    border: 1px solid rgba(10, 60, 120, 0.10);
    border-radius: 18px;
    padding: 18px 18px;
    box-shadow: 0 8px 24px rgba(20, 60, 120, 0.08);
    margin-bottom: 14px;
    transition: all 0.3s ease;
}
.card:hover {
    box-shadow: 0 12px 32px rgba(20, 60, 120, 0.14);
    transform: translateY(-2px);
}
.h1 {
    font-size: 26px;
    font-weight: 800;
    color: #073763;
    margin: 0 0 8px 0;
}
.sub {
    color: rgba(7,55,99,0.70);
    margin: 0 0 8px 0;
    font-size: 15px;
}
.small {
    font-size: 12px;
    color: rgba(7,55,99,0.70);
}
hr.soft {
    border: none;
    border-top: 1px solid rgba(10, 60, 120, 0.08);
    margin: 10px 0;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}
.kpi-box{
    background: linear-gradient(135deg, rgba(127,182,255,0.22), rgba(127,182,255,0.08));
    border: 2px solid rgba(127,182,255,0.35);
    border-radius: 16px;
    padding: 16px 14px;
    transition: all 0.3s ease;
}
.kpi-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(127,182,255,0.30);
    border-color: rgba(127,182,255,0.55);
}
.kpi-ttl{
    font-size: 13px;
    color: rgba(7,55,99,0.75);
    margin-bottom: 6px;
    font-weight: 600;
}
.kpi-val{
    font-size: 22px;
    font-weight: 900;
    color: #073763;
}
.delta-pos { color: #0b5ed7; font-weight: 800; } /* 파랑: 개선 */
.delta-neg { color: #d63384; font-weight: 800; } /* 빨강: 악화 */
.pill {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(127,182,255,0.28);
  background: rgba(127,182,255,0.10);
  color: #073763;
  font-size: 12px;
  font-weight: 700;
}
.scenario-card {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.05), rgba(139, 92, 246, 0.05));
    border: 1px solid rgba(37, 99, 235, 0.2);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 8px;
}
.optimal-result {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.05));
    border: 2px solid rgba(16, 185, 129, 0.3);
    border-radius: 14px;
    padding: 16px;
    margin-top: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# 1) 도메인 상수/매핑
# =========================================================
BUCKET_ORDER = ["3M", "6M", "1Y", "2Y", "3Y", "5Y+"]

# 버킷을 "대표 만기(년)"로 단순 치환
BUCKET_YEARS = {"3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y+": 7.0}
BUCKET_X = {b: i for i, b in enumerate(BUCKET_ORDER)}

DEFAULT_MARGIN_START = "6M"
DEFAULT_MARGIN_END = "2Y"

ASSET_COLOR = "#7fb6ff"
ASSET_EDGE = "#1f5fae"
LIAB_COLOR = "#c9ced6"
LIAB_EDGE = "#667085"
DUR_COLOR = "black"
SKY = "#7fb6ff"

ASSET_CF_COLOR = "#19c37d"
LIAB_CF_COLOR = "#ff9f1a"


# =========================================================
# 2) 샘플 데이터 생성 (Position)
# =========================================================
def generate_sample_positions(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # 확장된 자산 상품
    asset_products = [
        "대출-가계(변동금리)",
        "대출-가계(고정금리)", 
        "대출-기업(운전자금)",
        "대출-기업(시설자금)",
        "대출-기업(부동산PF)",
        "채권-국채",
        "채권-회사채(우량)",
        "채권-회사채(일반)",
        "채권-금융채",
        "신용카드(리볼빙)",
        "신용약정(미사용)",
    ]
    for p in asset_products:
        b = rng.choice(BUCKET_ORDER, p=[0.12, 0.18, 0.22, 0.22, 0.16, 0.10])
        bal = float(rng.uniform(3, 15)) * 1e12  # 3조~15조
        dur = float(rng.uniform(0.1, BUCKET_YEARS[b] * 0.9 + 0.2))
        rate = float(rng.uniform(0.025, 0.065))
        spread = float(rng.uniform(0.002, 0.020))
        
        # 금리 재조정 만기 (3M, 6M, 1Y 등)
        rate_maturity = rng.choice(["3M", "6M", "1Y", "2Y", "3Y"])
        
        # 마진 등급 계산 (spread 기반)
        if spread >= 0.015:
            margin_grade = "HIGH"
            margin_score = 0.8 + (spread - 0.015) * 10  # 0.8~1.0
        elif spread >= 0.008:
            margin_grade = "MEDIUM"
            margin_score = 0.5 + (spread - 0.008) * 4.3  # 0.5~0.8
        else:
            margin_grade = "LOW"
            margin_score = spread * 62.5  # 0.0~0.5
        
        margin_score = min(max(margin_score, 0.0), 1.0)
        
        rows.append(
            {
                "asof": pd.Timestamp("2026-01-01"),
                "type": "asset",
                "product": p,
                "maturity_bucket": b,
                "balance": bal,
                "duration": dur,
                "rate": rate,
                "spread": spread,
                "rate_maturity": rate_maturity,
                "margin_grade": margin_grade,
                "margin_score": margin_score,
            }
        )

    # 확장된 부채 상품
    liab_products = [
        "예금-요구불(개인)",
        "예금-요구불(법인)",
        "예금-저축성(개인)",
        "예금-정기(개인)",
        "예금-정기(법인)",
        "예금-외화(USD)",
        "채권-은행채(선순위)",
        "채권-은행채(후순위)",
        "차입-RP",
        "차입-콜머니",
        "차입-장기차입금",
        "지급보증(미사용)",
    ]
    for p in liab_products:
        b = rng.choice(BUCKET_ORDER, p=[0.20, 0.25, 0.20, 0.18, 0.10, 0.07])
        bal = float(rng.uniform(4, 18)) * 1e12  # 4조~18조
        dur = float(rng.uniform(0.05, BUCKET_YEARS[b] * 0.8 + 0.2))
        rate = float(rng.uniform(0.015, 0.045))
        spread = float(rng.uniform(0.000, 0.012))
        
        # 금리 재조정 만기
        rate_maturity = rng.choice(["3M", "6M", "1Y", "2Y"])
        
        # 마진 등급 (부채는 낮은 금리가 좋음)
        if rate <= 0.025:
            margin_grade = "HIGH"
            margin_score = 1.0 - rate * 10  # 0.75~1.0
        elif rate <= 0.035:
            margin_grade = "MEDIUM"
            margin_score = 0.75 - (rate - 0.025) * 5  # 0.50~0.75
        else:
            margin_grade = "LOW"
            margin_score = 0.50 - (rate - 0.035) * 5  # 0.0~0.50
        
        margin_score = min(max(margin_score, 0.0), 1.0)
        
        rows.append(
            {
                "asof": pd.Timestamp("2026-01-01"),
                "type": "liability",
                "product": p,
                "maturity_bucket": b,
                "balance": bal,
                "duration": dur,
                "rate": rate,
                "spread": spread,
                "rate_maturity": rate_maturity,
                "margin_grade": margin_grade,
                "margin_score": margin_score,
            }
        )

    # HQLA - LCR 103.5%를 위한 규모 조정
    # 30일 순유출이 약 10조라고 가정 → HQLA = 10.35조 필요
    hqla_products = ["현금", "국채(HQLA1)", "예치금(중앙은행)", "우량회사채(HQLA2A)"]
    hqla_balances = [2.5, 3.0, 2.5, 2.35]  # 총 10.35조
    
    for i, p in enumerate(hqla_products):
        bal = float(hqla_balances[i]) * 1e12
        margin_score = 1.0 - i * 0.05  # 현금 1.0, 국채 0.95, 예치금 0.90, 회사채 0.85
        rows.append(
            {
                "asof": pd.Timestamp("2026-01-01"),
                "type": "hqla",
                "product": p,
                "maturity_bucket": "0D",
                "balance": bal,
                "duration": 0.0,
                "rate": 0.0,
                "spread": 0.0,
                "rate_maturity": "0D",
                "margin_grade": "HIGH",
                "margin_score": margin_score,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 2-1) Excel 파일에서 데이터 로드
# =========================================================
DEFAULT_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "ALM_input_template.xlsx")


def load_positions_from_excel(excel_path: str = None) -> pd.DataFrame:
    """
    Excel 파일의 POSITIONS 시트와 HQLA 시트에서 포지션 데이터를 로드합니다.
    
    Args:
        excel_path: Excel 파일 경로. None이면 기본 경로 사용.
    
    Returns:
        positions DataFrame (type, product, balance, rate, duration, maturity_bucket 등)
    """
    if excel_path is None:
        excel_path = DEFAULT_EXCEL_PATH
    
    if not os.path.exists(excel_path):
        st.warning(f"Excel 파일을 찾을 수 없습니다: {excel_path}")
        return generate_sample_positions(seed=7)
    
    try:
        xl = pd.ExcelFile(excel_path)
        
        # POSITIONS 시트 로드 (첫 행이 컬럼명)
        positions_df = pd.read_excel(xl, sheet_name='POSITIONS')
        
        # 컬럼명 정규화 (괄호와 한글 제거)
        col_mapping = {
            'asof(YYYY-MM-DD)': 'asof',
            'type(asset/liability)': 'type',
            'product_code(선택)': 'product_code',
            'balance(원)': 'balance',
            'rate(연이율)': 'rate',
            'spread(선택)': 'spread',
            'maturity_date(선택)': 'maturity_date',
            'rate_maturity(선택)': 'rate_maturity',
            'duration(년)': 'duration',
            'notes(선택)': 'notes',
        }
        positions_df = positions_df.rename(columns=col_mapping)
        
        # 필수 컬럼 확인
        required_cols = ['type', 'product', 'balance']
        missing_cols = [c for c in required_cols if c not in positions_df.columns]
        if missing_cols:
            st.warning(f"POSITIONS 시트에 필수 컬럼이 없습니다: {missing_cols}")
            return generate_sample_positions(seed=7)
        
        # 유효한 type 값만 필터링 (설명 행 제거)
        valid_types = ['asset', 'liability']
        positions_df = positions_df[positions_df['type'].isin(valid_types)].copy()
        
        # balance가 숫자인 행만 유지
        positions_df = positions_df[pd.to_numeric(positions_df['balance'], errors='coerce').notna()].copy()
        positions_df['balance'] = pd.to_numeric(positions_df['balance'])
        
        # 기본값 설정
        if 'rate' not in positions_df.columns:
            positions_df['rate'] = 0.03
        else:
            positions_df['rate'] = pd.to_numeric(positions_df['rate'], errors='coerce').fillna(0.03)
        
        if 'spread' not in positions_df.columns:
            positions_df['spread'] = 0.005
        else:
            positions_df['spread'] = pd.to_numeric(positions_df['spread'], errors='coerce').fillna(0.005)
        
        if 'duration' not in positions_df.columns:
            positions_df['duration'] = 1.0
        else:
            positions_df['duration'] = pd.to_numeric(positions_df['duration'], errors='coerce').fillna(1.0)
        
        if 'maturity_bucket' not in positions_df.columns:
            positions_df['maturity_bucket'] = '1Y'
        else:
            positions_df['maturity_bucket'] = positions_df['maturity_bucket'].fillna('1Y')
        
        if 'rate_maturity' not in positions_df.columns:
            positions_df['rate_maturity'] = '3M'
        else:
            positions_df['rate_maturity'] = positions_df['rate_maturity'].fillna('3M')
        
        if 'asof' not in positions_df.columns:
            positions_df['asof'] = pd.Timestamp("2026-01-01")
        
        # 마진 등급/점수 계산
        def calc_margin_grade(spread_val):
            try:
                spread = float(spread_val)
                if spread >= 0.015:
                    return "HIGH"
                elif spread >= 0.008:
                    return "MEDIUM"
                else:
                    return "LOW"
            except:
                return "MEDIUM"
        
        def calc_margin_score(spread_val):
            try:
                spread = float(spread_val)
                if spread >= 0.015:
                    return min(0.8 + (spread - 0.015) * 10, 1.0)
                elif spread >= 0.008:
                    return 0.5 + (spread - 0.008) * 4.3
                else:
                    return spread * 62.5
            except:
                return 0.5
        
        positions_df['margin_grade'] = positions_df['spread'].apply(calc_margin_grade)
        positions_df['margin_score'] = positions_df['spread'].apply(calc_margin_score)
        
        # HQLA 시트 로드
        hqla_df = pd.read_excel(xl, sheet_name='HQLA')
        
        # HQLA 컬럼명 정규화
        hqla_col_mapping = {
            'asof(YYYY-MM-DD)': 'asof',
            'type(hqla)': 'type',
            'balance(원)': 'balance',
            'level(선택)': 'level',
            'haircut(선택,%)': 'haircut',
            'notes(선택)': 'notes',
        }
        hqla_df = hqla_df.rename(columns=hqla_col_mapping)
        
        # 유효한 type 값만 필터링 (hqla만)
        hqla_df = hqla_df[hqla_df['type'] == 'hqla'].copy()
        
        # balance가 숫자인 행만 유지
        hqla_df = hqla_df[pd.to_numeric(hqla_df['balance'], errors='coerce').notna()].copy()
        hqla_df['balance'] = pd.to_numeric(hqla_df['balance'])
        
        # HQLA 기본값 설정
        hqla_df['maturity_bucket'] = '0D'
        hqla_df['duration'] = 0.0
        hqla_df['rate'] = 0.0
        hqla_df['spread'] = 0.0
        hqla_df['rate_maturity'] = '0D'
        hqla_df['margin_grade'] = 'HIGH'
        hqla_df['margin_score'] = 1.0
        
        # 두 DataFrame 결합
        combined_df = pd.concat([positions_df, hqla_df], ignore_index=True)
        
        return combined_df
        
    except Exception as e:
        st.warning(f"Excel 파일 로드 중 오류 발생: {str(e)}")
        return generate_sample_positions(seed=7)


def load_yield_curve_from_excel(excel_path: str = None, curve_name: str = "BASE") -> Tuple[List[float], List[float]]:
    """
    Excel 파일의 YIELD_CURVE 시트에서 금리 커브를 로드합니다.
    
    Args:
        excel_path: Excel 파일 경로. None이면 기본 경로 사용.
        curve_name: 로드할 커브 이름 (BASE, STRESS 등)
    
    Returns:
        (curve_x, curve_y) - 테너(년), 금리(연율)
    """
    if excel_path is None:
        excel_path = DEFAULT_EXCEL_PATH
    
    # 기본값
    default_curve_x = [0.25, 1.0, 5.0, 10.0]
    default_curve_y = [0.032, 0.035, 0.040, 0.042]
    
    if not os.path.exists(excel_path):
        return default_curve_x, default_curve_y
    
    try:
        xl = pd.ExcelFile(excel_path)
        yield_df = pd.read_excel(xl, sheet_name='YIELD_CURVE')
        
        # 컬럼명 정규화
        col_mapping = {
            'asof(YYYY-MM-DD)': 'asof',
            'rate(연이율)': 'rate',
            'source(선택)': 'source',
            'notes(선택)': 'notes',
        }
        yield_df = yield_df.rename(columns=col_mapping)
        
        # curve_name 필터링 (유효한 값만)
        if 'curve_name' in yield_df.columns:
            # tenor_years가 숫자인 행만 필터링 (설명 행 제거)
            yield_df = yield_df[pd.to_numeric(yield_df['tenor_years'], errors='coerce').notna()].copy()
            
            filtered = yield_df[yield_df['curve_name'] == curve_name].copy()
            if filtered.empty:
                return default_curve_x, default_curve_y
            
            # tenor_years와 rate로 정렬
            filtered['tenor_years'] = pd.to_numeric(filtered['tenor_years'])
            filtered['rate'] = pd.to_numeric(filtered['rate'])
            filtered = filtered.sort_values('tenor_years')
            
            curve_x = filtered['tenor_years'].tolist()
            curve_y = filtered['rate'].tolist()
            
            if len(curve_x) > 0 and len(curve_y) > 0:
                return curve_x, curve_y
        
        return default_curve_x, default_curve_y
        
    except Exception as e:
        return default_curve_x, default_curve_y


def load_behavioral_params_from_excel(excel_path: str = None) -> Dict[str, float]:
    """
    Excel 파일의 BEHAVIORAL_PARAMS 시트에서 행동 파라미터를 로드합니다.
    
    Args:
        excel_path: Excel 파일 경로. None이면 기본 경로 사용.
    
    Returns:
        행동 파라미터 딕셔너리
    """
    if excel_path is None:
        excel_path = DEFAULT_EXCEL_PATH
    
    # 기본값
    default_params = {
        "loan_prepay_rate": 0.03,
        "loan_maturity_repay_rate": 0.85,
        "borrow_refinance_rate": 0.70,
        "credit_line_usage_rate": 0.02,
        "guarantee_usage_rate": 0.01,
        "core_deposit_ratio": 0.60,
        "deposit_rollover_rate": 0.75,
        "deposit_early_withdraw_rate": 0.02,
        "runoff_rate": 0.01,
        "early_termination": 0.005,
    }
    
    if not os.path.exists(excel_path):
        return default_params
    
    try:
        xl = pd.ExcelFile(excel_path)
        behav_df = pd.read_excel(xl, sheet_name='BEHAVIORAL_PARAMS')
        
        # 컬럼명 정규화
        col_mapping = {
            'asof(YYYY-MM-DD)': 'asof',
            'applies_to(선택)': 'applies_to',
            'notes(선택)': 'notes',
        }
        behav_df = behav_df.rename(columns=col_mapping)
        
        # param_value가 숫자인 행만 필터링 (설명 행 제거)
        behav_df = behav_df[pd.to_numeric(behav_df['param_value'], errors='coerce').notna()].copy()
        
        # 딕셔너리로 변환
        params = default_params.copy()
        for _, row in behav_df.iterrows():
            param_name = str(row['param_name'])
            try:
                param_value = float(row['param_value'])
                if param_name in params:
                    params[param_name] = param_value
            except:
                pass
        
        return params
        
    except Exception as e:
        return default_params


def get_available_excel_curves(excel_path: str = None) -> List[str]:
    """
    Excel 파일에서 사용 가능한 금리 커브 이름 목록을 반환합니다.
    """
    if excel_path is None:
        excel_path = DEFAULT_EXCEL_PATH
    
    if not os.path.exists(excel_path):
        return ["BASE"]
    
    try:
        xl = pd.ExcelFile(excel_path)
        yield_df = pd.read_excel(xl, sheet_name='YIELD_CURVE')
        
        if 'curve_name' in yield_df.columns:
            # tenor_years가 숫자인 행만 필터링 (설명 행 제거)
            yield_df = yield_df[pd.to_numeric(yield_df['tenor_years'], errors='coerce').notna()]
            curves = yield_df['curve_name'].dropna().unique().tolist()
            return curves if curves else ["BASE"]
        
        return ["BASE"]
        
    except Exception as e:
        return ["BASE"]


# =========================================================
# 3) Yield Curve 입력 -> 일자별 할인율 생성 (선형 보간)
# =========================================================
def build_yield_curve_inputs() -> Tuple[List[float], List[float]]:
    """
    사용자 입력: 3M, 1Y, 5Y, 10Y (연율)
    -> curve_x(년), curve_y(연율)
    """
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        r_3m = st.number_input("3M 금리(연)", min_value=-0.02, max_value=0.20, value=0.032, step=0.001, format="%.3f")
    with c2:
        r_1y = st.number_input("1Y 금리(연)", min_value=-0.02, max_value=0.20, value=0.035, step=0.001, format="%.3f")
    with c3:
        r_5y = st.number_input("5Y 금리(연)", min_value=-0.02, max_value=0.20, value=0.040, step=0.001, format="%.3f")
    with c4:
        r_10y = st.number_input("10Y 금리(연)", min_value=-0.02, max_value=0.20, value=0.042, step=0.001, format="%.3f")

    curve_x = [0.25, 1.0, 5.0, 10.0]
    curve_y = [float(r_3m), float(r_1y), float(r_5y), float(r_10y)]
    return curve_x, curve_y


def curve_rate_for_years(t_years: np.ndarray, curve_x: List[float], curve_y: List[float]) -> np.ndarray:
    """
    선형 보간 기반 금리 커브: r(t)
    - t_years: 연 단위 array
    """
    return np.interp(t_years, curve_x, curve_y)


def discount_factors_for_dates(
    dates: pd.DatetimeIndex,
    valuation_date: pd.Timestamp,
    curve_x: List[float],
    curve_y: List[float],
) -> pd.Series:
    """
    dates 각각에 대해 할인계수 DF(t)=1/(1+r(t))^(t)
    - r(t)는 커브 선형 보간
    - t는 year fraction
    """
    t_days = (dates - valuation_date).days.values.astype(float)
    t_years = np.maximum(t_days / 365.0, 0.0)

    r = curve_rate_for_years(t_years, curve_x, curve_y)
    df = 1.0 / np.power(1.0 + r, t_years)
    return pd.Series(df, index=dates)


# =========================================================
# 4) 고성능 Cashflow 엔진 (벡터화) + BASE/+1bp 활용 가능
# =========================================================
def build_cashflow_schedule_fast(
    positions: pd.DataFrame,
    start_date: str,
    end_date: str,
    behavioral: Dict[str, float],
    rate_shock_bp: float = 0.0,
    scenario: str = "BASE",
) -> pd.DataFrame:
    """
    벡터화 기반 캐시플로우 생성 (확장된 행동비율 적용)
    
    행동비율:
    - loan_prepay_rate: 대출 조기상환율
    - loan_maturity_repay_rate: 대출 만기상환율
    - borrow_refinance_rate: 차입 차환율 (갱신율)
    - credit_line_usage_rate: 신용약정 추가사용률
    - guarantee_usage_rate: 지급보증 추가사용률
    - core_deposit_ratio: 핵심예금비율
    - deposit_rollover_rate: 만기재예치율
    - deposit_early_withdraw_rate: 중도해지율
    - runoff_rate: 일반 유출율
    - early_termination: 조기종료율
    """
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    if n == 0:
        return pd.DataFrame()

    days = np.arange(n, dtype=float)
    shock = float(rate_shock_bp) / 10000.0

    # 확장된 행동비율 파라미터
    loan_prepay_d = float(behavioral.get("loan_prepay_rate", 0.03)) / 365.0
    loan_maturity_repay = float(behavioral.get("loan_maturity_repay_rate", 0.85))
    
    borrow_refinance = float(behavioral.get("borrow_refinance_rate", 0.70))
    credit_line_usage_d = float(behavioral.get("credit_line_usage_rate", 0.02)) / 365.0
    guarantee_usage_d = float(behavioral.get("guarantee_usage_rate", 0.01)) / 365.0
    
    core_deposit = float(behavioral.get("core_deposit_ratio", 0.60))
    deposit_rollover = float(behavioral.get("deposit_rollover_rate", 0.75))
    deposit_early_withdraw_d = float(behavioral.get("deposit_early_withdraw_rate", 0.02)) / 365.0
    
    runoff_d = float(behavioral.get("runoff_rate", 0.01)) / 365.0
    early_term_d = float(behavioral.get("early_termination", 0.005)) / 365.0

    all_cfs = []

    pos = positions[positions["type"].isin(["asset", "liability"])].copy()
    if pos.empty:
        return pd.DataFrame()

    for _, row in pos.iterrows():
        pos_type = str(row["type"])
        product = str(row.get("product", ""))
        is_asset = pos_type == "asset"
        sign = 1.0 if is_asset else -1.0

        # 상품별 행동비율 적용
        decay = 0.0
        rollover = 0.70  # 기본값
        
        if is_asset:
            if "대출" in product:
                decay = loan_prepay_d
                rollover = 1.0 - loan_maturity_repay
            elif "신용약정" in product:
                decay = -credit_line_usage_d  # 음수 = 잔액 증가
                rollover = 0.95
            else:
                decay = runoff_d * 0.5
                rollover = 0.80
        else:
            if "예금" in product:
                if "요구불" in product:
                    decay = runoff_d * (1.0 - core_deposit)
                    rollover = 0.95
                else:
                    decay = deposit_early_withdraw_d
                    rollover = 1.0 - deposit_rollover
            elif "차입" in product:
                decay = early_term_d
                rollover = 1.0 - borrow_refinance
            elif "지급보증" in product:
                decay = -guarantee_usage_d  # 음수 = 잔액 증가
                rollover = 0.90
            else:
                decay = runoff_d + early_term_d
                rollover = 0.75

        decay = max(decay, -0.1)  # 음수 허용 (잔액 증가)

        bal0 = float(row.get("balance", 0.0))
        if bal0 <= 0:
            continue

        # 잔액 경로 (증가 가능)
        bal_path = bal0 * np.power(1.0 - decay, days)
        bal_path = np.maximum(bal_path, 0.0)  # 음수 방지

        eff_rate = float(row.get("rate", 0.0)) + shock
        
        # ============================================================
        # 여러 계좌의 집합 가정: 매일 일정 비율의 계좌가 만기 도래
        # ============================================================
        
        # 1단계: 매일 만기 도래하는 잔액 계산 (균등 분산)
        years = float(BUCKET_YEARS.get(str(row.get("maturity_bucket", "1Y")), 1.0))
        maturity_days = int(years * 365)
        
        # 전체 잔액을 만기일수로 균등 분배 (매일 만기 도래)
        daily_maturity_amount = bal0 / max(maturity_days, 1)
        
        # 2단계: 행동비율 적용
        if is_asset:
            if "대출" in product:
                # 대출: 조기상환 + 만기상환 + 재대출
                daily_prepay = bal_path * loan_prepay_d  # 조기상환
                daily_maturity = np.zeros(n, dtype=float)
                # 매일 일정량씩 만기 도래
                for i in range(min(maturity_days, n)):
                    daily_maturity[i] = daily_maturity_amount * loan_maturity_repay  # 상환
                    # 재대출 (1 - 상환율) → 만기일에 다시 원금 CF 발생
                    refinance_amount = daily_maturity_amount * (1.0 - loan_maturity_repay)
                    refinance_day = min(i + maturity_days, n - 1)
                    daily_maturity[refinance_day] += refinance_amount
                
                principal = sign * (daily_prepay + daily_maturity)
            
            elif "신용약정" in product:
                # 신용약정: 잔액 증가 (사용)
                daily_usage = bal_path * (-credit_line_usage_d)  # 음수 = CF 유출
                principal = sign * daily_usage
            
            else:
                # 기타 자산: 소량 유출
                daily_runoff = bal_path * runoff_d * 0.5
                principal = sign * daily_runoff
        
        else:  # 부채
            if "예금" in product:
                if "요구불" in product:
                    # 요구불: 핵심예금 제외한 부분만 유출
                    daily_runoff = bal_path * runoff_d * (1.0 - core_deposit)
                    principal = sign * daily_runoff
                else:
                    # 정기예금: 중도해지 + 만기 유출/재예치
                    daily_early = bal_path * deposit_early_withdraw_d  # 중도해지
                    daily_maturity = np.zeros(n, dtype=float)
                    # 매일 일정량씩 만기 도래
                    for i in range(min(maturity_days, n)):
                        outflow = daily_maturity_amount * (1.0 - deposit_rollover)  # 유출
                        daily_maturity[i] = outflow
                        # 재예치 → 만기일에 다시 원금 CF 발생
                        rollover_amount = daily_maturity_amount * deposit_rollover
                        rollover_day = min(i + maturity_days, n - 1)
                        daily_maturity[rollover_day] += rollover_amount
                    
                    principal = sign * (daily_early + daily_maturity)
            
            elif "차입" in product:
                # 차입: 조기종료 + 만기 상환/차환
                daily_early = bal_path * early_term_d
                daily_maturity = np.zeros(n, dtype=float)
                for i in range(min(maturity_days, n)):
                    repay = daily_maturity_amount * (1.0 - borrow_refinance)  # 상환
                    daily_maturity[i] = repay
                    # 차환 → 만기일에 다시 원금 CF 발생
                    refinance_amount = daily_maturity_amount * borrow_refinance
                    refinance_day = min(i + maturity_days, n - 1)
                    daily_maturity[refinance_day] += refinance_amount
                
                principal = sign * (daily_early + daily_maturity)
            
            elif "지급보증" in product:
                # 지급보증: 잔액 증가 (실행)
                daily_usage = bal_path * (-guarantee_usage_d)  # 음수 = CF 유출
                principal = sign * daily_usage
            
            else:
                # 기타 부채: 소량 유출
                daily_runoff = bal_path * (runoff_d + early_term_d)
                principal = sign * daily_runoff
        
        # 3단계: 이자 CF
        interest = sign * (bal_path * eff_rate / 365.0)
        
        # 4단계: 총 CF
        cf = interest + principal

        all_cfs.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "type": pos_type,
                    "product": product,
                    "maturity_bucket": str(row.get("maturity_bucket", "")),
                    "balance0": bal0,
                    "duration": float(row.get("duration", 0.0)),
                    "cashflow": cf,
                    "interest": interest,
                    "principal": principal,
                    "scenario": scenario,
                }
            )
        )

    return pd.concat(all_cfs, ignore_index=True) if all_cfs else pd.DataFrame()


def pv_from_cashflows_with_curve(
    cashflows: pd.DataFrame,
    valuation_date: pd.Timestamp,
    curve_x: List[float],
    curve_y: List[float],
) -> float:
    """
    Net cashflow를 valuation_date 이후로 집계한 뒤,
    Discount Factor(커브 기반)로 PV 계산.
    """
    if cashflows.empty:
        return 0.0

    df = cashflows.copy()
    df["date"] = pd.to_datetime(df["date"])

    fut = df[df["date"] >= valuation_date]
    if fut.empty:
        return 0.0

    net = fut.groupby("date")["cashflow"].sum().sort_index()
    dts = pd.DatetimeIndex(net.index)

    disc = discount_factors_for_dates(dts, valuation_date, curve_x, curve_y)
    pv = float(np.sum(net.values.astype(float) * disc.values.astype(float)))
    return pv


def pv_breakdown_by_type_with_curve(
    cashflows: pd.DataFrame,
    valuation_date: pd.Timestamp,
    curve_x: List[float],
    curve_y: List[float],
) -> Dict[str, float]:
    """
    DV01을 자산/부채/순으로 보여주기 위해 type별 PV 분해
    """
    if cashflows.empty:
        return {"asset": 0.0, "liability": 0.0, "net": 0.0}

    df = cashflows.copy()
    df["date"] = pd.to_datetime(df["date"])
    fut = df[df["date"] >= valuation_date]
    if fut.empty:
        return {"asset": 0.0, "liability": 0.0, "net": 0.0}

    # 할인계수
    all_dates = pd.DatetimeIndex(sorted(fut["date"].unique()))
    disc = discount_factors_for_dates(all_dates, valuation_date, curve_x, curve_y)

    out = {}
    for t in ["asset", "liability"]:
        sub = fut[fut["type"] == t]
        if sub.empty:
            out[t] = 0.0
        else:
            net = sub.groupby("date")["cashflow"].sum().reindex(all_dates).fillna(0.0)
            out[t] = float(np.sum(net.values.astype(float) * disc.values.astype(float)))

    out["net"] = out["asset"] + out["liability"]
    return out


# =========================================================
# 5) KPI (간이) + DV01
# =========================================================
def compute_kpis_pro(
    positions: pd.DataFrame,
    cashflows: pd.DataFrame,
    valuation_date: pd.Timestamp,
    curve_x: List[float],
    curve_y: List[float],
    lcr_horizon_days: int,
    stress_horizon_days: int,
) -> Dict[str, float]:
    """
    - HQLA
    - NII(누적): valuation_date까지 interest 누적
    - NPV: 커브 기반 PV
    - DV01: (NPV(+1bp) - NPV(BASE))  (단위: 금액/1bp)
    - LCR(간이): HQLA / 30일 순유출
    - Stress survive(간이): HQLA + 누적 net cashflow가 음수로 내려가는지
    """
    # HQLA
    hqla = float(positions[positions["type"] == "hqla"]["balance"].sum())

    if cashflows.empty:
        return {
            "HQLA": hqla,
            "NII_YTD": 0.0,
            "NPV": 0.0,
            "DV01_NET": 0.0,
            "DV01_ASSET": 0.0,
            "DV01_LIAB": 0.0,
            "LCR": float("inf"),
            "NetOutflow_30D": 0.0,
            "Stress_Survive": 1.0,
        }

    df = cashflows.copy()
    df["date"] = pd.to_datetime(df["date"])

    # NII 누적
    nii = float(df.loc[df["date"] <= valuation_date, "interest"].sum())

    # NPV (BASE)
    npv = pv_from_cashflows_with_curve(df, valuation_date, curve_x, curve_y)

    # DV01
    behavioral_dummy = {
        "loan_prepay_rate": st.session_state.get("_loan_prepay_rate", 0.03),
        "loan_maturity_repay_rate": st.session_state.get("_loan_maturity_repay_rate", 0.85),
        "borrow_refinance_rate": st.session_state.get("_borrow_refinance_rate", 0.70),
        "credit_line_usage_rate": st.session_state.get("_credit_line_usage_rate", 0.02),
        "guarantee_usage_rate": st.session_state.get("_guarantee_usage_rate", 0.01),
        "core_deposit_ratio": st.session_state.get("_core_deposit_ratio", 0.60),
        "deposit_rollover_rate": st.session_state.get("_deposit_rollover_rate", 0.75),
        "deposit_early_withdraw_rate": st.session_state.get("_deposit_early_withdraw_rate", 0.02),
        "runoff_rate": st.session_state.get("_runoff_rate", 0.01),
        "early_termination": st.session_state.get("_early_term", 0.005),
    }
    start_date = pd.Timestamp(df["date"].min()).date().isoformat()
    end_date = pd.Timestamp(df["date"].max()).date().isoformat()

    cf_base = build_cashflow_schedule_fast(positions, start_date, end_date, behavioral_dummy, rate_shock_bp=0.0, scenario="BASE_DV01")
    cf_up1 = build_cashflow_schedule_fast(positions, start_date, end_date, behavioral_dummy, rate_shock_bp=1.0, scenario="UP1BP_DV01")

    pv_base_break = pv_breakdown_by_type_with_curve(cf_base, valuation_date, curve_x, curve_y)
    pv_up1_break = pv_breakdown_by_type_with_curve(cf_up1, valuation_date, curve_x, curve_y)

    dv01_asset = pv_up1_break["asset"] - pv_base_break["asset"]
    dv01_liab = pv_up1_break["liability"] - pv_base_break["liability"]
    dv01_net = pv_up1_break["net"] - pv_base_break["net"]

    # LCR(간이): 30일 순유출
    h_end = valuation_date + pd.Timedelta(days=int(lcr_horizon_days))
    win = df[(df["date"] > valuation_date) & (df["date"] <= h_end)]
    net_outflow_30d = -float(win["cashflow"].sum())
    net_outflow_30d = max(net_outflow_30d, 0.0)
    lcr = (hqla / net_outflow_30d) if net_outflow_30d > 0 else float("inf")

    # Stress survive(간이): stress_horizon_days 누적 net + HQLA의 최저점
    st_end = valuation_date + pd.Timedelta(days=int(stress_horizon_days))
    st_win = df[(df["date"] > valuation_date) & (df["date"] <= st_end)]
    daily_net = st_win.groupby("date")["cashflow"].sum().sort_index()
    cum = daily_net.cumsum()
    min_buffer = float((hqla + cum).min()) if len(cum) else hqla
    survive = 1.0 if min_buffer >= 0 else 0.0

    return {
        "HQLA": hqla,
        "NII_YTD": nii,
        "NPV": npv,
        "DV01_NET": dv01_net,
        "DV01_ASSET": dv01_asset,
        "DV01_LIAB": dv01_liab,
        "LCR": lcr,
        "NetOutflow_30D": net_outflow_30d,
        "Stress_Survive": survive,
    }


# =========================================================
# 6) 🆕 금리 시나리오 분석 (복수 시나리오 동시 비교)
# =========================================================
def run_rate_scenario_analysis(
    positions: pd.DataFrame,
    start_date: str,
    end_date: str,
    behavioral: Dict[str, float],
    valuation_date: pd.Timestamp,
    curve_x: List[float],
    curve_y: List[float],
    scenarios: Dict[str, float],  # {"시나리오명": bp_shock}
) -> pd.DataFrame:
    """
    여러 금리 시나리오를 동시 실행하여 KPI 비교표 생성
    """
    results = []
    
    for scenario_name, bp_shock in scenarios.items():
        cf = build_cashflow_schedule_fast(
            positions, start_date, end_date, behavioral,
            rate_shock_bp=bp_shock, scenario=scenario_name
        )
        
        kpi = compute_kpis_pro(
            positions, cf, valuation_date, curve_x, curve_y, 30, 90
        )
        
        results.append({
            "시나리오": scenario_name,
            "금리충격(bp)": bp_shock,
            "NPV(조)": kpi["NPV"] / 1e12,
            "NII(조)": kpi["NII_YTD"] / 1e12,
            "DV01_NET(억/bp)": kpi["DV01_NET"] / 1e8,
            "LCR": kpi["LCR"],
            "생존여부": "YES" if kpi["Stress_Survive"] >= 0.5 else "NO",
        })
    
    return pd.DataFrame(results)


# =========================================================
# 7) 🆕 행동비율에 따른 과부족 금액 분석
# =========================================================
def run_behavioral_gap_analysis(
    positions: pd.DataFrame,
    start_date: str,
    end_date: str,
    base_behavioral: Dict[str, float],
    valuation_date: pd.Timestamp,
    curve_x: List[float],
    curve_y: List[float],
    param_name: str,  # "runoff_rate", "rollover_rate", etc.
    param_range: np.ndarray,  # [0.1, 0.2, 0.3, ...]
) -> pd.DataFrame:
    """
    특정 행동 파라미터를 변화시키며 자금 과부족(GAP) 분석
    """
    results = []
    
    for param_value in param_range:
        behavioral = base_behavioral.copy()
        behavioral[param_name] = float(param_value)
        
        cf = build_cashflow_schedule_fast(
            positions, start_date, end_date, behavioral,
            rate_shock_bp=0.0, scenario=f"{param_name}={param_value:.2%}"
        )
        
        if cf.empty:
            continue
        
        # 30일 / 90일 / 180일 누적 GAP 계산
        df = cf.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        gaps = {}
        for horizon_days in [30, 90, 180]:
            h_end = valuation_date + pd.Timedelta(days=horizon_days)
            win = df[(df["date"] > valuation_date) & (df["date"] <= h_end)]
            gap = float(win["cashflow"].sum())
            gaps[f"GAP_{horizon_days}D"] = gap / 1e12
        
        hqla = float(positions[positions["type"] == "hqla"]["balance"].sum()) / 1e12
        
        results.append({
            f"{param_name}": param_value,
            "HQLA(조)": hqla,
            "30일GAP(조)": gaps["GAP_30D"],
            "90일GAP(조)": gaps["GAP_90D"],
            "180일GAP(조)": gaps["GAP_180D"],
            "30일과부족": hqla + gaps["GAP_30D"],
            "90일과부족": hqla + gaps["GAP_90D"],
            "180일과부족": hqla + gaps["GAP_180D"],
        })
    
    return pd.DataFrame(results)


# =========================================================
# 8) 🆕 민감도 분석 (토네이도 차트)
# =========================================================
def run_sensitivity_analysis(
    positions: pd.DataFrame,
    start_date: str,
    end_date: str,
    base_behavioral: Dict[str, float],
    valuation_date: pd.Timestamp,
    curve_x: List[float],
    curve_y: List[float],
    target_metric: str = "NPV",  # "NPV", "LCR", "NII_YTD"
) -> pd.DataFrame:
    """
    주요 파라미터들을 ±20% 변동시켰을 때 목표 지표의 변화 측정
    """
    # 베이스라인 계산
    cf_base = build_cashflow_schedule_fast(
        positions, start_date, end_date, base_behavioral,
        rate_shock_bp=0.0, scenario="BASE"
    )
    kpi_base = compute_kpis_pro(positions, cf_base, valuation_date, curve_x, curve_y, 30, 90)
    base_value = kpi_base[target_metric]
    
    # 새로운 행동비율 파라미터 사용
    params_to_test = {
        "loan_prepay_rate": base_behavioral.get("loan_prepay_rate", 0.03),
        "loan_maturity_repay_rate": base_behavioral.get("loan_maturity_repay_rate", 0.85),
        "borrow_refinance_rate": base_behavioral.get("borrow_refinance_rate", 0.70),
        "credit_line_usage_rate": base_behavioral.get("credit_line_usage_rate", 0.02),
        "deposit_rollover_rate": base_behavioral.get("deposit_rollover_rate", 0.75),
        "deposit_early_withdraw_rate": base_behavioral.get("deposit_early_withdraw_rate", 0.02),
        "core_deposit_ratio": base_behavioral.get("core_deposit_ratio", 0.60),
        "runoff_rate": base_behavioral.get("runoff_rate", 0.01),
    }
    
    results = []
    
    for param_name, base_val in params_to_test.items():
        # -20% 케이스
        behavioral_down = base_behavioral.copy()
        behavioral_down[param_name] = base_val * 0.8
        cf_down = build_cashflow_schedule_fast(
            positions, start_date, end_date, behavioral_down,
            rate_shock_bp=0.0, scenario=f"{param_name}_down"
        )
        kpi_down = compute_kpis_pro(positions, cf_down, valuation_date, curve_x, curve_y, 30, 90)
        
        # +20% 케이스
        behavioral_up = base_behavioral.copy()
        behavioral_up[param_name] = base_val * 1.2
        cf_up = build_cashflow_schedule_fast(
            positions, start_date, end_date, behavioral_up,
            rate_shock_bp=0.0, scenario=f"{param_name}_up"
        )
        kpi_up = compute_kpis_pro(positions, cf_up, valuation_date, curve_x, curve_y, 30, 90)
        
        impact_down = ((kpi_down[target_metric] - base_value) / base_value * 100) if base_value != 0 else 0
        impact_up = ((kpi_up[target_metric] - base_value) / base_value * 100) if base_value != 0 else 0
        
        results.append({
            "파라미터": param_name,
            "기준값": base_val,
            "-20% 영향(%)": impact_down,
            "+20% 영향(%)": impact_up,
            "민감도": abs(impact_up - impact_down),
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("민감도", ascending=False).reset_index(drop=True)
    return df


# =========================================================
# 9) 🆕 최적화 시뮬레이션 (목표 LCR/NII 달성)
# =========================================================
def optimize_behavioral_params(
    positions: pd.DataFrame,
    start_date: str,
    end_date: str,
    base_behavioral: Dict[str, float],
    valuation_date: pd.Timestamp,
    curve_x: List[float],
    curve_y: List[float],
    target_lcr: float = 1.2,
    target_nii_min: float = 0.0,  # 조 단위
) -> Dict:
    """
    scipy.optimize를 사용하여 LCR 목표를 달성하면서 NII를 최대화하는
    행동 파라미터 조합 탐색
    
    최적화 목표:
    - LCR >= target_lcr 제약 하에서
    - NII 최대화
    """
    
    # 초기 LCR 확인
    try:
        cf_initial = build_cashflow_schedule_fast(
            positions, start_date, end_date, base_behavioral,
            rate_shock_bp=0.0, scenario="INITIAL"
        )
        kpi_initial = compute_kpis_pro(positions, cf_initial, valuation_date, curve_x, curve_y, 30, 90)
        initial_lcr = kpi_initial["LCR"]
        
        # 목표 LCR이 너무 높으면 조정
        if target_lcr > initial_lcr * 1.5:
            adjusted_target = initial_lcr * 1.2
            warning_msg = f"목표 LCR {target_lcr:.2f}가 너무 높아 {adjusted_target:.2f}로 조정되었습니다."
            target_lcr = adjusted_target
        else:
            warning_msg = None
    except Exception as e:
        return {
            "success": False,
            "message": f"초기 KPI 계산 실패: {str(e)}",
        }
    
    def objective(params):
        """NII를 최대화하기 위해 음수 반환"""
        loan_prepay, deposit_rollover, runoff, early = params
        
        behavioral_temp = base_behavioral.copy()
        behavioral_temp.update({
            "loan_prepay_rate": float(loan_prepay),
            "deposit_rollover_rate": float(deposit_rollover),
            "runoff_rate": float(runoff),
            "early_termination": float(early),
        })
        
        try:
            cf = build_cashflow_schedule_fast(
                positions, start_date, end_date, behavioral_temp,
                rate_shock_bp=0.0, scenario="OPT"
            )
            kpi = compute_kpis_pro(positions, cf, valuation_date, curve_x, curve_y, 30, 90)
            
            # NII를 최대화 (음수로 반환)
            return -kpi["NII_YTD"]
        except Exception as e:
            return 1e15
    
    def constraint_lcr(params):
        """LCR >= target_lcr 제약 (soft)"""
        loan_prepay, deposit_rollover, runoff, early = params
        
        behavioral_temp = base_behavioral.copy()
        behavioral_temp.update({
            "loan_prepay_rate": float(loan_prepay),
            "deposit_rollover_rate": float(deposit_rollover),
            "runoff_rate": float(runoff),
            "early_termination": float(early),
        })
        
        try:
            cf = build_cashflow_schedule_fast(
                positions, start_date, end_date, behavioral_temp,
                rate_shock_bp=0.0, scenario="OPT"
            )
            kpi = compute_kpis_pro(positions, cf, valuation_date, curve_x, curve_y, 30, 90)
            
            # LCR - target_lcr >= 0 이어야 함
            return kpi["LCR"] - target_lcr
        except Exception as e:
            return -1e15
    
    # 초기값 (더 보수적으로 설정)
    x0 = [
        base_behavioral.get("loan_prepay_rate", 0.03),
        base_behavioral.get("deposit_rollover_rate", 0.75),
        base_behavioral.get("runoff_rate", 0.01),
        base_behavioral.get("early_termination", 0.005),
    ]
    
    # 파라미터 범위 (더 넓게 설정)
    bounds = [
        (0.001, 0.30),  # loan_prepay_rate
        (0.30, 0.99),   # deposit_rollover_rate (1.0은 제외)
        (0.001, 0.30),  # runoff_rate
        (0.001, 0.30),  # early_termination
    ]
    
    # 제약 조건
    constraints = [
        {"type": "ineq", "fun": constraint_lcr},
    ]
    
    # 최적화 실행 (여러 방법 시도)
    result = None
    methods = ["SLSQP", "trust-constr"]
    
    for method in methods:
        try:
            if method == "SLSQP":
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 150, "ftol": 1e-6},
                )
            else:  # trust-constr
                from scipy.optimize import NonlinearConstraint
                nlc = NonlinearConstraint(
                    lambda x: constraint_lcr(x),
                    0,  # lower bound
                    np.inf,  # upper bound
                )
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=[nlc],
                    options={"maxiter": 150},
                )
            
            if result.success:
                break
        except Exception as e:
            continue
    
    if result is not None and result.success:
        # 모든 base_behavioral 파라미터를 포함하여 업데이트
        optimal_params = base_behavioral.copy()
        optimal_params.update({
            "loan_prepay_rate": float(result.x[0]),
            "deposit_rollover_rate": float(result.x[1]),
            "runoff_rate": float(result.x[2]),
            "early_termination": float(result.x[3]),
        })
        
        # 최적 파라미터로 KPI 재계산
        try:
            cf_opt = build_cashflow_schedule_fast(
                positions, start_date, end_date, optimal_params,
                rate_shock_bp=0.0, scenario="OPTIMAL"
            )
            kpi_opt = compute_kpis_pro(positions, cf_opt, valuation_date, curve_x, curve_y, 30, 90)
            
            message = "최적화 성공"
            if warning_msg:
                message += f" ({warning_msg})"
            
            return {
                "success": True,
                "optimal_params": optimal_params,
                "optimal_kpi": kpi_opt,
                "message": message,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"최적 파라미터 KPI 계산 실패: {str(e)}",
            }
    else:
        # 최적화 실패 시 초기값 반환
        message = "최적화 실패"
        if result is not None:
            message += f": {result.message}"
        
        # 현재 상태가 제약을 만족하는지 확인
        current_satisfies = constraint_lcr(x0) >= 0
        
        if current_satisfies:
            message += f" (현재 LCR {initial_lcr:.2f}가 목표 {target_lcr:.2f}를 이미 만족합니다)"
            return {
                "success": True,
                "optimal_params": base_behavioral.copy(),
                "optimal_kpi": kpi_initial,
                "message": message,
            }
        else:
            message += f" (현재 LCR {initial_lcr:.2f}, 목표 {target_lcr:.2f}는 달성 불가능합니다. 목표를 낮춰주세요)"
            return {
                "success": False,
                "message": message,
            }


# =========================================================
# 10) 🆕 SVG 애니메이션 함수 (ALM Flow Animation)
# =========================================================
def build_svg_animation(
    positions: pd.DataFrame,
    cf: pd.DataFrame,
    day_index: int,
    total_days: int,
    base_seconds_per_cycle: float,
) -> str:
    """
    일자별 ALM Flow Animation SVG 생성
    - 상품 겹침 방지: Y축 간격 증가
    - 회전 속도: 기존 대비 1/5 (5배 느리게)
    """
    def escape_xml(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
    
    def _scale01(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float((x - lo) / (hi - lo))
    
    # 버킷별 X 종료 위치 계산 (현재시점에서 만기까지)
    def bucket_end_x(bucket: str, axis_start: float, axis_end: float) -> float:
        """버킷의 종료 X 위치 계산"""
        idx = BUCKET_X.get(str(bucket), 0)
        n_seg = max(1, len(BUCKET_ORDER) - 1)
        seg_w = (axis_end - axis_start) / n_seg
        # 버킷 끝 위치 (다음 버킷 시작점)
        return axis_start + seg_w * (idx + 1)
    
    def speed_seconds_per_cycle_by_maturity(pos_type: str, maturity_bucket: str, base_seconds: float) -> float:
        """회전 속도를 기존 대비 5배 느리게 (1/5 속도)"""
        years = BUCKET_YEARS.get(maturity_bucket, 1.0)
        s = _scale01(years, 0.25, 7.0)
        dur = float(base_seconds * (0.55 + 1.75 * s))
        if pos_type == "liability":
            dur = dur * 0.92
        # 5배 느리게
        dur = dur * 5.0
        return float(max(5.0, dur))
    
    def duration_ratio(duration: float, bucket: str) -> float:
        years = BUCKET_YEARS.get(bucket, 1.0)
        denom = max(years, 0.25)
        r = duration / denom
        return float(min(max(r, 0.05), 0.95))
    
    # 캔버스 - 타임라인 포함 충분한 높이
    W, H = 1520, 1850  # 높이 증가 (1750 -> 1850)
    pad = 22
    
    # 레이아웃
    left_w = 260  # 왼쪽 HQLA 패널 폭
    main_w = W - pad * 2 - left_w - 16
    
    x_left = pad
    x_main = x_left + left_w + 16
    y_top = pad
    
    # 상하 공간 배분
    top_h = 1050  # 자산/부채 영역
    bottom_h = 380  # 타임라인 영역 (320 -> 380으로 증가)
    y_bottom = y_top + top_h + 14
    
    asset_h = int(top_h * 0.50)  # 자산/부채 영역 동일 비율
    liab_h = top_h - asset_h
    
    y_asset0 = y_top
    y_asset1 = y_top + asset_h
    y_liab0 = y_asset1
    y_liab1 = y_top + top_h
    
    # 데이터 분리
    assets = positions[positions["type"] == "asset"].reset_index(drop=True)
    liabs = positions[positions["type"] == "liability"].reset_index(drop=True)
    hqla = positions[positions["type"] == "hqla"].reset_index(drop=True)
    
    # 누적 GAP -> Cash 반영
    cf_full = cf.iloc[: day_index + 1].copy() if not cf.empty else cf.iloc[:1].copy()
    cum_gap = float(cf_full["gap_cf"].sum())
    
    # 초기 Cash
    cash0 = 0.0
    if not hqla.empty:
        cash_rows = hqla[hqla["product"].astype(str).str.lower().str.contains("cash|현금")]
        if not cash_rows.empty:
            cash0 = float(cash_rows.iloc[0]["balance"])
    cash_t = cash0 + cum_gap
    cash_display = max(0.0, cash_t)
    funding_need = max(0.0, -cash_t)
    
    # HQLA 기타
    hqla_other = hqla.copy()
    if not hqla_other.empty:
        hqla_other = hqla_other[~(hqla_other["product"].astype(str).str.lower().str.contains("cash|현금"))].reset_index(drop=True)
    hqla_other_total = float(hqla_other["balance"].sum()) if not hqla_other.empty else 0.0
    
    # Progress
    progress = 0.0 if total_days <= 0 else min(max(day_index / total_days, 0.0), 1.0)
    prog_w = int((W - 2 * pad) * progress)
    
    # 컬러
    asset_stroke = "#1f5fae"
    liab_stroke = "#667085"
    asset_fill = "rgba(127,182,255,0.20)"
    liab_fill = "rgba(201,206,214,0.42)"
    duration_stroke = "#111111"
    
    # 축 설정
    axis_x0 = x_main + 26
    axis_x1 = x_main + main_w - 26
    axis_y = y_top + top_h - 30
    
    # 마진 밴드
    m0 = BUCKET_ORDER.index("6M") if "6M" in BUCKET_ORDER else 1
    m1 = BUCKET_ORDER.index("2Y") if "2Y" in BUCKET_ORDER else 3
    n_seg = max(1, len(BUCKET_ORDER) - 1)
    seg_w = (axis_x1 - axis_x0) / n_seg
    margin_band_x0 = axis_x0 + seg_w * m0
    margin_band_x1 = axis_x0 + seg_w * m1

    def bucket_x(bucket: str) -> float:
        idx = BUCKET_X.get(str(bucket), 0)
        return axis_x0 + seg_w * idx
    
    # 캡슐 설정
    cap_h = 25  # 캡슐 높이 (25px)
    cap_rx = 5  # 모서리 둥글기
    dasharray = "10 6"
    stroke_width = 2
    
    # 금리재조정 만기 X 위치 계산
    def rate_maturity_x(rate_mat: str) -> float:
        """금리재조정 만기의 X 위치"""
        idx = BUCKET_X.get(str(rate_mat), 0)
        return axis_x0 + seg_w * idx
    
    # 행 높이 설정 - 간격 50px
    row_height = 50
    
    # 캡슐 요소 - 현재시점(axis_x0)에서 만기(bucket)까지 뻗는 박스 (화살표 없음)
    def capsule_element(y: float, pos_type: str, product: str, bucket: str, balance: float, duration: float, rate_maturity: str = None) -> str:
        # 캡슐 시작: 현재시점 (axis_x0)
        x_start = axis_x0
        # 캡슐 끝: 만기 버킷 끝
        x_end = bucket_end_x(bucket, axis_x0, axis_x1)
        
        w = x_end - x_start - 8  # 약간의 여백
        h = float(cap_h)
        rx = float(cap_rx)
        
        dur_sec = speed_seconds_per_cycle_by_maturity(pos_type, bucket, base_seconds_per_cycle)
        
        stroke = asset_stroke if pos_type == "asset" else liab_stroke
        fill = asset_fill if pos_type == "asset" else liab_fill
        label_color = "#073763" if pos_type == "asset" else "#2b2f36"
        
        anim = f'<animate attributeName="stroke-dashoffset" from="0" to="-720" dur="{dur_sec}s" repeatCount="indefinite" />'
        
        bal_조 = balance / 1e12
        t1 = product
        t2 = f"{bucket} | {bal_조:,.0f}조"
        
        # 캡슐 중심 X
        capsule_center_x = x_start + w / 2
        
        # 금리재조정 만기 X 위치 계산
        rate_x = None
        if rate_maturity and rate_maturity in BUCKET_ORDER:
            rate_x = rate_maturity_x(rate_maturity)
        
        # 금리재조정 만기 직각선 (rate_maturity가 있으면)
        rate_line = ""
        if rate_x and x_start < rate_x < x_start + w:
            rate_line = f'''
            <line x1="{rate_x}" y1="{y - h/2 + 3}" x2="{rate_x}" y2="{y + h/2 - 3}"
                  stroke="#e63946" stroke-width="2" stroke-linecap="round"/>
            <circle cx="{rate_x}" cy="{y}" r="3" fill="#e63946"/>
            '''
        
        return f'''
        <g>
          <rect x="{x_start}" y="{y - h/2}" width="{w}" height="{h}" rx="{rx}"
                fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"
                stroke-dasharray="{dasharray}">
            {anim}
          </rect>
          {rate_line}
          <text x="{capsule_center_x}" y="{y - 2}" text-anchor="middle" font-size="10" font-weight="700" fill="{label_color}">
            {escape_xml(t1)}
          </text>
          <text x="{capsule_center_x}" y="{y + 10}" text-anchor="middle" font-size="9" font-weight="600" fill="rgba(7,55,99,0.60)">
            {escape_xml(t2)}
          </text>
        </g>
        '''
    
    # CF 타임라인
    vmax = float(max(cf["asset_cf"].max(), abs(cf["liability_cf"].min()), 1.0)) if not cf.empty else 1.0
    tl_x0 = x_left + 50  # 왼쪽 여백 추가
    tl_x1 = x_main + main_w - 20
    tl_y0 = y_bottom + 70  # 헤더 공간 확보
    tl_y1 = y_bottom + bottom_h - 50  # 하단 X축 레이블 공간
    tl_mid = (tl_y0 + tl_y1) / 2
    
    bar_w = (tl_x1 - tl_x0) / max(1, total_days + 1)
    tl_bars = []
    for i in range(len(cf_full)):
        a = float(cf_full.iloc[i]["asset_cf"])
        l = float(cf_full.iloc[i]["liability_cf"])
        ah = (abs(a) / vmax) * (tl_y1 - tl_y0) * 0.38
        lh = (abs(l) / vmax) * (tl_y1 - tl_y0) * 0.38
        x = tl_x0 + i * bar_w + bar_w * 0.15
        bw = max(1.0, bar_w * 0.70)
        
        tl_bars.append(f'<rect x="{x}" y="{tl_mid - ah}" width="{bw}" height="{ah}" fill="rgba(25,195,125,0.75)" rx="2"/>')
        tl_bars.append(f'<rect x="{x}" y="{tl_mid}" width="{bw}" height="{lh}" fill="rgba(255,176,32,0.75)" rx="2"/>')
    
    # X축 일자 눈금 생성
    tl_x_ticks = []
    tick_interval = max(1, total_days // 10)  # 약 10개 눈금
    for i in range(0, total_days + 1, tick_interval):
        x = tl_x0 + i * bar_w + bar_w * 0.5
        tl_x_ticks.append(f'''
          <line x1="{x}" y1="{tl_y1}" x2="{x}" y2="{tl_y1 + 6}" stroke="rgba(10,60,120,0.3)" stroke-width="1"/>
          <text x="{x}" y="{tl_y1 + 20}" text-anchor="middle" font-size="10" font-weight="600" fill="rgba(7,55,99,0.7)">D{i}</text>
        ''')
    
    marker_x = tl_x0 + day_index * bar_w + bar_w * 0.5
    marker = f'''
      <line x1="{marker_x}" y1="{tl_y0}" x2="{marker_x}" y2="{tl_y1}"
            stroke="rgba(59,130,246,0.9)" stroke-width="2.5"/>
      <circle cx="{marker_x}" cy="{tl_y0 - 6}" r="5" fill="rgba(59,130,246,0.9)"/>
      <text x="{marker_x}" y="{tl_y0 - 14}" text-anchor="middle" font-size="10" font-weight="700" fill="rgba(59,130,246,1)">Day {day_index}</text>
    '''
    
    # 축 + 마진밴드
    bucket_ticks = []
    for i, b in enumerate(BUCKET_ORDER):
        x = axis_x0 + seg_w * i
        bucket_ticks.append(f'''
          <line x1="{x}" y1="{axis_y-8}" x2="{x}" y2="{axis_y+8}" stroke="rgba(10,60,120,0.18)" stroke-width="1"/>
          <text x="{x}" y="{axis_y+24}" text-anchor="middle" font-size="11" font-weight="900" fill="rgba(7,55,99,0.82)">{b}</text>
        ''')
    
    margin_band = f'''
      <rect x="{margin_band_x0}" y="{y_top + 70}" width="{max(0.0, margin_band_x1 - margin_band_x0)}" height="{top_h - 110}"
            fill="rgba(25,195,125,0.08)" stroke="rgba(25,195,125,0.28)" stroke-width="1.2" rx="14"/>
      <text x="{(margin_band_x0 + margin_band_x1)/2}" y="{y_top + 60}" text-anchor="middle"
            font-size="11" font-weight="900" fill="rgba(25,195,125,0.95)">MARGIN ZONE</text>
    '''
    
    axis_line = f'''
      <line x1="{axis_x0}" y1="{axis_y}" x2="{axis_x1}" y2="{axis_y}"
            stroke="rgba(10,60,120,0.22)" stroke-width="2"/>
      {''.join(bucket_ticks)}
    '''
    
    # Progress bar
    prog = f'''
    <rect x="{pad}" y="{pad}" width="{W-2*pad}" height="12" rx="6" fill="rgba(10,60,120,0.06)" />
    <rect x="{pad}" y="{pad}" width="{prog_w}" height="12" rx="6" fill="rgba(127,182,255,0.80)" />
    <text x="{pad}" y="{pad-6}" font-size="12" font-weight="900" fill="#073763">Day Flow</text>
    <text x="{W-pad}" y="{pad-6}" text-anchor="end" font-size="12" font-weight="900" fill="#073763">{day_index}/{total_days} days</text>
    '''
    
    title = f'''
    <text x="{pad}" y="{pad+46}" font-size="20" font-weight="900" fill="#073763">
      Bank ALM Visual (Top: Assets / Bottom: Liabilities / Left: HQLA + Cash Account)
    </text>
    <text x="{pad}" y="{pad+70}" font-size="12" font-weight="600" fill="rgba(7,55,99,0.70)">
      외곽선 흐름 속도는 만기에 비례해 길수록 느리게, 짧을수록 빠르게 동작합니다. (5배 느린 속도)
    </text>
    '''
    
    # 패널
    panels = f'''
      <rect x="{x_left}" y="{y_top+84}" width="{left_w}" height="{top_h-84}" rx="18" fill="rgba(127,182,255,0.07)" stroke="rgba(10,60,120,0.10)"/>
      <rect x="{x_main}" y="{y_top+84}" width="{main_w}" height="{top_h-84}" rx="18" fill="white" stroke="rgba(10,60,120,0.10)"/>
      <rect x="{x_left}" y="{y_bottom}" width="{(x_main + main_w) - x_left}" height="{bottom_h}" rx="18" fill="white" stroke="rgba(10,60,120,0.10)"/>
    '''
    
    # HQLA 패널
    cash_bar_w = left_w - 44
    cash_ratio = 0.0
    denom_for_cash = max(1.0, cash0 + max(0.0, cum_gap))
    if denom_for_cash > 0:
        cash_ratio = min(max(cash_display / denom_for_cash, 0.0), 1.0)
    cash_fill_w = cash_bar_w * cash_ratio
    
    other_lines = []
    for _, r in hqla_other.iterrows():
        other_lines.append(f"{str(r['product'])}: {float(r['balance'])/1e12:,.2f}조")
    other_text = "<br/>".join([escape_xml(s) for s in other_lines]) if other_lines else "기타 HQLA 없음"
    
    hqla_panel = f'''
      <text x="{x_left+18}" y="{y_top+118}" font-size="14" font-weight="900" fill="#073763">HQLA</text>
      <text x="{x_left+18}" y="{y_top+150}" font-size="13" font-weight="900" fill="#073763">Cash Account (Dynamic)</text>
      <rect x="{x_left+18}" y="{y_top+162}" width="{cash_bar_w}" height="16" rx="8" fill="rgba(10,60,120,0.06)"/>
      <rect x="{x_left+18}" y="{y_top+162}" width="{cash_fill_w}" height="16" rx="8" fill="rgba(127,182,255,0.85)"/>
      
      <text x="{x_left+18}" y="{y_top+202}" font-size="12" font-weight="900" fill="rgba(7,55,99,0.88)">
        Cash0: {cash0/1e12:,.2f}조
      </text>
      <text x="{x_left+18}" y="{y_top+224}" font-size="12" font-weight="900" fill="rgba(7,55,99,0.88)">
        Cum GAP: {cum_gap/1e12:,.2f}조
      </text>
      <text x="{x_left+18}" y="{y_top+246}" font-size="12" font-weight="900" fill="rgba(7,55,99,0.88)">
        Cash(t): {cash_t/1e12:,.2f}조
      </text>
      <text x="{x_left+18}" y="{y_top+268}" font-size="12" font-weight="900" fill="rgba(255,92,92,0.92)">
        Funding Need: {funding_need/1e12:,.2f}조
      </text>
      
      <line x1="{x_left+18}" y1="{y_top+286}" x2="{x_left+left_w-18}" y2="{y_top+286}" stroke="rgba(10,60,120,0.14)"/>
      
      <text x="{x_left+18}" y="{y_top+312}" font-size="13" font-weight="900" fill="#073763">Other HQLA (Static)</text>
      <text x="{x_left+18}" y="{y_top+334}" font-size="12" font-weight="800" fill="rgba(7,55,99,0.78)">Total: {hqla_other_total/1e12:,.2f}조</text>
      
      <foreignObject x="{x_left+18}" y="{y_top+342}" width="{left_w-36}" height="120">
        <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:12px; font-weight:700; color:rgba(7,55,99,0.75); line-height:1.45;">
          {other_text}
        </div>
      </foreignObject>
    '''
    
    # 메인 라벨 + 더 두꺼운 구분선
    main_labels = f'''
      <text x="{x_main+18}" y="{y_top+118}" font-size="14" font-weight="900" fill="#073763">ASSETS (Top)</text>
      <text x="{x_main+18}" y="{y_liab0+118}" font-size="14" font-weight="900" fill="#073763">LIABILITIES (Bottom)</text>
      <line x1="{x_main+14}" y1="{y_liab0}" x2="{x_main+main_w-14}" y2="{y_liab0}"
            stroke="rgba(10,60,120,0.25)" stroke-width="3"/>
    '''
    
    # 캡슐 생성 - 모든 상품이 왼쪽(현재시점)에서 시작하여 만기까지 뻗음
    shapes = []
    
    def calculate_y_positions(df: pd.DataFrame, y_area_start: float, y_area_end: float) -> Dict[int, float]:
        """
        상품별 Y 위치 계산 - 단순히 순서대로 위에서 아래로 배치
        """
        if df.empty:
            return {}
        
        n = len(df)
        available_height = y_area_end - y_area_start
        actual_row_height = min(row_height, available_height / max(n, 1))
        
        y_map = {}
        for i, (idx, row) in enumerate(df.iterrows()):
            y_pos = y_area_start + (i + 0.5) * actual_row_height
            y_map[idx] = y_pos
        
        return y_map
    
    # 자산/부채 영역별 Y 범위 (레이블 공간 확보)
    asset_y_start = y_asset0 + 130  # ASSETS (Top) 레이블 공간
    asset_y_end = y_asset1 - 20
    liab_y_start = y_liab0 + 130  # LIABILITIES (Bottom) 레이블 공간  
    liab_y_end = y_liab1 - 50  # 축 레이블 공간
    
    asset_y_map = calculate_y_positions(assets, asset_y_start, asset_y_end)
    liab_y_map = calculate_y_positions(liabs, liab_y_start, liab_y_end)

    # 자산 캡슐 생성
    for i, r in assets.iterrows():
        y = asset_y_map.get(i, (asset_y_start + asset_y_end) / 2)
        rate_mat = str(r.get("rate_maturity", "")) if "rate_maturity" in r else None
        shapes.append(capsule_element(
            y=y, 
            pos_type="asset", 
            product=str(r["product"]), 
            bucket=str(r["maturity_bucket"]), 
            balance=float(r["balance"]), 
            duration=float(r["duration"]),
            rate_maturity=rate_mat
        ))
    
    # 부채 캡슐 생성
    for i, r in liabs.iterrows():
        y = liab_y_map.get(i, (liab_y_start + liab_y_end) / 2)
        rate_mat = str(r.get("rate_maturity", "")) if "rate_maturity" in r else None
        shapes.append(capsule_element(
            y=y, 
            pos_type="liability", 
            product=str(r["product"]), 
            bucket=str(r["maturity_bucket"]), 
            balance=float(r["balance"]), 
            duration=float(r["duration"]),
            rate_maturity=rate_mat
        ))
    
    # 타임라인 헤더 - 배경 박스와 함께 표시
    tl_header = f'''
      <rect x="{x_left+10}" y="{y_bottom+8}" width="{(x_main + main_w) - x_left - 300}" height="50" rx="8" fill="rgba(255,255,255,0.95)"/>
      <text x="{x_left+18}" y="{y_bottom+30}" font-size="15" font-weight="900" fill="#073763">📊 Daily Cashflow Timeline</text>
      <text x="{x_left+18}" y="{y_bottom+50}" font-size="11" font-weight="700" fill="rgba(7,55,99,0.70)">Asset CF (green) | Liability CF (orange) | Cum.GAP: {cum_gap/1e12:,.2f}조 | Cash(t): {cash_t/1e12:,.2f}조</text>
    '''
    
    # 타임라인 프레임 (X축 눈금 포함)
    tl_frame = f'''
      <rect x="{tl_x0 - 10}" y="{tl_y0 - 25}" width="{(tl_x1-tl_x0) + 20}" height="{tl_y1 - tl_y0 + 60}"
            rx="12" fill="rgba(248,250,252,0.95)" stroke="rgba(10,60,120,0.15)" stroke-width="1.5"/>
      <line x1="{tl_x0}" y1="{tl_mid}" x2="{tl_x1}" y2="{tl_mid}" stroke="rgba(10,60,120,0.15)" stroke-width="1" stroke-dasharray="4 2"/>
      <line x1="{tl_x0}" y1="{tl_y1}" x2="{tl_x1}" y2="{tl_y1}" stroke="rgba(10,60,120,0.2)" stroke-width="1"/>
      {''.join(tl_bars)}
      {''.join(tl_x_ticks)}
      {marker}
    '''
    
    # 레전드 (우측 상단)
    legend = f'''
      <rect x="{tl_x1 - 280}" y="{y_bottom + 12}" width="260" height="32" rx="8"
            fill="rgba(255,255,255,0.95)" stroke="rgba(10,60,120,0.15)"/>
      <circle cx="{tl_x1 - 260}" cy="{y_bottom + 28}" r="5" fill="rgba(25,195,125,0.85)"/>
      <text x="{tl_x1 - 248}" y="{y_bottom + 32}" font-size="10" font-weight="700" fill="rgba(7,55,99,0.8)">Asset CF</text>
      <circle cx="{tl_x1 - 180}" cy="{y_bottom + 28}" r="5" fill="rgba(255,176,32,0.85)"/>
      <text x="{tl_x1 - 168}" y="{y_bottom + 32}" font-size="10" font-weight="700" fill="rgba(7,55,99,0.8)">Liability CF</text>
      <circle cx="{tl_x1 - 90}" cy="{y_bottom + 28}" r="5" fill="rgba(59,130,246,0.9)"/>
      <text x="{tl_x1 - 78}" y="{y_bottom + 32}" font-size="10" font-weight="700" fill="rgba(7,55,99,0.8)">Today</text>
    '''
    
    svg = f'''
    <div style="width:100%; overflow:hidden;">
      <svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="{W}" height="{H}" rx="18" fill="white" />
        {prog}
        {title}
        {panels}
        {hqla_panel}
        {main_labels}
        {margin_band}
        {axis_line}
        {''.join(shapes)}
        {tl_header}
        {legend}
        {tl_frame}
      </svg>
    </div>
    '''
    return svg


# =========================================================
# 11) 1-page Diagram: Overlap 방지 동적 배치 (matplotlib)
# =========================================================
def plot_onepage_diagram_dynamic(
    positions: pd.DataFrame,
    margin_start: str,
    margin_end: str,
) -> plt.Figure:
    """
    - 좌: HQLA
    - 중: Assets
    - 우: Liabilities
    - 상품 수 증가해도 overlap 덜 나도록 y를 동적으로 배치하고 ylim 자동 조절
    """
    fig = plt.figure(figsize=(16, 6.8), dpi=140)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 2.0, 2.0], wspace=0.18)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    # Left: HQLA
    ax_left.set_title("HQLA", fontsize=12, fontweight="bold", color="#073763")
    ax_left.axis("off")

    hqla = positions[positions["type"] == "hqla"].copy()
    total = float(hqla["balance"].sum())

    ax_left.add_patch(
        Rectangle((0.05, 0.55), 0.9, 0.40, transform=ax_left.transAxes,
                  facecolor="#e8f3ff", edgecolor=SKY, linewidth=1.4)
    )
    ax_left.text(0.08, 0.90, "고유동성자산(HQLA)", transform=ax_left.transAxes,
                 fontsize=11, fontweight="bold", color="#073763")
    y = 0.82
    for _, r in hqla.iterrows():
        ax_left.text(0.10, y, f"- {r['product']}: {float(r['balance'])/1e9:,.0f} 조",
                     transform=ax_left.transAxes, fontsize=9, color="#073763")
        y -= 0.08
    ax_left.text(0.10, 0.58, f"합계: {total/1e9:,.0f} 조", transform=ax_left.transAxes,
                 fontsize=10, fontweight="bold", color="#073763")

    ms = BUCKET_X.get(margin_start, 1)
    me = BUCKET_X.get(margin_end, 3)

    def _setup_bucket_axis(ax, title: str, n_items: int, y_top_override: float = None):
        ax.set_title(title, fontsize=12, fontweight="bold", color="#073763")
        ax.set_xlim(-0.5, len(BUCKET_ORDER) - 0.5)

        y_top = y_top_override if y_top_override is not None else max(3.0, n_items * 0.9)
        ax.set_ylim(-0.5, y_top + 0.6)

        ax.set_yticks([])
        ax.set_xticks(range(len(BUCKET_ORDER)))
        ax.set_xticklabels(BUCKET_ORDER)
        ax.grid(axis="x", alpha=0.18)

        ax.add_patch(Rectangle((ms - 0.5, -0.5), (me - ms + 1), y_top + 1.2,
                               facecolor="#43d18b", alpha=0.20, edgecolor="none"))
        ax.text((ms + me) / 2, y_top + 0.25, "마진 우수 구간",
                ha="center", va="center", fontsize=9, color="#167a4b")

        return y_top

    assets = positions[positions["type"] == "asset"].reset_index(drop=True)
    liabs = positions[positions["type"] == "liability"].reset_index(drop=True)

    y_top_a = _setup_bucket_axis(ax_mid, "Assets 원만기 구조", len(assets))
    y_top_l = _setup_bucket_axis(ax_right, "Liabilities 원만기 구조", len(liabs))

    def _y_coords(n: int, y_top: float) -> np.ndarray:
        if n <= 0:
            return np.array([])
        return np.linspace(y_top - 0.5, 0.3, n)

    ay = _y_coords(len(assets), y_top_a)
    ly = _y_coords(len(liabs), y_top_l)

    # Assets draw
    for i, r in assets.iterrows():
        x = BUCKET_X.get(str(r["maturity_bucket"]), 2)
        y = float(ay[i])

        ax_mid.add_patch(
            Ellipse((x, y), width=1.85, height=0.52,
                    facecolor=ASSET_COLOR, edgecolor=ASSET_EDGE, alpha=0.55, linewidth=1.2)
        )
        ax_mid.text(x, y, f"{r['product']}\n{float(r['balance'])/1e9:,.0f}조",
                    ha="center", va="center", fontsize=8, color="#073763")

        dur = float(r["duration"])
        dur_x = x + min(0.60, dur / 3.0)
        ax_mid.add_patch(
            FancyArrowPatch((x - 0.55, y), (dur_x, y),
                            arrowstyle="->", mutation_scale=10, linewidth=1.4, color=DUR_COLOR)
        )
        ax_mid.text(x - 0.55, y + 0.30, f"Dur {dur:.2f}y", fontsize=7, color="black")

    # Liabilities draw
    for i, r in liabs.iterrows():
        x = BUCKET_X.get(str(r["maturity_bucket"]), 2)
        y = float(ly[i])

        ax_right.add_patch(
            Ellipse((x, y), width=1.85, height=0.52,
                    facecolor=LIAB_COLOR, edgecolor=LIAB_EDGE, alpha=0.75, linewidth=1.2)
        )
        ax_right.text(x, y, f"{r['product']}\n{float(r['balance'])/1e9:,.0f}조",
                      ha="center", va="center", fontsize=8, color="#2b2f36")

        dur = float(r["duration"])
        dur_x = x + min(0.60, dur / 3.0)
        ax_right.add_patch(
            FancyArrowPatch((x - 0.55, y), (dur_x, y),
                            arrowstyle="->", mutation_scale=10, linewidth=1.4, color=DUR_COLOR)
        )
        ax_right.text(x - 0.55, y + 0.30, f"Dur {dur:.2f}y", fontsize=7, color="black")

    fig.suptitle("ALM One-Page — 구조/만기/듀레이션/마진 구간 (Dynamic Layout)", fontsize=14, fontweight="bold", color="#073763")
    return fig


# =========================================================
# 11) Sankey Diagram
# =========================================================
def plot_sankey_funding(positions: pd.DataFrame) -> go.Figure:
    """
    개념도: Liability(자금원천) -> Asset(운용) 흐름
    """
    assets = positions[positions["type"] == "asset"].copy()
    liabs = positions[positions["type"] == "liability"].copy()

    if assets.empty or liabs.empty:
        fig = go.Figure()
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), title="Sankey (데이터 부족)")
        return fig

    a_vals = assets["balance"].values.astype(float)
    l_vals = liabs["balance"].values.astype(float)

    a_share = a_vals / a_vals.sum()
    l_share = l_vals / l_vals.sum()

    labels = [f"부채:{p}" for p in liabs["product"].tolist()] + [f"자산:{p}" for p in assets["product"].tolist()]
    n_l = len(liabs)
    n_a = len(assets)

    source = []
    target = []
    value = []

    total_asset = float(a_vals.sum())
    for i in range(n_l):
        for j in range(n_a):
            source.append(i)
            target.append(n_l + j)
            value.append(float(l_share[i] * total_asset * a_share[j]))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(pad=18, thickness=16, label=labels),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        title="자금흐름 Sankey (개념도: 부채→자산)",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# =========================================================
# 12) Cashflow Timeline
# =========================================================
def plot_cashflow_timeline(cashflows: pd.DataFrame, valuation_date: pd.Timestamp, window_days: int = 90) -> go.Figure:
    df = cashflows.copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), title="현금흐름(데이터 없음)")
        return fig

    df["date"] = pd.to_datetime(df["date"])

    start = valuation_date - pd.Timedelta(days=window_days)
    end = valuation_date + pd.Timedelta(days=window_days)
    win = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    a = win[win["type"] == "asset"].groupby("date")["cashflow"].sum()
    l = win[win["type"] == "liability"].groupby("date")["cashflow"].sum()
    idx = pd.date_range(start, end, freq="D")
    a = a.reindex(idx).fillna(0.0)
    l = l.reindex(idx).fillna(0.0)

    net = a + l
    cum = net.cumsum()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=a.values, name="Asset CF", marker_color=ASSET_CF_COLOR, opacity=0.55))
    fig.add_trace(go.Bar(x=idx, y=l.values, name="Liability CF", marker_color=LIAB_CF_COLOR, opacity=0.55))
    fig.add_trace(go.Scatter(x=idx, y=net.values, name="Net CF(GAP)", mode="lines", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=idx, y=cum.values, name="Cumulative Net", mode="lines", line=dict(width=2, dash="dot")))
    fig.add_vline(x=valuation_date, line_width=2, line_dash="dash", line_color="rgba(7,55,99,0.55)")

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        barmode="relative",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title="일자별 현금흐름 타임라인 (자산 위/부채 아래) + Net(GAP)",
        xaxis_title="Date",
        yaxis_title="Cashflow",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(10,60,120,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(10,60,120,0.06)")
    return fig


# =========================================================
# 13) 표시 유틸
# =========================================================
def fmt_조(x: float) -> str:
    if math.isinf(x):
        return "INF"
    return f"{x/1e12:,.2f}조"

def fmt_num(x: float) -> str:
    if math.isinf(x):
        return "INF"
    return f"{x:,.2f}"

def fmt_bp_amount(x: float) -> str:
    return f"{x/1e12:,.3f}조/1bp"

def delta_class(x: float) -> str:
    return "delta-pos" if x >= 0 else "delta-neg"


# =========================================================
# 14) 메인 UI
# =========================================================
@st.cache_data(show_spinner=False)
def cached_sample_positions(seed: int = 7) -> pd.DataFrame:
    return generate_sample_positions(seed=seed)


@st.cache_data(show_spinner=False)
def cached_excel_positions(excel_path: str = None) -> pd.DataFrame:
    """Excel 파일에서 포지션 데이터를 캐시하여 로드"""
    return load_positions_from_excel(excel_path)


@st.cache_data(show_spinner=False)
def cached_excel_yield_curve(excel_path: str = None, curve_name: str = "BASE") -> Tuple[List[float], List[float]]:
    """Excel 파일에서 Yield Curve를 캐시하여 로드"""
    return load_yield_curve_from_excel(excel_path, curve_name)


@st.cache_data(show_spinner=False)
def cached_excel_behavioral_params(excel_path: str = None) -> Dict[str, float]:
    """Excel 파일에서 행동 파라미터를 캐시하여 로드"""
    return load_behavioral_params_from_excel(excel_path)


def main():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="h1">🚀 ALM Visualizer PRO — Advanced Simulation Suite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">금리 시나리오 분석 | 행동비율 과부족 분석 | 민감도 분석 | 최적화 시뮬레이션 | DV01 | Yield Curve</div>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # 🆕 Excel 템플릿에서 기본값 로드
    # -----------------------------
    excel_exists = os.path.exists(DEFAULT_EXCEL_PATH)
    
    # Excel에서 기본값 로드 (있는 경우)
    if excel_exists:
        excel_yield_curve_x, excel_yield_curve_y = cached_excel_yield_curve(DEFAULT_EXCEL_PATH, "BASE")
        excel_behavioral_defaults = cached_excel_behavioral_params(DEFAULT_EXCEL_PATH)
    else:
        excel_yield_curve_x = [0.25, 1.0, 5.0, 10.0]
        excel_yield_curve_y = [0.032, 0.035, 0.040, 0.042]
        excel_behavioral_defaults = {
            "loan_prepay_rate": 0.03,
            "loan_maturity_repay_rate": 0.85,
            "borrow_refinance_rate": 0.70,
            "credit_line_usage_rate": 0.02,
            "guarantee_usage_rate": 0.01,
            "core_deposit_ratio": 0.60,
            "deposit_rollover_rate": 0.75,
            "deposit_early_withdraw_rate": 0.02,
            "runoff_rate": 0.01,
            "early_termination": 0.005,
        }
    
    # Excel Yield Curve를 테너별 기본값으로 변환
    def get_excel_rate_for_tenor(tenor_years, default_val):
        """Excel 커브에서 특정 테너의 금리를 가져옴"""
        try:
            for i, x in enumerate(excel_yield_curve_x):
                if abs(x - tenor_years) < 0.01:
                    return excel_yield_curve_y[i] * 100  # % 단위로 변환
            return default_val
        except:
            return default_val
    
    excel_r3m = get_excel_rate_for_tenor(0.25, 3.2)
    excel_r1y = get_excel_rate_for_tenor(1.0, 3.5)
    excel_r5y = get_excel_rate_for_tenor(5.0, 4.0)
    excel_r10y = get_excel_rate_for_tenor(10.0, 4.2)

    # -----------------------------
    # 🆕 사이드바에 모든 변수 배치
    # -----------------------------
    with st.sidebar:
        st.markdown("## ⚙️ 변수 설정")
        
        # Excel 데이터 소스 표시
        if excel_exists:
            st.caption("📊 기본값: ALM_input_template.xlsx")
        
        # ==========================================
        # 🆕 시나리오 선택 (최상단)
        # ==========================================
        st.markdown("### 🚨 위기 시나리오 선택")
        
        scenario_type = st.selectbox(
            "시나리오 유형",
            ["정상(Normal)", "은행위기(Bank Crisis)", "시장위기(Market Crisis)", "결합위기(Combined Crisis)"],
            key="scenario_type"
        )
        
        # 시나리오별 기본값 정의 (정상 시나리오는 Excel 데이터 사용)
        SCENARIO_DEFAULTS = {
            "정상(Normal)": {
                # 행동비율 - Excel 템플릿에서 로드
                "loan_prepay_rate": excel_behavioral_defaults.get("loan_prepay_rate", 0.03),
                "loan_maturity_repay_rate": excel_behavioral_defaults.get("loan_maturity_repay_rate", 0.85),
                "borrow_refinance_rate": excel_behavioral_defaults.get("borrow_refinance_rate", 0.70),
                "credit_line_usage_rate": excel_behavioral_defaults.get("credit_line_usage_rate", 0.02),
                "guarantee_usage_rate": excel_behavioral_defaults.get("guarantee_usage_rate", 0.01),
                "core_deposit_ratio": excel_behavioral_defaults.get("core_deposit_ratio", 0.60),
                "deposit_rollover_rate": excel_behavioral_defaults.get("deposit_rollover_rate", 0.75),
                "deposit_early_withdraw_rate": excel_behavioral_defaults.get("deposit_early_withdraw_rate", 0.02),
                "runoff_rate": excel_behavioral_defaults.get("runoff_rate", 0.01),
                "early_termination": excel_behavioral_defaults.get("early_termination", 0.005),
                # 금리 - Excel 템플릿에서 로드
                "r_3m": excel_r3m,
                "r_1y": excel_r1y,
                "r_5y": excel_r5y,
                "r_10y": excel_r10y,
                "stress_shock_bp": 150,
                # 설명
                "description": "평상시 운영 상황 (Excel 템플릿 기준)"
            },
            "은행위기(Bank Crisis)": {
                # 행동비율 - 예금 유출 급증, 대출 조기상환 감소
                "loan_prepay_rate": 0.01,  # 대출 조기상환 감소 (유동성 확보 어려움)
                "loan_maturity_repay_rate": 0.70,  # 만기상환율 하락
                "borrow_refinance_rate": 0.40,  # 차입 차환 어려움
                "credit_line_usage_rate": 0.08,  # 신용약정 사용 급증
                "guarantee_usage_rate": 0.05,  # 지급보증 사용 증가
                "core_deposit_ratio": 0.35,  # 핵심예금 비율 급락
                "deposit_rollover_rate": 0.45,  # 예금 재예치율 급락
                "deposit_early_withdraw_rate": 0.15,  # 중도해지율 급증
                "runoff_rate": 0.12,  # 일반 유출 급증
                "early_termination": 0.08,  # 조기종료 급증
                # 금리 - 신용스프레드 확대
                "r_3m": 4.5,
                "r_1y": 5.0,
                "r_5y": 5.5,
                "r_10y": 5.8,
                "stress_shock_bp": 250,
                # 설명
                "description": "은행 신뢰도 하락으로 예금 유출 급증"
            },
            "시장위기(Market Crisis)": {
                # 행동비율 - 시장 유동성 경색
                "loan_prepay_rate": 0.05,  # 대출 조기상환 증가 (자금 필요)
                "loan_maturity_repay_rate": 0.75,  # 만기상환율 소폭 하락
                "borrow_refinance_rate": 0.50,  # 시장 차입 어려움
                "credit_line_usage_rate": 0.06,  # 신용약정 사용 증가
                "guarantee_usage_rate": 0.03,  # 지급보증 사용 증가
                "core_deposit_ratio": 0.50,  # 핵심예금 비율 하락
                "deposit_rollover_rate": 0.60,  # 예금 재예치율 하락
                "deposit_early_withdraw_rate": 0.08,  # 중도해지율 증가
                "runoff_rate": 0.06,  # 일반 유출 증가
                "early_termination": 0.04,  # 조기종료 증가
                # 금리 - 급격한 금리 상승
                "r_3m": 5.0,
                "r_1y": 5.5,
                "r_5y": 6.0,
                "r_10y": 6.5,
                "stress_shock_bp": 300,
                # 설명
                "description": "시장 전반 유동성 경색 및 금리 급등"
            },
            "결합위기(Combined Crisis)": {
                # 행동비율 - 은행위기 + 시장위기 결합 (최악)
                "loan_prepay_rate": 0.02,  # 대출 조기상환 극히 저조
                "loan_maturity_repay_rate": 0.60,  # 만기상환율 급락
                "borrow_refinance_rate": 0.25,  # 차입 차환 거의 불가
                "credit_line_usage_rate": 0.10,  # 신용약정 최대 사용
                "guarantee_usage_rate": 0.08,  # 지급보증 최대 사용
                "core_deposit_ratio": 0.25,  # 핵심예금 비율 최저
                "deposit_rollover_rate": 0.30,  # 예금 재예치율 최저
                "deposit_early_withdraw_rate": 0.25,  # 중도해지율 최대
                "runoff_rate": 0.20,  # 일반 유출 최대
                "early_termination": 0.15,  # 조기종료 최대
                # 금리 - 극심한 금리 상승
                "r_3m": 6.0,
                "r_1y": 6.5,
                "r_5y": 7.0,
                "r_10y": 7.5,
                "stress_shock_bp": 400,
                # 설명
                "description": "은행위기 + 시장위기 동시 발생 (극단적 상황)"
            }
        }
        
        # 선택된 시나리오 기본값 가져오기
        scenario_defaults = SCENARIO_DEFAULTS[scenario_type]
        
        # 시나리오 설명 표시
        if scenario_type == "정상(Normal)":
            st.success(f"📊 {scenario_defaults['description']}")
        elif scenario_type == "은행위기(Bank Crisis)":
            st.warning(f"🏦 {scenario_defaults['description']}")
        elif scenario_type == "시장위기(Market Crisis)":
            st.warning(f"📉 {scenario_defaults['description']}")
        else:
            st.error(f"🔥 {scenario_defaults['description']}")
        
        # 시나리오 적용 버튼
        apply_scenario = st.button("🔄 시나리오 값 적용", type="primary", use_container_width=True)
        
        if apply_scenario:
            st.session_state["scenario_applied"] = scenario_type
            st.toast(f"✅ {scenario_type} 시나리오가 적용되었습니다!", icon="✅")
        
        st.markdown("---")
        
        # 탭으로 구분
        var_tabs = st.tabs(["🎯 행동비율", "📈 Yield Curve", "🔧 기타 설정"])
        
        # 시나리오 적용 여부에 따라 기본값 결정
        if "scenario_applied" in st.session_state and st.session_state.get("scenario_applied") == scenario_type:
            defaults = scenario_defaults
        else:
            defaults = SCENARIO_DEFAULTS["정상(Normal)"]
        
        # 탭 1: 행동비율 파라미터
        with var_tabs[0]:
            st.markdown("**📊 대출 관련**")
            loan_prepay_rate = st.slider("대출 조기상환율(연)", 0.0, 0.30, 
                                         scenario_defaults["loan_prepay_rate"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.03, 
                                         0.005, key="loan_prepay")
            loan_maturity_repay_rate = st.slider("대출 만기상환율", 0.50, 1.0, 
                                                  scenario_defaults["loan_maturity_repay_rate"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.85, 
                                                  0.05, key="loan_maturity")
            
            st.markdown("---")
            
            st.markdown("**💳 차입 및 약정**")
            borrow_refinance_rate = st.slider("차입 차환율", 0.30, 1.0, 
                                               scenario_defaults["borrow_refinance_rate"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.70, 
                                               0.05, key="borrow_ref")
            credit_line_usage_rate = st.slider("신용약정 추가사용률(연)", 0.0, 0.15, 
                                                scenario_defaults["credit_line_usage_rate"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.02, 
                                                0.005, key="credit_usage")
            guarantee_usage_rate = st.slider("지급보증 추가사용률(연)", 0.0, 0.15, 
                                              scenario_defaults["guarantee_usage_rate"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.01, 
                                              0.005, key="guarantee_usage")
            
            st.markdown("---")
            
            st.markdown("**🏦 예금 관련**")
            core_deposit_ratio = st.slider("핵심예금비율", 0.20, 0.90, 
                                            scenario_defaults["core_deposit_ratio"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.60, 
                                            0.05, key="core_deposit")
            deposit_rollover_rate = st.slider("만기재예치율", 0.20, 1.0, 
                                               scenario_defaults["deposit_rollover_rate"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.75, 
                                               0.05, key="deposit_rollover")
            deposit_early_withdraw_rate = st.slider("중도해지율(연)", 0.0, 0.30, 
                                                     scenario_defaults["deposit_early_withdraw_rate"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.02, 
                                                     0.005, key="deposit_early")
            
            st.markdown("---")
            
            st.markdown("**⚙️ 기타**")
            runoff_rate = st.slider("일반 유출율(연)", 0.0, 0.30, 
                                     scenario_defaults["runoff_rate"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.01, 
                                     0.005, key="runoff")
            early_termination = st.slider("조기종료율(연)", 0.0, 0.30, 
                                           scenario_defaults["early_termination"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 0.005, 
                                           0.005, key="early_term")
        
        # 탭 2: Yield Curve
        with var_tabs[1]:
            st.markdown("**Yield Curve 입력**")
            st.markdown("선형 보간을 통해 할인 계수 생성")
            
            r_3m = st.number_input("3M 금리(%)", 0.0, 15.0, 
                                    scenario_defaults["r_3m"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 3.0, 
                                    0.1, key="r3m") / 100.0
            r_1y = st.number_input("1Y 금리(%)", 0.0, 15.0, 
                                    scenario_defaults["r_1y"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 3.2, 
                                    0.1, key="r1y") / 100.0
            r_5y = st.number_input("5Y 금리(%)", 0.0, 15.0, 
                                    scenario_defaults["r_5y"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 3.8, 
                                    0.1, key="r5y") / 100.0
            r_10y = st.number_input("10Y 금리(%)", 0.0, 15.0, 
                                     scenario_defaults["r_10y"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 4.2, 
                                     0.1, key="r10y") / 100.0
            
            curve_x = [0.25, 1.0, 5.0, 10.0]
            curve_y = [r_3m, r_1y, r_5y, r_10y]
            
            # 현재 Yield Curve 시각화
            st.markdown("**현재 Yield Curve**")
            curve_fig = go.Figure()
            curve_fig.add_trace(go.Scatter(
                x=curve_x,
                y=[y * 100 for y in curve_y],
                mode='lines+markers',
                name='Yield Curve',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=8)
            ))
            curve_fig.update_layout(
                xaxis_title="만기(년)",
                yaxis_title="금리(%)",
                height=200,
                margin=dict(t=10, b=30, l=40, r=10)
            )
            st.plotly_chart(curve_fig, use_container_width=True)
        
        # 탭 3: 기타 설정
        with var_tabs[2]:
            st.markdown("**마진 구간**")
            margin_start = st.selectbox("마진 시작 버킷", BUCKET_ORDER, index=BUCKET_ORDER.index(DEFAULT_MARGIN_START), key="margin_start")
            margin_end = st.selectbox("마진 종료 버킷", BUCKET_ORDER, index=BUCKET_ORDER.index(DEFAULT_MARGIN_END), key="margin_end")
            
            st.markdown("**금리 쇼크**")
            stress_shock_bp = st.slider("STRESS 금리쇼크(bp)", 0, 500, 
                                         scenario_defaults["stress_shock_bp"] if apply_scenario or st.session_state.get("scenario_applied") == scenario_type else 150, 
                                         25, key="stress_shock")
            
            st.markdown("**분석 기간**")
            lcr_h = st.slider("LCR Horizon(일)", 10, 60, 30, 5, key="lcr_h")
            stress_h = st.slider("Stress Horizon(일)", 30, 180, 90, 10, key="stress_h")
        
        st.markdown("---")
        
        # 현재 시나리오 상태 표시
        current_scenario = st.session_state.get("scenario_applied", "정상(Normal)")
        if current_scenario == "정상(Normal)":
            st.info(f"📊 현재 적용: **{current_scenario}**")
        elif current_scenario == "은행위기(Bank Crisis)":
            st.warning(f"🏦 현재 적용: **{current_scenario}**")
        elif current_scenario == "시장위기(Market Crisis)":
            st.warning(f"📉 현재 적용: **{current_scenario}**")
        else:
            st.error(f"🔥 현재 적용: **{current_scenario}**")
        
        st.info("💡 슬라이더를 조정하면 모든 분석에 즉시 반영됩니다.")
        
        # 세션에 저장
        st.session_state["_loan_prepay_rate"] = float(loan_prepay_rate)
        st.session_state["_loan_maturity_repay_rate"] = float(loan_maturity_repay_rate)
        st.session_state["_borrow_refinance_rate"] = float(borrow_refinance_rate)
        st.session_state["_credit_line_usage_rate"] = float(credit_line_usage_rate)
        st.session_state["_guarantee_usage_rate"] = float(guarantee_usage_rate)
        st.session_state["_core_deposit_ratio"] = float(core_deposit_ratio)
        st.session_state["_deposit_rollover_rate"] = float(deposit_rollover_rate)
        st.session_state["_deposit_early_withdraw_rate"] = float(deposit_early_withdraw_rate)
        st.session_state["_runoff_rate"] = float(runoff_rate)
        st.session_state["_early_term"] = float(early_termination)
        st.session_state["_current_scenario"] = scenario_type
        
        behavioral = {
            "loan_prepay_rate": float(loan_prepay_rate),
            "loan_maturity_repay_rate": float(loan_maturity_repay_rate),
            "borrow_refinance_rate": float(borrow_refinance_rate),
            "credit_line_usage_rate": float(credit_line_usage_rate),
            "guarantee_usage_rate": float(guarantee_usage_rate),
            "core_deposit_ratio": float(core_deposit_ratio),
            "deposit_rollover_rate": float(deposit_rollover_rate),
            "deposit_early_withdraw_rate": float(deposit_early_withdraw_rate),
            "runoff_rate": float(runoff_rate),
            "early_termination": float(early_termination),
        }

    # -----------------------------
    # A) 데이터 모드 (Excel 템플릿 기반)
    # -----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Excel 템플릿 파일 존재 확인
    excel_exists = os.path.exists(DEFAULT_EXCEL_PATH)
    
    if excel_exists:
        mode_options = ["📊 Excel 템플릿 사용 (ALM_input_template.xlsx)", "🔢 샘플 데이터 생성", "📁 파일 업로드"]
        default_idx = 0
    else:
        mode_options = ["🔢 샘플 데이터 생성", "📁 파일 업로드"]
        default_idx = 0
    
    mode = st.radio("데이터 입력 방식", mode_options, horizontal=True, index=default_idx)
    
    # Excel 파일 정보 표시
    if excel_exists and "Excel 템플릿" in mode:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ Excel 템플릿 파일 감지됨: `ALM_input_template.xlsx`")
        with col2:
            # Excel 파일에서 로드된 데이터 정보 표시
            try:
                temp_positions = cached_excel_positions(DEFAULT_EXCEL_PATH)
                asset_count = len(temp_positions[temp_positions["type"] == "asset"])
                liab_count = len(temp_positions[temp_positions["type"] == "liability"])
                hqla_count = len(temp_positions[temp_positions["type"] == "hqla"])
                st.caption(f"자산: {asset_count}개 | 부채: {liab_count}개 | HQLA: {hqla_count}개")
            except:
                pass
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 데이터 로드
    if "Excel 템플릿" in mode:
        # Excel 템플릿에서 데이터 로드
        positions = cached_excel_positions(DEFAULT_EXCEL_PATH)
        
        # Excel에서 Yield Curve 로드 (사이드바 값과 병합)
        excel_curve_x, excel_curve_y = cached_excel_yield_curve(DEFAULT_EXCEL_PATH, "BASE")
        
        # Excel에서 행동 파라미터 로드 (사이드바에서 덮어쓸 수 있음)
        excel_behavioral = cached_excel_behavioral_params(DEFAULT_EXCEL_PATH)
        
        # 사이드바 값이 기본값과 다르면 사이드바 값을 우선 사용 (이미 behavioral에 설정됨)
        # 그렇지 않으면 Excel 값 사용
        if not st.session_state.get("scenario_applied"):
            # 시나리오가 적용되지 않은 경우, Excel 값을 behavioral에 반영
            for key in excel_behavioral:
                if key in behavioral:
                    # 사이드바 기본값과 같으면 Excel 값 사용
                    default_vals = {
                        "loan_prepay_rate": 0.03,
                        "loan_maturity_repay_rate": 0.85,
                        "borrow_refinance_rate": 0.70,
                        "credit_line_usage_rate": 0.02,
                        "guarantee_usage_rate": 0.01,
                        "core_deposit_ratio": 0.60,
                        "deposit_rollover_rate": 0.75,
                        "deposit_early_withdraw_rate": 0.02,
                        "runoff_rate": 0.01,
                        "early_termination": 0.005,
                    }
                    if abs(behavioral[key] - default_vals.get(key, 0)) < 0.0001:
                        behavioral[key] = excel_behavioral[key]
        
        st.info(f"📊 **Excel 템플릿 데이터 로드 완료** - 포지션: {len(positions)}개 | "
                f"자산: {len(positions[positions['type']=='asset'])}개 | "
                f"부채: {len(positions[positions['type']=='liability'])}개 | "
                f"HQLA: {len(positions[positions['type']=='hqla'])}개")
        
    elif "샘플 데이터" in mode:
        # 기존 샘플 데이터 생성
        positions = cached_sample_positions(seed=7)
        st.info("🔢 **샘플 데이터 생성 완료** - 랜덤 시드 기반 가상 데이터입니다.")
        
    else:
        # 파일 업로드
        up = st.file_uploader("배치 결과 업로드", type=["csv", "xlsx"])
        if up is None:
            st.info("📁 파일을 업로드하면 자동으로 시각화/지표 계산이 수행됩니다.")
            return
        if up.name.lower().endswith(".csv"):
            positions = pd.read_csv(up)
        else:
            # 업로드된 Excel 파일 처리
            try:
                xl = pd.ExcelFile(up)
                if 'POSITIONS' in xl.sheet_names:
                    # ALM 템플릿 형식
                    positions = load_positions_from_excel(up)
                else:
                    # 일반 Excel 파일
                    positions = pd.read_excel(up)
            except:
                positions = pd.read_excel(up)

        required = {"type", "product", "balance"}
        miss = required - set(positions.columns)
        if miss:
            st.error(f"필수 컬럼 누락: {sorted(list(miss))}")
            return
        if "rate" not in positions.columns:
            positions["rate"] = 0.03
        if "spread" not in positions.columns:
            positions["spread"] = 0.005
        if "duration" not in positions.columns:
            positions["duration"] = 1.0
        if "maturity_bucket" not in positions.columns:
            positions["maturity_bucket"] = "1Y"
    
    # 평가일 설정
    start_date = pd.Timestamp("2026-01-01")
    end_date = pd.Timestamp("2026-12-31")
    valuation_date = pd.Timestamp("2026-03-15")
    
    # 상품 필터 (사이드바 미포함 - 전체 표시)
    prods = positions[positions["type"].isin(["asset", "liability"])]["product"].unique().tolist()
    positions_f = positions.copy()

    # -----------------------------
    # F) BASE vs STRESS 병렬 계산 (진행바 숨김)
    # -----------------------------
    # 진행바를 표시하지 않고 백그라운드에서 계산
    with st.spinner("🔄 BASE/STRESS 시나리오 계산 중..."):
        base_cf = build_cashflow_schedule_fast(
            positions_f, str(start_date.date()), str(end_date.date()),
            behavioral, rate_shock_bp=0.0, scenario="BASE"
        )

        stress_cf = build_cashflow_schedule_fast(
            positions_f, str(start_date.date()), str(end_date.date()),
            behavioral, rate_shock_bp=float(stress_shock_bp), scenario="STRESS"
        )

        base_k = compute_kpis_pro(
            positions_f, base_cf, valuation_date, curve_x, curve_y, int(lcr_h), int(stress_h)
        )

        stress_k = compute_kpis_pro(
            positions_f, stress_cf, valuation_date, curve_x, curve_y, int(lcr_h), int(stress_h)
        )
        
        # CF 결과를 cashflows_df로 참조 (CF 결과 분석 탭에서 사용)
        cashflows_df = base_cf.copy()

    delta = {k: float(stress_k.get(k, 0.0) - base_k.get(k, 0.0)) for k in stress_k.keys()}

    # -----------------------------
    # G) KPI 대시보드 (컴팩트 테이블 형태)
    # -----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # 현재 시나리오 표시
    current_scenario = st.session_state.get("_current_scenario", "정상(Normal)")
    scenario_colors = {
        "정상(Normal)": "#10b981",
        "은행위기(Bank Crisis)": "#f59e0b",
        "시장위기(Market Crisis)": "#f59e0b", 
        "결합위기(Combined Crisis)": "#ef4444"
    }
    scenario_icons = {
        "정상(Normal)": "📊",
        "은행위기(Bank Crisis)": "🏦",
        "시장위기(Market Crisis)": "📉",
        "결합위기(Combined Crisis)": "🔥"
    }
    scenario_color = scenario_colors.get(current_scenario, "#3b82f6")
    scenario_icon = scenario_icons.get(current_scenario, "📊")
    
    # Delta 포맷팅 함수
    def fmt_delta(value, fmt_fn, reverse=False):
        """Delta 값을 색상과 화살표로 포맷팅"""
        if math.isnan(value) or math.isinf(value):
            return '<span style="color:#94a3b8;">-</span>'
        is_positive = value >= 0
        if reverse:  # 값이 감소하면 좋은 경우 (예: 유출)
            is_positive = not is_positive
        color = "#10b981" if is_positive else "#ef4444"
        arrow = "▲" if value >= 0 else "▼"
        return f'<span style="color:{color};font-weight:600;">{arrow} {fmt_fn(abs(value))}</span>'
    
    # LCR 특별 처리
    def fmt_lcr_display(value):
        if math.isinf(value) or value > 10:
            return "∞ (안정)"
        return f"{value:.1%}"
    
    # Stress 생존 표시
    def fmt_survive(value):
        if value >= 0.5:
            return '<span style="color:#10b981;font-weight:700;">✓ 생존</span>'
        else:
            return '<span style="color:#ef4444;font-weight:700;">✗ 위험</span>'
    
    st.markdown(f"""
    <style>
    .kpi-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
    }}
    .kpi-table th {{
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 10px 12px;
        text-align: center;
        font-weight: 700;
        color: #475569;
        border-bottom: 2px solid #e2e8f0;
    }}
    .kpi-table th:first-child {{
        text-align: left;
        border-radius: 8px 0 0 0;
    }}
    .kpi-table th:last-child {{
        border-radius: 0 8px 0 0;
    }}
    .kpi-table td {{
        padding: 12px;
        text-align: center;
        border-bottom: 1px solid #f1f5f9;
    }}
    .kpi-table td:first-child {{
        text-align: left;
        font-weight: 600;
        color: #334155;
        background: #fafbfc;
    }}
    .kpi-table tr:last-child td {{
        border-bottom: none;
    }}
    .kpi-table tr:last-child td:first-child {{
        border-radius: 0 0 0 8px;
    }}
    .kpi-table .val-base {{
        font-weight: 700;
        color: #1e40af;
        font-size: 14px;
    }}
    .kpi-table .val-stress {{
        font-weight: 700;
        color: #9333ea;
        font-size: 14px;
    }}
    .scenario-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding: 12px 16px;
        background: linear-gradient(135deg, {scenario_color}15 0%, {scenario_color}08 100%);
        border: 1px solid {scenario_color}40;
        border-radius: 10px;
    }}
    </style>
    
    <div class="scenario-header">
        <span style="font-size: 15px; font-weight: 800; color: {scenario_color};">
            {scenario_icon} {current_scenario}
        </span>
        <span style="font-size: 12px; color: #64748b;">
            금리쇼크 +{stress_shock_bp}bp | 예금유출 {deposit_early_withdraw_rate*100:.1f}% | 핵심예금 {core_deposit_ratio*100:.0f}%
        </span>
    </div>
    
    <table class="kpi-table">
        <thead>
            <tr>
                <th style="width:22%;">지표</th>
                <th style="width:22%;">BASE</th>
                <th style="width:22%;">STRESS (+{stress_shock_bp}bp)</th>
                <th style="width:18%;">변화 (Δ)</th>
                <th style="width:16%;">상태</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>💰 HQLA</td>
                <td class="val-base">{fmt_조(base_k["HQLA"])}</td>
                <td class="val-stress">{fmt_조(stress_k["HQLA"])}</td>
                <td>{fmt_delta(delta["HQLA"], fmt_조)}</td>
                <td>{fmt_survive(1) if base_k["HQLA"] > 0 else fmt_survive(0)}</td>
            </tr>
            <tr>
                <td>📈 NII (순이자수익)</td>
                <td class="val-base">{fmt_조(base_k["NII_YTD"])}</td>
                <td class="val-stress">{fmt_조(stress_k["NII_YTD"])}</td>
                <td>{fmt_delta(delta["NII_YTD"], fmt_조)}</td>
                <td>{'<span style="color:#10b981;">●</span>' if base_k["NII_YTD"] > 0 else '<span style="color:#ef4444;">●</span>'}</td>
            </tr>
            <tr>
                <td>💎 NPV (순현재가치)</td>
                <td class="val-base">{fmt_조(base_k["NPV"])}</td>
                <td class="val-stress">{fmt_조(stress_k["NPV"])}</td>
                <td>{fmt_delta(delta["NPV"], fmt_조)}</td>
                <td>{'<span style="color:#10b981;">●</span>' if base_k["NPV"] >= 0 else '<span style="color:#f59e0b;">●</span>'}</td>
            </tr>
            <tr>
                <td>📊 DV01 (Net)</td>
                <td class="val-base">{base_k["DV01_NET"]/1e8:+.2f}억/bp</td>
                <td class="val-stress">{stress_k["DV01_NET"]/1e8:+.2f}억/bp</td>
                <td><span style="color:#64748b;font-size:11px;">자산 {base_k["DV01_ASSET"]/1e8:+.1f} / 부채 {base_k["DV01_LIAB"]/1e8:+.1f}</span></td>
                <td>{'<span style="color:#10b981;">●</span>' if abs(base_k["DV01_NET"]) < 1e10 else '<span style="color:#f59e0b;">●</span>'}</td>
            </tr>
            <tr>
                <td>🏦 LCR</td>
                <td class="val-base">{fmt_lcr_display(base_k["LCR"])}</td>
                <td class="val-stress">{fmt_lcr_display(stress_k["LCR"])}</td>
                <td><span style="color:#64748b;font-size:11px;">30일유출 {fmt_조(base_k["NetOutflow_30D"])}</span></td>
                <td>{fmt_survive(1) if base_k["LCR"] >= 1.0 else fmt_survive(0)}</td>
            </tr>
            <tr>
                <td>🛡️ Stress 생존</td>
                <td class="val-base">{fmt_survive(base_k["Stress_Survive"])}</td>
                <td class="val-stress">{fmt_survive(stress_k["Stress_Survive"])}</td>
                <td colspan="2" style="text-align:center;">
                    <span style="font-size:11px;color:#64748b;">
                        {stress_h}일 스트레스 테스트 기준
                    </span>
                </td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # G-2) Asset/Liability Composition 도넛차트
    # -----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    comp_col1, comp_col2 = st.columns(2)
    
    # 자산 구성 (Asset Composition)
    with comp_col1:
        assets_for_pie = positions_f[positions_f["type"] == "asset"].copy()
        
        # 상품 카테고리 분류
        def categorize_asset(product):
            product_lower = product.lower()
            if "국채" in product or "gov" in product_lower:
                return "Gov Bond"
            elif "카드" in product or "credit" in product_lower or "리볼빙" in product:
                return "Credit Card"
            elif "가계" in product or "household" in product_lower:
                return "Household Loan"
            elif "mortgage" in product_lower or "주택" in product:
                return "Mortgage"
            elif "기업" in product or "corporate" in product_lower:
                return "Corporate Loan"
            elif "채권" in product or "bond" in product_lower:
                return "Corporate Bond"
            else:
                return "Other"
        
        assets_for_pie["category"] = assets_for_pie["product"].apply(categorize_asset)
        asset_comp = assets_for_pie.groupby("category")["balance"].sum().reset_index()
        
        # 색상 매핑 (파란색 계열)
        asset_colors = {
            "Gov Bond": "#00bcd4",      # 청록색
            "Credit Card": "#4fc3f7",   # 밝은 파랑
            "Household Loan": "#2196f3", # 파랑
            "Mortgage": "#1565c0",      # 진한 파랑
            "Corporate Loan": "#0d47a1", # 아주 진한 파랑
            "Corporate Bond": "#64b5f6", # 연한 파랑
            "Other": "#90caf9"          # 매우 연한 파랑
        }
        
        fig_asset = go.Figure(data=[go.Pie(
            labels=asset_comp["category"],
            values=asset_comp["balance"],
            hole=0.6,
            marker=dict(colors=[asset_colors.get(cat, "#7fb6ff") for cat in asset_comp["category"]]),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>"
        )])
        
        fig_asset.update_layout(
            title=dict(text="ASSET COMPOSITION", font=dict(size=14, color="#073763", family="Arial Black")),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(t=40, b=80, l=20, r=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_asset, use_container_width=True)
    
    # 부채 구성 (Liability Composition)
    with comp_col2:
        liabs_for_pie = positions_f[positions_f["type"] == "liability"].copy()
        
        # 상품 카테고리 분류
        def categorize_liability(product):
            product_lower = product.lower()
            if "회사채" in product or "corporate bond" in product_lower:
                return "Corporate Bond"
            elif "차입" in product or "borrow" in product_lower:
                return "Borrowing"
            elif "저축" in product or "saving" in product_lower:
                return "Savings"
            elif "요구불" in product or "demand" in product_lower or "MMDA" in product:
                return "Demand Deposit"
            elif "정기" in product or "time" in product_lower or "예금" in product:
                return "Time Deposit"
            else:
                return "Other"
        
        liabs_for_pie["category"] = liabs_for_pie["product"].apply(categorize_liability)
        liab_comp = liabs_for_pie.groupby("category")["balance"].sum().reset_index()
        
        # 색상 매핑 (주황/갈색 계열)
        liab_colors = {
            "Corporate Bond": "#ff9800",   # 주황색
            "Borrowing": "#e65100",        # 진한 주황
            "Savings": "#8d6e63",          # 갈색
            "Demand Deposit": "#a1887f",   # 연한 갈색
            "Time Deposit": "#4e342e",     # 진한 갈색
            "Other": "#bcaaa4"             # 매우 연한 갈색
        }
        
        fig_liab = go.Figure(data=[go.Pie(
            labels=liab_comp["category"],
            values=liab_comp["balance"],
            hole=0.6,
            marker=dict(colors=[liab_colors.get(cat, "#ff9f1a") for cat in liab_comp["category"]]),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>"
        )])
        
        fig_liab.update_layout(
            title=dict(text="LIABILITY COMPOSITION", font=dict(size=14, color="#073763", family="Arial Black")),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(t=40, b=80, l=20, r=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_liab, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # G-3) Risk Analysis (Liquidity Gap & Interest Rate Repricing Gap)
    # -----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='h1' style='font-size:18px;'>RISK ANALYSIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub' style='font-size:13px;'>Detailed breakdown of liquidity and interest rate sensitivity across time buckets.</div>", unsafe_allow_html=True)
    
    risk_col1, risk_col2 = st.columns(2)
    
    # 만기 버킷별 자산/부채 집계
    bucket_labels = ["0-3M", "3-6M", "6-12M", "1-3Y", "3-5Y", ">5Y"]
    bucket_map = {"3M": "0-3M", "6M": "3-6M", "1Y": "6-12M", "2Y": "1-3Y", "3Y": "3-5Y", "5Y+": ">5Y"}
    
    assets_by_bucket = positions_f[positions_f["type"] == "asset"].copy()
    assets_by_bucket["bucket_label"] = assets_by_bucket["maturity_bucket"].map(bucket_map)
    asset_bucket_sum = assets_by_bucket.groupby("bucket_label")["balance"].sum()
    
    liabs_by_bucket = positions_f[positions_f["type"] == "liability"].copy()
    liabs_by_bucket["bucket_label"] = liabs_by_bucket["maturity_bucket"].map(bucket_map)
    liab_bucket_sum = liabs_by_bucket.groupby("bucket_label")["balance"].sum()
    
    # 버킷별 Gap 계산
    gap_data = []
    cumulative_gap = 0
    for bucket in bucket_labels:
        asset_val = asset_bucket_sum.get(bucket, 0) / 1e9  # 천원 단위로 변환 (k)
        liab_val = liab_bucket_sum.get(bucket, 0) / 1e9
        gap = asset_val - liab_val
        cumulative_gap += gap
        gap_data.append({
            "bucket": bucket,
            "asset": asset_val,
            "liability": liab_val,
            "gap": gap,
            "cumulative": cumulative_gap
        })
    
    gap_df = pd.DataFrame(gap_data)
    
    # Liquidity Gap Analysis (왼쪽)
    with risk_col1:
        fig_liq = go.Figure()
        
        # Gap 바 차트
        fig_liq.add_trace(go.Bar(
            x=gap_df["bucket"],
            y=gap_df["gap"],
            name="Gap",
            marker_color=["#3b82f6" if g >= 0 else "#3b82f6" for g in gap_df["gap"]],
            yaxis="y"
        ))
        
        # Cumulative 라인
        fig_liq.add_trace(go.Scatter(
            x=gap_df["bucket"],
            y=gap_df["cumulative"],
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#00bcd4", width=2),
            marker=dict(size=8, color="#00bcd4"),
            yaxis="y2"
        ))
        
        fig_liq.update_layout(
            title=dict(
                text="LIQUIDITY GAP ANALYSIS",
                font=dict(size=14, color="#073763", family="Arial Black")
            ),
            xaxis=dict(title="", tickfont=dict(size=10)),
            yaxis=dict(
                title="",
                tickformat=".0f",
                ticksuffix="k",
                side="left"
            ),
            yaxis2=dict(
                title="",
                tickformat=".0f",
                ticksuffix="k",
                overlaying="y",
                side="right"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            margin=dict(t=60, b=40, l=60, r=60),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.5)",
            bargap=0.3
        )
        
        # 격자 추가
        fig_liq.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
        fig_liq.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
        
        # 설명 텍스트
        fig_liq.add_annotation(
            text="Projected cash flow mismatch across maturity buckets.",
            xref="paper", yref="paper",
            x=0, y=1.12,
            showarrow=False,
            font=dict(size=10, color="rgba(7,55,99,0.6)")
        )
        
        st.plotly_chart(fig_liq, use_container_width=True)
    
    # Interest Rate Repricing Gap (오른쪽)
    with risk_col2:
        # rate_maturity 기반 재조정 갭 계산
        repricing_bucket_map = {"3M": "0-3M", "6M": "3-6M", "1Y": "6-12M", "2Y": "1-3Y", "3Y": "3-5Y"}
        
        # 자산 금리 재조정
        assets_repricing = positions_f[positions_f["type"] == "asset"].copy()
        if "rate_maturity" in assets_repricing.columns:
            assets_repricing["repricing_bucket"] = assets_repricing["rate_maturity"].map(repricing_bucket_map)
        else:
            assets_repricing["repricing_bucket"] = assets_repricing["maturity_bucket"].map(bucket_map)
        asset_repricing_sum = assets_repricing.groupby("repricing_bucket")["balance"].sum()
        
        # 부채 금리 재조정
        liabs_repricing = positions_f[positions_f["type"] == "liability"].copy()
        if "rate_maturity" in liabs_repricing.columns:
            liabs_repricing["repricing_bucket"] = liabs_repricing["rate_maturity"].map(repricing_bucket_map)
        else:
            liabs_repricing["repricing_bucket"] = liabs_repricing["maturity_bucket"].map(bucket_map)
        liab_repricing_sum = liabs_repricing.groupby("repricing_bucket")["balance"].sum()
        
        # 재조정 갭 데이터
        repricing_data = []
        cumulative_repricing = 0
        for bucket in bucket_labels:
            asset_val = asset_repricing_sum.get(bucket, 0) / 1e9
            liab_val = liab_repricing_sum.get(bucket, 0) / 1e9
            gap = asset_val - liab_val
            cumulative_repricing += gap
            repricing_data.append({
                "bucket": bucket,
                "asset": asset_val,
                "liability": liab_val,
                "gap": gap,
                "cumulative": cumulative_repricing
            })
        
        repricing_df = pd.DataFrame(repricing_data)
        
        fig_rate = go.Figure()
        
        # Gap 바 차트
        fig_rate.add_trace(go.Bar(
            x=repricing_df["bucket"],
            y=repricing_df["gap"],
            name="Gap",
            marker_color=["#3b82f6" if g >= 0 else "#3b82f6" for g in repricing_df["gap"]],
            yaxis="y"
        ))
        
        # Cumulative 라인
        fig_rate.add_trace(go.Scatter(
            x=repricing_df["bucket"],
            y=repricing_df["cumulative"],
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#00bcd4", width=2),
            marker=dict(size=8, color="#00bcd4"),
            yaxis="y2"
        ))
        
        fig_rate.update_layout(
            title=dict(
                text="INTEREST RATE REPRICING GAP",
                font=dict(size=14, color="#073763", family="Arial Black")
            ),
            xaxis=dict(title="", tickfont=dict(size=10)),
            yaxis=dict(
                title="",
                tickformat=".0f",
                ticksuffix="k",
                side="left"
            ),
            yaxis2=dict(
                title="",
                tickformat=".0f",
                ticksuffix="k",
                overlaying="y",
                side="right"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            margin=dict(t=60, b=40, l=60, r=60),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.5)",
            bargap=0.3
        )
        
        # 격자 추가
        fig_rate.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
        fig_rate.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
        
        # 설명 텍스트
        fig_rate.add_annotation(
            text="Asset/Liability mismatch based on repricing periods.",
            xref="paper", yref="paper",
            x=0, y=1.12,
            showarrow=False,
            font=dict(size=10, color="rgba(7,55,99,0.6)")
        )
        
        st.plotly_chart(fig_rate, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # H) 🆕 고급 시뮬레이션 탭
    # -----------------------------
    
    # total_days 계산 (애니메이션은 60일로 제한)
    total_days_full = (end_date - start_date).days
    total_days = min(60, total_days_full)  # 애니메이션용 60일 제한
    
    sim_tabs = st.tabs([
        "📋 CF 결과 분석",
        "🎬 ALM Flow Animation",
        "📊 데이터 분석",
        "📈 Cashflow Timeline",
        "🌊 Sankey(자금흐름)",
        "🎯 금리 시나리오 분석",
        "💰 행동비율 과부족 분석",
        "🔬 민감도 분석",
        "⚡ 최적화 시뮬레이션"
    ])

    # 탭 0: CF 결과 분석 (새로 추가)
    with sim_tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📋 Cashflow 결과 종합 분석")
        
        if cashflows_df.empty:
            st.warning("⚠️ CF 데이터가 없습니다. 시뮬레이션을 실행해주세요.")
        else:
            # CF 서브탭
            cf_result_tabs = st.tabs([
                "📊 집계 CF", 
                "📈 일별 추이", 
                "🏢 상품별 분석", 
                "📅 기간별 분석",
                "💾 데이터 다운로드"
            ])
            
            # 서브탭 1: 집계 CF
            with cf_result_tabs[0]:
                st.markdown("#### 📊 일별 집계 Cashflow")
                
                # 자산/부채 집계
                agg_cf = cashflows_df.groupby(['date', 'type']).agg({
                    'cashflow': 'sum',
                    'interest': 'sum',
                    'principal': 'sum'
                }).reset_index()
                
                # Pivot
                cf_pivot = agg_cf.pivot(index='date', columns='type', values='cashflow').fillna(0)
                cf_pivot['gap'] = cf_pivot.get('asset', 0) - abs(cf_pivot.get('liability', 0))
                cf_pivot['cumulative_gap'] = cf_pivot['gap'].cumsum()
                cf_pivot = cf_pivot.reset_index()
                
                # 요약 통계
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_asset_cf = cf_pivot['asset'].sum() if 'asset' in cf_pivot else 0
                    st.metric("총 자산 CF", f"{total_asset_cf/1e9:,.1f}조")
                with col2:
                    total_liab_cf = cf_pivot['liability'].sum() if 'liability' in cf_pivot else 0
                    st.metric("총 부채 CF", f"{total_liab_cf/1e9:,.1f}조")
                with col3:
                    total_gap = cf_pivot['gap'].sum()
                    st.metric("총 GAP", f"{total_gap/1e9:,.1f}조", 
                             delta=f"{total_gap/1e9:,.1f}조")
                with col4:
                    final_cum_gap = cf_pivot['cumulative_gap'].iloc[-1]
                    st.metric("최종 누적 GAP", f"{final_cum_gap/1e9:,.1f}조")
                
                st.markdown("---")
                
                # 데이터 테이블
                st.markdown("**일별 집계 데이터**")
                display_cf = cf_pivot.copy()
                display_cf['date'] = pd.to_datetime(display_cf['date']).dt.strftime('%Y-%m-%d')
                if 'asset' in display_cf:
                    display_cf['asset_조'] = (display_cf['asset'] / 1e12).round(2)
                if 'liability' in display_cf:
                    display_cf['liability_조'] = (display_cf['liability'] / 1e12).round(2)
                display_cf['gap_조'] = (display_cf['gap'] / 1e12).round(2)
                display_cf['cum_gap_조'] = (display_cf['cumulative_gap'] / 1e12).round(2)
                
                show_cols = ['date', 'asset_조', 'liability_조', 'gap_조', 'cum_gap_조']
                show_cols = [c for c in show_cols if c in display_cf.columns]
                st.dataframe(display_cf[show_cols], use_container_width=True, height=400)
            
            # 서브탭 2: 일별 추이
            with cf_result_tabs[1]:
                st.markdown("#### 📈 일별 Cashflow 추이")
                
                # 그래프
                fig = go.Figure()
                
                if 'asset' in cf_pivot.columns:
                    fig.add_trace(go.Bar(
                        x=cf_pivot['date'],
                        y=cf_pivot['asset'] / 1e12,
                        name='자산 CF',
                        marker_color='#19c37d'
                    ))
                
                if 'liability' in cf_pivot.columns:
                    fig.add_trace(go.Bar(
                        x=cf_pivot['date'],
                        y=cf_pivot['liability'] / 1e12,
                        name='부채 CF',
                        marker_color='#ff9f1a'
                    ))
                
                fig.add_trace(go.Scatter(
                    x=cf_pivot['date'],
                    y=cf_pivot['gap'] / 1e12,
                    name='GAP',
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=3),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='일별 Cashflow 및 GAP',
                    xaxis_title='날짜',
                    yaxis_title='CF (조)',
                    yaxis2=dict(
                        title='GAP (조)',
                        overlaying='y',
                        side='right'
                    ),
                    barmode='relative',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 누적 GAP 차트
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=cf_pivot['date'],
                    y=cf_pivot['cumulative_gap'] / 1e12,
                    name='누적 GAP',
                    fill='tozeroy',
                    line=dict(color='#8b5cf6', width=2)
                ))
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                fig2.update_layout(
                    title='누적 GAP 추이',
                    xaxis_title='날짜',
                    yaxis_title='누적 GAP (조)',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # 서브탭 3: 상품별 분석
            with cf_result_tabs[2]:
                st.markdown("#### 🏢 상품별 Cashflow 분석")
                
                # 상품별 집계
                prod_cf = cashflows_df.groupby(['product', 'type']).agg({
                    'cashflow': 'sum',
                    'interest': 'sum',
                    'principal': 'sum',
                    'balance0': 'sum'
                }).reset_index()
                
                # 필터
                col1, col2 = st.columns(2)
                with col1:
                    type_filter = st.selectbox("유형 선택", ["전체", "asset", "liability"], key="cf_type_filter")
                with col2:
                    sort_by = st.selectbox("정렬 기준", ["총 CF", "이자", "원금", "잔액"], key="cf_sort")
                
                # 필터링
                filtered_prod = prod_cf.copy()
                if type_filter != "전체":
                    filtered_prod = filtered_prod[filtered_prod['type'] == type_filter]
                
                # 정렬
                sort_map = {"총 CF": "cashflow", "이자": "interest", "원금": "principal", "잔액": "balance0"}
                filtered_prod = filtered_prod.sort_values(sort_map[sort_by], ascending=False)
                
                # 포맷팅
                filtered_prod['총CF(조)'] = (filtered_prod['cashflow'] / 1e12).round(2)
                filtered_prod['이자(조)'] = (filtered_prod['interest'] / 1e12).round(2)
                filtered_prod['원금(조)'] = (filtered_prod['principal'] / 1e12).round(2)
                filtered_prod['잔액(조)'] = (filtered_prod['balance0'] / 1e12).round(2)
                
                # 테이블
                display_cols = ['product', 'type', '총CF(조)', '이자(조)', '원금(조)', '잔액(조)']
                st.dataframe(filtered_prod[display_cols], use_container_width=True, height=400)
                
                # 차트
                fig = go.Figure()
                
                top_10 = filtered_prod.head(10)
                fig.add_trace(go.Bar(
                    x=top_10['product'],
                    y=top_10['총CF(조)'],
                    text=top_10['총CF(조)'].round(1),
                    textposition='auto',
                    marker_color=['#19c37d' if t == 'asset' else '#ff9f1a' 
                                 for t in top_10['type']]
                ))
                
                fig.update_layout(
                    title=f'상품별 총 CF Top 10 ({type_filter})',
                    xaxis_title='상품',
                    yaxis_title='총 CF (조)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 서브탭 4: 기간별 분석
            with cf_result_tabs[3]:
                st.markdown("#### 📅 기간별 Cashflow 분석")
                
                # 기간 선택
                col1, col2 = st.columns(2)
                with col1:
                    period_type = st.selectbox("집계 기간", ["주별", "월별", "분기별"], key="period_type")
                with col2:
                    metric_type = st.selectbox("측정 지표", ["총 CF", "이자", "원금", "GAP"], key="metric_type")
                
                # 기간별 집계
                period_map = {"주별": 'W', "월별": 'M', "분기별": 'Q'}
                freq = period_map[period_type]
                
                period_cf = cashflows_df.copy()
                period_cf['date'] = pd.to_datetime(period_cf['date'])
                period_cf['period'] = period_cf['date'].dt.to_period(freq)
                
                period_agg = period_cf.groupby(['period', 'type']).agg({
                    'cashflow': 'sum',
                    'interest': 'sum',
                    'principal': 'sum'
                }).reset_index()
                
                # Pivot
                metric_map = {"총 CF": "cashflow", "이자": "interest", "원금": "principal", "GAP": "cashflow"}
                metric_col = metric_map[metric_type]
                
                period_pivot = period_agg.pivot(index='period', columns='type', values=metric_col).fillna(0)
                
                if metric_type == "GAP":
                    period_pivot['value'] = period_pivot.get('asset', 0) - abs(period_pivot.get('liability', 0))
                    
                period_pivot = period_pivot.reset_index()
                period_pivot['period_str'] = period_pivot['period'].astype(str)
                
                # 차트
                fig = go.Figure()
                
                if metric_type == "GAP":
                    fig.add_trace(go.Bar(
                        x=period_pivot['period_str'],
                        y=period_pivot['value'] / 1e12,
                        name='GAP',
                        marker_color='#3b82f6'
                    ))
                else:
                    if 'asset' in period_pivot.columns:
                        fig.add_trace(go.Bar(
                            x=period_pivot['period_str'],
                            y=period_pivot['asset'] / 1e12,
                            name='자산',
                            marker_color='#19c37d'
                        ))
                    if 'liability' in period_pivot.columns:
                        fig.add_trace(go.Bar(
                            x=period_pivot['period_str'],
                            y=period_pivot['liability'] / 1e12,
                            name='부채',
                            marker_color='#ff9f1a'
                        ))
                
                fig.update_layout(
                    title=f'{period_type} {metric_type} 추이',
                    xaxis_title='기간',
                    yaxis_title=f'{metric_type} (조)',
                    barmode='group',
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계 테이블
                st.markdown("**기간별 통계**")
                if metric_type == "GAP":
                    stats_df = pd.DataFrame({
                        '기간': period_pivot['period_str'],
                        'GAP(조)': (period_pivot['value'] / 1e12).round(2)
                    })
                else:
                    stats_df = period_pivot.copy()
                    stats_df['기간'] = stats_df['period_str']
                    if 'asset' in stats_df.columns:
                        stats_df['자산(조)'] = (stats_df['asset'] / 1e12).round(2)
                    if 'liability' in stats_df.columns:
                        stats_df['부채(조)'] = (stats_df['liability'] / 1e12).round(2)
                    
                    display_cols = ['기간']
                    if '자산(조)' in stats_df.columns:
                        display_cols.append('자산(조)')
                    if '부채(조)' in stats_df.columns:
                        display_cols.append('부채(조)')
                    stats_df = stats_df[display_cols]
                
                st.dataframe(stats_df, use_container_width=True)
            
            # 서브탭 5: 데이터 다운로드
            with cf_result_tabs[4]:
                st.markdown("#### 💾 Cashflow 데이터 다운로드")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**상세 CF 데이터**")
                    st.info("모든 계약의 일별 상세 CF 데이터를 다운로드합니다.")
                    
                    csv_detail = cashflows_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 상세 CF 다운로드 (CSV)",
                        data=csv_detail,
                        file_name=f"cashflow_detail_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("**집계 CF 데이터**")
                    st.info("일별 집계된 CF 데이터를 다운로드합니다.")
                    
                    csv_agg = cf_pivot.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 집계 CF 다운로드 (CSV)",
                        data=csv_agg,
                        file_name=f"cashflow_aggregated_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.markdown("---")
                
                # 엑셀 다운로드 (선택사항)
                st.markdown("**📊 Excel 형식 다운로드 (상세 + 집계)**")
                
                try:
                    from io import BytesIO
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        cashflows_df.to_excel(writer, sheet_name='상세CF', index=False)
                        cf_pivot.to_excel(writer, sheet_name='집계CF', index=False)
                        if not prod_cf.empty:
                            prod_cf.to_excel(writer, sheet_name='상품별CF', index=False)
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 전체 데이터 다운로드 (Excel)",
                        data=excel_data,
                        file_name=f"cashflow_full_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except ImportError:
                    st.warning("Excel 다운로드를 위해서는 openpyxl 패키지가 필요합니다.")
                
                st.markdown("---")
                st.markdown("**데이터 미리보기**")
                st.dataframe(cashflows_df.head(100), use_container_width=True, height=300)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # 탭 1: ALM Flow Animation (일자별 시뮬레이션)
    with sim_tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎬 ALM Flow Animation - 일자별 시뮬레이션")
        st.markdown("외곽선 흐름 속도는 만기에 비례하여 길수록 느리게, 짧을수록 빠르게 동작합니다.")
        
        # 애니메이션 컨트롤
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        # 세션 상태 초기화
        if "anim_running" not in st.session_state:
            st.session_state["anim_running"] = False
        if "anim_day" not in st.session_state:
            st.session_state["anim_day"] = 0
        if "anim_fps" not in st.session_state:
            st.session_state["anim_fps"] = 5
        if "base_seconds_per_cycle" not in st.session_state:
            st.session_state["base_seconds_per_cycle"] = 12.0
        
        with col1:
            play_btn = st.button("▶ Play", type="primary", key="play_anim")
        with col2:
            pause_btn = st.button("⏸ Pause", key="pause_anim")
        with col3:
            step_btn = st.button("⏭ +1 Day", key="step_anim")
        with col4:
            reset_btn = st.button("🔄 Reset", key="reset_anim")
        
        col5, col6 = st.columns([1, 1])
        with col5:
            anim_fps = st.slider("재생 속도 (FPS)", 1, 15, st.session_state["anim_fps"], key="fps_slider")
            st.session_state["anim_fps"] = anim_fps
        with col6:
            base_seconds = st.slider("회전 시간 (초/사이클)", 4.0, 30.0, st.session_state["base_seconds_per_cycle"], 0.5, key="cycle_slider")
            st.session_state["base_seconds_per_cycle"] = base_seconds
        
        # 버튼 로직
        if play_btn:
            st.session_state["anim_running"] = True
        if pause_btn:
            st.session_state["anim_running"] = False
        if reset_btn:
            st.session_state["anim_running"] = False
            st.session_state["anim_day"] = 0
        if step_btn:
            st.session_state["anim_running"] = False
            st.session_state["anim_day"] = min(st.session_state["anim_day"] + 1, total_days)
        
        # 현재 day
        current_day = st.session_state["anim_day"]
        
        # 자동 진행
        if st.session_state["anim_running"]:
            current_day = min(current_day + 1, total_days)
            st.session_state["anim_day"] = current_day
        
        # 일자별 cashflow 데이터 생성 (간단한 시뮬레이션)
        from datetime import datetime, timedelta
        
        def generate_daily_cf(seed: int, days: int):
            rng = np.random.default_rng(seed + 101)
            start = datetime.today().date()
            dates = [start + timedelta(days=i) for i in range(days + 1)]
            
            # 조 단위로 수정: 일 평균 0.2조 정도의 CF
            base_a = rng.normal(loc=0.20, scale=0.06, size=len(dates))
            base_l = rng.normal(loc=0.17, scale=0.055, size=len(dates))
            
            shock_center = min(30, days)
            shock = np.exp(-0.5 * ((np.arange(len(dates)) - shock_center) / 6.0) ** 2)
            asset_cf = (base_a + 0.35 * shock) * 1e12  # 조 단위
            liab_cf = (base_l + 0.30 * shock) * 1e12  # 조 단위
            
            return pd.DataFrame({
                "date": pd.to_datetime(dates),
                "asset_cf": asset_cf,
                "liability_cf": -liab_cf,
                "gap_cf": asset_cf - liab_cf
            })
        
        daily_cf = generate_daily_cf(7, total_days)
        
        # 현재까지의 누적 계산
        cf_to_date = daily_cf.iloc[:current_day+1] if current_day < len(daily_cf) else daily_cf
        cum_gap = float(cf_to_date["gap_cf"].sum())
        
        # HQLA 계산
        hqla_balance = float(positions_f[positions_f["type"] == "hqla"]["balance"].sum())
        
        # 초기 현금 (HQLA 중 현금)
        cash_rows = positions_f[positions_f["type"] == "hqla"]
        cash0 = float(cash_rows[cash_rows["product"].str.contains("현금", na=False)]["balance"].sum())
        
        # 현재 시점 현금 = 초기 현금 + 누적 GAP
        cash_t = cash0 + cum_gap
        
        # ==========================================
        # LCR 예측 기초자료 DATA 기반 LCR 계산
        # ==========================================
        # LCR 예측 탭의 샘플 데이터와 동일한 기초자료 사용
        lcr_base_data = {
            "고유동성자산": [80, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
                          100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 
                          116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 
                          132, 133, 134, 135, 136, 137, 138, 138],
            "현금유출": [110, 110, 111.5, 113, 114.5, 116, 117.5, 119, 120.5, 122, 123.5, 125, 126.5, 128, 129.5, 
                      131, 132.5, 134, 135.5, 137, 138.5, 140, 141.5, 143, 144.5, 146, 147.5, 149, 150.5, 
                      152, 153.5, 155, 156.5, 158, 159.5, 161, 162.5, 164, 165.5, 167, 168.5, 170, 171.5, 
                      173, 174.5, 176, 177.5, 179, 180.5, 182, 183.5, 185, 186.5, 188, 189.5, 191, 192.5, 
                      194, 195.5, 197, 198.5],
            "현금유입": [30, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
                      50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
                      70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
        }
        
        # 현재 day에 해당하는 LCR 계산 (기초자료 DATA 기반)
        day_idx = min(current_day, len(lcr_base_data["고유동성자산"]) - 1)
        lcr_hqla = lcr_base_data["고유동성자산"][day_idx]
        lcr_outflow = lcr_base_data["현금유출"][day_idx]
        lcr_inflow = lcr_base_data["현금유입"][day_idx]
        lcr_net_outflow = lcr_outflow - lcr_inflow
        
        # LCR = 고유동성자산 / (현금유출 - 현금유입) * 100%
        lcr = (lcr_hqla / lcr_net_outflow) * 100 if lcr_net_outflow > 0 else 999.99
        lcr = min(max(lcr, 0.0), 999.99)
        
        # 현재 HQLA (기초자료 기반, 조 단위)
        current_hqla = lcr_hqla * 1e12
        
        # NII 계산 (간소화: 누적 자산CF의 일부를 이자로 가정)
        nii_ytd = float(cf_to_date["asset_cf"].sum() * 0.03)  # 3% 가정
        
        # ==========================================
        # 🆕 실시간 KPI 대시보드 (화면 상단)
        # ==========================================
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 16px; margin-bottom: 20px; color: white;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 12px; backdrop-filter: blur(10px);">
                    <div style="font-size: 13px; opacity: 0.9; margin-bottom: 5px;">📊 NII (누적)</div>
                    <div style="font-size: 24px; font-weight: 800;">{:.2f}조</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 12px; backdrop-filter: blur(10px);">
                    <div style="font-size: 13px; opacity: 0.9; margin-bottom: 5px;">💧 LCR</div>
                    <div style="font-size: 24px; font-weight: 800;">{:.1f}%</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 12px; backdrop-filter: blur(10px);">
                    <div style="font-size: 13px; opacity: 0.9; margin-bottom: 5px;">🏦 HQLA</div>
                    <div style="font-size: 24px; font-weight: 800;">{:.2f}조</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 12px; backdrop-filter: blur(10px);">
                    <div style="font-size: 13px; opacity: 0.9; margin-bottom: 5px;">💰 누적 GAP</div>
                    <div style="font-size: 24px; font-weight: 800;">{:.2f}조</div>
                </div>
            </div>
        </div>
        """.format(nii_ytd/1e12, lcr, current_hqla/1e12, cum_gap/1e12), unsafe_allow_html=True)
        
        # SVG 애니메이션 생성
        svg_anim = build_svg_animation(
            positions_f, 
            daily_cf, 
            current_day, 
            total_days, 
            st.session_state["base_seconds_per_cycle"]
        )
        
        # 진행 상태 표시
        progress_pct = int((current_day / max(1, total_days)) * 100)
        st.progress(progress_pct / 100)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: rgba(127,182,255,0.1); border-radius: 10px; margin: 10px 0;">
            <span style="font-weight: 800; color: #073763;">Day {current_day} / {total_days} ({progress_pct}%)</span>
            <span style="margin-left: 20px;">상태: <b>{"▶ Running" if st.session_state["anim_running"] else "⏸ Paused"}</b></span>
        </div>
        """, unsafe_allow_html=True)
        
        # SVG 렌더링 (캔버스 높이 1750에 맞춤)
        st.components.v1.html(svg_anim, height=1900, scrolling=True)
        
        # 자동 재생 로직
        if st.session_state["anim_running"]:
            if current_day >= total_days:
                st.session_state["anim_running"] = False
            else:
                time.sleep(1.0 / max(1, anim_fps))
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 탭 3: 데이터 분석
    with sim_tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 분석 데이터 상세")
        
        # 서브탭
        data_subtabs = st.tabs(["자산 상품", "부채 상품", "HQLA", "전체 요약"])
        
        with data_subtabs[0]:
            st.markdown("#### 💰 자산 상품 분석")
            assets_df = positions_f[positions_f["type"] == "asset"].copy()
            
            if not assets_df.empty:
                # 요약 통계
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("자산 상품 수", len(assets_df))
                with col2:
                    st.metric("총 잔액", f"{assets_df['balance'].sum()/1e9:,.1f} 조")
                with col3:
                    avg_duration = assets_df['duration'].mean()
                    st.metric("평균 듀레이션", f"{avg_duration:.2f} 년")
                with col4:
                    if 'rate' in assets_df.columns:
                        avg_rate = assets_df['rate'].mean()
                        st.metric("평균 금리", f"{avg_rate*100:.2f}%")
                
                # 상세 데이터
                st.markdown("**상세 데이터**")
                display_cols = ['product', 'maturity_bucket', 'balance', 'duration']
                if 'rate' in assets_df.columns:
                    display_cols.append('rate')
                if 'spread' in assets_df.columns:
                    display_cols.append('spread')
                
                display_df = assets_df[display_cols].copy()
                display_df['balance'] = display_df['balance'].apply(lambda x: f"{x/1e9:,.2f} 조")
                if 'rate' in display_df.columns:
                    display_df['rate'] = display_df['rate'].apply(lambda x: f"{x*100:.2f}%")
                if 'spread' in display_df.columns:
                    display_df['spread'] = display_df['spread'].apply(lambda x: f"{x*100:.2f}%")
                
                st.dataframe(display_df, use_container_width=True)
                
                # 시각화
                col1, col2 = st.columns(2)
                with col1:
                    # 만기별 잔액 분포
                    bucket_summary = assets_df.groupby('maturity_bucket')['balance'].sum().reset_index()
                    fig = go.Figure(data=[
                        go.Bar(x=bucket_summary['maturity_bucket'], 
                               y=bucket_summary['balance']/1e9,
                               marker_color='#7fb6ff')
                    ])
                    fig.update_layout(
                        title="자산 만기별 잔액 분포",
                        xaxis_title="만기 버킷",
                        yaxis_title="잔액 (조)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 상품별 비중
                    product_summary = assets_df.groupby('product')['balance'].sum()
                    fig = go.Figure(data=[
                        go.Pie(labels=product_summary.index, 
                               values=product_summary.values,
                               hole=0.4)
                    ])
                    fig.update_layout(title="자산 상품별 비중", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("자산 데이터가 없습니다.")
        
        with data_subtabs[1]:
            st.markdown("#### 💳 부채 상품 분석")
            liabs_df = positions_f[positions_f["type"] == "liability"].copy()
            
            if not liabs_df.empty:
                # 요약 통계
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("부채 상품 수", len(liabs_df))
                with col2:
                    st.metric("총 잔액", f"{liabs_df['balance'].sum()/1e9:,.1f} 조")
                with col3:
                    avg_duration = liabs_df['duration'].mean()
                    st.metric("평균 듀레이션", f"{avg_duration:.2f} 년")
                with col4:
                    if 'rate' in liabs_df.columns:
                        avg_rate = liabs_df['rate'].mean()
                        st.metric("평균 금리", f"{avg_rate*100:.2f}%")
                
                # 상세 데이터
                st.markdown("**상세 데이터**")
                display_cols = ['product', 'maturity_bucket', 'balance', 'duration']
                if 'rate' in liabs_df.columns:
                    display_cols.append('rate')
                if 'spread' in liabs_df.columns:
                    display_cols.append('spread')
                
                display_df = liabs_df[display_cols].copy()
                display_df['balance'] = display_df['balance'].apply(lambda x: f"{x/1e9:,.2f} 조")
                if 'rate' in display_df.columns:
                    display_df['rate'] = display_df['rate'].apply(lambda x: f"{x*100:.2f}%")
                if 'spread' in display_df.columns:
                    display_df['spread'] = display_df['spread'].apply(lambda x: f"{x*100:.2f}%")
                
                st.dataframe(display_df, use_container_width=True)
                
                # 시각화
                col1, col2 = st.columns(2)
                with col1:
                    # 만기별 잔액 분포
                    bucket_summary = liabs_df.groupby('maturity_bucket')['balance'].sum().reset_index()
                    fig = go.Figure(data=[
                        go.Bar(x=bucket_summary['maturity_bucket'], 
                               y=bucket_summary['balance']/1e9,
                               marker_color='#c9ced6')
                    ])
                    fig.update_layout(
                        title="부채 만기별 잔액 분포",
                        xaxis_title="만기 버킷",
                        yaxis_title="잔액 (조)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 상품별 비중
                    product_summary = liabs_df.groupby('product')['balance'].sum()
                    fig = go.Figure(data=[
                        go.Pie(labels=product_summary.index, 
                               values=product_summary.values,
                               hole=0.4)
                    ])
                    fig.update_layout(title="부채 상품별 비중", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("부채 데이터가 없습니다.")
        
        with data_subtabs[2]:
            st.markdown("#### 🏦 HQLA (고유동성자산) 분석")
            hqla_df = positions_f[positions_f["type"] == "hqla"].copy()
            
            if not hqla_df.empty:
                # 요약 통계
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("HQLA 항목 수", len(hqla_df))
                with col2:
                    st.metric("총 HQLA", f"{hqla_df['balance'].sum()/1e9:,.1f} 조")
                with col3:
                    lcr_ratio = base_k.get("LCR", 0)
                    st.metric("LCR 비율", f"{lcr_ratio:.2f}")
                
                # 상세 데이터
                st.markdown("**상세 데이터**")
                display_df = hqla_df[['product', 'balance']].copy()
                display_df['balance'] = display_df['balance'].apply(lambda x: f"{x/1e9:,.2f} 조")
                display_df['비중(%)'] = (hqla_df['balance'] / hqla_df['balance'].sum() * 100).apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True)
                
                # 시각화
                fig = go.Figure(data=[
                    go.Bar(x=hqla_df['product'], 
                           y=hqla_df['balance']/1e9,
                           marker_color='#19c37d')
                ])
                fig.update_layout(
                    title="HQLA 항목별 잔액",
                    xaxis_title="항목",
                    yaxis_title="잔액 (조)",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("HQLA 데이터가 없습니다.")
        
        with data_subtabs[3]:
            st.markdown("#### 📈 전체 포트폴리오 요약")
            
            # 전체 통계
            total_assets = float(positions_f[positions_f["type"] == "asset"]["balance"].sum())
            total_liabs = float(positions_f[positions_f["type"] == "liability"]["balance"].sum())
            total_hqla = float(positions_f[positions_f["type"] == "hqla"]["balance"].sum())
            net_position = total_assets - total_liabs
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("총 자산", f"{total_assets/1e9:,.1f} 조")
            with col2:
                st.metric("총 부채", f"{total_liabs/1e9:,.1f} 조")
            with col3:
                st.metric("순 포지션", f"{net_position/1e9:,.1f} 조")
            with col4:
                st.metric("HQLA", f"{total_hqla/1e9:,.1f} 조")
            with col5:
                leverage = (total_assets / max(net_position, 1)) if net_position > 0 else 0
                st.metric("레버리지", f"{leverage:.2f}x")
            
            # 만기 구조 비교
            st.markdown("**자산-부채 만기 구조 비교**")
            
            assets_by_bucket = positions_f[positions_f["type"] == "asset"].groupby('maturity_bucket')['balance'].sum()
            liabs_by_bucket = positions_f[positions_f["type"] == "liability"].groupby('maturity_bucket')['balance'].sum()
            
            all_buckets = sorted(set(list(assets_by_bucket.index) + list(liabs_by_bucket.index)), 
                                key=lambda x: BUCKET_ORDER.index(x) if x in BUCKET_ORDER else 999)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='자산',
                x=all_buckets,
                y=[assets_by_bucket.get(b, 0)/1e9 for b in all_buckets],
                marker_color='#7fb6ff'
            ))
            fig.add_trace(go.Bar(
                name='부채',
                x=all_buckets,
                y=[liabs_by_bucket.get(b, 0)/1e9 for b in all_buckets],
                marker_color='#c9ced6'
            ))
            fig.update_layout(
                title="만기 버킷별 자산-부채 비교",
                xaxis_title="만기 버킷",
                yaxis_title="잔액 (조)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 듀레이션 GAP 분석
            st.markdown("**듀레이션 GAP 분석**")
            
            assets_dur = positions_f[positions_f["type"] == "asset"]
            liabs_dur = positions_f[positions_f["type"] == "liability"]
            
            if not assets_dur.empty and not liabs_dur.empty:
                # 가중평균 듀레이션
                asset_weighted_dur = (assets_dur['duration'] * assets_dur['balance']).sum() / assets_dur['balance'].sum()
                liab_weighted_dur = (liabs_dur['duration'] * liabs_dur['balance']).sum() / liabs_dur['balance'].sum()
                duration_gap = asset_weighted_dur - liab_weighted_dur
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("자산 가중평균 듀레이션", f"{asset_weighted_dur:.2f} 년")
                with col2:
                    st.metric("부채 가중평균 듀레이션", f"{liab_weighted_dur:.2f} 년")
                with col3:
                    st.metric("듀레이션 GAP", f"{duration_gap:.2f} 년", 
                             delta=f"{'양(+)의 GAP' if duration_gap > 0 else '음(-)의 GAP'}")
            
            # 데이터 다운로드
            st.markdown("**데이터 다운로드**")
            
            # CSV 생성
            csv = positions_f.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 전체 데이터 CSV 다운로드",
                data=csv,
                file_name=f"alm_positions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

    # 탭 3: Cashflow Timeline + LCR 예측
    with sim_tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Cashflow Timeline & LCR 예측")
        
        # 서브탭 생성
        cf_timeline_tabs = st.tabs(["📊 Cashflow Timeline", "🏦 LCR 예측"])
        
        # 서브탭 1: Cashflow Timeline
        with cf_timeline_tabs[0]:
            cL, cR = st.columns([1, 1], gap="large")
            with cL:
                st.markdown("<b>BASE</b>", unsafe_allow_html=True)
                st.plotly_chart(plot_cashflow_timeline(base_cf, valuation_date, window_days=90), use_container_width=True)
            with cR:
                st.markdown(f"<b>STRESS (+{stress_shock_bp}bp)</b>", unsafe_allow_html=True)
                st.plotly_chart(plot_cashflow_timeline(stress_cf, valuation_date, window_days=90), use_container_width=True)

            st.caption("해석 포인트: 평가일 이동 시 잔여 CF PV(NPV), DV01(1bp 민감도), 30일 순유출(LCR), 스트레스 버퍼가 동시 반영됩니다.")
        
        # 서브탭 2: LCR 예측
        with cf_timeline_tabs[1]:
            st.markdown("#### 🏦 LCR(유동성커버리지비율) 예측 시뮬레이션")
            st.markdown("일자별 고유동성자산, 현금유출, 현금유입을 기반으로 LCR을 예측합니다.")
            st.markdown("**LCR = 고유동성자산(A) / (현금유출(B) - 현금유입(C)) × 100%**")
            
            st.markdown("---")
            
            # 기초 자료 DATA 입력 방식 선택
            data_input_method = st.radio(
                "데이터 입력 방식",
                ["직접 입력", "샘플 데이터 사용", "CF 시뮬레이션 연동"],
                horizontal=True,
                key="lcr_data_method"
            )
            
            if data_input_method == "직접 입력":
                st.markdown("##### 📝 기초자료 DATA 직접 입력")
                st.info("💡 각 행에 일자별 데이터를 입력하세요. 탭으로 구분된 데이터를 붙여넣기 할 수 있습니다.")
                
                # 기본 템플릿 데이터 (D+0부터 D+60까지 61일)
                default_data = {
                    "일자": ["D+0"] + [f"D+{i}" for i in range(1, 61)],
                    "고유동성자산(A)": [80] + [80 + i for i in range(1, 61)],
                    "현금유출(B)": [110] + [110 + i * 1.5 for i in range(1, 61)],
                    "현금유입(C)": [30] + [30 + i for i in range(1, 61)]
                }
                default_df = pd.DataFrame(default_data)
                
                # 데이터 에디터
                edited_df = st.data_editor(
                    default_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    height=400,
                    key="lcr_data_editor"
                )
                
                lcr_input_df = edited_df.copy()
                
            elif data_input_method == "샘플 데이터 사용":
                st.markdown("##### 📋 샘플 기초자료 DATA")
                
                # 제공된 샘플 데이터 (D+0부터 D+60까지)
                sample_data = {
                    "일자": ["D+0"] + [f"D+{i}" for i in range(1, 61)],
                    "고유동성자산(A)": [80, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
                                      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 
                                      116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 
                                      132, 133, 134, 135, 136, 137, 138, 138],
                    "현금유출(B)": [110, 110, 111.5, 113, 114.5, 116, 117.5, 119, 120.5, 122, 123.5, 125, 126.5, 128, 129.5, 
                                  131, 132.5, 134, 135.5, 137, 138.5, 140, 141.5, 143, 144.5, 146, 147.5, 149, 150.5, 
                                  152, 153.5, 155, 156.5, 158, 159.5, 161, 162.5, 164, 165.5, 167, 168.5, 170, 171.5, 
                                  173, 174.5, 176, 177.5, 179, 180.5, 182, 183.5, 185, 186.5, 188, 189.5, 191, 192.5, 
                                  194, 195.5, 197, 198.5],
                    "현금유입(C)": [30, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
                                  50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
                                  70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
                }
                lcr_input_df = pd.DataFrame(sample_data)
                
                st.dataframe(lcr_input_df, use_container_width=True, height=300)
                
            else:  # CF 시뮬레이션 연동
                st.markdown("##### 🔗 CF 시뮬레이션 결과 연동")
                st.info("💡 CF 시뮬레이션에서 산출된 일별 현금흐름을 기반으로 LCR을 계산합니다.")
                
                # 초기 HQLA 설정
                initial_hqla_cf = st.number_input("초기 고유동성자산(조)", min_value=1.0, max_value=500.0, value=80.0, step=1.0, key="lcr_init_hqla")
                initial_outflow_cf = st.number_input("초기 현금유출(조)", min_value=1.0, max_value=500.0, value=110.0, step=1.0, key="lcr_init_outflow")
                initial_inflow_cf = st.number_input("초기 현금유입(조)", min_value=1.0, max_value=500.0, value=30.0, step=1.0, key="lcr_init_inflow")
                
                # CF 데이터에서 LCR 데이터 생성
                if not cashflows_df.empty:
                    agg_cf_lcr = cashflows_df.groupby(['date', 'type']).agg({'cashflow': 'sum'}).reset_index()
                    cf_pivot_lcr = agg_cf_lcr.pivot(index='date', columns='type', values='cashflow').fillna(0).reset_index()
                    
                    # D+0 초기값 추가
                    lcr_data_list = [{
                        "일자": "D+0",
                        "고유동성자산(A)": round(initial_hqla_cf, 2),
                        "현금유출(B)": round(initial_outflow_cf, 2),
                        "현금유입(C)": round(initial_inflow_cf, 2)
                    }]
                    
                    cumulative_hqla = initial_hqla_cf
                    
                    for idx, row in cf_pivot_lcr.iterrows():
                        day_num = idx + 1
                        if day_num > 60:
                            break
                        
                        asset_cf = row.get('asset', 0) / 1e12  # 조 단위
                        liab_cf = abs(row.get('liability', 0)) / 1e12
                        
                        # HQLA는 자산CF 유입으로 증가
                        cumulative_hqla += asset_cf
                        
                        lcr_data_list.append({
                            "일자": f"D+{day_num}",
                            "고유동성자산(A)": round(cumulative_hqla, 2),
                            "현금유출(B)": round(initial_outflow_cf + liab_cf * day_num, 2),  # 누적 유출
                            "현금유입(C)": round(initial_inflow_cf + asset_cf * day_num, 2)  # 누적 유입
                        })
                    
                    lcr_input_df = pd.DataFrame(lcr_data_list)
                    st.dataframe(lcr_input_df, use_container_width=True, height=300)
                else:
                    st.warning("⚠️ CF 시뮬레이션 데이터가 없습니다. 먼저 CF 결과 분석 탭에서 데이터를 확인하세요.")
                    lcr_input_df = pd.DataFrame()
            
            st.markdown("---")
            
            # LCR 계산 및 시각화
            if 'lcr_input_df' in dir() and not lcr_input_df.empty:
                # LCR 계산
                lcr_input_df["순유출(B-C)"] = lcr_input_df["현금유출(B)"] - lcr_input_df["현금유입(C)"]
                lcr_input_df["LCR(%)"] = lcr_input_df.apply(
                    lambda row: round((row["고유동성자산(A)"] / row["순유출(B-C)"]) * 100, 2) 
                    if row["순유출(B-C)"] > 0 else 999.99, 
                    axis=1
                )
                
                # CF 영향 반영 옵션
                apply_cf_impact = st.checkbox("✅ CF 시뮬레이션 GAP을 HQLA에 누적 반영", value=False, key="lcr_apply_cf")
                
                if apply_cf_impact and not cashflows_df.empty:
                    # CF GAP 누적 계산
                    agg_cf_gap = cashflows_df.groupby(['date', 'type']).agg({'cashflow': 'sum'}).reset_index()
                    cf_pivot_gap = agg_cf_gap.pivot(index='date', columns='type', values='cashflow').fillna(0)
                    cf_pivot_gap['gap'] = cf_pivot_gap.get('asset', 0) - abs(cf_pivot_gap.get('liability', 0))
                    
                    cumulative_cf_gap = 0.0
                    adjusted_hqla_list = []
                    adjusted_lcr_list = []
                    
                    for idx in range(len(lcr_input_df)):
                        if idx < len(cf_pivot_gap):
                            cumulative_cf_gap += cf_pivot_gap['gap'].iloc[idx] / 1e12
                        
                        adjusted_hqla = lcr_input_df["고유동성자산(A)"].iloc[idx] + cumulative_cf_gap
                        net_outflow = lcr_input_df["순유출(B-C)"].iloc[idx]
                        
                        adjusted_lcr = (adjusted_hqla / net_outflow) * 100 if net_outflow > 0 else 999.99
                        
                        adjusted_hqla_list.append(round(adjusted_hqla, 2))
                        adjusted_lcr_list.append(round(adjusted_lcr, 2))
                    
                    lcr_input_df["조정HQLA"] = adjusted_hqla_list
                    lcr_input_df["조정LCR(%)"] = adjusted_lcr_list
                
                # KPI 요약
                st.markdown("#### 📊 LCR 예측 결과")
                
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                
                lcr_col = "조정LCR(%)" if "조정LCR(%)" in lcr_input_df.columns else "LCR(%)"
                
                with kpi_col1:
                    st.metric("D+0 LCR (현재)", f"{lcr_input_df[lcr_col].iloc[0]:.2f}%")
                with kpi_col2:
                    st.metric("D+30 LCR", f"{lcr_input_df[lcr_col].iloc[min(30, len(lcr_input_df)-1)]:.2f}%")
                with kpi_col3:
                    st.metric("D+60 LCR", f"{lcr_input_df[lcr_col].iloc[-1]:.2f}%",
                             delta=f"{lcr_input_df[lcr_col].iloc[-1] - lcr_input_df[lcr_col].iloc[0]:.2f}%")
                with kpi_col4:
                    below_100 = (lcr_input_df[lcr_col] < 100).sum()
                    st.metric("규제미달 일수", f"{below_100}일", 
                             delta="위험" if below_100 > 0 else "안전",
                             delta_color="inverse" if below_100 > 0 else "normal")
                
                # LCR 추이 차트
                st.markdown("#### 📈 LCR 추이 차트")
                
                fig_lcr = go.Figure()
                
                # LCR 라인
                fig_lcr.add_trace(go.Scatter(
                    x=lcr_input_df["일자"],
                    y=lcr_input_df["LCR(%)"],
                    name="기본 LCR(%)",
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=4)
                ))
                
                if "조정LCR(%)" in lcr_input_df.columns:
                    fig_lcr.add_trace(go.Scatter(
                        x=lcr_input_df["일자"],
                        y=lcr_input_df["조정LCR(%)"],
                        name="조정 LCR(%)",
                        mode="lines+markers",
                        line=dict(color="#10b981", width=2, dash="dot"),
                        marker=dict(size=4)
                    ))
                
                # 100% 기준선
                fig_lcr.add_hline(y=100, line_dash="dash", line_color="red", 
                                annotation_text="규제 기준 (100%)", 
                                annotation_position="top right")
                
                fig_lcr.update_layout(
                    title="일자별 LCR 예측",
                    xaxis=dict(title="", tickangle=45, tickfont=dict(size=9)),
                    yaxis=dict(title="LCR(%)", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_lcr, use_container_width=True)
                
                # 구성요소 차트
                st.markdown("#### 📊 구성요소 추이")
                
                fig_comp = go.Figure()
                
                hqla_col = "조정HQLA" if "조정HQLA" in lcr_input_df.columns else "고유동성자산(A)"
                
                fig_comp.add_trace(go.Scatter(
                    x=lcr_input_df["일자"],
                    y=lcr_input_df[hqla_col],
                    name="고유동성자산(A)",
                    mode="lines",
                    line=dict(color="#10b981", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(16, 185, 129, 0.2)"
                ))
                
                fig_comp.add_trace(go.Scatter(
                    x=lcr_input_df["일자"],
                    y=lcr_input_df["현금유출(B)"],
                    name="현금유출(B)",
                    mode="lines",
                    line=dict(color="#ef4444", width=2)
                ))
                
                fig_comp.add_trace(go.Scatter(
                    x=lcr_input_df["일자"],
                    y=lcr_input_df["현금유입(C)"],
                    name="현금유입(C)",
                    mode="lines",
                    line=dict(color="#f59e0b", width=2)
                ))
                
                fig_comp.add_trace(go.Scatter(
                    x=lcr_input_df["일자"],
                    y=lcr_input_df["순유출(B-C)"],
                    name="순유출(B-C)",
                    mode="lines",
                    line=dict(color="#8b5cf6", width=2, dash="dash")
                ))
                
                fig_comp.update_layout(
                    title="일자별 구성요소 추이",
                    xaxis=dict(title="", tickangle=45, tickfont=dict(size=9)),
                    yaxis=dict(title="금액(조)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # 데이터 테이블 (가로 형태)
                st.markdown("#### 📋 기초자료 DATA (가로 형태)")
                
                # 가로 형태로 전치
                display_cols = ["일자", "LCR(%)", "고유동성자산(A)", "현금유출(B)", "현금유입(C)"]
                if "조정LCR(%)" in lcr_input_df.columns:
                    display_cols.insert(2, "조정LCR(%)")
                    display_cols.insert(4, "조정HQLA")
                
                pivot_df = lcr_input_df[display_cols].set_index("일자").T
                st.dataframe(pivot_df, use_container_width=True)
                
                # 다운로드
                st.markdown("---")
                csv_lcr = lcr_input_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 LCR 예측 데이터 다운로드 (CSV)",
                    data=csv_lcr,
                    file_name=f"lcr_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)

    # 탭 4: Sankey
    with sim_tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(plot_sankey_funding(positions_f), use_container_width=True)
        st.caption("Sankey는 실제 '어떤 예금이 어떤 대출의 재원이 되는지'의 정합 매칭이 있어야 더 강력합니다. 현재는 부채 비중으로 자산에 배분하는 개념도입니다.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 탭 6: 🆕 금리 시나리오 분석
    with sim_tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 금리 시나리오 분석")
        st.markdown("복수의 금리 충격 시나리오를 동시에 비교하여 리스크 프로파일을 분석합니다.")
        
        # 시나리오 정의
        scenarios = {
            "BASE (0bp)": 0,
            "소폭 상승 (+50bp)": 50,
            "중간 상승 (+100bp)": 100,
            "급격 상승 (+200bp)": 200,
            "극단 상승 (+300bp)": 300,
            "소폭 하락 (-50bp)": -50,
        }
        
        with st.spinner("시나리오 분석 실행 중..."):
            scenario_results = run_rate_scenario_analysis(
                positions_f,
                str(start_date.date()),
                str(end_date.date()),
                behavioral,
                valuation_date,
                curve_x,
                curve_y,
                scenarios
            )
        
        st.dataframe(scenario_results, use_container_width=True)
        
        # 시각화
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scenario_results["시나리오"],
            y=scenario_results["NPV(조)"],
            name="NPV",
            marker_color="#2563eb"
        ))
        fig.update_layout(
            title="시나리오별 NPV 비교",
            xaxis_title="시나리오",
            yaxis_title="NPV (조)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # 탭 7: 🆕 행동비율 과부족 분석
    with sim_tabs[6]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💰 행동비율에 따른 자금 과부족 분석")
        st.markdown("행동 파라미터 변화에 따른 자금 GAP 및 과부족 금액을 분석합니다.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            param_choice = st.selectbox(
                "분석할 파라미터",
                ["runoff_rate", "deposit_rollover_rate", "loan_prepay_rate", "early_termination"],
                format_func=lambda x: {
                    "runoff_rate": "유출율",
                    "deposit_rollover_rate": "예금 재가입률",
                    "loan_prepay_rate": "대출 조기상환율",
                    "early_termination": "중도해지율"
                }[x]
            )
        with col2:
            param_steps = st.slider("분석 구간 수", 5, 20, 10)
        
        # 파라미터 범위 생성
        base_val = behavioral.get(param_choice, 0.1)
        if param_choice == "deposit_rollover_rate":
            param_range = np.linspace(0.3, 1.0, param_steps)
        else:
            param_range = np.linspace(0.001, 0.30, param_steps)
        
        with st.spinner(f"{param_choice} 파라미터 분석 중..."):
            gap_results = run_behavioral_gap_analysis(
                positions_f,
                str(start_date.date()),
                str(end_date.date()),
                behavioral,
                valuation_date,
                curve_x,
                curve_y,
                param_choice,
                param_range
            )
        
        st.dataframe(gap_results, use_container_width=True)
        
        # 시각화
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gap_results[param_choice],
            y=gap_results["30일과부족"],
            mode="lines+markers",
            name="30일 과부족",
            line=dict(color="#10b981", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=gap_results[param_choice],
            y=gap_results["90일과부족"],
            mode="lines+markers",
            name="90일 과부족",
            line=dict(color="#f59e0b", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=gap_results[param_choice],
            y=gap_results["180일과부족"],
            mode="lines+markers",
            name="180일 과부족",
            line=dict(color="#ef4444", width=3)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="과부족 기준선")
        fig.update_layout(
            title=f"{param_choice} 변화에 따른 기간별 자금 과부족",
            xaxis_title=param_choice,
            yaxis_title="과부족 금액 (조)",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # 탭 8: 🆕 민감도 분석
    with sim_tabs[7]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔬 민감도 분석 (Tornado Chart)")
        st.markdown("주요 파라미터들의 ±20% 변동이 목표 지표에 미치는 영향을 분석합니다.")
        
        target_metric = st.selectbox(
            "분석 대상 지표",
            ["NPV", "NII_YTD", "LCR"],
            format_func=lambda x: {"NPV": "순현재가치", "NII_YTD": "순이자수익", "LCR": "유동성커버리지비율"}[x]
        )
        
        with st.spinner("민감도 분석 실행 중..."):
            sensitivity_results = run_sensitivity_analysis(
                positions_f,
                str(start_date.date()),
                str(end_date.date()),
                behavioral,
                valuation_date,
                curve_x,
                curve_y,
                target_metric
            )
        
        st.dataframe(sensitivity_results, use_container_width=True)
        
        # Tornado Chart
        fig = go.Figure()
        
        for idx, row in sensitivity_results.iterrows():
            fig.add_trace(go.Bar(
                name=row["파라미터"],
                x=[row["-20% 영향(%)"]],
                y=[row["파라미터"]],
                orientation='h',
                marker=dict(color='#ef4444'),
                showlegend=False
            ))
            fig.add_trace(go.Bar(
                name=row["파라미터"],
                x=[row["+20% 영향(%)"]],
                y=[row["파라미터"]],
                orientation='h',
                marker=dict(color='#10b981'),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"{target_metric} 민감도 분석 (Tornado Chart)",
            xaxis_title="영향도 (%)",
            yaxis_title="파라미터",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # 탭 9: 🆕 최적화 시뮬레이션
    with sim_tabs[8]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚡ 최적화 시뮬레이션")
        st.markdown("목표 LCR을 달성하면서 NII를 최대화하는 최적 행동 파라미터 조합을 탐색합니다.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            target_lcr_input = st.number_input("목표 LCR", min_value=1.0, max_value=3.0, value=1.2, step=0.1)
        with col2:
            run_optimization = st.button("🚀 최적화 실행", type="primary")
        
        if run_optimization:
            with st.spinner("최적화 실행 중... (수십 초 소요될 수 있습니다)"):
                opt_result = optimize_behavioral_params(
                    positions_f,
                    str(start_date.date()),
                    str(end_date.date()),
                    behavioral,
                    valuation_date,
                    curve_x,
                    curve_y,
                    target_lcr=target_lcr_input,
                    target_nii_min=0.0
                )
            
            if opt_result["success"]:
                st.success("✅ " + opt_result["message"])
                
                st.markdown('<div class="optimal-result">', unsafe_allow_html=True)
                st.markdown("#### 🎯 최적 파라미터")
                
                opt_params = opt_result["optimal_params"]
                opt_kpi = opt_result["optimal_kpi"]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**최적 행동 파라미터:**")
                    st.write(f"- 대출 조기상환율: {opt_params.get('loan_prepay_rate', 0):.4f}")
                    st.write(f"- 예금 재가입률: {opt_params.get('deposit_rollover_rate', 0):.4f}")
                    st.write(f"- 유출율: {opt_params.get('runoff_rate', 0):.4f}")
                    st.write(f"- 중도해지율: {opt_params.get('early_termination', 0):.4f}")
                
                with col2:
                    st.markdown("**최적화 결과 KPI:**")
                    st.write(f"- NPV: {fmt_조(opt_kpi['NPV'])}")
                    st.write(f"- NII: {fmt_조(opt_kpi['NII_YTD'])}")
                    st.write(f"- LCR: {fmt_num(opt_kpi['LCR'])}")
                    st.write(f"- 생존 여부: {'YES' if opt_kpi['Stress_Survive'] >= 0.5 else 'NO'}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 비교표
                st.markdown("#### 📊 현재 vs 최적 비교")
                
                # 안전한 나눗셈 함수
                def safe_divide(a, b):
                    if abs(b) < 1e-9:
                        return 0.0
                    return (a - b) / abs(b) * 100
                
                comparison_df = pd.DataFrame({
                    "지표": ["대출 조기상환율", "예금 재가입률", "유출율", "중도해지율", "NPV(조)", "NII(조)", "LCR"],
                    "현재": [
                        f"{behavioral.get('loan_prepay_rate', 0):.4f}",
                        f"{behavioral.get('deposit_rollover_rate', 0):.4f}",
                        f"{behavioral.get('runoff_rate', 0):.4f}",
                        f"{behavioral.get('early_termination', 0):.4f}",
                        f"{base_k['NPV']/1e9:.2f}",
                        f"{base_k['NII_YTD']/1e9:.2f}",
                        f"{base_k['LCR']:.2f}",
                    ],
                    "최적": [
                        f"{opt_params.get('loan_prepay_rate', 0):.4f}",
                        f"{opt_params.get('deposit_rollover_rate', 0):.4f}",
                        f"{opt_params.get('runoff_rate', 0):.4f}",
                        f"{opt_params.get('early_termination', 0):.4f}",
                        f"{opt_kpi['NPV']/1e9:.2f}",
                        f"{opt_kpi['NII_YTD']/1e9:.2f}",
                        f"{opt_kpi['LCR']:.2f}",
                    ],
                    "개선율(%)": [
                        f"{safe_divide(opt_params.get('loan_prepay_rate', 0), behavioral.get('loan_prepay_rate', 1)):.1f}",
                        f"{safe_divide(opt_params.get('deposit_rollover_rate', 0), behavioral.get('deposit_rollover_rate', 1)):.1f}",
                        f"{safe_divide(opt_params.get('runoff_rate', 0), behavioral.get('runoff_rate', 1)):.1f}",
                        f"{safe_divide(opt_params.get('early_termination', 0), behavioral.get('early_termination', 1)):.1f}",
                        f"{safe_divide(opt_kpi['NPV'], base_k['NPV']):.1f}",
                        f"{safe_divide(opt_kpi['NII_YTD'], base_k['NII_YTD']):.1f}",
                        f"{safe_divide(opt_kpi['LCR'], base_k['LCR']):.1f}",
                    ]
                })
                st.dataframe(comparison_df, use_container_width=True)
                
            else:
                st.error("❌ " + opt_result["message"])
                
                # 현재 LCR 확인 및 권장사항 표시
                current_lcr = base_k.get('LCR', 0)
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                            padding: 20px; border-radius: 12px; margin-top: 15px; color: #78350f;">
                    <h4 style="margin-top: 0; color: #78350f;">💡 최적화 실패 원인 및 해결 방법</h4>
                    <p style="margin-bottom: 10px;"><strong>현재 상태:</strong></p>
                    <ul style="margin-bottom: 15px;">
                        <li>현재 LCR: <strong>{:.2f}</strong></li>
                        <li>목표 LCR: <strong>{:.2f}</strong></li>
                    </ul>
                    <p style="margin-bottom: 10px;"><strong>권장사항:</strong></p>
                    <ol>
                        <li><strong>목표 LCR을 낮추기</strong>: 현재 LCR의 1.2배 이하로 설정해보세요 (권장: {:.2f})</li>
                        <li><strong>행동비율 조정</strong>: 사이드바에서 예금 재가입률을 높이거나 유출율을 낮춰보세요</li>
                        <li><strong>HQLA 증가</strong>: 고유동성자산을 늘려 LCR 개선을 시도하세요</li>
                    </ol>
                    <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                        💬 제약 조건이 너무 엄격하면 수학적으로 해가 존재하지 않을 수 있습니다.
                    </p>
                </div>
                """.format(current_lcr, target_lcr_input, current_lcr * 1.2), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


