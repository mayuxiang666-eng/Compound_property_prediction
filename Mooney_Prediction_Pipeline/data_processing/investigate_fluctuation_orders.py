import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration (confirmed by user)
THRESHOLD_STD = 6.0  # MU, 标准差阈值
TOP_N_ORDERS = 8      # 需要调查的订单数量
DATE_RANGE_DAYS = 180  # 最近约6个月

def load_mny_data():
    """加载最近6个月的实验室 MNY 结果。"""
    script_dir = os.path.dirname(__file__)
    sql_path = os.path.join(script_dir, 'get master MNY test.sql')
    with open(sql_path, 'r', encoding='utf-8') as f:
        mny_sql = f.read()
    # 使用 pipeline_orchestrator 中的连接函数
    from pipeline_orchestrator import connect_datamart_encrypted, query_with_retry
    conn = connect_datamart_encrypted('mustangmaster')
    df = pd.read_sql(mny_sql, conn)
    if 'test_result_start_time' in df.columns:
        cutoff = datetime.now() - timedelta(days=DATE_RANGE_DAYS)
        df = df[pd.to_datetime(df['test_result_start_time']) >= cutoff]
    return df

def select_high_fluctuation_orders(mny_df):
    """计算每个 OrderID 的 MNY 标准差，并挑选出波动大的订单。"""
    mny_df['order_id'] = mny_df['order_id'].astype(str).str.strip()
    std_series = mny_df.groupby('order_id')['test_result'].std().reset_index()
    std_series = std_series.rename(columns={'test_result': 'mny_std'})
    high_std = std_series[std_series['mny_std'] > THRESHOLD_STD]
    top_orders = high_std.nlargest(TOP_N_ORDERS, 'mny_std')
    return top_orders

def export_orders(mny_df, selected_orders):
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'fluctuation_orders_summary.csv')
    selected_orders.to_csv(summary_path, index=False)
    detail = mny_df[mny_df['order_id'].isin(selected_orders['order_id'])]
    detail_path = os.path.join(out_dir, 'fluctuation_orders_detail.csv')
    detail.to_csv(detail_path, index=False)
    # Use ASCII-friendly output to avoid encoding errors on Windows console
    print(f"Exported summary: {summary_path}\nExported detailed rows: {detail_path}")

def main():
    mny_df = load_mny_data()
    top_orders = select_high_fluctuation_orders(mny_df)
    if top_orders.empty:
        print('未找到满足阈值的订单')
        return
    export_orders(mny_df, top_orders)

if __name__ == '__main__':
    main()
