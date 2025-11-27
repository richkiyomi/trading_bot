"""
Iron Condor P&L Dashboard
Streamlit web application for tracking and analyzing iron condor trading performance.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional
import os

from models import (
    get_db_session, IronCondor, IronCondorLeg, Order, PositionSnapshot,
    CondorStatus, ExitReason
)

# Page config - optimized for mobile
st.set_page_config(
    page_title="Iron Condor P&L Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Iron Condor P&L Dashboard - Track your trading performance"
    }
)

# Custom CSS for better styling and mobile responsiveness
st.markdown("""
    <style>
    /* Base styles */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #00cc00;
        font-weight: bold;
    }
    .negative {
        color: #ff0000;
        font-weight: bold;
    }
    
    /* Mobile responsiveness */
    @media screen and (max-width: 768px) {
        /* Make tables scrollable on mobile */
        .stDataFrame {
            overflow-x: auto;
            display: block;
        }
        
        /* Stack columns vertically on mobile */
        .element-container {
            width: 100% !important;
        }
        
        /* Adjust metric cards for mobile */
        .metric-container {
            padding: 0.5rem;
        }
        
        /* Make sidebar collapsible on mobile */
        .css-1d391kg {
            width: 100% !important;
        }
        
        /* Adjust plotly charts for mobile */
        .js-plotly-plot {
            width: 100% !important;
            height: auto !important;
        }
        
        /* Better spacing on mobile */
        .main .block-container {
            padding: 1rem;
        }
        
        /* Make text more readable on mobile */
        h1, h2, h3 {
            font-size: 1.5rem !important;
        }
    }
    
    /* Tablet responsiveness */
    @media screen and (min-width: 769px) and (max-width: 1024px) {
        .stDataFrame {
            overflow-x: auto;
        }
    }
    
    /* Ensure proper viewport */
    html {
        -webkit-text-size-adjust: 100%;
        -ms-text-size-adjust: 100%;
    }
    </style>
""", unsafe_allow_html=True)


def format_currency(value: Optional[float]) -> str:
    """Format value as currency."""
    if value is None:
        return "$0.00"
    return f"${float(value):,.2f}"


def format_percentage(value: Optional[float]) -> str:
    """Format value as percentage."""
    if value is None:
        return "0.00%"
    return f"{float(value):.2%}"


def get_realized_pnl_summary(session, start_date: Optional[date] = None, end_date: Optional[date] = None):
    """Get summary of realized P&L from closed condors."""
    query = session.query(IronCondor).filter(IronCondor.status == CondorStatus.CLOSED)
    
    if start_date:
        query = query.filter(IronCondor.exit_time >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        query = query.filter(IronCondor.exit_time <= datetime.combine(end_date, datetime.max.time()))
    
    condors = query.all()
    
    if not condors:
        return {
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'total_credit': 0.0,
            'total_debit': 0.0,
        }
    
    pnl_values = [float(c.realized_pnl) for c in condors if c.realized_pnl is not None]
    winning = [p for p in pnl_values if p > 0]
    losing = [p for p in pnl_values if p < 0]
    
    return {
        'total_pnl': sum(pnl_values),
        'total_trades': len(condors),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': len(winning) / len(condors) if condors else 0.0,
        'avg_win': sum(winning) / len(winning) if winning else 0.0,
        'avg_loss': sum(losing) / len(losing) if losing else 0.0,
        'largest_win': max(winning) if winning else 0.0,
        'largest_loss': min(losing) if losing else 0.0,
        'total_credit': sum(float(c.entry_net_credit) for c in condors),
        'total_debit': sum(float(c.exit_net_debit or 0) for c in condors),
    }


def get_unrealized_pnl_summary(session):
    """Get summary of unrealized P&L from open condors."""
    open_condors = session.query(IronCondor).filter(
        IronCondor.status.in_([CondorStatus.OPEN, CondorStatus.PENDING_OPEN])
    ).all()
    
    if not open_condors:
        return {
            'total_unrealized': 0.0,
            'open_positions': 0,
            'total_credit': 0.0,
        }
    
    # Get latest snapshot for each condor
    total_unrealized = 0.0
    for condor in open_condors:
        latest_snapshot = session.query(PositionSnapshot).filter(
            PositionSnapshot.condor_id == condor.id
        ).order_by(PositionSnapshot.snapshot_time.desc()).first()
        
        if latest_snapshot:
            total_unrealized += float(latest_snapshot.unrealized_pl)
        else:
            # Fallback: estimate from entry credit (conservative)
            total_unrealized += float(condor.entry_net_credit) * 0.5  # Assume 50% profit
    
    return {
        'total_unrealized': total_unrealized,
        'open_positions': len(open_condors),
        'total_credit': sum(float(c.entry_net_credit) for c in open_condors),
    }


def calculate_rr_ratio(max_profit: float, max_loss: float) -> Optional[float]:
    """Calculate Risk/Reward ratio (max_loss / max_profit)."""
    if max_profit > 0:
        return max_loss / max_profit
    return None


def get_condors_dataframe(session, status_filter: Optional[list] = None, 
                         symbol_filter: Optional[str] = None,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None,
                         rr_min: Optional[float] = None,
                         rr_max: Optional[float] = None):
    """Get condors as DataFrame for display."""
    query = session.query(IronCondor)
    
    if status_filter:
        query = query.filter(IronCondor.status.in_(status_filter))
    
    if symbol_filter:
        query = query.filter(IronCondor.underlying_symbol == symbol_filter.upper())
    
    if start_date:
        query = query.filter(IronCondor.created_at >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        query = query.filter(IronCondor.created_at <= datetime.combine(end_date, datetime.max.time()))
    
    condors = query.order_by(IronCondor.created_at.desc()).all()
    
    data = []
    for c in condors:
        # Get latest unrealized P/L for open positions
        unrealized_pl = None
        unrealized_pl_pct = None
        if c.status in [CondorStatus.OPEN, CondorStatus.PENDING_OPEN]:
            latest_snapshot = session.query(PositionSnapshot).filter(
                PositionSnapshot.condor_id == c.id
            ).order_by(PositionSnapshot.snapshot_time.desc()).first()
            if latest_snapshot:
                unrealized_pl = float(latest_snapshot.unrealized_pl)
                unrealized_pl_pct = float(latest_snapshot.unrealized_pl_pct)
        
        # Calculate R/R ratio
        max_profit = float(c.max_profit)
        max_loss = float(c.max_loss)
        rr_ratio = calculate_rr_ratio(max_profit, max_loss)
        
        # Filter by R/R if specified
        if rr_min is not None and rr_ratio is not None and rr_ratio < rr_min:
            continue
        if rr_max is not None and rr_ratio is not None and rr_ratio > rr_max:
            continue
        
        data.append({
            'ID': c.id,
            'Symbol': c.underlying_symbol,
            'Status': c.status.value,
            'Expiration': c.expiration_date,
            'Entry Date': c.entry_time.date() if c.entry_time else None,
            'Exit Date': c.exit_time.date() if c.exit_time else None,
            'Exit Reason': c.exit_reason.value if c.exit_reason else None,
            'Entry Credit': float(c.entry_net_credit),
            'Exit Debit': float(c.exit_net_debit) if c.exit_net_debit else None,
            'Realized P/L': float(c.realized_pnl) if c.realized_pnl else None,
            'Unrealized P/L': unrealized_pl,
            'Unrealized P/L %': unrealized_pl_pct,
            'Max Profit': float(c.max_profit),
            'Max Loss': float(c.max_loss),
            'R/R Ratio': rr_ratio,
            'Credit Ratio': float(c.credit_ratio),
            'Wing Width': float(c.wing_width),
            'DTE at Entry': c.dte_at_entry,
        })
    
    return pd.DataFrame(data)


def get_pnl_by_symbol(session):
    """Get P&L breakdown by symbol."""
    closed_condors = session.query(IronCondor).filter(
        IronCondor.status == CondorStatus.CLOSED
    ).all()
    
    symbol_data = {}
    for c in closed_condors:
        symbol = c.underlying_symbol
        if symbol not in symbol_data:
            symbol_data[symbol] = {
                'trades': 0,
                'total_pnl': 0.0,
                'wins': 0,
                'losses': 0,
            }
        
        symbol_data[symbol]['trades'] += 1
        if c.realized_pnl:
            pnl = float(c.realized_pnl)
            symbol_data[symbol]['total_pnl'] += pnl
            if pnl > 0:
                symbol_data[symbol]['wins'] += 1
            else:
                symbol_data[symbol]['losses'] += 1
    
    if not symbol_data:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            'Symbol': symbol,
            'Total P/L': data['total_pnl'],
            'Trades': data['trades'],
            'Wins': data['wins'],
            'Losses': data['losses'],
            'Win Rate': data['wins'] / data['trades'] if data['trades'] > 0 else 0,
            'Avg P/L': data['total_pnl'] / data['trades'] if data['trades'] > 0 else 0,
        }
        for symbol, data in symbol_data.items()
    ])
    
    return df.sort_values('Total P/L', ascending=False)


def get_pnl_timeline(session, start_date: Optional[date] = None, end_date: Optional[date] = None):
    """Get P&L timeline (cumulative)."""
    query = session.query(IronCondor).filter(
        IronCondor.status == CondorStatus.CLOSED
    ).filter(IronCondor.realized_pnl.isnot(None))
    
    if start_date:
        query = query.filter(IronCondor.exit_time >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        query = query.filter(IronCondor.exit_time <= datetime.combine(end_date, datetime.max.time()))
    
    condors = query.order_by(IronCondor.exit_time.asc()).all()
    
    if not condors:
        return pd.DataFrame()
    
    cumulative_pnl = 0.0
    timeline_data = []
    
    for c in condors:
        pnl = float(c.realized_pnl)
        cumulative_pnl += pnl
        timeline_data.append({
            'Date': c.exit_time.date() if c.exit_time else c.created_at.date(),
            'P/L': pnl,
            'Cumulative P/L': cumulative_pnl,
            'Symbol': c.underlying_symbol,
        })
    
    return pd.DataFrame(timeline_data)


def get_rr_analysis(session, start_date: Optional[date] = None, end_date: Optional[date] = None):
    """Get R/R analysis data for closed trades."""
    query = session.query(IronCondor).filter(
        IronCondor.status == CondorStatus.CLOSED
    ).filter(IronCondor.realized_pnl.isnot(None))
    
    if start_date:
        query = query.filter(IronCondor.exit_time >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        query = query.filter(IronCondor.exit_time <= datetime.combine(end_date, datetime.max.time()))
    
    condors = query.all()
    
    if not condors:
        return pd.DataFrame()
    
    data = []
    for c in condors:
        max_profit = float(c.max_profit)
        max_loss = float(c.max_loss)
        rr_ratio = calculate_rr_ratio(max_profit, max_loss)
        realized_pnl = float(c.realized_pnl) if c.realized_pnl else 0.0
        
        # Calculate actual R/R (realized P/L vs max loss)
        actual_rr = None
        if realized_pnl > 0 and max_loss > 0:
            actual_rr = realized_pnl / max_loss
        elif realized_pnl < 0 and max_profit > 0:
            actual_rr = abs(realized_pnl) / max_profit
        
        data.append({
            'ID': c.id,
            'Symbol': c.underlying_symbol,
            'R/R Ratio': rr_ratio,
            'Theoretical R/R': f"{rr_ratio:.2f}:1" if rr_ratio else "N/A",
            'Max Profit': max_profit,
            'Max Loss': max_loss,
            'Entry Credit': float(c.entry_net_credit),
            'Realized P/L': realized_pnl,
            'Actual R/R': actual_rr,
            'P/L % of Max Profit': (realized_pnl / max_profit) if max_profit > 0 else None,
            'P/L % of Max Loss': (abs(realized_pnl) / max_loss) if max_loss > 0 and realized_pnl < 0 else None,
            'Credit Ratio': float(c.credit_ratio),
            'Exit Reason': c.exit_reason.value if c.exit_reason else None,
            'Exit Date': c.exit_time.date() if c.exit_time else None,
        })
    
    return pd.DataFrame(data)


def get_rr_summary_stats(session, start_date: Optional[date] = None, end_date: Optional[date] = None):
    """Get summary statistics for R/R ratios."""
    df = get_rr_analysis(session, start_date, end_date)
    
    if df.empty or 'R/R Ratio' not in df.columns:
        return {
            'avg_rr': 0.0,
            'median_rr': 0.0,
            'min_rr': 0.0,
            'max_rr': 0.0,
            'best_rr_trade_pnl': 0.0,
            'worst_rr_trade_pnl': 0.0,
            'rr_buckets': {},
        }
    
    valid_rr = df[df['R/R Ratio'].notna()]['R/R Ratio']
    
    if valid_rr.empty:
        return {
            'avg_rr': 0.0,
            'median_rr': 0.0,
            'min_rr': 0.0,
            'max_rr': 0.0,
            'best_rr_trade_pnl': 0.0,
            'worst_rr_trade_pnl': 0.0,
            'rr_buckets': {},
        }
    
    # R/R buckets
    buckets = {
        'Excellent (< 0.2:1)': len(df[(df['R/R Ratio'].notna()) & (df['R/R Ratio'] < 0.2)]),
        'Good (0.2-0.3:1)': len(df[(df['R/R Ratio'].notna()) & (df['R/R Ratio'] >= 0.2) & (df['R/R Ratio'] < 0.3)]),
        'Fair (0.3-0.5:1)': len(df[(df['R/R Ratio'].notna()) & (df['R/R Ratio'] >= 0.3) & (df['R/R Ratio'] < 0.5)]),
        'Poor (0.5-1.0:1)': len(df[(df['R/R Ratio'].notna()) & (df['R/R Ratio'] >= 0.5) & (df['R/R Ratio'] < 1.0)]),
        'Very Poor (> 1.0:1)': len(df[(df['R/R Ratio'].notna()) & (df['R/R Ratio'] >= 1.0)]),
    }
    
    # Best and worst R/R trades
    best_rr_trade = df.loc[df['R/R Ratio'].idxmin()] if not df[df['R/R Ratio'].notna()].empty else None
    worst_rr_trade = df.loc[df['R/R Ratio'].idxmax()] if not df[df['R/R Ratio'].notna()].empty else None
    
    return {
        'avg_rr': float(valid_rr.mean()),
        'median_rr': float(valid_rr.median()),
        'min_rr': float(valid_rr.min()),
        'max_rr': float(valid_rr.max()),
        'best_rr_trade_pnl': float(best_rr_trade['Realized P/L']) if best_rr_trade is not None else 0.0,
        'worst_rr_trade_pnl': float(worst_rr_trade['Realized P/L']) if worst_rr_trade is not None else 0.0,
        'rr_buckets': buckets,
    }


def main():
    st.title("📊 Iron Condor P&L Dashboard")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(date.today() - timedelta(days=30), date.today()),
        help="Filter trades by entry date"
    )
    
    start_date = date_range[0] if isinstance(date_range, tuple) else None
    end_date = date_range[1] if isinstance(date_range, tuple) else None
    
    # Status filter
    status_options = [s.value for s in CondorStatus]
    selected_statuses = st.sidebar.multiselect(
        "Status",
        options=status_options,
        default=status_options,
        help="Filter by condor status"
    )
    
    # Symbol filter
    session = get_db_session()
    try:
        all_symbols = sorted(set([c.underlying_symbol for c in session.query(IronCondor.underlying_symbol).distinct().all()]))
        selected_symbol = st.sidebar.selectbox(
            "Symbol (Optional)",
            options=["All"] + all_symbols,
            help="Filter by underlying symbol"
        )
        symbol_filter = None if selected_symbol == "All" else selected_symbol
    finally:
        session.close()
    
    # R/R Ratio filter
    st.sidebar.subheader("R/R Ratio Filter")
    use_rr_filter = st.sidebar.checkbox("Filter by R/R Ratio", value=False)
    rr_min = None
    rr_max = None
    if use_rr_filter:
        rr_min = st.sidebar.number_input("Min R/R", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="Minimum R/R ratio (e.g., 0.2)")
        rr_max = st.sidebar.number_input("Max R/R", min_value=0.0, max_value=10.0, value=5.0, step=0.1, help="Maximum R/R ratio (e.g., 1.0)")
        if rr_min == 0.0:
            rr_min = None
        if rr_max >= 10.0:
            rr_max = None
    
    # Get data
    session = get_db_session()
    try:
        # Overview metrics
        realized_summary = get_realized_pnl_summary(session, start_date, end_date)
        unrealized_summary = get_unrealized_pnl_summary(session)
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pnl = realized_summary['total_pnl'] + unrealized_summary['total_unrealized']
            pnl_color = "positive" if total_pnl >= 0 else "negative"
            st.metric(
                "Total P&L",
                format_currency(total_pnl),
                delta=format_currency(unrealized_summary['total_unrealized']) + " unrealized"
            )
        
        with col2:
            st.metric(
                "Realized P&L",
                format_currency(realized_summary['total_pnl']),
                delta=f"{realized_summary['total_trades']} closed trades"
            )
        
        with col3:
            st.metric(
                "Unrealized P&L",
                format_currency(unrealized_summary['total_unrealized']),
                delta=f"{unrealized_summary['open_positions']} open positions"
            )
        
        with col4:
            win_rate = realized_summary['win_rate']
            st.metric(
                "Win Rate",
                format_percentage(win_rate),
                delta=f"{realized_summary['winning_trades']}/{realized_summary['total_trades']} wins"
            )
        
        # Detailed metrics
        st.subheader("Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Trades", realized_summary['total_trades'])
        with col2:
            st.metric("Avg Win", format_currency(realized_summary['avg_win']))
        with col3:
            st.metric("Avg Loss", format_currency(realized_summary['avg_loss']))
        with col4:
            st.metric("Largest Win", format_currency(realized_summary['largest_win']))
        with col5:
            st.metric("Largest Loss", format_currency(realized_summary['largest_loss']))
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📋 Trade Details", "📊 By Symbol", "📅 Timeline", "⚖️ R/R Analysis"])
        
        with tab1:
            st.subheader("P&L Distribution")
            
            # Get closed trades for distribution
            closed_condors = session.query(IronCondor).filter(
                IronCondor.status == CondorStatus.CLOSED,
                IronCondor.realized_pnl.isnot(None)
            ).all()
            
            if closed_condors:
                pnl_values = [float(c.realized_pnl) for c in closed_condors]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        x=pnl_values,
                        nbins=20,
                        title="P&L Distribution",
                        labels={'x': 'P&L ($)', 'count': 'Frequency'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(
                        y=pnl_values,
                        title="P&L Box Plot",
                        labels={'y': 'P&L ($)'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Trade Details")
            
            df = get_condors_dataframe(
                session,
                status_filter=selected_statuses if selected_statuses else None,
                symbol_filter=symbol_filter,
                start_date=start_date,
                end_date=end_date,
                rr_min=rr_min if use_rr_filter else None,
                rr_max=rr_max if use_rr_filter else None
            )
            
            if not df.empty:
                # Format currency columns
                currency_cols = ['Entry Credit', 'Exit Debit', 'Realized P/L', 'Unrealized P/L', 
                                'Max Profit', 'Max Loss']
                for col in currency_cols:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
                
                if 'Unrealized P/L %' in df.columns:
                    df['Unrealized P/L %'] = df['Unrealized P/L %'].apply(
                        lambda x: format_percentage(x) if pd.notna(x) else "N/A"
                    )
                
                if 'Credit Ratio' in df.columns:
                    df['Credit Ratio'] = df['Credit Ratio'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "N/A")
                
                if 'R/R Ratio' in df.columns:
                    df['R/R Ratio'] = df['R/R Ratio'].apply(lambda x: f"{x:.2f}:1" if pd.notna(x) else "N/A")
                
                st.dataframe(df, use_container_width=True, height=400)
                
                # Export button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"iron_condor_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No trades found matching the selected filters.")
        
        with tab3:
            st.subheader("Performance by Symbol")
            
            symbol_df = get_pnl_by_symbol(session)
            
            if not symbol_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(symbol_df, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        symbol_df,
                        x='Symbol',
                        y='Total P/L',
                        title="Total P&L by Symbol",
                        color='Total P/L',
                        color_continuous_scale=['red', 'green'] if symbol_df['Total P/L'].min() < 0 else ['green'],
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No closed trades to analyze by symbol.")
        
        with tab4:
            st.subheader("P&L Timeline")
            
            timeline_df = get_pnl_timeline(session, start_date, end_date)
            
            if not timeline_df.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=timeline_df['Date'],
                    y=timeline_df['Cumulative P/L'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break Even")
                
                fig.update_layout(
                    title="Cumulative P&L Over Time",
                    xaxis_title="Date",
                    yaxis_title="Cumulative P&L ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Daily P&L chart
                fig2 = px.bar(
                    timeline_df,
                    x='Date',
                    y='P/L',
                    title="Daily P&L",
                    color='P/L',
                    color_continuous_scale=['red', 'green'],
                    labels={'P/L': 'P&L ($)'}
                )
                fig2.add_hline(y=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No closed trades to show timeline.")
        
        with tab5:
            st.subheader("Risk/Reward Ratio Analysis")
            
            rr_df = get_rr_analysis(session, start_date, end_date)
            rr_summary = get_rr_summary_stats(session, start_date, end_date)
            
            if not rr_df.empty:
                # R/R Summary Metrics
                st.markdown("### R/R Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average R/R", f"{rr_summary['avg_rr']:.2f}:1")
                with col2:
                    st.metric("Median R/R", f"{rr_summary['median_rr']:.2f}:1")
                with col3:
                    st.metric("Best R/R", f"{rr_summary['min_rr']:.2f}:1", 
                             delta=format_currency(rr_summary['best_rr_trade_pnl']))
                with col4:
                    st.metric("Worst R/R", f"{rr_summary['max_rr']:.2f}:1",
                             delta=format_currency(rr_summary['worst_rr_trade_pnl']))
                
                # R/R Distribution
                st.markdown("### R/R Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    valid_rr = rr_df[rr_df['R/R Ratio'].notna()]['R/R Ratio']
                    if not valid_rr.empty:
                        fig = px.histogram(
                            x=valid_rr,
                            nbins=20,
                            title="R/R Ratio Distribution",
                            labels={'x': 'R/R Ratio', 'count': 'Frequency'},
                            color_discrete_sequence=['#ff7f0e']
                        )
                        fig.add_vline(x=0.3, line_dash="dash", line_color="green", 
                                     annotation_text="Good (<0.3)", annotation_position="top")
                        fig.add_vline(x=1.0, line_dash="dash", line_color="red", 
                                     annotation_text="Poor (>1.0)", annotation_position="top")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # R/R Buckets
                    buckets_df = pd.DataFrame([
                        {'Category': k, 'Count': v}
                        for k, v in rr_summary['rr_buckets'].items()
                    ])
                    if not buckets_df.empty:
                        fig = px.bar(
                            buckets_df,
                            x='Category',
                            y='Count',
                            title="R/R Ratio Categories",
                            color='Count',
                            color_continuous_scale='RdYlGn',
                            labels={'Count': 'Number of Trades'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # R/R vs Performance Correlation
                st.markdown("### R/R vs Actual Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter plot: R/R vs Realized P/L
                    valid_data = rr_df[rr_df['R/R Ratio'].notna()].copy()
                    if not valid_data.empty:
                        fig = px.scatter(
                            valid_data,
                            x='R/R Ratio',
                            y='Realized P/L',
                            color='Realized P/L',
                            size='Entry Credit',
                            hover_data=['Symbol', 'Theoretical R/R', 'Exit Reason'],
                            title="R/R Ratio vs Realized P/L",
                            color_continuous_scale=['red', 'green'],
                            labels={'R/R Ratio': 'R/R Ratio', 'Realized P/L': 'Realized P/L ($)'}
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="black")
                        fig.add_vline(x=0.3, line_dash="dash", line_color="green", 
                                     annotation_text="Good R/R", annotation_position="top")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # P/L % of Max Profit/Loss
                    valid_data = valid_data.copy()
                    valid_data['P/L %'] = valid_data.apply(
                        lambda row: row['P/L % of Max Profit'] if row['Realized P/L'] >= 0 
                        else -row['P/L % of Max Loss'] if pd.notna(row['P/L % of Max Loss']) else None,
                        axis=1
                    )
                    valid_pct = valid_data[valid_data['P/L %'].notna()]
                    if not valid_pct.empty:
                        fig = px.scatter(
                            valid_pct,
                            x='R/R Ratio',
                            y='P/L %',
                            color='Realized P/L',
                            size='Entry Credit',
                            hover_data=['Symbol', 'Theoretical R/R'],
                            title="R/R Ratio vs P/L % of Max",
                            color_continuous_scale=['red', 'green'],
                            labels={'R/R Ratio': 'R/R Ratio', 'P/L %': 'P/L % of Max Profit/Loss'}
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="black")
                        fig.add_hline(y=0.7, line_dash="dash", line_color="blue", 
                                     annotation_text="70% Target", annotation_position="right")
                        st.plotly_chart(fig, use_container_width=True)
                
                # R/R by Symbol
                st.markdown("### R/R Analysis by Symbol")
                symbol_rr = rr_df.groupby('Symbol').agg({
                    'R/R Ratio': ['mean', 'min', 'max', 'count'],
                    'Realized P/L': 'sum'
                }).round(2)
                symbol_rr.columns = ['Avg R/R', 'Min R/R', 'Max R/R', 'Trades', 'Total P/L']
                symbol_rr = symbol_rr.sort_values('Avg R/R')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(symbol_rr, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        symbol_rr.reset_index(),
                        x='Symbol',
                        y='Avg R/R',
                        title="Average R/R Ratio by Symbol",
                        color='Total P/L',
                        color_continuous_scale=['red', 'green'],
                        labels={'Avg R/R': 'Average R/R Ratio'}
                    )
                    fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                                 annotation_text="Good R/R", annotation_position="right")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed R/R Table
                st.markdown("### Detailed R/R Analysis")
                display_rr_df = rr_df.copy()
                
                # Format columns
                if 'R/R Ratio' in display_rr_df.columns:
                    display_rr_df['R/R Ratio'] = display_rr_df['R/R Ratio'].apply(
                        lambda x: f"{x:.2f}:1" if pd.notna(x) else "N/A"
                    )
                
                currency_cols = ['Max Profit', 'Max Loss', 'Entry Credit', 'Realized P/L']
                for col in currency_cols:
                    if col in display_rr_df.columns:
                        display_rr_df[col] = display_rr_df[col].apply(
                            lambda x: format_currency(x) if pd.notna(x) else "N/A"
                        )
                
                if 'Actual R/R' in display_rr_df.columns:
                    display_rr_df['Actual R/R'] = display_rr_df['Actual R/R'].apply(
                        lambda x: f"{x:.2f}:1" if pd.notna(x) else "N/A"
                    )
                
                if 'P/L % of Max Profit' in display_rr_df.columns:
                    display_rr_df['P/L % of Max Profit'] = display_rr_df['P/L % of Max Profit'].apply(
                        lambda x: format_percentage(x) if pd.notna(x) else "N/A"
                    )
                
                if 'P/L % of Max Loss' in display_rr_df.columns:
                    display_rr_df['P/L % of Max Loss'] = display_rr_df['P/L % of Max Loss'].apply(
                        lambda x: format_percentage(x) if pd.notna(x) else "N/A"
                    )
                
                if 'Credit Ratio' in display_rr_df.columns:
                    display_rr_df['Credit Ratio'] = display_rr_df['Credit Ratio'].apply(
                        lambda x: f"{float(x):.4f}" if pd.notna(x) else "N/A"
                    )
                
                st.dataframe(display_rr_df, use_container_width=True, height=400)
                
                # Export button
                csv = rr_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download R/R Analysis as CSV",
                    data=csv,
                    file_name=f"rr_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No closed trades available for R/R analysis.")
        
    finally:
        session.close()


if __name__ == "__main__":
    main()

