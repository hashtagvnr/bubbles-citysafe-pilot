import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import datetime
from dateutil.relativedelta import relativedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Bubbles Civic Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("gloucester_crime_data.csv")
        df['date_obj'] = pd.to_datetime(df['month'])
        df['time_step'] = (df['date_obj'].dt.year - df['date_obj'].dt.year.min()) * 12 + df['date_obj'].dt.month
        df['month_num'] = df['date_obj'].dt.month
        return df
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("System Offline. Please run main.py.")
    st.stop()

def get_color(category):
    if 'shoplifting' in category: return [0, 0, 255, 160] 
    elif 'violence' in category: return [255, 0, 0, 160] 
    elif 'anti-social' in category: return [255, 165, 0, 160] 
    elif 'drugs' in category: return [0, 255, 0, 160] 
    elif 'burglary' in category: return [128, 0, 128, 160] 
    else: return [128, 128, 128, 140] 

df['color'] = df['category'].apply(get_color)

# --- 3. SIDEBAR ---
st.sidebar.title("🏙️ Bubbles CitySafe")
user_role = st.sidebar.radio("Select Role:", ["📊 Civic Analyst", "👔 City Operations Manager"])
st.sidebar.markdown("---")

if 'fetched_town' in df.columns:
    towns = ['All Gloucestershire'] + sorted(list(df['fetched_town'].unique()))
    selected_town = st.sidebar.selectbox("Target District", towns)
else:
    selected_town = 'All Gloucestershire'

if selected_town != 'All Gloucestershire':
    town_data = df[df['fetched_town'] == selected_town].copy()
else:
    town_data = df.copy()

# --- MODE 1: CIVIC ANALYST (Trend & Reputation) ---
if user_role == "📊 Civic Analyst":
    st.title(f"📊 Civic Health Monitor: {selected_town}")
    st.markdown("Analyze crime trends impacting local business and reputation.")
    
    # Inputs
    all_cats = list(df['category'].unique())
    # Default to Shoplifting as it affects Council revenue most
    default_index = all_cats.index('shoplifting') if 'shoplifting' in all_cats else 0
    selected_cat = st.sidebar.selectbox("Issue Category", all_cats, index=default_index)
    future_date = st.sidebar.date_input("Forecast Date", datetime.date(2026, 6, 1))
    
    # Logic
    cat_data = town_data[town_data['category'] == selected_cat].copy()
    if not cat_data.empty:
        cat_data = cat_data[cat_data['date_obj'] >= cat_data['date_obj'].min()]
        trend_data = cat_data.groupby(['date_obj', 'time_step', 'month_num']).size().reset_index(name='count')
        
        if len(trend_data) > 6:
            X = trend_data[['time_step']]
            y = trend_data['count']
            model = LinearRegression()
            model.fit(X, y)
            trend_data['trend_line'] = model.predict(X)
            trend_data['seasonal_diff'] = trend_data['count'] - trend_data['trend_line']
            seasonal_adjustments = trend_data.groupby('month_num')['seasonal_diff'].mean()
            
            last_date = trend_data['date_obj'].max()
            last_step = trend_data['time_step'].max()
            future_rows = []
            curr_date = last_date
            curr_step = last_step
            
            while curr_date.date() < future_date:
                curr_date += relativedelta(months=1)
                curr_step += 1
                base = model.predict([[curr_step]])[0]
                adj = seasonal_adjustments.get(curr_date.month, 0)
                val = int(base + adj)
                future_rows.append({'date_obj': curr_date, 'count': max(0, val), 'type': 'Forecast'})
                
            full_chart = pd.concat([trend_data.assign(type='Actual')[['date_obj', 'count', 'type']], pd.DataFrame(future_rows)])
            predicted_vol = future_rows[-1]['count'] if future_rows else 0
            
            # REPUTATION SCORE LOGIC
            # If crime goes down, Score goes up.
            start_vol = trend_data.iloc[0]['count']
            if predicted_vol < start_vol:
                rep_direction = "IMPROVING"
                rep_color = "normal" # Green
            else:
                rep_direction = "DECLINING"
                rep_color = "inverse" # Red
                
        else:
            full_chart = trend_data.assign(type='Actual')[['date_obj', 'count', 'type']]
            predicted_vol = 0
            rep_direction = "STABLE"
            
        col1, col2 = st.columns([3, 1])
        with col1:
            chart = alt.Chart(full_chart).mark_line(point=True).encode(
                x=alt.X('date_obj:T', axis=alt.Axis(format='%b %Y', labelAngle=-45), title='Timeline'),
                y=alt.Y('count:Q', scale=alt.Scale(zero=False), title='Incident Volume'),
                color=alt.Color('type', scale=alt.Scale(range=['#1f77b4', '#ff0000']))
            ).properties(height=350, title="Civic Safety Trajectory").interactive()
            st.altair_chart(chart, use_container_width=True)
            
        with col2:
            st.metric("Forecasted Incidents", predicted_vol)
            st.metric("Public Safety Score", rep_direction, delta_color=rep_color)
            
            target_month = future_date.strftime("%B")
            hist_pool = cat_data[cat_data['date_obj'].dt.month_name() == target_month]
            if len(hist_pool) == 0: hist_pool = cat_data
            
            if predicted_vol > 0 and not hist_pool.empty:
                map_data = hist_pool.sample(n=predicted_vol, replace=True, random_state=42)
                st.subheader("Hotspot Map")
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(latitude=map_data['latitude'].mean(), longitude=map_data['longitude'].mean(), zoom=11),
                    layers=[pdk.Layer("ScatterplotLayer", map_data, get_position=['longitude', 'latitude'], get_color='color', get_radius=50, opacity=0.6, filled=True)]
                ))
            else:
                st.warning("Low prediction volume.")
    else:
        st.warning("No data for this category.")

# --- MODE 2: CITY OPERATIONS MANAGER (COUNCIL EDITION) ---
elif user_role == "👔 City Operations Manager":
    st.title(f"👔 Resource & Budget Optimization: {selected_town}")
    st.markdown("Optimize Street Warden deployment to protect the High Street economy.")

    col_main1, col_main2 = st.columns([1, 2])
    
    with col_main1:
        st.subheader("1. Deployment Settings")
        op_date = st.date_input("Target Month", datetime.date(2026, 1, 1))
        target_month_name = op_date.strftime("%B")
        
        st.markdown("---")
        st.subheader("2. Cost Factors")
        
        # COUNCIL REALITY: Wardens are cheaper than Police (£20 vs £60)
        # But Agency staff (Contractors) are expensive (£35)
        warden_rate = st.number_input("In-House Warden Rate (£/hr)", value=20, help="Cost of Council employee")
        agency_rate = st.number_input("Agency Contractor Rate (£/hr)", value=35, help="Cost of hiring external security")
        
        shift_duration = st.number_input("Shift Duration (Hours)", value=8)
        days_active = st.slider("Active Days per Month", 4, 31, 26, help="Mon-Sat (High Street Hours)")
        
        total_monthly_hours = shift_duration * days_active
        st.caption(f"Total Hours per Warden: {total_monthly_hours}")

    historical_ops_data = town_data[town_data['date_obj'].dt.month_name() == target_month_name]
    
    if not historical_ops_data.empty:
        # Filter for "Street Level" crimes relevant to wardens (Shoplifting, Anti-Social)
        # Wardens don't deal with Burglary or Cyber crime
        warden_crimes = historical_ops_data[historical_ops_data['category'].isin(['shoplifting', 'anti-social-behaviour', 'drugs', 'violence-and-sexual-offences', 'public-order'])]
        
        if warden_crimes.empty:
            warden_crimes = historical_ops_data # Fallback
            
        avg_monthly_volume = len(warden_crimes) / warden_crimes['date_obj'].dt.year.nunique()
        
        with col_main2:
            st.subheader("3. Optimization Engine")
            
            # Capacity: Wardens walk beat. Can handle ~5 incidents/shift interaction
            warden_capacity = 5
            
            units_needed = int(avg_monthly_volume / 30 / warden_capacity * 2) # Rough heuristic for daily need
            if units_needed < 2: units_needed = 2
            
            # SLIDER
            deployed_units = st.slider("Deploy Street Wardens", 1, 15, units_needed)
            
            # --- BUSINESS VALUE LOGIC ---
            # Assumption: Each incident prevented saves the High Street money.
            # Avg Shoplifting value = £50. Avg Anti-Social damage = £100.
            # Weighted avg value saved per incident:
            avg_value_saved = 75 
            
            # If we cover the demand, we save value. If we are understaffed, we lose value.
            coverage_ratio = deployed_units / units_needed
            if coverage_ratio > 1: coverage_ratio = 1 # Can't save more than 100%
            
            protected_value = int(avg_monthly_volume * coverage_ratio * avg_value_saved)
            
            # --- COST LOGIC ---
            in_house_cost = deployed_units * total_monthly_hours * warden_rate
            agency_cost = deployed_units * total_monthly_hours * agency_rate
            
            # SAVINGS (In-House vs Agency)
            contractor_savings = agency_cost - in_house_cost
            
            # METRICS
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Street Incidents", int(avg_monthly_volume))
            m2.metric("Optimal Warden Count", units_needed)
            m3.metric("High Street Value Protected", f"£{protected_value:,}", help="Estimated revenue saved by preventing theft/disorder")
            
            st.markdown("---")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Agency Cost (Benchmark)", f"£{agency_cost:,.0f}")
            c2.metric("Bubbles Optimized Cost", f"£{in_house_cost:,.0f}")
            c3.metric("Efficiency Savings", f"£{contractor_savings:,.0f}", delta="Council Budget Saved")
            
            st.success(f"By optimizing deployment, you protect **£{protected_value:,}** in local business value while saving **£{contractor_savings:,}** vs agency rates.")

        # --- MAP ---
        st.subheader(f"📍 Warden Patrol Zones ({target_month_name})")
        coords = warden_crimes[['latitude', 'longitude']]
        if len(coords) >= deployed_units:
            kmeans = KMeans(n_clusters=deployed_units, random_state=42, n_init=10)
            kmeans.fit(coords)
            hubs = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
            
            # Risk = Red (Crime)
            layer_risk = pdk.Layer("ScatterplotLayer", warden_crimes, get_position=['longitude', 'latitude'], get_color=[255, 0, 0, 80], get_radius=30)
            # Wardens = Green (Safety)
            layer_hubs = pdk.Layer("ScatterplotLayer", hubs, get_position=['longitude', 'latitude'], get_color=[0, 255, 0, 255], get_radius=250, pickable=True)
            
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=coords['latitude'].mean(), longitude=coords['longitude'].mean(), zoom=13),
                layers=[layer_risk, layer_hubs],
                tooltip={"text": "Warden Start Point"}
            ))
            st.caption(f"Green zones indicate optimal start points for {deployed_units} Street Wardens to deter retail crime.")
            
    else:
        st.error(f"No historical data available for {target_month_name}.")