import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Inventory Management System", page_icon="📊", layout="wide")

# Restoring the comprehensive CSS from the original code
st.markdown("""
<style>
    .metric-container { background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #666; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .status-excess { background-color: #e3f2fd; border-left: 5px solid #2196F3; padding: 10px; border-radius: 5px; }
    .status-short { background-color: #ffebee; border-left: 5px solid #F44336; padding: 10px; border-radius: 5px; }
    .status-normal { background-color: #e8f5e8; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px; }
    .highlight-box { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA HANDLER CLASS ---
class DataHandler:
    @staticmethod
    def get_column_mapping():
        """Centralized column mapping configuration"""
        return {
            'part_no': ['part_no', 'part no', 'material', 'item_code', 'code'],
            'description': ['description', 'desc', 'part_description', 'item_name'],
            'rm_qty': ['rm_in_qty', 'norm_qty', 'required_qty', 'target_qty', 'rm_qty'],
            'current_qty': ['current_qty', 'stock_qty', 'qty', 'available_qty', 'stock'],
            'unit_price': ['unit_price', 'price', 'rate', 'cost', 'unit_cost'],
            'vendor_name': ['vendor_name', 'vendor', 'supplier'],
            'category': ['category', 'part_category', 'group'],
            'stock_value': ['stock_value', 'value', 'total_value', 'amount']
        }

    @staticmethod
    def safe_float(val):
        try:
            return float(str(val).replace(',', '').replace('₹', '').replace('%', '').strip())
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def standardize_df(df, required_cols):
        """Generic standardizer that maps columns based on the config"""
        if df is None or df.empty: return []
        
        # Normalize headers
        df.columns = [str(c).strip().lower() for c in df.columns]
        mapping = DataHandler.get_column_mapping()
        
        standardized_data = []
        found_map = {}

        # 1. Map columns
        for target, synonyms in mapping.items():
            for syn in synonyms:
                if syn in df.columns:
                    found_map[target] = syn
                    break
        
        # 2. Check requirements
        missing = [col for col in required_cols if col not in found_map]
        if missing:
            return None, f"Missing columns: {', '.join(missing)}"

        # 3. Process Data
        for _, row in df.iterrows():
            item = {}
            # Extract mapped columns
            for target, source in found_map.items():
                val = row[source]
                if target in ['rm_qty', 'current_qty', 'unit_price', 'stock_value']:
                    item[target] = DataHandler.safe_float(val)
                else:
                    item[target] = str(val).strip() if pd.notna(val) else "N/A"
            
            # Logic: Derive Unit Price if missing but Value exists
            if item.get('unit_price', 0) == 0 and item.get('stock_value', 0) > 0 and item.get('current_qty', 0) > 0:
                item['unit_price'] = item['stock_value'] / item['current_qty']
            
            # Logic: Default Price
            if item.get('unit_price', 0) == 0:
                item['unit_price'] = 1.0 # Default to avoid div by zero

            if item.get('part_no') and item['part_no'] not in ['nan', 'N/A', '']:
                standardized_data.append(item)
                
        return standardized_data, None

    @staticmethod
    def load_sample_pfep():
        """Restored Sample Data Feature"""
        data = [
            {"part_no": "A001", "description": "Steel Plate 5mm", "rm_qty": 500, "unit_price": 120, "vendor_name": "MetalCorp", "category": "Raw Material"},
            {"part_no": "A002", "description": "Bolt M10", "rm_qty": 10000, "unit_price": 5, "vendor_name": "Fasteners Inc", "category": "Hardware"},
            {"part_no": "B003", "description": "Engine Oil 5L", "rm_qty": 50, "unit_price": 2500, "vendor_name": "LubeTech", "category": "Consumables"},
            {"part_no": "C004", "description": "Circuit Board X1", "rm_qty": 200, "unit_price": 4500, "vendor_name": "ElectroSystems", "category": "Electronics"},
            {"part_no": "D005", "description": "Packaging Box", "rm_qty": 5000, "unit_price": 15, "vendor_name": "PackIt", "category": "Packaging"}
        ]
        return data

    @staticmethod
    def load_sample_inventory():
        """Restored Sample Data Feature"""
        data = [
            {"part_no": "A001", "current_qty": 450, "stock_value": 54000}, # Normal
            {"part_no": "A002", "current_qty": 2000, "stock_value": 10000}, # Short
            {"part_no": "B003", "current_qty": 100, "stock_value": 250000}, # Excess
            {"part_no": "C004", "current_qty": 195, "stock_value": 877500}, # Normal
            {"part_no": "D005", "current_qty": 8000, "stock_value": 120000} # Excess
        ]
        return data

# --- 3. ANALYZER ENGINE ---
class Analyzer:
    @staticmethod
    def analyze(pfep_data, inv_data, tolerance):
        pfep_map = {p['part_no'].upper(): p for p in pfep_data}
        results = []
        
        for inv in inv_data:
            p_no = inv['part_no'].upper()
            pfep = pfep_map.get(p_no)
            
            if not pfep: continue # Skip if not in master
            
            curr_qty = inv.get('current_qty', 0)
            norm_qty = pfep.get('rm_qty', 0)
            price = pfep.get('unit_price', 1)
            
            # Tolerance Logic
            lower_bound = norm_qty * (1 - tolerance/100)
            upper_bound = norm_qty * (1 + tolerance/100)
            
            status = "Within Norms"
            deviation_qty = 0
            
            if curr_qty < lower_bound:
                status = "Short Inventory"
                deviation_qty = lower_bound - curr_qty # How much needed
            elif curr_qty > upper_bound:
                status = "Excess Inventory"
                deviation_qty = curr_qty - upper_bound # How much extra
            
            # Financials
            stock_value = curr_qty * price
            deviation_value = deviation_qty * price
            
            # Actionable Insights Logic
            action = "Monitor"
            if status == "Short Inventory" and deviation_value > 50000:
                action = "URGENT ORDER"
            elif status == "Excess Inventory" and deviation_value > 100000:
                action = "LIQUIDATE / HOLD ORDERS"
                
            results.append({
                'Part No': p_no,
                'Description': pfep.get('description', ''),
                'Category': pfep.get('category', 'General'),
                'Vendor': pfep.get('vendor_name', 'Unknown'),
                'Unit Price': price,
                'Norm Qty': norm_qty,
                'Current Qty': curr_qty,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound,
                'Status': status,
                'Stock Value': stock_value,
                'Deviation Qty': deviation_qty,
                'Deviation Value': deviation_value,
                'Action': action
            })
            
        return pd.DataFrame(results)

# --- 4. UI COMPONENTS ---
class DashboardUI:
    @staticmethod
    def render_sidebar():
        st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2821/2821637.png", width=50)
        st.sidebar.title("Controls")
        
        # Authentication Logic
        if 'user_role' not in st.session_state:
            role = st.sidebar.selectbox("Select Role", ["Select", "Admin", "User"])
            if role == "Admin":
                pwd = st.sidebar.text_input("Password", type="password")
                if st.sidebar.button("Login"):
                    if pwd == "admin123": # Simplified auth
                        st.session_state.user_role = "Admin"
                        st.rerun()
                    else:
                        st.sidebar.error("Wrong password")
            elif role == "User":
                if st.sidebar.button("Enter as User"):
                    st.session_state.user_role = "User"
                    st.rerun()
            return False
        
        st.sidebar.success(f"👤 Logged as: {st.session_state.user_role}")
        
        # Advanced Settings (Restored)
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Analysis Settings")
        tolerance = st.sidebar.slider("Tolerance (+/- %)", 0, 50, 30, help="Allowable deviation from Norm")
        critical_threshold = st.sidebar.number_input("Critical Value Threshold (₹)", 10000, 1000000, 100000)
        
        # Data Locking (Restored)
        if st.session_state.user_role == "Admin":
            locked = st.sidebar.checkbox("🔒 Lock Master Data", value=st.session_state.get('pfep_locked', False))
            st.session_state.pfep_locked = locked
        
        if st.sidebar.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()
            
        return {'tolerance': tolerance, 'critical_threshold': critical_threshold}

    @staticmethod
    def render_data_upload():
        st.header("🗂️ Data Management")
        
        t1, t2 = st.tabs(["📁 Master Data (PFEP)", "📦 Current Inventory"])
        
        # PFEP Tab
        with t1:
            is_locked = st.session_state.get('pfep_locked', False)
            if is_locked:
                st.info("🔒 Master Data is Locked by Admin.")
                if st.session_state.get('pfep_data'):
                    st.dataframe(pd.DataFrame(st.session_state.pfep_data).head())
            else:
                col1, col2 = st.columns(2)
                with col1:
                    f = st.file_uploader("Upload PFEP Excel/CSV", key="pfep_up")
                    if f:
                        df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                        data, err = DataHandler.standardize_df(df, ['part_no', 'rm_qty'])
                        if data:
                            st.session_state.pfep_data = data
                            st.success(f"✅ Loaded {len(data)} parts from file")
                        else:
                            st.error(err)
                with col2:
                    st.markdown("OR")
                    if st.button("🧪 Load Sample PFEP Data"):
                        st.session_state.pfep_data = DataHandler.load_sample_pfep()
                        st.success("✅ Sample PFEP Loaded")
                        st.rerun()

        # Inventory Tab
        with t2:
            if not st.session_state.get('pfep_data'):
                st.error("Please load Master Data (PFEP) first.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    f = st.file_uploader("Upload Inventory Excel/CSV", key="inv_up")
                    if f:
                        df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                        data, err = DataHandler.standardize_df(df, ['part_no', 'current_qty'])
                        if data:
                            st.session_state.inv_data = data
                            st.success(f"✅ Loaded {len(data)} inventory records")
                        else:
                            st.error(err)
                with col2:
                    st.markdown("OR")
                    if st.button("🧪 Load Sample Inventory"):
                        st.session_state.inv_data = DataHandler.load_sample_inventory()
                        st.success("✅ Sample Inventory Loaded")
                        st.rerun()

    @staticmethod
    def render_dashboard(df, settings):
        st.markdown("---")
        
        # --- Sidebar Filters (Restored) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Filters")
        
        # Vendor Filter
        all_vendors = list(df['Vendor'].unique())
        sel_vendors = st.sidebar.multiselect("Filter Vendor", all_vendors, default=all_vendors)
        
        # Category Filter
        all_cats = list(df['Category'].unique())
        sel_cats = st.sidebar.multiselect("Filter Category", all_cats, default=all_cats)
        
        # Apply Filters
        filtered_df = df[
            (df['Vendor'].isin(sel_vendors)) & 
            (df['Category'].isin(sel_cats))
        ]
        
        # --- Top KPIs ---
        total_val = filtered_df['Stock Value'].sum()
        excess_val = filtered_df[filtered_df['Status']=='Excess Inventory']['Deviation Value'].sum()
        short_val = filtered_df[filtered_df['Status']=='Short Inventory']['Deviation Value'].sum() # Value needed
        
        st.markdown(f"""
        <div class="highlight-box">
            <h3>📊 Executive Summary</h3>
            <b>Total Parts:</b> {len(filtered_df)} | 
            <b>Total Value:</b> ₹{total_val:,.0f} | 
            <b>Net Impact:</b> ₹{excess_val + short_val:,.0f}
        </div>
        """, unsafe_allow_html=True)
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Stock Value", f"₹{total_val/1e5:.2f} Lakhs")
        kpi2.metric("Excess Capital (Stuck)", f"₹{excess_val/1e5:.2f} Lakhs", delta="- High Risk", delta_color="inverse")
        kpi3.metric("Shortage Value (Needed)", f"₹{short_val/1e5:.2f} Lakhs", delta="Critical", delta_color="inverse")
        
        # --- Main Tabs ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Charts", "💡 Actionable Insights", "🔮 Forecasting", "📋 Detailed Data", "📥 Export"
        ])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                # Top 10 Value Chart
                top_val = filtered_df.nlargest(10, 'Stock Value')
                fig = px.bar(top_val, x='Part No', y='Stock Value', color='Status', 
                             title="🏆 Top 10 High Value Parts",
                             hover_data=['Description'],
                             color_discrete_map={'Excess Inventory':'#2196F3', 'Short Inventory':'#F44336', 'Within Norms':'#4CAF50'})
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                # Status Pie Chart
                counts = filtered_df['Status'].value_counts()
                fig = px.pie(values=counts, names=counts.index, title="🧩 Inventory Status Distribution",
                             color=counts.index,
                             color_discrete_map={'Excess Inventory':'#2196F3', 'Short Inventory':'#F44336', 'Within Norms':'#4CAF50'})
                st.plotly_chart(fig, use_container_width=True)
                
            # Vendor Analysis Chart
            st.subheader("🏢 Vendor Performance Analysis")
            vendor_perf = filtered_df[filtered_df['Status']!='Within Norms'].groupby('Vendor')['Deviation Value'].sum().reset_index()
            if not vendor_perf.empty:
                fig_v = px.bar(vendor_perf.nlargest(10, 'Deviation Value'), x='Vendor', y='Deviation Value',
                               title="Top Vendors with Deviation (Excess/Short Value)", color='Deviation Value')
                st.plotly_chart(fig_v, use_container_width=True)

        with tab2:
            st.header("💡 Actionable Insights")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("🚨 Critical Shortages (Urgent Orders)")
                urgent = filtered_df[(filtered_df['Status']=='Short Inventory') & 
                                     (filtered_df['Deviation Value'] > 5000)] # threshold
                if not urgent.empty:
                    st.dataframe(urgent[['Part No', 'Description', 'Vendor', 'Current Qty', 'Norm Qty', 'Deviation Value']].style.format({'Deviation Value': "₹{:,.0f}"}))
                else:
                    st.success("No critical shortages found.")
                    
            with col_b:
                st.subheader("💰 Excess Liquidation Candidates")
                excess = filtered_df[(filtered_df['Status']=='Excess Inventory') & 
                                     (filtered_df['Deviation Value'] > settings['critical_threshold'])]
                if not excess.empty:
                    st.dataframe(excess[['Part No', 'Description', 'Current Qty', 'Norm Qty', 'Deviation Value']].style.format({'Deviation Value': "₹{:,.0f}"}))
                else:
                    st.success("No high-value excess inventory found.")

        with tab3:
            st.header("🔮 Forecasting & Reorder (Simulated)")
            # Simulated Logic for Reorder Days
            forecast_df = filtered_df.copy()
            # Simulation: Assume daily consumption is roughly Norm Qty / 30
            forecast_df['Daily Usage'] = forecast_df['Norm Qty'] / 30 
            forecast_df['Days Cover'] = np.where(forecast_df['Daily Usage'] > 0, 
                                                 forecast_df['Current Qty'] / forecast_df['Daily Usage'], 
                                                 999)
            
            reorder_needed = forecast_df[forecast_df['Days Cover'] < 10].sort_values('Days Cover')
            
            st.warning(f"⚠️ {len(reorder_needed)} items have less than 10 days of stock cover.")
            st.dataframe(reorder_needed[['Part No', 'Description', 'Current Qty', 'Daily Usage', 'Days Cover']].style.format({'Daily Usage': "{:.1f}", 'Days Cover': "{:.1f} Days"}))

        with tab4:
            st.header("📋 Detailed Data Table")
            st.dataframe(filtered_df.style.format({
                'Unit Price': "₹{:,.2f}",
                'Stock Value': "₹{:,.0f}",
                'Deviation Value': "₹{:,.0f}",
                'Lower Bound': "{:,.0f}",
                'Upper Bound': "{:,.0f}"
            }), use_container_width=True)

        with tab5:
            st.header("📥 Export Reports")
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer) as writer:
                filtered_df.to_excel(writer, sheet_name='Full Analysis', index=False)
                filtered_df[filtered_df['Status']=='Excess Inventory'].to_excel(writer, sheet_name='Excess List', index=False)
                filtered_df[filtered_df['Status']=='Short Inventory'].to_excel(writer, sheet_name='Shortage List', index=False)
                
            st.download_button(
                label="📥 Download Complete Analysis (Excel)",
                data=buffer,
                file_name=f"Inventory_Analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )

# --- 5. MAIN EXECUTION ---
def main():
    # Render Sidebar & Get Settings
    settings = DashboardUI.render_sidebar()
    
    # If not logged in, stop here
    if not st.session_state.get('user_role'):
        return

    st.title(f"🏭 Inventory Management System ({st.session_state.user_role} View)")
    
    # Render Data Upload Section
    DashboardUI.render_data_upload()
    
    # Render Dashboard if data exists
    if st.session_state.get('pfep_data') and st.session_state.get('inv_data'):
        # Run Analysis
        try:
            results_df = Analyzer.analyze(
                st.session_state.pfep_data, 
                st.session_state.inv_data, 
                settings['tolerance']
            )
            DashboardUI.render_dashboard(results_df, settings)
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Check if your Part Numbers match between PFEP and Inventory files.")

if __name__ == "__main__":
    main()
