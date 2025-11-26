import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Inventory IMS", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; }
    .status-excess { color: #2196F3; font-weight: bold; }
    .status-short { color: #F44336; font-weight: bold; }
    .status-ok { color: #4CAF50; font-weight: bold; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STATUS_COLORS = {'Within Norms': '#4CAF50', 'Excess Inventory': '#2196F3', 'Short Inventory': '#F44336'}

# --- UTILITY CLASS ---
class Utils:
    @staticmethod
    def safe_float(val):
        try:
            return float(str(val).replace(',', '').replace('₹', '').replace('%', '').strip())
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def standardize_df(df, col_map):
        """Generic column mapper based on a dictionary of target: [synonyms]"""
        if df is None or df.empty: return []
        
        df.columns = [str(c).strip().lower() for c in df.columns]
        standardized = []
        
        # Invert map for O(1) lookup
        lookup = {}
        for target, synonyms in col_map.items():
            for syn in synonyms:
                if syn.lower() in df.columns:
                    lookup[target] = syn.lower()
                    break
        
        if 'part_no' not in lookup: return [] # Critical fail

        for _, row in df.iterrows():
            item = {}
            for target, source_col in lookup.items():
                val = row[source_col]
                # Apply specific conversions
                if target in ['rm_qty', 'unit_price', 'current_qty', 'stock_value', 'rm_days']:
                    item[target] = Utils.safe_float(val)
                else:
                    item[target] = str(val).strip() if pd.notna(val) else ""
            
            # Logic for derived fields or defaults
            if not item.get('unit_price') and item.get('stock_value') and item.get('current_qty'):
                item['unit_price'] = item['stock_value'] / item['current_qty'] if item['current_qty'] > 0 else 0
            
            if item.get('part_no'): standardized.append(item)
            
        return standardized

    @staticmethod
    def get_state(key, default=None):
        if key not in st.session_state: st.session_state[key] = default
        return st.session_state[key]

    @staticmethod
    def set_state(key, value):
        st.session_state[key] = value

# --- ANALYZER ENGINE ---
class Analyzer:
    def run_analysis(pfep_data, inventory_data, tolerance=30):
        results = []
        pfep_map = {p['part_no'].upper(): p for p in pfep_data}
        
        for inv in inventory_data:
            p_no = inv['part_no'].upper()
            pfep = pfep_map.get(p_no)
            if not pfep: continue

            curr_qty = inv.get('current_qty', 0)
            norm_qty = pfep.get('rm_qty', 0)
            unit_price = pfep.get('unit_price', 1) or inv.get('unit_price', 1)
            
            lower = norm_qty * (1 - tolerance/100)
            upper = norm_qty * (1 + tolerance/100)
            
            status = 'Within Norms'
            if curr_qty < lower: status = 'Short Inventory'
            elif curr_qty > upper: status = 'Excess Inventory'
            
            deviation = curr_qty - upper if status == 'Excess Inventory' else (lower - curr_qty if status == 'Short Inventory' else 0)
            dev_val = deviation * unit_price
            
            results.append({
                'PART NO': p_no,
                'PART DESCRIPTION': pfep.get('description', ''),
                'Vendor': pfep.get('vendor_name', 'Unknown'),
                'Status': status,
                'Current Qty': curr_qty,
                'Norm Qty': norm_qty,
                'Stock Value': curr_qty * unit_price,
                'Deviation Qty': deviation,
                'Deviation Value': dev_val,
                'Unit Price': unit_price
            })
        return results

# --- MAIN APPLICATION ---
class InventoryApp:
    def __init__(self):
        self.pfep_cols = {
            'part_no': ['part_no', 'material', 'item_code'],
            'description': ['description', 'desc'],
            'rm_qty': ['rm_in_qty', 'norm_qty', 'required_qty'],
            'unit_price': ['unit_price', 'rate', 'price', 'cost'],
            'vendor_name': ['vendor_name', 'vendor', 'supplier']
        }
        self.inv_cols = {
            'part_no': ['part_no', 'material', 'item_code'],
            'current_qty': ['current_qty', 'qty', 'stock'],
            'stock_value': ['stock_value', 'value', 'amount']
        }

    def sidebar_auth_and_config(self):
        st.sidebar.title("🔐 Access")
        role = Utils.get_state('user_role')
        
        if not role:
            sel_role = st.sidebar.selectbox("Role", ["Select", "Admin", "User"])
            if sel_role == "Admin":
                pwd = st.sidebar.text_input("Password", type="password")
                if st.sidebar.button("Login") and pwd == "admin123": # Simple auth
                    Utils.set_state('user_role', 'Admin')
                    st.rerun()
            elif sel_role == "User":
                if st.sidebar.button("Enter"):
                    Utils.set_state('user_role', 'User')
                    st.rerun()
        else:
            st.sidebar.success(f"Logged as {role}")
            if st.sidebar.button("Logout"):
                st.session_state.clear()
                st.rerun()

            # Filters
            st.sidebar.markdown("---")
            if role == "Admin":
                Utils.set_state('tolerance', st.sidebar.slider("Tolerance %", 0, 50, 30))
            
            # Data Status
            pfep = Utils.get_state('pfep_data')
            inv = Utils.get_state('inv_data')
            st.sidebar.info(f"📁 PFEP: {len(pfep) if pfep else 0} records")
            st.sidebar.info(f"📦 Inventory: {len(inv) if inv else 0} records")

    def data_upload_section(self):
        role = Utils.get_state('user_role')
        t1, t2 = st.tabs(["📁 PFEP Master", "📦 Inventory Data"])
        
        # PFEP Upload (Admin Only)
        with t1:
            if role == "Admin":
                f = st.file_uploader("Upload PFEP (Excel/CSV)", key="pfep_up")
                if f:
                    df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                    data = Utils.standardize_df(df, self.pfep_cols)
                    if data:
                        Utils.set_state('pfep_data', data)
                        st.success(f"✅ Loaded {len(data)} PFEP records")
            else:
                st.warning("Admin access required for PFEP upload.")

        # Inventory Upload
        with t2:
            if Utils.get_state('pfep_data'):
                f = st.file_uploader("Upload Inventory (Excel/CSV)", key="inv_up")
                if f:
                    df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                    data = Utils.standardize_df(df, self.inv_cols)
                    if data:
                        Utils.set_state('inv_data', data)
                        st.success(f"✅ Loaded {len(data)} Inventory records")
            else:
                st.error("Please load PFEP data first.")

    def dashboard(self):
        pfep = Utils.get_state('pfep_data')
        inv = Utils.get_state('inv_data')
        
        if not pfep or not inv: return

        # Run Analysis
        tol = Utils.get_state('tolerance', 30)
        res = Analyzer.run_analysis(pfep, inv, tol)
        df = pd.DataFrame(res)
        
        if df.empty:
            st.warning("No matching parts found between PFEP and Inventory.")
            return

        # --- METRICS ---
        st.markdown("### 📊 Executive Summary")
        c1, c2, c3, c4 = st.columns(4)
        total_val = df['Stock Value'].sum()
        excess_val = df[df['Status'] == 'Excess Inventory']['Deviation Value'].sum()
        short_val = df[df['Status'] == 'Short Inventory']['Deviation Value'].abs().sum()
        
        c1.metric("Total Items", len(df))
        c2.metric("Total Value", f"₹{total_val/1e5:.2f} L")
        c3.metric("Excess Value", f"₹{excess_val/1e5:.2f} L", delta_color="inverse")
        c4.metric("Shortage Value", f"₹{short_val/1e5:.2f} L", delta_color="inverse")

        # --- CHARTS ---
        t1, t2, t3 = st.tabs(["📉 Visual Analysis", "📋 Detailed Data", "📥 Export"])
        
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                # Top 10 by Value
                top_val = df.sort_values('Stock Value', ascending=False).head(10)
                fig = px.bar(top_val, x='PART NO', y='Stock Value', color='Status', 
                             title="Top 10 Parts by Value", color_discrete_map=STATUS_COLORS)
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                # Status Breakdown
                counts = df['Status'].value_counts()
                fig = px.pie(values=counts, names=counts.index, title="Inventory Status Distribution",
                             color=counts.index, color_discrete_map=STATUS_COLORS)
                st.plotly_chart(fig, use_container_width=True)
            
            # Vendor Analysis
            st.subheader("🏢 Top Vendors by Deviation")
            ven_df = df[df['Status'] != 'Within Norms'].groupby(['Vendor', 'Status'])['Deviation Value'].sum().reset_index()
            if not ven_df.empty:
                fig_v = px.bar(ven_df.sort_values('Deviation Value', ascending=False).head(15), 
                               x='Vendor', y='Deviation Value', color='Status', barmode='group',
                               color_discrete_map=STATUS_COLORS)
                st.plotly_chart(fig_v, use_container_width=True)

        with t2:
            st.dataframe(df.style.format({
                'Stock Value': "₹{:,.0f}", 'Deviation Value': "₹{:,.0f}", 'Current Qty': "{:,.0f}"
            }), use_container_width=True)

        with t3:
            st.subheader("Export Reports")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer) as writer:
                df.to_excel(writer, sheet_name='Full Analysis')
                df[df['Status'] == 'Excess Inventory'].to_excel(writer, sheet_name='Excess')
                df[df['Status'] == 'Short Inventory'].to_excel(writer, sheet_name='Short')
            
            st.download_button("📥 Download Analysis Excel", buffer, "inventory_analysis.xlsx")

    def run(self):
        self.sidebar_auth_and_config()
        if Utils.get_state('user_role'):
            st.title("🏭 Inventory Management System")
            self.data_upload_section()
            st.markdown("---")
            self.dashboard()
        else:
            st.info("Please Log in from the sidebar.")

if __name__ == "__main__":
    app = InventoryApp()
    app.run()
