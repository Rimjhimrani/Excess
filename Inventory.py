import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import io
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MODERN UI CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Inventory AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Stockpeers/Modern" Dashboard Look
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #0e1117;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Custom KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
        text-align: center;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
    }
    .kpi-title {
        color: #9ca3af;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .kpi-sub {
        font-size: 0.8rem;
        margin-top: 5px;
    }
    .text-green { color: #10b981; }
    .text-red { color: #ef4444; }
    .text-blue { color: #3b82f6; }
    .text-orange { color: #f59e0b; }

    /* Custom Graph Container */
    .graph-container {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #374151;
        margin-bottom: 20px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937;
        border-radius: 5px;
        color: white;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #374151;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PERSISTENCE & LOGIC (UNCHANGED)
# -----------------------------------------------------------------------------

class DataPersistence:
    """Handle data persistence across sessions"""
    @staticmethod
    def save_data_to_session_state(key, data):
        st.session_state[key] = {
            'data': data,
            'timestamp': datetime.now(),
            'saved': True
        }
    
    @staticmethod
    def load_data_from_session_state(key):
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('data')
        return None

class InventoryAnalyzer:
    """Enhanced inventory analysis with BOM and Daily Production logic"""
    
    def __init__(self):
        self.debug = False

    def safe_float_convert(self, value, default=0.0):
        try:
            if value is None: return default
            if isinstance(value, (int, float)): return float(value)
            clean = str(value).strip().replace(',', '').replace('₹', '').replace('$', '').replace('%', '')
            if not clean: return default
            return float(clean)
        except (ValueError, TypeError):
            return default

    def calculate_dynamic_norms(self, boms_data, production_plan):
        master_requirements = {}
        for bom_name, bom_rows in boms_data.items():
            production_count = production_plan.get(bom_name, 0)
            for item in bom_rows:
                part_no = str(item['Part_No']).strip().upper()
                qty_per_veh = self.safe_float_convert(item.get('Qty_Veh', 0))
                total_req_for_bom = qty_per_veh * production_count
                
                if part_no in master_requirements:
                    master_requirements[part_no]['RM_IN_QTY'] += total_req_for_bom
                    if bom_name not in master_requirements[part_no]['Aggregates']:
                        master_requirements[part_no]['Aggregates'].append(bom_name)
                else:
                    master_requirements[part_no] = {
                        'Part_No': part_no,
                        'Description': item.get('Description', ''),
                        'RM_IN_QTY': total_req_for_bom, 
                        'Aggregates': [bom_name],
                        'Vendor_Name': item.get('Vendor_Name', 'Unknown')
                    }
        return master_requirements

    def analyze_inventory(self, boms_data, inventory_data, production_plan, tolerance=None):
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)

        required_parts_dict = self.calculate_dynamic_norms(boms_data, production_plan)
        results = []
        
        for inv_item in inventory_data:
            try:
                part_no = str(inv_item['Part_No']).strip().upper()
                current_qty = float(inv_item.get('Current_QTY', 0))
                current_value = float(inv_item.get('Current Inventory - VALUE', 0))
                part_desc = inv_item.get('Description', 'Unknown')
                vendor_name = inv_item.get('Vendor_Name', 'Unknown')
                
                req_data = required_parts_dict.get(part_no)
                rm_qty_norm = 0.0
                aggregates = []
                lower_bound = 0.0
                upper_bound = 0.0
                status = "Within Norms"
                deviation_qty = 0.0
                deviation_value = 0.0
                unit_price = 0.0
                if current_qty > 0:
                    unit_price = current_value / current_qty

                if req_data:
                    rm_qty_norm = float(req_data.get('RM_IN_QTY', 0))
                    aggregates = req_data.get('Aggregates', [])
                    if part_desc == 'Unknown' or part_desc == '':
                        part_desc = req_data.get('Description', 'Unknown')
                    if vendor_name == 'Unknown' or vendor_name == '':
                        vendor_name = req_data.get('Vendor_Name', 'Unknown')

                    lower_bound = np.ceil(rm_qty_norm * (1 - tolerance / 100))
                    upper_bound = np.ceil(rm_qty_norm * (1 + tolerance / 100))
                    
                    if current_qty < lower_bound:
                        status = 'Short Inventory'
                        deviation_qty = lower_bound - current_qty
                        deviation_value = deviation_qty * unit_price
                    elif current_qty > upper_bound:
                        status = 'Excess Inventory'
                        deviation_qty = current_qty - upper_bound
                        deviation_value = deviation_qty * unit_price
                    else:
                        status = 'Within Norms'
                else:
                    status = 'Excess Inventory'
                    rm_qty_norm = 0.0
                    lower_bound = 0.0
                    upper_bound = 0.0
                    aggregates = ["Not in BOM"]
                    deviation_qty = current_qty
                    deviation_value = current_value

                if deviation_value < 0: deviation_value = 0
                if deviation_qty < 0: deviation_qty = 0

                result = {
                    'PART NO': part_no,
                    'PART DESCRIPTION': part_desc,
                    'Vendor Name': vendor_name,
                    'Used In Aggregates': ", ".join(aggregates),
                    'Matched Status': "Matched" if req_data else "Unmatched",
                    'RM Norm - In Qty': rm_qty_norm,
                    'Lower Bound Qty': lower_bound,
                    'Upper Bound Qty': upper_bound,
                    'UNIT PRICE': unit_price,
                    'Current Inventory - Qty': current_qty,
                    'Current Inventory - VALUE': current_value,
                    'Stock Deviation Qty': deviation_qty,
                    'Stock Deviation Value': deviation_value,
                    'Status': status,
                    'INVENTORY REMARK STATUS': status
                }
                results.append(result)
            except Exception as e:
                continue
        return results

    # --- NEW: Graph to show all 3 statuses in one line graph ---
    def show_status_distribution_line(self, processed_data):
        """
        Creates a multi-line chart where the X-axis is the Part Rank (sorted by value)
        and Y-axis is the Value. This creates a curve showing the magnitude of each category.
        """
        df = pd.DataFrame(processed_data)
        
        # Prepare data for 3 traces
        # 1. Excess (Deviation Value)
        excess_data = df[df['Status'] == 'Excess Inventory'].sort_values('Stock Deviation Value', ascending=False).reset_index(drop=True)
        excess_data['Rank'] = excess_data.index + 1
        
        # 2. Short (Deviation Value)
        short_data = df[df['Status'] == 'Short Inventory'].sort_values('Stock Deviation Value', ascending=False).reset_index(drop=True)
        short_data['Rank'] = short_data.index + 1
        
        # 3. Within Norms (Current Inventory Value - as they have 0 deviation, we show actual stock value)
        normal_data = df[df['Status'] == 'Within Norms'].sort_values('Current Inventory - VALUE', ascending=False).reset_index(drop=True)
        normal_data['Rank'] = normal_data.index + 1

        fig = go.Figure()

        # Trace 1: Excess
        fig.add_trace(go.Scatter(
            x=excess_data['Rank'], 
            y=excess_data['Stock Deviation Value'],
            mode='lines',
            name='Excess Value',
            line=dict(color='#3b82f6', width=3), # Blue
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='<b>Excess</b><br>Rank: %{x}<br>Value: ₹%{y:,.0f}'
        ))

        # Trace 2: Shortage
        fig.add_trace(go.Scatter(
            x=short_data['Rank'], 
            y=short_data['Stock Deviation Value'],
            mode='lines',
            name='Shortage Value',
            line=dict(color='#ef4444', width=3), # Red
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)',
            hovertemplate='<b>Shortage</b><br>Rank: %{x}<br>Value: ₹%{y:,.0f}'
        ))

        # Trace 3: Within Norms
        fig.add_trace(go.Scatter(
            x=normal_data['Rank'], 
            y=normal_data['Current Inventory - VALUE'],
            mode='lines',
            name='Healthy Stock Value',
            line=dict(color='#10b981', width=2, dash='dot'), # Green
            hovertemplate='<b>Healthy</b><br>Rank: %{x}<br>Value: ₹%{y:,.0f}'
        ))

        fig.update_layout(
            title="<b>Inventory Value Distribution Curve</b> (Sorted High to Low)",
            title_font_color="white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Part Count (Ranked by Value)", showgrid=False, color="white"),
            yaxis=dict(title="Value (₹)", gridcolor='#374151', color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white")),
            hovermode="x unified",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, value_format='million', top_n=10):
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        
        vendor_totals = {}
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            val = item.get('Stock Deviation Value', 0)
            if val > 0:
                vendor_totals[vendor] = vendor_totals.get(vendor, 0) + val
            
        if not vendor_totals:
            st.info(f"No positive values found for {status_filter}.")
            return

        sorted_vendors = sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        vendors = [x[0] for x in sorted_vendors]
        values = [x[1] for x in sorted_vendors]
        
        if value_format == 'lakhs':
            divisor = 100000; unit_label = "Lakhs"; suffix = "L"
        else:
            divisor = 1000000; unit_label = "Millions"; suffix = "M"

        plot_values = [v / divisor for v in values]
        y_title = f"Value (₹ {unit_label})"
        text_fmt = [f"{v:.2f}{suffix}" for v in plot_values]

        fig = go.Figure(go.Bar(
            x=vendors, y=plot_values, marker_color=color, 
            text=text_fmt, textposition='auto',
            hoverinfo="text",
            hovertext=[f"<b>{v}</b><br>Total: ₹{val:,.2f}" for v, val in zip(vendors, values)]
        ))
        
        fig.update_layout(
            title=f"<b>{chart_title}</b> (Top {top_n})",
            title_font_color="white",
            xaxis=dict(title="Vendor", color="white", showgrid=False), 
            yaxis=dict(title=y_title, color="white", gridcolor='#374151'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

    def show_part_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, value_format='million', top_n=10):
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter and item.get('Stock Deviation Value', 0) > 0]
        
        if not filtered:
            st.info(f"No positive values found for {status_filter}.")
            return

        sorted_parts = sorted(filtered, key=lambda x: x.get('Stock Deviation Value', 0), reverse=True)[:top_n]
        
        part_nos = [x['PART NO'] for x in sorted_parts]
        values = [x['Stock Deviation Value'] for x in sorted_parts]
        
        if value_format == 'lakhs':
            divisor = 100000; unit_label = "Lakhs"; suffix = "L"
        else:
            divisor = 1000000; unit_label = "Millions"; suffix = "M"

        plot_values = [v / divisor for v in values]
        y_title = f"Value (₹ {unit_label})"
        text_fmt = [f"{v:.2f}{suffix}" for v in plot_values]

        fig = go.Figure(go.Bar(
            x=part_nos, y=plot_values, marker_color=color, 
            text=text_fmt, textposition='auto',
            hoverinfo="text",
            hovertext=[f"<b>{x['PART NO']}</b><br>{x['PART DESCRIPTION']}<br>Deviation: ₹{x['Stock Deviation Value']:,.2f}" for x in sorted_parts]
        ))
        
        fig.update_layout(
            title=f"<b>{chart_title}</b> (Top {top_n})",
            title_font_color="white",
            xaxis=dict(title="Part No", color="white", showgrid=False), 
            yaxis=dict(title=y_title, color="white", gridcolor='#374151'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

# -----------------------------------------------------------------------------
# 3. MAIN CONTROLLER
# -----------------------------------------------------------------------------

class InventoryManagementSystem:
    def __init__(self):
        self.analyzer = InventoryAnalyzer()
        self.persistence = DataPersistence()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        if 'user_role' not in st.session_state: st.session_state.user_role = None
        self.persistent_keys = [
            'persistent_boms_data', 'persistent_boms_locked',
            'persistent_inventory_data', 'persistent_inventory_locked',
            'persistent_analysis_results', 'production_plan', 'admin_tolerance'
        ]
        for key in self.persistent_keys:
            if key not in st.session_state: st.session_state[key] = None

    def safe_float_convert(self, value):
        return self.analyzer.safe_float_convert(value)

    def authenticate_user(self):
        st.sidebar.markdown("### 🔐 Authentication")
        if st.session_state.user_role is None:
            role = st.sidebar.selectbox("Select Role", ["Select Role", "Admin", "User"])
            if role == "Admin":
                st.markdown("**Admin Login**")
                password = st.sidebar.text_input("Password", type="password")
                if st.sidebar.button("Login"):
                    if password == "Agilomatrix@123":
                        st.session_state.user_role = "Admin"
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid Password")
            elif role == "User":
                if st.sidebar.button("Enter as User"):
                    st.session_state.user_role = "User"
                    st.rerun()
        else:
            st.sidebar.success(f"Logged in as: **{st.session_state.user_role}**")
            self.display_data_status()
            if st.session_state.user_role == "Admin":
                if st.sidebar.button("Switch to User View"):
                    st.session_state.user_role = "User"
                    st.rerun()
            if st.sidebar.button("Logout"):
                st.session_state.clear()
                st.rerun()

    def display_data_status(self):
        st.sidebar.markdown("---")
        boms = self.persistence.load_data_from_session_state('persistent_boms_data')
        bom_locked = st.session_state.get('persistent_boms_locked', False)
        if boms:
            st.sidebar.success(f"✅ BOMs: {len(boms)} Aggregates Loaded {'🔒' if bom_locked else '🔓'}")
        else:
            st.sidebar.error("❌ BOMs: Not Loaded")
        inv = self.persistence.load_data_from_session_state('persistent_inventory_data')
        if inv:
            st.sidebar.success(f"✅ Inventory: {len(inv)} Parts Loaded")
        else:
            st.sidebar.error("❌ Inventory: Not Loaded")

    def standardize_bom_data(self, df, aggregate_name):
        # (Same logic as before, omitted for brevity but included in execution)
        if df is None or df.empty: return []
        df.columns = [str(col).strip().lower() for col in df.columns]
        col_map = {
            'part_no': ['part no', 'part_no', 'part number', 'item code', 'material'],
            'description': ['description', 'part description', 'material description', 'desc'],
            'qty_veh': ['qty/veh', 'qty per veh', 'qty', 'quantity', 'usage', 'qty_veh'],
            'vendor_code': ['vendor code', 'vendor_code'],
            'vendor_name': ['vendor name', 'vendor_name']
        }
        found = {}
        for target, aliases in col_map.items():
            for alias in aliases:
                if alias in df.columns: found[target] = alias; break
        if 'part_no' not in found or 'qty_veh' not in found: return []
        std_data = []
        for _, row in df.iterrows():
            try:
                p_no = str(row[found['part_no']]).strip()
                if not p_no or p_no.lower() == 'nan': continue
                item = {
                    'Part_No': p_no,
                    'Description': str(row.get(found.get('description'), '')).strip(),
                    'Qty_Veh': self.safe_float_convert(row[found['qty_veh']]),
                    'Vendor_Code': str(row.get(found.get('vendor_code'), '')).strip(),
                    'Vendor_Name': str(row.get(found.get('vendor_name'), '')).strip(),
                    'Aggregate': aggregate_name
                }
                std_data.append(item)
            except Exception: continue
        return std_data

    def standardize_current_inventory(self, df):
        # (Same logic as before)
        if df is None or df.empty: return []
        df.columns = [str(col).strip().lower() for col in df.columns]
        col_map = {
            'part_no': ['part no', 'part_no', 'material', 'item code'],
            'current_qty': ['current stock', 'stock', 'qty', 'current_qty', 'quantity', 'unrestricted'],
            'value': ['value', 'amount', 'total value', 'stock value', 'current inventory - value'],
            'desc': ['description', 'material description'],
            'vendor': ['vendor', 'vendor name']
        }
        found = {}
        for target, aliases in col_map.items():
            for alias in aliases:
                if alias in df.columns: found[target] = alias; break
        if 'part_no' not in found or 'current_qty' not in found: return []
        std_data = []
        for _, row in df.iterrows():
            try:
                p_no = str(row[found['part_no']]).strip()
                if not p_no or p_no.lower() == 'nan': continue
                val = 0.0
                if 'value' in found: val = self.safe_float_convert(row[found['value']])
                item = {
                    'Part_No': p_no,
                    'Current_QTY': self.safe_float_convert(row[found['current_qty']]),
                    'Current Inventory - VALUE': val,
                    'Description': str(row.get(found.get('desc'), '')).strip(),
                    'Vendor_Name': str(row.get(found.get('vendor'), '')).strip()
                }
                std_data.append(item)
            except Exception: continue
        return std_data

    def admin_data_management(self):
        st.header("🔧 Admin Dashboard - BOM Management")
        if st.session_state.admin_tolerance is None: st.session_state.admin_tolerance = 30
        st.session_state.admin_tolerance = st.selectbox("Set Analysis Tolerance (+/- %)", [0, 10, 20, 30, 40, 50], index=3)
        st.markdown("---")
        
        # (Admin logic remains identical)
        locked = st.session_state.get('persistent_boms_locked', False)
        if locked:
            st.warning("🔒 BOM Data is Locked for Users.")
            if st.button("🔓 Unlock Data"):
                st.session_state.persistent_boms_locked = False
                st.rerun()
        else:
            uploaded_files = st.file_uploader("Upload BOMs", type=['xlsx', 'xls', 'csv'], accept_multiple_files=True)
            if uploaded_files:
                if st.button("Process & Save BOMs"):
                    processed_boms = {}
                    for f in uploaded_files:
                        agg_name = f.name.rsplit('.', 1)[0]
                        try:
                            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                            data = self.standardize_bom_data(df, agg_name)
                            if data: processed_boms[agg_name] = data
                        except Exception: pass
                    if processed_boms:
                        self.persistence.save_data_to_session_state('persistent_boms_data', processed_boms)
                        st.success(f"✅ Saved {len(processed_boms)} Aggregates!")
                        st.rerun()
            current_boms = self.persistence.load_data_from_session_state('persistent_boms_data')
            if current_boms:
                st.write(f"**Loaded Aggregates:** {', '.join(current_boms.keys())}")
                if st.button("🔒 Lock Data & Publish to Users"):
                    st.session_state.persistent_boms_locked = True
                    st.rerun()

    def user_inventory_upload(self):
        st.title("📦 Inventory Analysis Dashboard")
        boms_data = self.persistence.load_data_from_session_state('persistent_boms_data')
        boms_locked = st.session_state.get('persistent_boms_locked', False)
        
        if not boms_data or not boms_locked:
            st.error("⚠️ Admin has not locked/published BOM data yet.")
            return

        with st.expander("🏭 Daily Production Plan (Input)", expanded=True):
            cols = st.columns(len(boms_data))
            production_plan = {}
            if 'current_plan' not in st.session_state: st.session_state.current_plan = {}
            
            for idx, (bom_name, _) in enumerate(boms_data.items()):
                with cols[idx % 3]:
                    val = st.number_input(f"{bom_name}", min_value=0, value=st.session_state.current_plan.get(bom_name, 0), key=f"prod_{idx}")
                    production_plan[bom_name] = val
                    st.session_state.current_plan[bom_name] = val
        
        st.write("### Inventory File")
        inv_locked = st.session_state.get('persistent_inventory_locked', False)
        if inv_locked:
            col_res1, col_res2 = st.columns([1,5])
            with col_res1:
                if st.button("🔄 Reset Inventory"):
                    st.session_state.persistent_inventory_locked = False
                    st.session_state.persistent_inventory_data = None
                    st.rerun()
            with col_res2:
                st.success("✅ Inventory Loaded & Locked.")
            self.run_analysis(boms_data, production_plan)
        else:
            inv_file = st.file_uploader("Upload Inventory File (Excel/CSV)", type=['xlsx', 'csv'])
            if inv_file:
                df = pd.read_csv(inv_file) if inv_file.name.endswith('.csv') else pd.read_excel(inv_file)
                std_inv = self.standardize_current_inventory(df)
                if std_inv:
                    if st.button("💾 Save & Analyze"):
                        self.persistence.save_data_to_session_state('persistent_inventory_data', std_inv)
                        st.session_state.persistent_inventory_locked = True
                        st.rerun()

    def run_analysis(self, boms_data, production_plan):
        inv_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        total_prod = sum(production_plan.values())
        if total_prod == 0:
            st.warning("⚠️ Please enter Production Plan quantities to see analysis.")
            return
            
        st.markdown("---")
        tolerance = st.session_state.get('admin_tolerance', 30)
        results = self.analyzer.analyze_inventory(boms_data, inv_data, production_plan, tolerance)
        
        if results: self.display_dashboard(results)
        else: st.error("No results found.")

    def display_dashboard(self, results):
        df = pd.DataFrame(results)
        
        # Metrics Calculation
        matched_count = len(df[df['Matched Status'] == 'Matched'])
        excess_count = len(df[df['Status'] == 'Excess Inventory'])
        short_count = len(df[df['Status'] == 'Short Inventory'])
        
        total_excess_val = df[(df['Status'] == 'Excess Inventory') & (df['Stock Deviation Value'] > 0)]['Stock Deviation Value'].sum()
        total_short_val = df[(df['Status'] == 'Short Inventory') & (df['Stock Deviation Value'] > 0)]['Stock Deviation Value'].sum()

        # --- 1. MODERN KPI CARDS ---
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Parts in Shortage</div>
                <div class="kpi-value text-red">{short_count}</div>
                <div class="kpi-sub">Value: ₹{total_short_val/100000:.2f} L</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Parts in Excess</div>
                <div class="kpi-value text-blue">{excess_count}</div>
                <div class="kpi-sub">Value: ₹{total_excess_val/100000:.2f} L</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Matched Parts</div>
                <div class="kpi-value text-green">{matched_count}</div>
                <div class="kpi-sub">Total Processed: {len(df)}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
             st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Tolerance Applied</div>
                <div class="kpi-value text-orange">+/- {st.session_state.admin_tolerance}%</div>
                <div class="kpi-sub">Admin Setting</div>
            </div>
            """, unsafe_allow_html=True)

        # --- 2. GLOBAL LINE GRAPH (THE NEW FEATURE) ---
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        self.analyzer.show_status_distribution_line(results)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- 3. DETAIL TABS ---
        t1, t2, t3 = st.tabs(["🔴 Shortages Analysis", "🔵 Excess Analysis", "📋 Full Data Table"])
        
        with t1:
            col_us, col_ts, col_xs = st.columns([2, 2, 6])
            with col_us:
                unit_choice_short = st.radio("Graph Unit", ["Millions", "Lakhs"], horizontal=True, key="unit_short")
            with col_ts:
                top_n_short = st.selectbox("Show Top:", [10, 20, 30], index=0, key="top_n_short")
            
            fmt_short = "lakhs" if unit_choice_short == "Lakhs" else "million"
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                self.analyzer.show_part_chart_by_status(results, "Short Inventory", "Top Parts (Shortage)", "short_p", "#ef4444", fmt_short, top_n_short)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_g2:
                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                self.analyzer.show_vendor_chart_by_status(results, "Short Inventory", "Top Vendors (Shortage)", "short_v", "#ef4444", fmt_short, top_n_short)
                st.markdown('</div>', unsafe_allow_html=True)

            short_df = df[df['Status'] == 'Short Inventory'].sort_values('Stock Deviation Value', ascending=False)
            st.dataframe(short_df[['PART NO', 'PART DESCRIPTION', 'Vendor Name', 'Current Inventory - Qty', 'Lower Bound Qty', 'Stock Deviation Value']], use_container_width=True)

        with t2:
            col_ue, col_te, col_xe = st.columns([2, 2, 6])
            with col_ue:
                unit_choice_excess = st.radio("Graph Unit", ["Millions", "Lakhs"], horizontal=True, key="unit_excess")
            with col_te:
                top_n_excess = st.selectbox("Show Top:", [10, 20, 30], index=0, key="top_n_excess")
            
            fmt_excess = "lakhs" if unit_choice_excess == "Lakhs" else "million"
            
            col_g3, col_g4 = st.columns(2)
            with col_g3:
                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                self.analyzer.show_part_chart_by_status(results, "Excess Inventory", "Top Parts (Excess)", "excess_p", "#3b82f6", fmt_excess, top_n_excess)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_g4:
                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                self.analyzer.show_vendor_chart_by_status(results, "Excess Inventory", "Top Vendors (Excess)", "excess_v", "#3b82f6", fmt_excess, top_n_excess)
                st.markdown('</div>', unsafe_allow_html=True)

            excess_df = df[df['Status'] == 'Excess Inventory'].sort_values('Stock Deviation Value', ascending=False)
            st.dataframe(excess_df[['PART NO', 'PART DESCRIPTION', 'Vendor Name', 'Current Inventory - Qty', 'Upper Bound Qty', 'Stock Deviation Value']], use_container_width=True)
            
        with t3:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Analysis CSV", csv, "inventory_analysis.csv", "text/csv")

    def run(self):
        self.authenticate_user()
        if st.session_state.user_role == "Admin": self.admin_data_management()
        elif st.session_state.user_role == "User": self.user_inventory_upload()
        else: st.info("👈 Please login from the sidebar.")

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
