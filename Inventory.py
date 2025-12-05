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

# Set page configuration
st.set_page_config(
    page_title="Inventory Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.graph-description {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
    font-style: italic;
    border-left: 4px solid #1f77b4;
}
.metric-container {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.status-card {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.status-excess {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
}
.status-short {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
}
.status-normal {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

class DataPersistence:
    """Handle data persistence across sessions"""
    
    @staticmethod
    def save_data_to_session_state(key, data):
        """Save data with timestamp to session state"""
        st.session_state[key] = {
            'data': data,
            'timestamp': datetime.now(),
            'saved': True
        }
    
    @staticmethod
    def load_data_from_session_state(key):
        """Load data from session state if it exists"""
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('data')
        return None

class InventoryAnalyzer:
    """Enhanced inventory analysis with BOM and Daily Production logic"""
    
    def __init__(self):
        self.debug = False
        self.status_colors = {
            'Within Norms': '#4CAF50',    # Green
            'Excess Inventory': '#2196F3', # Blue
            'Short Inventory': '#F44336'   # Red
        }

    def safe_float_convert(self, value, default=0.0):
        try:
            if value is None: return default
            if isinstance(value, (int, float)): return float(value)
            # Clean string
            clean = str(value).strip().replace(',', '').replace('₹', '').replace('$', '').replace('%', '')
            if not clean: return default
            return float(clean)
        except (ValueError, TypeError):
            return default

    def calculate_dynamic_norms(self, boms_data, production_plan):
        """
        Calculate total required quantity for every part based on BOMs and Production Plan.
        """
        master_requirements = {}

        # Iterate through every uploaded BOM
        for bom_name, bom_rows in boms_data.items():
            # Get the user-inputted production count for this BOM
            production_count = production_plan.get(bom_name, 0)
            
            if production_count > 0:
                for item in bom_rows:
                    part_no = str(item['Part_No']).strip().upper()
                    qty_per_veh = self.safe_float_convert(item.get('Qty_Veh', 0))
                    
                    # Calculate required quantity for this BOM based on daily plan
                    total_req_for_bom = qty_per_veh * production_count
                    
                    if part_no in master_requirements:
                        master_requirements[part_no]['RM_IN_QTY'] += total_req_for_bom
                        # Track which vehicle aggregates use this part
                        if bom_name not in master_requirements[part_no]['Aggregates']:
                            master_requirements[part_no]['Aggregates'].append(bom_name)
                    else:
                        master_requirements[part_no] = {
                            'Part_No': part_no,
                            'Description': item.get('Description', ''),
                            'RM_IN_QTY': total_req_for_bom, # This is the Norm
                            'Aggregates': [bom_name],
                            'Vendor_Code': item.get('Vendor_Code', ''),
                            'Vendor_Name': item.get('Vendor_Name', 'Unknown')
                        }
        return master_requirements

    def analyze_inventory(self, boms_data, inventory_data, production_plan, tolerance=None):
        """
        Analyze inventory using Dynamic Norms.
        STRICT FIX: All deviation values are calculated as POSITIVE numbers.
        Negative calculations are removed so sums work correctly.
        """
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)

        # 1. Calculate Dynamic Norms (The Requirement)
        required_parts_dict = self.calculate_dynamic_norms(boms_data, production_plan)
        
        results = []
        
        # 2. Normalize Inventory Data Dictionary
        inventory_dict = {str(item['Part_No']).strip().upper(): item for item in inventory_data}
        
        # 3. Intersection Only: Parts in BOTH BOM and Inventory
        all_parts = set(required_parts_dict.keys()) & set(inventory_dict.keys())

        for part_no in all_parts:
            try:
                # Get Data
                req_data = required_parts_dict.get(part_no, {})
                inv_data = inventory_dict.get(part_no, {})
                
                # Basic info
                part_desc = inv_data.get('Description') or req_data.get('Description') or "Unknown"
                vendor_name = inv_data.get('Vendor_Name') or req_data.get('Vendor_Name') or "Unknown"
                
                # Quantities
                rm_qty_norm = float(req_data.get('RM_IN_QTY', 0)) # Calculated Norm
                current_qty = float(inv_data.get('Current_QTY', 0))
                
                # Financials
                current_value = float(inv_data.get('Current Inventory - VALUE', 0))
                unit_price = 0.0
                if current_qty > 0:
                    unit_price = current_value / current_qty
                
                # Bounds
                lower_bound = rm_qty_norm * (1 - tolerance / 100)
                upper_bound = rm_qty_norm * (1 + tolerance / 100)

                # Deviation Analysis
                deviation_qty = 0
                deviation_value = 0
                status = 'Within Norms'

                if current_qty < lower_bound:
                    status = 'Short Inventory'
                    # Positive Magnitude: How much are we missing?
                    deviation_qty = lower_bound - current_qty 
                    # Positive Value: Cost to fill the shortage
                    deviation_value = deviation_qty * unit_price 
                elif current_qty > upper_bound:
                    status = 'Excess Inventory'
                    # Positive Magnitude: How much extra do we have?
                    deviation_qty = current_qty - upper_bound
                    # Positive Value: Value of excess stock
                    deviation_value = deviation_qty * unit_price
                
                # Double check to ensure we only have positive values
                if deviation_value < 0:
                    deviation_value = 0

                # Create Result Record
                result = {
                    'PART NO': part_no,
                    'PART DESCRIPTION': part_desc,
                    'Vendor Name': vendor_name,
                    'Vendor_Code': inv_data.get('Vendor_Code', ''),
                    'Used In Aggregates': ", ".join(req_data.get('Aggregates', [])),
                    
                    'RM Norm - In Qty': rm_qty_norm,
                    'Lower Bound Qty': lower_bound,
                    'Upper Bound Qty': upper_bound,
                    
                    'UNIT PRICE': unit_price,
                    'Current Inventory - Qty': current_qty,
                    'Current Inventory - VALUE': current_value,
                    
                    'Stock Deviation Qty': deviation_qty,
                    'Stock Deviation Value': deviation_value, # STRICTLY POSITIVE
                    'Status': status,
                    'INVENTORY REMARK STATUS': status
                }
                results.append(result)
            except Exception as e:
                continue
                
        return results

    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, value_format='lakhs'):
        """Show top 10 vendors by deviation value (Strictly Positive)"""
        # Filter by status
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        
        vendor_totals = {}
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            val = item.get('Stock Deviation Value', 0)
            
            # STRICT FILTER: Ignore negative values
            if val > 0:
                vendor_totals[vendor] = vendor_totals.get(vendor, 0) + val
            
        if not vendor_totals:
            st.info(f"No positive values found for {status_filter}.")
            return

        # Sort top 10
        sorted_vendors = sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        vendors = [x[0] for x in sorted_vendors]
        values = [x[1] for x in sorted_vendors]
        
        # Formatting
        if value_format == 'lakhs':
            plot_values = [v/100000 for v in values]
            y_title = "Value (₹ Lakhs)"
            text_fmt = [f"{v:.2f}L" for v in plot_values]
        else:
            plot_values = values
            y_title = "Value (₹)"
            text_fmt = [f"{v:,.0f}" for v in plot_values]

        fig = go.Figure(go.Bar(
            x=vendors,
            y=plot_values,
            marker_color=color,
            text=text_fmt,
            textposition='auto'
        ))
        
        fig.update_layout(
            title=chart_title,
            xaxis_title="Vendor",
            yaxis_title=y_title
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)


class InventoryManagementSystem:
    """Main Application Controller"""
    
    def __init__(self):
        self.analyzer = InventoryAnalyzer()
        self.persistence = DataPersistence()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'default_tolerance': 30,
                'chart_theme': 'plotly'
            }
        
        # Persistent Data Keys for BOM System
        self.persistent_keys = [
            'persistent_boms_data',    # Stores multiple BOMs: {'ModelA': [rows], 'ModelB': [rows]}
            'persistent_boms_locked',
            'persistent_inventory_data', 
            'persistent_inventory_locked',
            'persistent_analysis_results',
            'production_plan',
            'admin_tolerance'
        ]
        
        for key in self.persistent_keys:
            if key not in st.session_state:
                st.session_state[key] = None

    def safe_float_convert(self, value):
        return self.analyzer.safe_float_convert(value)

    def authenticate_user(self):
        """Authentication and Role Switching"""
        st.sidebar.markdown("### 🔐 Authentication")
        
        if st.session_state.user_role is None:
            role = st.sidebar.selectbox("Select Role", ["Select Role", "Admin", "User"])
            
            if role == "Admin":
                st.markdown("**Admin Login**")
                password = st.sidebar.text_input("Password", type="password")
                if st.sidebar.button("Login"):
                    if password == "Agilomatrix@123": # Change password as needed
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
        """Display status of loaded data"""
        st.sidebar.markdown("---")
        
        # BOM Status
        boms = self.persistence.load_data_from_session_state('persistent_boms_data')
        bom_locked = st.session_state.get('persistent_boms_locked', False)
        
        if boms:
            st.sidebar.success(f"✅ BOMs: {len(boms)} Aggregates Loaded {'🔒' if bom_locked else '🔓'}")
        else:
            st.sidebar.error("❌ BOMs: Not Loaded")
            
        # Inventory Status
        inv = self.persistence.load_data_from_session_state('persistent_inventory_data')
        if inv:
            st.sidebar.success(f"✅ Inventory: {len(inv)} Parts Loaded")
        else:
            st.sidebar.error("❌ Inventory: Not Loaded")

    def standardize_bom_data(self, df, aggregate_name):
        """Standardize uploaded BOM files"""
        if df is None or df.empty: return []
        
        # Lowercase columns for matching
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        # Column Mappings
        col_map = {
            'part_no': ['part no', 'part_no', 'part number', 'item code', 'material'],
            'description': ['description', 'part description', 'material description', 'desc'],
            'qty_veh': ['qty/veh', 'qty per veh', 'qty', 'quantity', 'usage', 'qty_veh'],
            'vendor_code': ['vendor code', 'vendor_code'],
            'vendor_name': ['vendor name', 'vendor_name']
        }
        
        # Find columns
        found = {}
        for target, aliases in col_map.items():
            for alias in aliases:
                if alias in df.columns:
                    found[target] = alias
                    break
        
        if 'part_no' not in found or 'qty_veh' not in found:
            st.error(f"❌ File for '{aggregate_name}' is missing required columns: 'Part No' or 'Qty/Veh'.")
            return []
            
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
            except Exception:
                continue
        return std_data

    def standardize_current_inventory(self, df):
        """Standardize User Inventory File"""
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
                if alias in df.columns:
                    found[target] = alias
                    break
        
        if 'part_no' not in found or 'current_qty' not in found:
            st.error("❌ Inventory file missing 'Part No' or 'Qty' columns.")
            return []

        std_data = []
        for _, row in df.iterrows():
            try:
                p_no = str(row[found['part_no']]).strip()
                if not p_no or p_no.lower() == 'nan': continue
                
                val = 0.0
                if 'value' in found:
                    val = self.safe_float_convert(row[found['value']])
                
                item = {
                    'Part_No': p_no,
                    'Current_QTY': self.safe_float_convert(row[found['current_qty']]),
                    'Current Inventory - VALUE': val,
                    'Description': str(row.get(found.get('desc'), '')).strip(),
                    'Vendor_Name': str(row.get(found.get('vendor'), '')).strip()
                }
                std_data.append(item)
            except Exception:
                continue
        return std_data

    def admin_data_management(self):
        """Admin Interface"""
        st.header("🔧 Admin Dashboard - BOM Management")
        
        # Tolerance
        if st.session_state.admin_tolerance is None:
            st.session_state.admin_tolerance = 30
            
        st.session_state.admin_tolerance = st.selectbox(
            "Set Analysis Tolerance (+/- %)", 
            [0, 10, 20, 30, 40, 50],
            index=[0,10,20,30,40,50].index(st.session_state.admin_tolerance)
        )
        st.markdown("---")

        # BOM Management
        locked = st.session_state.get('persistent_boms_locked', False)
        if locked:
            st.warning("🔒 BOM Data is Locked for Users.")
            if st.button("🔓 Unlock Data"):
                st.session_state.persistent_boms_locked = False
                st.rerun()
        else:
            st.subheader("1. Upload BOM Files")
            st.info("Upload separate Excel/CSV files for each Vehicle/Aggregate (Max 5).")
            
            uploaded_files = st.file_uploader(
                "Upload BOMs", 
                type=['xlsx', 'xls', 'csv'], 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if len(uploaded_files) > 5:
                    st.error("Max 5 files allowed.")
                else:
                    if st.button("Process & Save BOMs"):
                        processed_boms = {}
                        valid = True
                        for f in uploaded_files:
                            agg_name = f.name.rsplit('.', 1)[0]
                            try:
                                df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                                data = self.standardize_bom_data(df, agg_name)
                                if data:
                                    processed_boms[agg_name] = data
                                else:
                                    valid = False
                            except Exception as e:
                                st.error(f"Error reading {f.name}: {e}")
                                valid = False
                        
                        if valid and processed_boms:
                            self.persistence.save_data_to_session_state('persistent_boms_data', processed_boms)
                            st.success(f"✅ Saved {len(processed_boms)} Aggregates!")
                            st.rerun()
            
            # Show existing data
            current_boms = self.persistence.load_data_from_session_state('persistent_boms_data')
            if current_boms:
                st.write(f"**Loaded Aggregates:** {', '.join(current_boms.keys())}")
                if st.button("🔒 Lock Data & Publish to Users"):
                    st.session_state.persistent_boms_locked = True
                    st.rerun()

    def user_inventory_upload(self):
        """User Interface"""
        st.header("📦 Inventory Analysis Dashboard")
        
        # Check Admin Data
        boms_data = self.persistence.load_data_from_session_state('persistent_boms_data')
        boms_locked = st.session_state.get('persistent_boms_locked', False)
        
        if not boms_data or not boms_locked:
            st.error("⚠️ Admin has not locked/published BOM data yet.")
            return

        # 1. Production Plan Input
        st.subheader("1. 🏭 Daily Production Plan")
        st.info("Enter the planned production quantity for each aggregate today.")
        
        cols = st.columns(len(boms_data))
        production_plan = {}
        
        if 'current_plan' not in st.session_state: st.session_state.current_plan = {}
        
        for idx, (bom_name, _) in enumerate(boms_data.items()):
            with cols[idx % 3]:
                val = st.number_input(
                    f"{bom_name}", 
                    min_value=0, 
                    value=st.session_state.current_plan.get(bom_name, 0),
                    key=f"prod_{idx}"
                )
                production_plan[bom_name] = val
                st.session_state.current_plan[bom_name] = val
        
        # 2. Inventory Upload
        st.subheader("2. 📊 Upload Current Inventory")
        inv_locked = st.session_state.get('persistent_inventory_locked', False)
        
        if inv_locked:
            st.success("✅ Inventory Loaded.")
            if st.button("🔄 Reset Inventory"):
                st.session_state.persistent_inventory_locked = False
                st.session_state.persistent_inventory_data = None
                st.rerun()
            
            # Proceed to Analysis
            self.run_analysis(boms_data, production_plan)
        else:
            inv_file = st.file_uploader("Upload Inventory File", type=['xlsx', 'csv'])
            if inv_file:
                df = pd.read_csv(inv_file) if inv_file.name.endswith('.csv') else pd.read_excel(inv_file)
                std_inv = self.standardize_current_inventory(df)
                
                if std_inv:
                    st.success(f"Read {len(std_inv)} rows.")
                    if st.button("💾 Save & Analyze"):
                        self.persistence.save_data_to_session_state('persistent_inventory_data', std_inv)
                        st.session_state.persistent_inventory_locked = True
                        st.rerun()

    def run_analysis(self, boms_data, production_plan):
        """Execute Analysis and Display Results"""
        inv_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        
        total_prod = sum(production_plan.values())
        if total_prod == 0:
            st.warning("⚠️ Please enter Production Plan quantities (Step 1) to see analysis.")
            return
            
        st.markdown("---")
        st.subheader("📈 Analysis Results")
        
        tolerance = st.session_state.get('admin_tolerance', 30)
        
        # Run Logic
        results = self.analyzer.analyze_inventory(boms_data, inv_data, production_plan, tolerance)
        
        if results:
            self.display_dashboard(results)
        else:
            st.error("No results found. Check matching Part Numbers between BOM and Inventory.")

    def display_dashboard(self, results):
        """Visualization Dashboard"""
        df = pd.DataFrame(results)
        
        # Metrics
        st.markdown("#### Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        total_parts = len(df)
        excess_count = len(df[df['Status'] == 'Excess Inventory'])
        
        # STRICT SUMMING: Only sum values that are > 0. 
        # (Though analyze_inventory should already ensure this, we add a filter here to be 100% sure)
        total_excess_val = df[(df['Status'] == 'Excess Inventory') & (df['Stock Deviation Value'] > 0)]['Stock Deviation Value'].sum()
        total_short_val = df[(df['Status'] == 'Short Inventory') & (df['Stock Deviation Value'] > 0)]['Stock Deviation Value'].sum()
        
        c1.metric("Total Parts", total_parts)
        c2.metric("Excess Parts", excess_count)
        c3.metric("Excess Value", f"₹{total_excess_val:,.0f}")
        c4.metric("Shortage Value", f"₹{total_short_val:,.0f}")
        
        st.markdown("---")
        
        # Tabs for tables
        t1, t2, t3 = st.tabs(["🔴 Shortages", "🔵 Excess", "📋 Full Details"])
        
        with t1:
            short_df = df[df['Status'] == 'Short Inventory'].sort_values('Stock Deviation Value', ascending=False)
            
            # Rename column for clarity in display
            display_short = short_df[[
                'PART NO', 'PART DESCRIPTION', 'Vendor Name', 'Current Inventory - Qty', 
                'Lower Bound Qty', 'Stock Deviation Value'
            ]].rename(columns={'Stock Deviation Value': 'Shortage Amount (₹)'})
            
            st.dataframe(display_short, use_container_width=True)
            
            # Chart
            self.analyzer.show_vendor_chart_by_status(results, "Short Inventory", "Top Vendors by Shortage Value", "short_v", "#F44336")
            
        with t2:
            excess_df = df[df['Status'] == 'Excess Inventory'].sort_values('Stock Deviation Value', ascending=False)
            
            display_excess = excess_df[[
                'PART NO', 'PART DESCRIPTION', 'Vendor Name', 'Current Inventory - Qty', 
                'Upper Bound Qty', 'Stock Deviation Value'
            ]].rename(columns={'Stock Deviation Value': 'Excess Amount (₹)'})
            
            st.dataframe(display_excess, use_container_width=True)
            
            # Chart
            self.analyzer.show_vendor_chart_by_status(results, "Excess Inventory", "Top Vendors by Excess Value", "excess_v", "#2196F3")
            
        with t3:
            st.dataframe(df, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Analysis CSV", csv, "inventory_analysis.csv", "text/csv")

    def run(self):
        st.title("🏭 Production-Based Inventory Analyzer")
        self.authenticate_user()
        
        if st.session_state.user_role == "Admin":
            self.admin_data_management()
        elif st.session_state.user_role == "User":
            self.user_inventory_upload()
        else:
            st.info("👈 Please login from the sidebar.")

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
