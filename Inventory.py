import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import re
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

# Custom CSS
st.markdown("""
<style>
.graph-description { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-style: italic; border-left: 4px solid #1f77b4; }
.metric-container { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.status-excess { background-color: #ffebee; border-left: 4px solid #f44336; }
.status-short { background-color: #fff3e0; border-left: 4px solid #ff9800; }
.status-normal { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
</style>
""", unsafe_allow_html=True)

# --- HELPER: ROBUST PART NUMBER NORMALIZATION ---
def normalize_part_key(part_no):
    """
    Creates a standardized key for matching.
    Removes spaces, dashes, dots, and ensures uppercase.
    Example: '123-456 A' -> '123456A'
    """
    if part_no is None: return ""
    # Convert to string
    s = str(part_no).strip().upper()
    # Remove float artifacts (e.g., "101.0" -> "101")
    if s.endswith(".0"):
        s = s[:-2]
    # Remove all non-alphanumeric characters (keep only A-Z and 0-9)
    clean_s = re.sub(r'[^A-Z0-9]', '', s)
    return clean_s

class DataPersistence:
    @staticmethod
    def save_data_to_session_state(key, data):
        st.session_state[key] = {'data': data, 'timestamp': datetime.now(), 'saved': True}
    
    @staticmethod
    def load_data_from_session_state(key):
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('data')
        return None

class InventoryAnalyzer:
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
        """
        Calculate norms keyed by the CLEAN Part Number.
        """
        master_requirements = {} # Key: clean_part_no

        for bom_name, bom_rows in boms_data.items():
            production_count = production_plan.get(bom_name, 0)
            
            if production_count > 0:
                for item in bom_rows:
                    # Use the Clean Key generated during standardization
                    clean_key = item.get('clean_key')
                    
                    # Fallback normalization if missing
                    if not clean_key: 
                        clean_key = normalize_part_key(item.get('Part_No'))

                    qty_per_veh = self.safe_float_convert(item.get('Qty_Veh', 0))
                    total_req_for_bom = qty_per_veh * production_count
                    
                    if clean_key in master_requirements:
                        master_requirements[clean_key]['RM_IN_QTY'] += total_req_for_bom
                        if bom_name not in master_requirements[clean_key]['Aggregates']:
                            master_requirements[clean_key]['Aggregates'].append(bom_name)
                    else:
                        master_requirements[clean_key] = {
                            'clean_key': clean_key,
                            'Display_Part_No': item.get('Part_No'), # Keep original for display
                            'Description': item.get('Description', ''),
                            'RM_IN_QTY': total_req_for_bom,
                            'Aggregates': [bom_name],
                            'Vendor_Code': item.get('Vendor_Code', ''),
                            'Vendor_Name': item.get('Vendor_Name', 'Unknown')
                        }
        return master_requirements

    def analyze_inventory(self, boms_data, inventory_data, production_plan, tolerance=None):
        if tolerance is None: tolerance = st.session_state.get("admin_tolerance", 30)

        # 1. Get Requirements (Keyed by Clean Key)
        required_parts_dict = self.calculate_dynamic_norms(boms_data, production_plan)
        
        # 2. Map Inventory (Keyed by Clean Key)
        inventory_dict = {}
        for item in inventory_data:
            clean_key = item.get('clean_key')
            if not clean_key: clean_key = normalize_part_key(item.get('Part_No'))
            
            # If duplicates exist in inventory file, sum them up
            if clean_key in inventory_dict:
                inventory_dict[clean_key]['Current_QTY'] += item.get('Current_QTY', 0)
                inventory_dict[clean_key]['Current Inventory - VALUE'] += item.get('Current Inventory - VALUE', 0)
            else:
                inventory_dict[clean_key] = item.copy()

        results = []
        
        # 3. Create Union of all keys (Requirements + Inventory)
        all_keys = set(required_parts_dict.keys()) | set(inventory_dict.keys())

        for key in all_keys:
            try:
                req_data = required_parts_dict.get(key, {})
                inv_data = inventory_dict.get(key, {})
                
                # Determine Display Name (Prefer Inventory, fallback to BOM)
                display_part_no = inv_data.get('Part_No') or req_data.get('Display_Part_No') or key
                part_desc = inv_data.get('Description') or req_data.get('Description') or "Unknown"
                vendor_name = inv_data.get('Vendor_Name') or req_data.get('Vendor_Name') or "Unknown"
                
                # Values
                rm_qty_norm = float(req_data.get('RM_IN_QTY', 0))
                current_qty = float(inv_data.get('Current_QTY', 0))
                current_value = float(inv_data.get('Current Inventory - VALUE', 0))
                
                # Calculate Unit Price
                unit_price = 0.0
                if current_qty > 0:
                    unit_price = current_value / current_qty
                
                # Bounds
                lower_bound = rm_qty_norm * (1 - tolerance / 100)
                upper_bound = rm_qty_norm * (1 + tolerance / 100)

                status = 'Within Norms'
                deviation_qty = 0
                deviation_value = 0

                # Logic:
                # 1. If in BOM but 0 Inventory -> Short
                # 2. If in Inventory but 0 BOM (Norm=0) -> Excess
                # 3. If in both, check bounds

                if current_qty < lower_bound:
                    status = 'Short Inventory'
                    deviation_qty = lower_bound - current_qty
                    deviation_value = deviation_qty * unit_price * -1 # Cost to buy
                elif current_qty > upper_bound:
                    status = 'Excess Inventory'
                    deviation_qty = current_qty - upper_bound
                    deviation_value = deviation_qty * unit_price # Value of excess
                
                # If Norm is 0 and we have stock, it's definitely Excess
                if rm_qty_norm == 0 and current_qty > 0:
                    status = 'Excess Inventory'
                    deviation_qty = current_qty
                    deviation_value = current_value

                result = {
                    'PART NO': display_part_no,
                    'PART DESCRIPTION': part_desc,
                    'Vendor Name': vendor_name,
                    'Used In Aggregates': ", ".join(req_data.get('Aggregates', [])) if req_data else "Not in BOM",
                    
                    'RM Norm - In Qty': rm_qty_norm,
                    'Lower Bound Qty': lower_bound,
                    'Upper Bound Qty': upper_bound,
                    
                    'UNIT PRICE': unit_price,
                    'Current Inventory - Qty': current_qty,
                    'Current Inventory - VALUE': current_value,
                    
                    'Stock Deviation Value': deviation_value,
                    'Status': status,
                    'INVENTORY REMARK STATUS': status
                }
                results.append(result)
            except Exception:
                continue
                
        return results

    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color):
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        vendor_totals = {}
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            val = abs(item.get('Stock Deviation Value', 0))
            vendor_totals[vendor] = vendor_totals.get(vendor, 0) + val
            
        if not vendor_totals: return
        
        sorted_vendors = sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        vendors, values = zip(*sorted_vendors)
        plot_values = [v/100000 for v in values] # Lakhs
        
        fig = go.Figure(go.Bar(
            x=vendors, y=plot_values, marker_color=color,
            text=[f"{v:.2f}L" for v in plot_values], textposition='auto'
        ))
        fig.update_layout(title=chart_title, yaxis_title="Value (₹ Lakhs)")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

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
            'persistent_analysis_results', 'production_plan'
        ]
        for key in self.persistent_keys:
            if key not in st.session_state: st.session_state[key] = None

    def safe_float_convert(self, value):
        return self.analyzer.safe_float_convert(value)

    # --- UPDATED STANDARDIZATION WITH CLEAN KEYS ---
    def standardize_bom_data(self, df, aggregate_name):
        if df is None or df.empty: return []
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        col_map = {
            'part_no': ['part no', 'part_no', 'part number', 'item code', 'material'],
            'description': ['description', 'part description', 'material description'],
            'qty_veh': ['qty/veh', 'qty per veh', 'qty', 'quantity', 'usage', 'qty_veh'],
            'vendor_name': ['vendor name', 'vendor', 'name']
        }
        
        found = {}
        for target, aliases in col_map.items():
            for alias in aliases:
                if alias in df.columns: found[target] = alias; break
        
        if 'part_no' not in found or 'qty_veh' not in found:
            st.error(f"❌ {aggregate_name}: Missing Part No or Qty/Veh"); return []

        std_data = []
        for _, row in df.iterrows():
            try:
                p_no = str(row[found['part_no']]).strip()
                if not p_no or p_no.lower() == 'nan': continue
                
                # Generate Clean Key here
                clean_key = normalize_part_key(p_no)
                
                item = {
                    'Part_No': p_no,
                    'clean_key': clean_key,
                    'Description': str(row.get(found.get('description'), '')).strip(),
                    'Qty_Veh': self.safe_float_convert(row[found['qty_veh']]),
                    'Vendor_Name': str(row.get(found.get('vendor_name'), '')).strip(),
                    'Aggregate': aggregate_name
                }
                std_data.append(item)
            except Exception: continue
        return std_data

    def standardize_current_inventory(self, df):
        if df is None or df.empty: return []
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        col_map = {
            'part_no': ['part no', 'part_no', 'material', 'item code'],
            'current_qty': ['stock', 'qty', 'current_qty', 'quantity', 'unrestricted'],
            'value': ['value', 'amount', 'total value', 'stock value', 'current inventory - value'],
            'desc': ['description', 'material description'],
            'vendor': ['vendor', 'vendor name']
        }
        
        found = {}
        for target, aliases in col_map.items():
            for alias in aliases:
                if alias in df.columns: found[target] = alias; break
        
        if 'part_no' not in found or 'current_qty' not in found:
            st.error("❌ Inventory missing Part No or Qty"); return []

        std_data = []
        for _, row in df.iterrows():
            try:
                p_no = str(row[found['part_no']]).strip()
                if not p_no or p_no.lower() == 'nan': continue
                
                # Generate Clean Key here
                clean_key = normalize_part_key(p_no)
                
                val = 0.0
                if 'value' in found: val = self.safe_float_convert(row[found['value']])
                
                item = {
                    'Part_No': p_no,
                    'clean_key': clean_key,
                    'Current_QTY': self.safe_float_convert(row[found['current_qty']]),
                    'Current Inventory - VALUE': val,
                    'Description': str(row.get(found.get('desc'), '')).strip(),
                    'Vendor_Name': str(row.get(found.get('vendor'), '')).strip()
                }
                std_data.append(item)
            except Exception: continue
        return std_data

    # --- UI METHODS ---
    def authenticate_user(self):
        st.sidebar.markdown("### 🔐 Authentication")
        if st.session_state.user_role is None:
            role = st.sidebar.selectbox("Role", ["Select", "Admin", "User"])
            if role == "Admin":
                pwd = st.sidebar.text_input("Password", type="password")
                if st.sidebar.button("Login"):
                    if pwd == "Agilomatrix@123":
                        st.session_state.user_role = "Admin"
                        st.rerun()
                    else: st.sidebar.error("Wrong Password")
            elif role == "User":
                if st.sidebar.button("Enter User"):
                    st.session_state.user_role = "User"
                    st.rerun()
        else:
            st.sidebar.success(f"Logged: {st.session_state.user_role}")
            if st.sidebar.button("Logout"):
                st.session_state.clear()
                st.rerun()

    def admin_data_management(self):
        st.header("🔧 Admin - BOM Management")
        
        # Tolerance
        if "admin_tolerance" not in st.session_state: st.session_state.admin_tolerance = 30
        st.session_state.admin_tolerance = st.selectbox("Tolerance %", [0,10,20,30,40,50], 
            index=[0,10,20,30,40,50].index(st.session_state.admin_tolerance))
        
        if st.session_state.get('persistent_boms_locked', False):
            st.warning("Locked."); 
            if st.button("Unlock"): 
                st.session_state.persistent_boms_locked = False; st.rerun()
        else:
            files = st.file_uploader("Upload BOMs", accept_multiple_files=True)
            if files:
                if st.button("Process"):
                    processed = {}
                    for f in files:
                        df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                        data = self.standardize_bom_data(df, f.name.rsplit('.', 1)[0])
                        if data: processed[f.name.rsplit('.', 1)[0]] = data
                    if processed:
                        self.persistence.save_data_to_session_state('persistent_boms_data', processed)
                        st.success("Saved!"); st.rerun()
            
            boms = self.persistence.load_data_from_session_state('persistent_boms_data')
            if boms:
                st.write(f"Loaded: {', '.join(boms.keys())}")
                if st.button("Lock"):
                    st.session_state.persistent_boms_locked = True; st.rerun()

    def user_inventory_upload(self):
        st.header("📦 Inventory Analysis")
        boms = self.persistence.load_data_from_session_state('persistent_boms_data')
        if not boms or not st.session_state.get('persistent_boms_locked'):
            st.error("Admin data missing."); return

        st.subheader("1. Production Plan")
        cols = st.columns(len(boms))
        plan = {}
        if 'current_plan' not in st.session_state: st.session_state.current_plan = {}
        
        for idx, (name, _) in enumerate(boms.items()):
            with cols[idx%3]:
                val = st.number_input(name, min_value=0, value=st.session_state.current_plan.get(name, 0), key=f"p_{idx}")
                plan[name] = val
                st.session_state.current_plan[name] = val
        
        st.subheader("2. Inventory")
        if st.session_state.get('persistent_inventory_locked'):
            st.success("Loaded.")
            if st.button("Reset"):
                st.session_state.persistent_inventory_locked = False; st.rerun()
            self.run_analysis(boms, plan)
        else:
            f = st.file_uploader("Upload Inventory")
            if f:
                df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                data = self.standardize_current_inventory(df)
                if data:
                    if st.button("Analyze"):
                        self.persistence.save_data_to_session_state('persistent_inventory_data', data)
                        st.session_state.persistent_inventory_locked = True; st.rerun()

    def run_analysis(self, boms, plan):
        inv = self.persistence.load_data_from_session_state('persistent_inventory_data')
        if sum(plan.values()) == 0: st.warning("Enter Production Plan"); return
        
        results = self.analyzer.analyze_inventory(boms, inv, plan)
        
        if results:
            # Dashboard
            df = pd.DataFrame(results)
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Items", len(df))
            c2.metric("Excess Items", len(df[df['Status']=='Excess Inventory']))
            c3.metric("Excess Value", f"₹{df[df['Status']=='Excess Inventory']['Stock Deviation Value'].sum():,.0f}")
            c4.metric("Shortage Value", f"₹{abs(df[df['Status']=='Short Inventory']['Stock Deviation Value'].sum()):,.0f}")
            
            t1, t2, t3 = st.tabs(["Short", "Excess", "All"])
            with t1:
                st.dataframe(df[df['Status']=='Short Inventory'])
                self.analyzer.show_vendor_chart_by_status(results, "Short Inventory", "Shortage Value", "k1", "#F44336")
            with t2:
                st.dataframe(df[df['Status']=='Excess Inventory'])
                self.analyzer.show_vendor_chart_by_status(results, "Excess Inventory", "Excess Value", "k2", "#2196F3")
            with t3: st.dataframe(df)

    def run(self):
        self.authenticate_user()
        if st.session_state.user_role == "Admin": self.admin_data_management()
        elif st.session_state.user_role == "User": self.user_inventory_upload()

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
