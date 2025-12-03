import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import io
import re
from typing import Any

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
.status-card { padding: 15px; border-radius: 8px; margin: 10px 0; }
.status-excess { background-color: #ffebee; border-left: 4px solid #f44336; }
.status-short { background-color: #fff3e0; border-left: 4px solid #ff9800; }
.status-normal { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
.success-box { background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 15px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

class DataPersistence:
    """Handle data persistence across sessions"""
    @staticmethod
    def save_data_to_session_state(key, data):
        st.session_state[key] = {'data': data, 'timestamp': datetime.now(), 'saved': True}
    
    @staticmethod
    def load_data_from_session_state(key):
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('data')
        return None
    
    @staticmethod
    def get_data_timestamp(key):
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('timestamp')
        return None

class InventoryAnalyzer:
    """Enhanced inventory analysis"""
    def __init__(self):
        self.debug = False

    def safe_float_convert(self, value, default=0.0):
        try:
            if isinstance(value, str):
                value = re.sub(r'[₹$€£,]', '', value).strip()
                if '%' in value: return float(value.replace('%', '')) / 100
            return float(value)
        except (ValueError, TypeError):
            return default

    def analyze_inventory(self, pfep_data, current_inventory, tolerance=None):
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)
        
        results = []
        # Create lookup dictionary for Inventory
        inventory_dict = {str(item['Part_No']).strip().upper(): item for item in current_inventory}
        
        # We iterate through PFEP as the Master List
        for pfep_item in pfep_data:
            part_no = str(pfep_item.get('Part_No')).strip().upper()
            
            # Get Inventory Data (or defaults if missing)
            inv_item = inventory_dict.get(part_no, {})
            current_qty = float(inv_item.get('Current_QTY', 0)) or 0.0
            
            # Get PFEP Data
            # NOTE: AVG CONSUMPTION and RM_IN_QTY might have been updated by BOM calculation
            avg_per_day = self.safe_float_convert(pfep_item.get('AVG CONSUMPTION/DAY', 0))
            rm_days = self.safe_float_convert(pfep_item.get('RM_IN_DAYS', 0))
            
            # Recalculate RM Norm in Qty based on (potentially updated) Consumption * RM Days
            rm_qty = avg_per_day * rm_days
            
            unit_price = self.safe_float_convert(pfep_item.get('unit_price', 0)) or 1.0
            
            # Calculation
            current_value = current_qty * unit_price
            revised_norm_qty = rm_qty 
            
            lower_bound = revised_norm_qty * (1 - tolerance / 100)
            upper_bound = revised_norm_qty * (1 + tolerance / 100)
            
            deviation_qty = current_qty - revised_norm_qty
            deviation_value = deviation_qty * unit_price

            if current_qty < lower_bound:
                status = 'Short Inventory'
            elif current_qty > upper_bound:
                status = 'Excess Inventory'
            else:
                status = 'Within Norms'

            results.append({
                'PART NO': part_no,
                'PART DESCRIPTION': pfep_item.get('Description', ''),
                'Vendor Name': pfep_item.get('Vendor_Name', 'Unknown'),
                'Vendor_Code': pfep_item.get('Vendor_Code', ''),
                'AVG CONSUMPTION/DAY': avg_per_day,
                'RM IN DAYS': rm_days,
                'RM Norm - In Qty': revised_norm_qty, # This is the dynamically calculated norm
                'Revised Norm Qty': revised_norm_qty,
                'Lower Bound Qty': lower_bound,
                'Upper Bound Qty': upper_bound,
                'UNIT PRICE': unit_price,
                'Current Inventory - Qty': current_qty,
                'Current Inventory - VALUE': current_value,
                'Stock Deviation Value': deviation_value,
                'Status': status,
                'INVENTORY REMARK STATUS': status
            })
            
        return results

    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, value_format='lakhs'):
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        vendor_totals = {}
        
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            val = item.get('Stock Deviation Value', 0)
            # For charts, we usually want absolute magnitude
            if status_filter == "Short Inventory" and val < 0: val = abs(val)
            vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + val
            
        if not vendor_totals: return

        top_vendors = sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [v[0] for v in top_vendors]
        raw_vals = [v[1] for v in top_vendors]
        
        factor = 100000 if value_format == 'lakhs' else 1
        vals = [v/factor for v in raw_vals]

        fig = go.Figure(go.Bar(
            x=names, y=vals, marker_color=color,
            text=[f"{v:.1f}" for v in vals]
        ))
        fig.update_layout(title=chart_title, yaxis_title=f"Value ({value_format})")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

class InventoryManagementSystem:
    def __init__(self):
        self.analyzer = InventoryAnalyzer()
        self.persistence = DataPersistence()
        self.initialize_session_state()

    def initialize_session_state(self):
        if 'user_role' not in st.session_state: st.session_state.user_role = None
        # Persistent Data Keys
        self.persistent_keys = [
            'persistent_pfep_data', 'persistent_pfep_locked',
            'persistent_inventory_data', 'persistent_inventory_locked',
            'persistent_analysis_results',
            # New Keys for BOM
            'persistent_bom_data', 'persistent_bom_locked', 'bom_names'
        ]
        for key in self.persistent_keys:
            if key not in st.session_state: st.session_state[key] = None
        
        if 'bom_names' not in st.session_state or st.session_state.bom_names is None:
            st.session_state.bom_names = []

    def safe_float_convert(self, value):
        try:
            return float(value)
        except:
            return 0.0

    def standardize_pfep_data(self, df):
        # Basic standardization logic
        # Expects: Part No, Description, Vendor, RM Days, Unit Price
        # AVG CONSUMPTION will be overwritten by BOM logic later
        data = []
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        # Mappings
        map_cols = {
            'part_no': ['part_no', 'part no', 'material', 'item code'],
            'desc': ['description', 'desc', 'part description'],
            'rm_days': ['rm_in_days', 'rm days', 'inventory days'],
            'price': ['unit_price', 'price', 'rate', 'unit price'],
            'vendor': ['vendor_name', 'vendor', 'supplier']
        }
        
        # Find columns
        final_map = {}
        for k, v_list in map_cols.items():
            for v in v_list:
                if v in df.columns:
                    final_map[k] = v
                    break
        
        if 'part_no' not in final_map:
            st.error("PFEP must have a Part No column")
            return []

        for _, row in df.iterrows():
            data.append({
                'Part_No': str(row[final_map['part_no']]).strip(),
                'Description': str(row.get(final_map.get('desc'), '')).strip(),
                'RM_IN_DAYS': self.safe_float_convert(row.get(final_map.get('rm_days'), 7)),
                'unit_price': self.safe_float_convert(row.get(final_map.get('price'), 1.0)),
                'Vendor_Name': str(row.get(final_map.get('vendor'), 'Unknown')),
                # Initialize calculated fields
                'AVG CONSUMPTION/DAY': 0.0, 
                'RM_IN_QTY': 0.0
            })
        return data

    def standardize_bom_data(self, df):
        # BOM Needs: Part No, Part Desc, Qty/Veh
        data = []
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        map_cols = {
            'part_no': ['part no', 'part_no', 'material', 'child part'],
            'qty': ['qty/veh', 'qty', 'quantity', 'usage', 'qty per veh']
        }
        
        final_map = {}
        for k, v_list in map_cols.items():
            for v in v_list:
                if v in df.columns:
                    final_map[k] = v
                    break
                    
        if 'part_no' not in final_map or 'qty' not in final_map:
            st.error("BOM File must have 'Part No' and 'Qty/Veh' columns.")
            return None

        for _, row in df.iterrows():
            data.append({
                'Part_No': str(row[final_map['part_no']]).strip(),
                'Qty_Per_Veh': self.safe_float_convert(row[final_map['qty']])
            })
        return data

    def standardize_current_inventory(self, df):
        data = []
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        map_cols = {
            'part_no': ['part no', 'part_no', 'material'],
            'qty': ['current_qty', 'qty', 'stock', 'closing stock']
        }
        
        final_map = {}
        for k, v_list in map_cols.items():
            for v in v_list:
                if v in df.columns:
                    final_map[k] = v
                    break

        if 'part_no' not in final_map or 'qty' not in final_map:
            st.error("Inventory File must have Part No and Qty columns")
            return []

        for _, row in df.iterrows():
            data.append({
                'Part_No': str(row[final_map['part_no']]).strip(),
                'Current_QTY': self.safe_float_convert(row[final_map['qty']])
            })
        return data

    # ---------------- ADMIN: BOM & PFEP MANAGEMENT ----------------
    def admin_data_management(self):
        st.header("🔧 Admin Dashboard - Master Data")
        
        # Tabs for PFEP and BOM
        tab_pfep, tab_bom = st.tabs(["📄 PFEP Master", "🔩 BOM Configuration"])

        # --- PFEP SECTION ---
        with tab_pfep:
            st.subheader("1. Upload PFEP Master")
            pfep_locked = st.session_state.get('persistent_pfep_locked', False)
            
            if pfep_locked:
                st.success("✅ PFEP Data is Locked.")
                if st.button("Unlock PFEP"):
                    st.session_state.persistent_pfep_locked = False
                    st.rerun()
            else:
                pfep_file = st.file_uploader("Upload PFEP (Excel/CSV)", type=['xlsx', 'csv'])
                if pfep_file:
                    df = pd.read_csv(pfep_file) if pfep_file.name.endswith('.csv') else pd.read_excel(pfep_file)
                    std_data = self.standardize_pfep_data(df)
                    if std_data:
                        self.persistence.save_data_to_session_state('persistent_pfep_data', std_data)
                        st.success(f"Loaded {len(std_data)} parts.")
                        if st.button("Save & Lock PFEP"):
                            st.session_state.persistent_pfep_locked = True
                            st.rerun()

        # --- BOM SECTION (NEW REQUIREMENT) ---
        with tab_bom:
            st.subheader("2. Upload Bill of Materials (BOM)")
            bom_locked = st.session_state.get('persistent_bom_locked', False)
            
            if bom_locked:
                st.success("✅ BOM Configuration is Locked.")
                stored_boms = st.session_state.get('persistent_bom_data', [])
                st.info(f"{len(stored_boms)} BOMs Configured.")
                if st.button("Unlock BOMs"):
                    st.session_state.persistent_bom_locked = False
                    st.session_state.persistent_bom_data = None
                    st.rerun()
            else:
                st.markdown("Upload between **1 and 5** BOM files.")
                bom_files = st.file_uploader("Select BOM Files", type=['xlsx', 'csv'], accept_multiple_files=True)
                
                if bom_files:
                    if len(bom_files) > 5:
                        st.error("Maximum 5 BOM files allowed.")
                    else:
                        st.write(f"Selected {len(bom_files)} files.")
                        
                        # Process on button click
                        if st.button("Process & Lock BOMs"):
                            all_boms = []
                            bom_names = []
                            
                            valid = True
                            for f in bom_files:
                                try:
                                    df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                                    bom_data = self.standardize_bom_data(df)
                                    if bom_data:
                                        all_boms.append(bom_data)
                                        bom_names.append(f.name)
                                    else:
                                        valid = False
                                        break
                                except Exception as e:
                                    st.error(f"Error reading {f.name}: {str(e)}")
                                    valid = False
                            
                            if valid and len(all_boms) > 0:
                                self.persistence.save_data_to_session_state('persistent_bom_data', all_boms)
                                st.session_state.bom_names = bom_names
                                st.session_state.persistent_bom_locked = True
                                st.success("✅ BOMs processed and locked successfully!")
                                st.rerun()

    # ---------------- USER: PRODUCTION & INVENTORY ----------------
    def user_view(self):
        st.header("🏭 User Dashboard - Production & Inventory")
        
        # Check Admin Logic
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        boms_data = self.persistence.load_data_from_session_state('persistent_bom_data')
        bom_names = st.session_state.get('bom_names', [])
        
        if not pfep_data or not boms_data:
            st.error("⚠️ Admin has not locked PFEP or BOM data yet.")
            return

        # 1. Daily Production Input (Dynamic)
        st.subheader("1. Daily Production Plan")
        st.markdown("Enter the planned production quantity for each BOM model.")
        
        production_inputs = {}
        
        # Create columns based on number of BOMs
        cols = st.columns(len(boms_data))
        
        for idx, (bom, name) in enumerate(zip(boms_data, bom_names)):
            with cols[idx]:
                # Default label
                default_label = f"BOM {idx+1}: {name}"
                # User can rename (simulated by just text input here, or we accept the label matches the file)
                prod_qty = st.number_input(f"Qty/Veh ({name})", min_value=0, value=0, key=f"prod_{idx}")
                production_inputs[idx] = prod_qty

        # 2. Inventory Upload
        st.subheader("2. Upload Current Inventory")
        inv_file = st.file_uploader("Current Inventory File", type=['xlsx', 'csv'])
        
        if inv_file and st.button("🚀 Calculate & Analyze"):
            # A. Process Inventory
            df_inv = pd.read_csv(inv_file) if inv_file.name.endswith('.csv') else pd.read_excel(inv_file)
            std_inventory = self.standardize_current_inventory(df_inv)
            self.persistence.save_data_to_session_state('persistent_inventory_data', std_inventory)
            
            # B. Calculate Consumption based on BOMs + Production Input
            # Logic: For every part in PFEP, look at all BOMs. 
            # Total Consumption = Sum(BOM_Qty_Per_Veh * User_Daily_Prod)
            
            # Create a map for aggregated consumption
            part_consumption_map = {}
            
            for idx, bom_list in enumerate(boms_data):
                daily_prod = production_inputs.get(idx, 0)
                if daily_prod > 0:
                    for item in bom_list:
                        p_no = item['Part_No']
                        qty_per = item['Qty_Per_Veh']
                        consumption = qty_per * daily_prod
                        
                        part_consumption_map[p_no] = part_consumption_map.get(p_no, 0) + consumption
            
            # C. Update PFEP Data with calculated consumption
            # We create a COPY of PFEP data to not corrupt the master for other sessions/re-runs
            updated_pfep = []
            for item in pfep_data:
                new_item = item.copy()
                p_no = str(item['Part_No']).strip()
                
                # Get calculated consumption
                # Note: keys in map might need normalization
                # Try exact match first
                calc_cons = part_consumption_map.get(p_no, 0.0)
                
                # Update item
                new_item['AVG CONSUMPTION/DAY'] = calc_cons
                # Recalculate RM_IN_QTY (Norm) inside analyzer, but good to set here too
                new_item['RM_IN_QTY'] = calc_cons * new_item.get('RM_IN_DAYS', 0)
                
                updated_pfep.append(new_item)
            
            # D. Run Analysis
            results = self.analyzer.analyze_inventory(updated_pfep, std_inventory)
            self.persistence.save_data_to_session_state('persistent_analysis_results', results)
            st.success("Analysis Complete!")
            st.rerun()

    def display_results(self):
        results = self.persistence.load_data_from_session_state('persistent_analysis_results')
        if not results: return
        
        st.divider()
        st.header("📊 Analysis Results")
        
        df = pd.DataFrame(results)
        
        # Summary Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Parts", len(df))
        c2.metric("Total Value", f"₹{df['Current Inventory - VALUE'].sum():,.0f}")
        c3.metric("Excess Value", f"₹{df[df['Status']=='Excess Inventory']['Stock Deviation Value'].sum():,.0f}")
        c4.metric("Shortage Value", f"₹{abs(df[df['Status']=='Short Inventory']['Stock Deviation Value'].sum()):,.0f}")
        
        # Tabs
        t1, t2, t3 = st.tabs(["Details", "Visuals", "Export"])
        
        with t1:
            st.dataframe(df)
            
        with t2:
            st.subheader("Inventory Status Distribution")
            fig = px.pie(df, names='Status', title='Part Count by Status', color='Status',
                         color_discrete_map={'Within Norms':'#4CAF50', 'Excess Inventory':'#2196F3', 'Short Inventory':'#F44336'})
            st.plotly_chart(fig)
            
            st.subheader("Top Shortages (Value)")
            short = df[df['Status']=='Short Inventory'].sort_values(by='Stock Deviation Value') # Negative values
            if not short.empty:
                short['AbsValue'] = short['Stock Deviation Value'].abs()
                fig_short = px.bar(short.head(10), x='PART NO', y='AbsValue', title="Top 10 Shortages")
                st.plotly_chart(fig_short)

        with t3:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="inventory_analysis.csv", mime='text/csv')

    def run(self):
        st.title("🏭 Intelligent Inventory System")
        st.caption("With BOM-based Dynamic Consumption Calculation")
        
        # Authentication / Role Switching
        with st.sidebar:
            st.header("Login")
            role = st.selectbox("Role", ["Select", "Admin", "User"])
            if role == "Admin":
                pwd = st.text_input("Password", type="password")
                if pwd == "admin": # Simple check
                    st.session_state.user_role = "Admin"
                elif pwd:
                    st.error("Wrong Password")
            elif role == "User":
                st.session_state.user_role = "User"
            
            if st.button("Reset / Logout"):
                st.session_state.clear()
                st.rerun()

        # Routing
        if st.session_state.user_role == "Admin":
            self.admin_data_management()
        elif st.session_state.user_role == "User":
            self.user_view()
            self.display_results()
        else:
            st.info("Please select a role from the sidebar. (Admin Pwd: 'admin')")

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
