import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import io
import re

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
.metric-container { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.status-card { padding: 15px; border-radius: 8px; margin: 10px 0; }
.status-excess { background-color: #ffebee; border-left: 4px solid #f44336; }
.status-short { background-color: #fff3e0; border-left: 4px solid #ff9800; }
.status-normal { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
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

class InventoryAnalyzer:
    def safe_float_convert(self, value, default=0.0):
        try:
            if isinstance(value, str):
                value = re.sub(r'[₹$€£,]', '', value).strip()
                if '%' in value: return float(value.replace('%', '')) / 100
            return float(value)
        except (ValueError, TypeError):
            return default

    def analyze_inventory(self, master_data, current_inventory, tolerance=None):
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)
        
        results = []
        # Create lookup dictionary for Inventory
        inventory_dict = {str(item['Part_No']).strip().upper(): item for item in current_inventory}
        
        # Iterate through the Generated Master Data (from BOMs)
        for master_item in master_data:
            part_no = str(master_item.get('Part_No')).strip().upper()
            
            # Get Inventory Data (or defaults if missing)
            inv_item = inventory_dict.get(part_no, {})
            current_qty = float(inv_item.get('Current_QTY', 0)) or 0.0
            
            # Master Data details
            avg_per_day = self.safe_float_convert(master_item.get('AVG CONSUMPTION/DAY', 0))
            rm_days = self.safe_float_convert(master_item.get('RM_IN_DAYS', 7)) # Default 7 days if not in BOM
            unit_price = self.safe_float_convert(master_item.get('unit_price', 1.0)) # Default 1.0 if not in BOM
            
            # Calculate Norms
            rm_qty = avg_per_day * rm_days
            
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
                'PART DESCRIPTION': master_item.get('Description', ''),
                'Vendor Name': master_item.get('Vendor_Name', 'Unknown'),
                'AVG CONSUMPTION/DAY': avg_per_day,
                'RM IN DAYS': rm_days,
                'RM Norm - In Qty': revised_norm_qty, 
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

class InventoryManagementSystem:
    def __init__(self):
        self.analyzer = InventoryAnalyzer()
        self.persistence = DataPersistence()
        self.initialize_session_state()

    def initialize_session_state(self):
        if 'user_role' not in st.session_state: st.session_state.user_role = None
        
        self.persistent_keys = [
            'persistent_inventory_data', 
            'persistent_analysis_results',
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

    def standardize_bom_data(self, df):
        # We now look for extended fields in BOM since PFEP is gone
        data = []
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        map_cols = {
            'part_no': ['part no', 'part_no', 'material', 'child part'],
            'desc': ['part desc', 'description', 'part description', 'material description'],
            'qty': ['qty/veh', 'qty', 'quantity', 'usage', 'qty per veh'],
            # Optional fields that might be in BOM now
            'price': ['unit price', 'price', 'rate'],
            'vendor': ['vendor', 'supplier'],
            'rm_days': ['rm days', 'inventory days', 'norm days']
        }
        
        final_map = {}
        for k, v_list in map_cols.items():
            for v in v_list:
                if v in df.columns:
                    final_map[k] = v
                    break
                    
        if 'part_no' not in final_map or 'qty' not in final_map:
            st.error(f"BOM File missing required columns: Part No or Qty/Veh.")
            return None

        for _, row in df.iterrows():
            data.append({
                'Part_No': str(row[final_map['part_no']]).strip(),
                'Description': str(row.get(final_map.get('desc'), '')).strip(),
                'Qty_Per_Veh': self.safe_float_convert(row[final_map['qty']]),
                # Optional fields with defaults
                'unit_price': self.safe_float_convert(row.get(final_map.get('price'), 1.0)),
                'Vendor_Name': str(row.get(final_map.get('vendor'), 'Unknown')),
                'RM_IN_DAYS': self.safe_float_convert(row.get(final_map.get('rm_days'), 7)) # Default 7 days
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

    # ---------------- ADMIN: ONLY BOM UPLOAD ----------------
    def admin_data_management(self):
        st.header("🔧 Admin Dashboard")
        st.subheader("Upload Bill of Materials (BOM)")
        st.info("Upload BOMs containing: Part No, Description, Qty/Veh. (Optional: Price, Vendor, RM Days)")
        
        bom_locked = st.session_state.get('persistent_bom_locked', False)
        
        if bom_locked:
            st.success("✅ BOM Configuration is Locked.")
            stored_boms = st.session_state.get('persistent_bom_data', [])
            st.write(f"📁 {len(stored_boms)} BOM files loaded.")
            if st.button("Unlock BOMs"):
                st.session_state.persistent_bom_locked = False
                st.session_state.persistent_bom_data = None
                st.rerun()
        else:
            st.markdown("Upload **Min 1** and **Max 5** BOM files.")
            bom_files = st.file_uploader("Select BOM Files", type=['xlsx', 'csv'], accept_multiple_files=True)
            
            if bom_files:
                if len(bom_files) > 5:
                    st.error("Maximum 5 BOM files allowed.")
                else:
                    st.write(f"Selected {len(bom_files)} files.")
                    
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
        st.header("🏭 User Dashboard")
        
        boms_data = self.persistence.load_data_from_session_state('persistent_bom_data')
        bom_names = st.session_state.get('bom_names', [])
        
        if not boms_data:
            st.error("⚠️ Admin has not uploaded and locked BOM data yet.")
            return

        # 1. Daily Production Input (Dynamic)
        st.subheader("1. Daily Production Plan")
        st.markdown("Enter the daily production quantity for each BOM model.")
        
        production_inputs = {}
        cols = st.columns(len(boms_data))
        
        # Display inputs dynamically based on uploaded BOMs
        for idx, (bom, name) in enumerate(zip(boms_data, bom_names)):
            with cols[idx]:
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
            
            # B. Generate Master Data from BOMs + Production Input
            # We aggregate all BOMs into a single "Part Consumption Map"
            
            master_parts = {} # Key: Part_No, Value: {Details + Total Consumption}
            
            for idx, bom_list in enumerate(boms_data):
                daily_prod = production_inputs.get(idx, 0)
                
                for item in bom_list:
                    p_no = item['Part_No']
                    qty_per_veh = item['Qty_Per_Veh']
                    
                    # Calculate daily consumption for this specific BOM
                    consumption = qty_per_veh * daily_prod
                    
                    if p_no not in master_parts:
                        # First time seeing this part, initialize it
                        master_parts[p_no] = {
                            'Part_No': p_no,
                            'Description': item['Description'],
                            'Vendor_Name': item['Vendor_Name'],
                            'unit_price': item['unit_price'],
                            'RM_IN_DAYS': item['RM_IN_DAYS'],
                            'AVG CONSUMPTION/DAY': 0.0
                        }
                    
                    # Add consumption (if part exists in multiple BOMs, we sum it up)
                    master_parts[p_no]['AVG CONSUMPTION/DAY'] += consumption
            
            # Convert dict back to list for analyzer
            generated_master_data = list(master_parts.values())
            
            # C. Run Analysis
            results = self.analyzer.analyze_inventory(generated_master_data, std_inventory)
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
        c1.metric("Total Parts (From BOMs)", len(df))
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
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Excess (Value)")
                excess = df[df['Status']=='Excess Inventory'].sort_values(by='Stock Deviation Value', ascending=False).head(10)
                if not excess.empty:
                    fig_excess = px.bar(excess, x='PART NO', y='Stock Deviation Value', title="Top 10 Excess Items")
                    st.plotly_chart(fig_excess, use_container_width=True)
            
            with col2:
                st.subheader("Top Shortages (Value)")
                short = df[df['Status']=='Short Inventory'].sort_values(by='Stock Deviation Value').head(10)
                if not short.empty:
                    short['AbsValue'] = short['Stock Deviation Value'].abs()
                    fig_short = px.bar(short, x='PART NO', y='AbsValue', title="Top 10 Shortage Items")
                    st.plotly_chart(fig_short, use_container_width=True)

        with t3:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="inventory_analysis.csv", mime='text/csv')

    def run(self):
        st.title("🏭 Intelligent Inventory System")
        st.caption("BOM-Driven Analysis | Password: 'admin'")
        
        # Authentication
        with st.sidebar:
            st.header("Login")
            role = st.selectbox("Role", ["Select", "Admin", "User"])
            if role == "Admin":
                pwd = st.text_input("Password", type="password")
                if pwd == "admin": 
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
            st.info("Please select a role from the sidebar.")

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
