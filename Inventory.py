import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import io
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

# Custom CSS (Kept exactly as original)
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
    
    @staticmethod
    def is_data_saved(key):
        """Check if data is saved"""
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('saved', False)
        return False

class InventoryAnalyzer:
    """Enhanced inventory analysis with comprehensive reporting"""
    
    def __init__(self):
        self.debug = False
        
    def safe_float_convert(self, value, default=0.0):
        try:
            if isinstance(value, str):
                value = value.strip()
                value = re.sub(r'[₹$€£,]', '', value)
                if '%' in value:
                    return float(value.replace('%', '')) / 100
            return float(value)
        except (ValueError, TypeError):
            return default

    def analyze_inventory(self, master_data, current_inventory, tolerance=None):
        """
        Analyze inventory using Generated Master Data (from BOMs) and Inventory Dump.
        Only analyses parts present in master_data (BOMs).
        """
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)
            
        results = []
        # Create lookup dictionary for Inventory
        inventory_dict = {str(item['Part_No']).strip().upper(): item for item in current_inventory}
        
        # Iterate through Master Data (BOM Parts) only
        for master_item in master_data:
            part_no = str(master_item.get('Part_No')).strip().upper()
            
            # 1. Get Inventory Data (Match Part No)
            inventory_item = inventory_dict.get(part_no, {})
            current_qty = float(inventory_item.get('Current_QTY', 0)) or 0.0
            stock_value_from_file = float(inventory_item.get('Current Inventory - VALUE', 0)) or 0.0
            
            # 2. Get Calculated Data from BOM
            part_desc = master_item.get('Description', '')
            
            # BOM doesn't usually have Price/RM Days, use defaults or try to get from inventory
            unit_price = self.safe_float_convert(master_item.get('unit_price', 0))
            if unit_price == 0 and current_qty > 0 and stock_value_from_file > 0:
                unit_price = stock_value_from_file / current_qty
            elif unit_price == 0:
                unit_price = 1.0 # Default fallback
                
            rm_days = self.safe_float_convert(master_item.get('RM_IN_DAYS', 7.0)) # Default 7 days
            
            # This is the Calculated Consumption from User Input * BOM Qty
            avg_per_day = self.safe_float_convert(master_item.get('AVG CONSUMPTION/DAY', 0))
            
            # 3. Calculate Norms
            rm_qty = avg_per_day * rm_days # Norm Qty
            
            # If inventory file has no value, calculate it
            current_value = stock_value_from_file if stock_value_from_file > 0 else (current_qty * unit_price)
            
            # Norms with tolerance
            lower_bound = rm_qty * (1 - tolerance / 100)
            upper_bound = rm_qty * (1 + tolerance / 100)
            revised_norm_qty = upper_bound

            deviation_qty = current_qty - revised_norm_qty
            deviation_value = deviation_qty * unit_price

            # Status based on range
            if current_qty < lower_bound:
                status = 'Short Inventory'
            elif current_qty > upper_bound:
                status = 'Excess Inventory'
            else:
                status = 'Within Norms'

            result = {
                'PART NO': part_no,
                'PART DESCRIPTION': part_desc,
                'Vendor Name': master_item.get('Vendor_Name', 'Unknown'),
                'Vendor_Code': master_item.get('Vendor_Code', ''),
                'AVG CONSUMPTION/DAY': avg_per_day,
                'RM IN DAYS': rm_days,
                'RM Norm - In Qty': rm_qty,
                'Revised Norm Qty': revised_norm_qty,
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
            
        return results 
        
    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, value_format='lakhs'):
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        vendor_totals = {}
        
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            # Use absolute value for charts
            val = abs(item.get('Stock Deviation Value', 0))
            vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + val
            
        if not vendor_totals:
            st.info(f"No vendors found in '{status_filter}' status.")
            return

        top_vendors = sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        vendor_names = [v[0] for v in top_vendors]
        raw_values = [v[1] for v in top_vendors]
        
        factor = 100000 if value_format == 'lakhs' else 1
        values = [v/factor for v in raw_values]
        
        y_title = f"Value ({value_format})"

        fig = go.Figure(go.Bar(
            x=vendor_names, y=values, marker_color=color,
            text=[f"{v:.1f}" for v in values],
            hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
        ))
        fig.update_layout(title=chart_title, yaxis_title=y_title)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

class InventoryManagementSystem:
    """Main application class"""
    
    def __init__(self):
        self.analyzer = InventoryAnalyzer()
        self.persistence = DataPersistence()
        self.initialize_session_state()
        self.status_colors = {
            'Within Norms': '#4CAF50',
            'Excess Inventory': '#2196F3',
            'Short Inventory': '#F44336'
        }
        
    def initialize_session_state(self):
        if 'user_role' not in st.session_state: st.session_state.user_role = None
        if 'admin_tolerance' not in st.session_state: st.session_state.admin_tolerance = 30
        
        # Keys for BOM Data
        self.persistent_keys = [
            'persistent_bom_data', 'persistent_bom_locked', 'bom_filenames',
            'persistent_inventory_data', 'persistent_inventory_locked',
            'persistent_analysis_results'
        ]
        for key in self.persistent_keys:
            if key not in st.session_state: st.session_state[key] = None

    def safe_float_convert(self, value):
        try:
            return float(value)
        except:
            return 0.0

    def standardize_bom_data(self, df):
        """Standardize BOM columns (Part No, Description, Qty/Veh)"""
        if df is None or df.empty: return []
        
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        # Mappings
        map_cols = {
            'part_no': ['part no', 'part_no', 'part_number', 'material', 'child part'],
            'desc': ['part description', 'part desc', 'description', 'desc', 'material description'],
            'qty': ['qty/veh', 'qty/veh - 1', 'qty', 'quantity', 'usage', 'qty per veh', 'qty - 1'],
            # Optional (to maintain compatibility if provided)
            'price': ['unit price', 'price', 'rate'],
            'vendor': ['vendor', 'supplier'],
            'rm_days': ['rm days', 'inventory days']
        }
        
        final_map = {}
        for k, v_list in map_cols.items():
            for v in v_list:
                if v in df.columns:
                    final_map[k] = v
                    break
        
        if 'part_no' not in final_map or 'qty' not in final_map:
            return [] # Invalid BOM file

        data = []
        for _, row in df.iterrows():
            item = {
                'Part_No': str(row[final_map['part_no']]).strip(),
                'Description': str(row.get(final_map.get('desc'), '')).strip(),
                'Qty_Per_Veh': self.safe_float_convert(row[final_map['qty']]),
                'unit_price': self.safe_float_convert(row.get(final_map.get('price'), 0.0)),
                'Vendor_Name': str(row.get(final_map.get('vendor'), 'Unknown')),
                'RM_IN_DAYS': self.safe_float_convert(row.get(final_map.get('rm_days'), 7.0)) # Default
            }
            if item['Part_No'] and item['Part_No'].lower() not in ['nan', 'none', '']:
                data.append(item)
        return data

    def standardize_current_inventory(self, df):
        if df is None or df.empty: return []
        
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        map_cols = {
            'part_no': ['part no', 'part_no', 'material', 'code'],
            'qty': ['current_qty', 'qty', 'quantity', 'stock', 'closing stock'],
            'value': ['stock value', 'value', 'amount', 'current inventory - value', 'current inventory-value']
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

        data = []
        for _, row in df.iterrows():
            data.append({
                'Part_No': str(row[final_map['part_no']]).strip(),
                'Current_QTY': self.safe_float_convert(row[final_map['qty']]),
                'Current Inventory - VALUE': self.safe_float_convert(row.get(final_map.get('value'), 0)),
                'Description': str(row.get('description', ''))
            })
        return data

    def admin_data_management(self):
        """Admin Interface for BOM Upload (1 to 5 files)"""
        st.header("🔧 Admin Dashboard - BOM Management")
        
        bom_locked = st.session_state.get('persistent_bom_locked', False)
        
        if bom_locked:
            st.warning("🔒 BOM Data is Locked. Users can now perform analysis.")
            bom_data = self.persistence.load_data_from_session_state('persistent_bom_data')
            st.info(f"Loaded {len(bom_data)} BOM Files.")
            
            if st.button("🔓 Unlock Data"):
                st.session_state.persistent_bom_locked = False
                st.session_state.persistent_bom_data = None
                st.session_state.bom_filenames = None
                st.session_state.persistent_analysis_results = None
                st.rerun()
        else:
            st.markdown("### Upload Bill of Materials (BOM)")
            st.markdown("Please upload **minimum 1** and **maximum 5** BOM files.")
            st.markdown("Files must contain: **Part No, Part Desc, Qty/Veh**")
            
            uploaded_files = st.file_uploader("Select BOM Files", type=['xlsx', 'xls', 'csv'], accept_multiple_files=True)
            
            if uploaded_files:
                if len(uploaded_files) > 5:
                    st.error("❌ Maximum 5 files allowed.")
                else:
                    st.write(f"📂 Selected {len(uploaded_files)} files.")
                    
                    if st.button("Process & Lock BOMs"):
                        all_boms = []
                        bom_names = []
                        valid_upload = True
                        
                        for file in uploaded_files:
                            try:
                                if file.name.endswith('.csv'):
                                    df = pd.read_csv(file)
                                else:
                                    df = pd.read_excel(file)
                                
                                processed_bom = self.standardize_bom_data(df)
                                if processed_bom:
                                    all_boms.append(processed_bom)
                                    bom_names.append(file.name)
                                else:
                                    st.error(f"❌ File {file.name} missing required columns.")
                                    valid_upload = False
                            except Exception as e:
                                st.error(f"❌ Error reading {file.name}: {e}")
                                valid_upload = False
                        
                        if valid_upload and len(all_boms) > 0:
                            self.persistence.save_data_to_session_state('persistent_bom_data', all_boms)
                            st.session_state.bom_filenames = bom_names
                            st.session_state.persistent_bom_locked = True
                            st.success("✅ BOMs processed and locked successfully!")
                            st.rerun()

    def user_inventory_upload(self):
        """User Interface: Daily Production & Inventory Upload"""
        st.header("📦 Inventory Analysis")
        
        # Check Admin Data
        bom_data = self.persistence.load_data_from_session_state('persistent_bom_data')
        bom_names = st.session_state.get('bom_filenames', [])
        is_locked = st.session_state.get('persistent_bom_locked', False)
        
        if not bom_data or not is_locked:
            st.error("⚠️ Admin has not uploaded and locked BOM data yet.")
            return
            
        st.success(f"✅ Loaded {len(bom_data)} BOM Models.")
        
        # 1. Daily Production Inputs
        st.subheader("1. Daily Production Plan")
        st.info("Enter the daily production quantity for each BOM model.")
        
        production_inputs = {}
        cols = st.columns(len(bom_data))
        
        for idx, (bom, name) in enumerate(zip(bom_data, bom_names)):
            with cols[idx]:
                # User can identify BOM by filename
                prod_qty = st.number_input(f"Daily Qty for {name}", min_value=0, value=0, key=f"prod_qty_{idx}")
                production_inputs[idx] = prod_qty
                
        # 2. Inventory Upload
        st.subheader("2. Upload Current Inventory")
        inv_file = st.file_uploader("Current Inventory File", type=['xlsx', 'xls', 'csv'])
        
        if inv_file is not None:
            if st.button("🚀 Calculate & Analyze", type="primary"):
                # Load Inventory
                try:
                    if inv_file.name.endswith('.csv'):
                        df_inv = pd.read_csv(inv_file)
                    else:
                        df_inv = pd.read_excel(inv_file)
                    
                    std_inventory = self.standardize_current_inventory(df_inv)
                    self.persistence.save_data_to_session_state('persistent_inventory_data', std_inventory)
                    
                    # 3. Generate Master List from BOMs + Daily Production
                    # Logic: Sum (BOM_Qty * Daily_Prod) for each part across all BOMs
                    
                    master_parts_map = {} # Key: Part_No, Value: Item Dict
                    
                    for idx, bom_list in enumerate(bom_data):
                        daily_prod = production_inputs.get(idx, 0)
                        
                        for item in bom_list:
                            p_no = str(item['Part_No']).strip().upper()
                            qty_per_veh = item['Qty_Per_Veh']
                            
                            # Consumption for this specific BOM
                            consumption = qty_per_veh * daily_prod
                            
                            if p_no not in master_parts_map:
                                # First time seeing part, add details
                                master_parts_map[p_no] = {
                                    'Part_No': p_no,
                                    'Description': item['Description'],
                                    'Vendor_Name': item['Vendor_Name'],
                                    'Vendor_Code': '',
                                    'unit_price': item['unit_price'],
                                    'RM_IN_DAYS': item['RM_IN_DAYS'],
                                    'AVG CONSUMPTION/DAY': 0.0
                                }
                            
                            # Aggregate Consumption
                            master_parts_map[p_no]['AVG CONSUMPTION/DAY'] += consumption
                            
                    # Convert map to list for analyzer
                    generated_master_data = list(master_parts_map.values())
                    
                    # 4. Run Analysis
                    results = self.analyzer.analyze_inventory(generated_master_data, std_inventory)
                    self.persistence.save_data_to_session_state('persistent_analysis_results', results)
                    st.success("✅ Analysis Complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing inventory: {e}")

        # Display results if available
        if st.session_state.get('persistent_analysis_results'):
            self.display_comprehensive_analysis(st.session_state.persistent_analysis_results)

    def display_comprehensive_analysis(self, analysis_results):
        """Display the full dashboard (Original logic maintained)"""
        st.markdown("---")
        st.header("📊 Executive Summary Dashboard")
        
        df = pd.DataFrame(analysis_results)
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parts (In BOMs)", len(df))
        with col2:
            st.metric("Total Inventory Value", f"₹{df['Current Inventory - VALUE'].sum():,.0f}")
        with col3:
            st.metric("Excess Value", f"₹{df[df['Status']=='Excess Inventory']['Stock Deviation Value'].sum():,.0f}")
        with col4:
            # Shortage is negative deviation, take absolute
            short_val = abs(df[df['Status']=='Short Inventory']['Stock Deviation Value'].sum())
            st.metric("Shortage Value", f"₹{short_val:,.0f}")
            
        # Detailed Tables
        st.subheader("📋 Detailed Analysis")
        tab1, tab2, tab3 = st.tabs(["🔍 All Parts", "🔴 Shortages", "🔵 Excess"])
        
        with tab1:
            st.dataframe(df, use_container_width=True)
        with tab2:
            short_df = df[df['Status'] == 'Short Inventory'].sort_values('Stock Deviation Value')
            st.dataframe(short_df, use_container_width=True)
        with tab3:
            excess_df = df[df['Status'] == 'Excess Inventory'].sort_values('Stock Deviation Value', ascending=False)
            st.dataframe(excess_df, use_container_width=True)
            
        # Charts Section
        st.markdown("---")
        st.subheader("📈 Visual Analytics")
        
        c1, c2 = st.columns(2)
        with c1:
            # Top 10 Shortages
            short_top = df[df['Status'] == 'Short Inventory'].copy()
            if not short_top.empty:
                short_top['AbsVal'] = short_top['Stock Deviation Value'].abs()
                short_top = short_top.sort_values('AbsVal', ascending=False).head(10)
                fig = px.bar(short_top, x='PART NO', y='AbsVal', title="Top 10 Shortages (Value)", color_discrete_sequence=['#F44336'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No shortages found.")
                
        with c2:
            # Top 10 Excess
            excess_top = df[df['Status'] == 'Excess Inventory'].copy()
            if not excess_top.empty:
                excess_top = excess_top.sort_values('Stock Deviation Value', ascending=False).head(10)
                fig = px.bar(excess_top, x='PART NO', y='Stock Deviation Value', title="Top 10 Excess (Value)", color_discrete_sequence=['#2196F3'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No excess inventory found.")
                
        # Vendor Charts (Original logic)
        st.subheader("🏢 Vendor Performance")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            self.analyzer.show_vendor_chart_by_status(analysis_results, "Short Inventory", "Top Vendors (Shortage)", "v_short", "#F44336")
        with col_v2:
            self.analyzer.show_vendor_chart_by_status(analysis_results, "Excess Inventory", "Top Vendors (Excess)", "v_excess", "#2196F3")
            
        # Export
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Analysis (CSV)", data=csv, file_name="inventory_analysis.csv", mime="text/csv")


    def authenticate_user(self):
        """Authentication system"""
        st.sidebar.markdown("### 🔐 Authentication")
        
        if st.session_state.user_role is None:
            role = st.sidebar.selectbox("Select Role", ["Select Role", "Admin", "User"])
            
            if role == "Admin":
                password = st.sidebar.text_input("Admin Password", type="password")
                if st.sidebar.button("Login"):
                    if password == "Agilomatrix@123":
                        st.session_state.user_role = "Admin"
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid password")
            
            elif role == "User":
                if st.sidebar.button("Enter as User"):
                    st.session_state.user_role = "User"
                    st.rerun()
        else:
            st.sidebar.success(f"Logged in as {st.session_state.user_role}")
            if st.sidebar.button("Logout"):
                st.session_state.clear()
                st.rerun()

    def run(self):
        st.title("📊 Inventory Analyzer (BOM Based)")
        st.markdown("<p style='font-size:18px; font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
        st.markdown("---")

        self.authenticate_user()

        if st.session_state.user_role == "Admin":
            self.admin_data_management()
        elif st.session_state.user_role == "User":
            self.user_inventory_upload()
        else:
            st.info("👋 Please select your role and authenticate to access the system.")

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
