import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import re
from typing import Union, Any, Optional, List, Dict
import io

# ... [Keep your existing Imports, Logging, and CSS setup] ...

class InventoryAnalyzer:
    """Enhanced inventory analysis with BOM-based logic"""
    
    def __init__(self):
        self.debug = False
        self.persistence = None # Will be set by main app
        self.status_colors = {
            'Within Norms': '#4CAF50',
            'Excess Inventory': '#2196F3',
            'Short Inventory': '#F44336'
        }

    def safe_float_convert(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def calculate_dynamic_norms(self, boms_data, production_plan):
        """
        Calculate total required quantity for every part based on BOMs and Production Plan.
        Returns a dictionary keyed by Part_No containing the summed requirement.
        """
        master_requirements = {}

        # Iterate through every uploaded BOM
        for bom_name, bom_df in boms_data.items():
            # Get the user-inputted production count for this BOM
            production_count = production_plan.get(bom_name, 0)
            
            if production_count > 0:
                for item in bom_df:
                    part_no = str(item['Part_No']).strip().upper()
                    qty_per_veh = self.safe_float_convert(item['Qty_Veh'])
                    
                    # Calculate required quantity for this BOM
                    total_req_for_bom = qty_per_veh * production_count
                    
                    if part_no in master_requirements:
                        master_requirements[part_no]['RM_IN_QTY'] += total_req_for_bom
                        # Append Aggregate name for reference
                        if bom_name not in master_requirements[part_no]['Aggregates']:
                            master_requirements[part_no]['Aggregates'].append(bom_name)
                    else:
                        master_requirements[part_no] = {
                            'Part_No': part_no,
                            'Description': item.get('Description', ''),
                            'RM_IN_QTY': total_req_for_bom,
                            'Aggregates': [bom_name],
                            'Vendor_Name': item.get('Vendor_Name', 'Unknown'), # Optional if in BOM
                            'Vendor_Code': item.get('Vendor_Code', ''),
                        }
        return master_requirements

    def analyze_inventory(self, boms_data, inventory_data, production_plan, tolerance=None):
        """
        Analyze inventory against BOM-based requirements.
        """
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)

        # 1. Calculate Dynamic Norms (The "Required" List)
        required_parts_dict = self.calculate_dynamic_norms(boms_data, production_plan)
        
        results = []
        
        # 2. Normalize Inventory Data
        # We assume Inventory Data contains the Unit Price info (derived from Value/Qty)
        inventory_dict = {str(item['Part_No']).strip().upper(): item for item in inventory_data}
        
        # Get all unique part numbers from both lists
        all_parts = set(required_parts_dict.keys()) | set(inventory_dict.keys())

        for part_no in all_parts:
            try:
                # Get Data from Sources
                req_data = required_parts_dict.get(part_no, {})
                inv_data = inventory_dict.get(part_no, {})
                
                # Basic Details
                part_desc = inv_data.get('Description') or req_data.get('Description') or "Unknown"
                
                # Quantities
                rm_qty = float(req_data.get('RM_IN_QTY', 0)) # This is our Target/Norm
                current_qty = float(inv_data.get('Current_QTY', 0))
                
                # Financials (Derive Unit Price from Inventory if possible)
                current_value = float(inv_data.get('Current Inventory - VALUE', 0))
                unit_price = 0.0
                if current_qty > 0:
                    unit_price = current_value / current_qty
                
                # Norms with tolerance
                lower_bound = rm_qty * (1 - tolerance / 100)
                upper_bound = rm_qty * (1 + tolerance / 100)
                
                # Deviation
                # Logic: If Short, how much needed to reach Norm? If Excess, how much over Norm?
                deviation_qty = 0
                status = 'Within Norms'
                
                if current_qty < lower_bound:
                    status = 'Short Inventory'
                    deviation_qty = lower_bound - current_qty # Quantity needed
                    deviation_value = deviation_qty * unit_price * -1 # Negative value for shortage
                elif current_qty > upper_bound:
                    status = 'Excess Inventory'
                    deviation_qty = current_qty - upper_bound # Quantity excess
                    deviation_value = deviation_qty * unit_price
                else:
                    deviation_value = 0

                # Construct Result
                result = {
                    'PART NO': part_no,
                    'PART DESCRIPTION': part_desc,
                    'Vendor Name': inv_data.get('Vendor_Code', 'Unknown'), # Use Inventory Vendor info
                    'Used In Aggregates': ", ".join(req_data.get('Aggregates', [])),
                    'RM Norm - In Qty': rm_qty,
                    'Revised Norm Qty': upper_bound, # Visualization helper
                    'Lower Bound Qty': lower_bound,
                    'Upper Bound Qty': upper_bound,
                    'UNIT PRICE': unit_price,
                    'Current Inventory - Qty': current_qty,
                    'Current Inventory - VALUE': current_value,
                    'Stock Deviation Value': deviation_value,
                    'VALUE(Unit Price* Short/Excess Inventory)': deviation_value,
                    'Status': status,
                    'INVENTORY REMARK STATUS': status
                }
                results.append(result)
            except Exception as e:
                # logging.error(f"Error analyzing {part_no}: {e}")
                continue
                
        return results

    # ... [Keep get_vendor_summary and show_vendor_chart_by_status as they were] ...
    def get_vendor_summary(self, processed_data):
        # (Same as before, ensure it handles missing Vendor Names gracefully)
        return super().get_vendor_summary(processed_data) if hasattr(super(), 'get_vendor_summary') else self._local_vendor_summary(processed_data)

    def _local_vendor_summary(self, processed_data):
        # Internal backup if inheritance is tricky in copy-paste
        from collections import defaultdict
        summary = defaultdict(lambda: {'total_parts':0, 'short_parts':0, 'excess_parts':0, 'normal_parts':0, 'total_value':0.0})
        for item in processed_data:
            vendor = item.get('Vendor Name', 'Unknown')
            status = item.get('Status', 'Unknown')
            val = item.get('Current Inventory - VALUE', 0)
            summary[vendor]['total_parts'] += 1
            summary[vendor]['total_value'] += val
            if status == "Short Inventory": summary[vendor]['short_parts'] += 1
            elif status == "Excess Inventory": summary[vendor]['excess_parts'] += 1
            elif status == "Within Norms": summary[vendor]['normal_parts'] += 1
        return summary
    
    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, value_format='lakhs'):
        # (Paste the exact previous implementation of this method here)
        # Using the logic from previous response
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        if not filtered: return
        
        vendor_totals = {}
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            val = abs(item.get('Stock Deviation Value', 0)) # Use deviation value
            vendor_totals[vendor] = vendor_totals.get(vendor, 0) + val
            
        # Top 10
        top_vendors = sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        if not top_vendors: return
        
        vendors, values = zip(*top_vendors)
        
        # Formatting
        display_values = [v/100000 for v in values] if value_format == 'lakhs' else values
        
        fig = go.Figure(go.Bar(
            x=vendors, y=display_values, marker_color=color,
            text=[f"{v:.1f}L" for v in display_values], textposition='auto'
        ))
        fig.update_layout(title=chart_title, yaxis_title="Value (Lakhs)")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)


class InventoryManagementSystem:
    # ... [Keep __init__, initialize_session_state, safe_helpers] ...
    
    def __init__(self):
        self.debug = True
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
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {'default_tolerance': 30, 'chart_theme': 'plotly'}
        
        # UPDATED KEYS FOR BOM
        self.persistent_keys = [
            'persistent_boms_data', # Dict of BOMs
            'persistent_boms_locked',
            'persistent_inventory_data', 
            'persistent_inventory_locked',
            'persistent_analysis_results',
            'production_plan' # User input
        ]
        for key in self.persistent_keys:
            if key not in st.session_state: st.session_state[key] = None

    # ... [Keep authenticate_user] ...

    def display_data_status(self):
        """Display current data loading status in sidebar (Updated for BOMs)"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Data Status")
        
        # Check persistent BOM data
        boms_data = self.persistence.load_data_from_session_state('persistent_boms_data')
        boms_locked = st.session_state.get('persistent_boms_locked', False)
        
        if boms_data:
            bom_count = len(boms_data)
            lock_icon = "🔒" if boms_locked else "🔓"
            st.sidebar.success(f"✅ BOM Data: {bom_count} Aggregates {lock_icon}")
        else:
            st.sidebar.error("❌ BOM Data: Not loaded")
        
        # Check persistent inventory data
        inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        if inventory_data:
            st.sidebar.success(f"✅ Inventory: {len(inventory_data)} parts")
        else:
            st.sidebar.error("❌ Inventory: Not loaded")

    def standardize_bom_data(self, df, aggregate_name):
        """Standardize BOM columns (Part No, Desc, Qty/Veh)"""
        if df is None or df.empty: return []
        
        # Clean column names
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        # Mappings
        col_map = {
            'part_no': ['part no', 'part_no', 'part number', 'material', 'item code'],
            'description': ['part description', 'description', 'desc', 'material description'],
            'qty_veh': ['qty/veh', 'qty per veh', 'quantity', 'qty', 'usage'],
            # Optional
            'vendor_code': ['vendor code', 'vendor'],
            'vendor_name': ['vendor name']
        }
        
        found_map = {}
        for target, alts in col_map.items():
            for alt in alts:
                if alt in df.columns:
                    found_map[target] = alt
                    break
        
        if 'part_no' not in found_map or 'qty_veh' not in found_map:
            st.error(f"❌ BOM '{aggregate_name}' missing required columns (Part No, Qty/Veh).")
            return []

        std_data = []
        for _, row in df.iterrows():
            try:
                p_no = str(row[found_map['part_no']]).strip()
                if not p_no or p_no.lower() == 'nan': continue
                
                item = {
                    'Part_No': p_no,
                    'Description': str(row.get(found_map.get('description'), '')).strip(),
                    'Qty_Veh': self.safe_float_convert(row[found_map['qty_veh']]),
                    'Aggregate': aggregate_name,
                    'Vendor_Code': str(row.get(found_map.get('vendor_code'), '')).strip(),
                    'Vendor_Name': str(row.get(found_map.get('vendor_name'), '')).strip(),
                }
                std_data.append(item)
            except Exception: continue
            
        return std_data

    def admin_data_management(self):
        """Admin Interface for BOM Uploads"""
        st.header("🔧 Admin Dashboard - BOM Management")
        
        # Tolerance Setting
        if "admin_tolerance" not in st.session_state: st.session_state.admin_tolerance = 30
        st.session_state.admin_tolerance = st.selectbox(
            "Analysis Tolerance (+/- %)", [0, 10, 20, 30, 40, 50], 
            index=[0,10,20,30,40,50].index(st.session_state.admin_tolerance)
        )
        st.markdown("---")

        # Check Lock
        boms_locked = st.session_state.get('persistent_boms_locked', False)
        if boms_locked:
            st.warning("🔒 BOM Data is locked for Users.")
            if st.button("🔓 Unlock to Edit"):
                st.session_state.persistent_boms_locked = False
                st.rerun()
            return

        st.subheader("📁 Upload Bill of Materials (BOM)")
        st.info("Upload 1 to 5 BOM files. Each file represents an Aggregate/Product.")

        uploaded_files = st.file_uploader(
            "Choose BOM Excel files", type=['xlsx', 'xls', 'csv'], 
            accept_multiple_files=True
        )

        if uploaded_files:
            if len(uploaded_files) > 5:
                st.error("❌ Maximum 5 files allowed.")
            else:
                st.write(f"📂 {len(uploaded_files)} files selected.")
                
                if st.button("Process & Save BOMs"):
                    processed_boms = {}
                    valid_upload = True
                    
                    for u_file in uploaded_files:
                        # Extract Aggregate Name from Filename (remove extension)
                        agg_name = u_file.name.rsplit('.', 1)[0]
                        try:
                            df = pd.read_csv(u_file) if u_file.name.endswith('.csv') else pd.read_excel(u_file)
                            std_data = self.standardize_bom_data(df, agg_name)
                            if std_data:
                                processed_boms[agg_name] = std_data
                            else:
                                valid_upload = False
                        except Exception as e:
                            st.error(f"Error reading {u_file.name}: {e}")
                            valid_upload = False
                    
                    if valid_upload and processed_boms:
                        self.persistence.save_data_to_session_state('persistent_boms_data', processed_boms)
                        st.success(f"✅ Successfully saved {len(processed_boms)} BOMs!")
                        
                        # Preview
                        for name, data in processed_boms.items():
                            with st.expander(f"📄 BOM: {name} ({len(data)} parts)"):
                                st.dataframe(pd.DataFrame(data).head())
                                
                        if st.button("🔒 Lock Data for Users"):
                            st.session_state.persistent_boms_locked = True
                            st.rerun()
                    else:
                        st.error("Failed to process some files. Check column headers.")

        # Show current status if no file uploaded
        current_boms = self.persistence.load_data_from_session_state('persistent_boms_data')
        if current_boms and not uploaded_files:
            st.success(f"Currently Loaded: {', '.join(current_boms.keys())}")
            if st.button("🔒 Lock Existing Data"):
                st.session_state.persistent_boms_locked = True
                st.rerun()

    def user_inventory_upload(self):
        """User Interface: Daily Production Input + Inventory Upload"""
        st.header("📦 Inventory Analysis System")
        
        # 1. Check Admin Data
        boms_data = self.persistence.load_data_from_session_state('persistent_boms_data')
        boms_locked = st.session_state.get('persistent_boms_locked', False)
        
        if not boms_data or not boms_locked:
            st.error("❌ Admin has not locked BOM data yet.")
            return

        # 2. Daily Production Input (The Core Logic Change)
        st.subheader("🏭 Daily Production Plan")
        st.info("Enter the planned production quantity for each Aggregate.")
        
        col_container = st.container()
        cols = col_container.columns(len(boms_data))
        
        production_plan = {}
        
        # Initialize session state for plan if needed
        if 'current_plan' not in st.session_state: st.session_state.current_plan = {}

        for idx, (bom_name, _) in enumerate(boms_data.items()):
            with cols[idx % len(cols)]:
                # Use session state to remember inputs
                val = st.number_input(
                    f"{bom_name}", 
                    min_value=0, 
                    value=st.session_state.current_plan.get(bom_name, 0),
                    key=f"prod_input_{idx}"
                )
                production_plan[bom_name] = val
                st.session_state.current_plan[bom_name] = val

        # 3. Inventory Upload (Standard)
        st.subheader("📊 Upload Current Inventory")
        
        # Check if inventory is already loaded
        inv_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        inv_locked = st.session_state.get('persistent_inventory_locked', False)
        
        if inv_locked and inv_data:
            st.info("Inventory Loaded. Proceeding to Analysis...")
            # Pass everything to analysis
            self.display_analysis_interface(boms_data, inv_data, production_plan)
            
            if st.button("🔄 Reset Inventory"):
                st.session_state.persistent_inventory_locked = False
                st.rerun()
            return

        uploaded_file = st.file_uploader("Choose Inventory File", type=['xlsx', 'xls', 'csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            std_inv = self.standardize_current_inventory(df) # (Use your existing method)
            
            if std_inv:
                st.success(f"✅ Loaded {len(std_inv)} inventory records.")
                if st.button("💾 Save & Analyze"):
                    self.persistence.save_data_to_session_state('persistent_inventory_data', std_inv)
                    st.session_state.persistent_inventory_locked = True
                    st.rerun()

    def display_analysis_interface(self, boms_data=None, inventory_data=None, production_plan=None):
        """Orchestrates the analysis display"""
        
        # Reload if arguments missing (dashboard refresh)
        if not boms_data:
            boms_data = self.persistence.load_data_from_session_state('persistent_boms_data')
            inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
            production_plan = st.session_state.get('current_plan', {})

        if not boms_data or not inventory_data: return

        st.markdown("---")
        st.subheader("📈 Analysis Results")
        
        # Check if plan has values
        total_prod = sum(production_plan.values())
        if total_prod == 0:
            st.warning("⚠️ Please enter Production Plan quantities above to see requirements.")
            return

        tolerance = st.session_state.get('admin_tolerance', 30)
        st.info(f"Analysis Parameters: Tolerance ±{tolerance}% | Total Production: {total_prod} Units")

        # Run Analysis
        with st.spinner("Calculating Material Requirements & Variances..."):
            results = self.analyzer.analyze_inventory(
                boms_data, 
                inventory_data, 
                production_plan, 
                tolerance
            )
        
        if results:
            self.persistence.save_data_to_session_state('persistent_analysis_results', results)
            # Call your existing display methods
            self.display_analysis_results() # This calls your existing display logic
        else:
            st.error("No results generated. Check part number matching between BOM and Inventory.")

    # ... [Keep all other existing display methods: display_analysis_results, standardize_current_inventory, etc.] ...
    # ... [Keep safe_float_convert, safe_print, etc.] ...

    # Ensure standardize_current_inventory is available (from your previous code)
    def standardize_current_inventory(self, df):
        # (Paste your existing standardize_current_inventory method here)
        # This is critical because we need 'Current Inventory - VALUE' to calculate Unit Price
        if df is None or df.empty: return []
        
        col_map = {
            'part_no': ['part_no', 'part no', 'material', 'item code'],
            'current_qty': ['current_qty', 'stock', 'qty', 'closing stock'],
            'value': ['value', 'amount', 'stock value', 'total value']
        }
        
        # ... [Standardization logic similar to your previous code] ...
        # Simplified for brevity in this snippet:
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        mapped = {}
        for k, v in col_map.items():
            for alias in v:
                if alias in df.columns: mapped[k] = alias; break
        
        if 'part_no' not in mapped or 'current_qty' not in mapped:
            st.error("Inventory file missing Part No or Qty column"); return []

        data = []
        for _, row in df.iterrows():
            try:
                item = {
                    'Part_No': str(row[mapped['part_no']]).strip(),
                    'Current_QTY': self.safe_float_convert(row[mapped['current_qty']]),
                    'Current Inventory - VALUE': self.safe_float_convert(row.get(mapped.get('value'), 0)),
                    'Description': str(row.get('description', '')),
                    'Vendor_Code': str(row.get('vendor', ''))
                }
                data.append(item)
            except: continue
        return data

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
