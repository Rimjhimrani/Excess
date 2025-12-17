import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import pickle
import base64
import uuid
import io
import re
from typing import Union, Any, Optional, List, Dict
from decimal import Decimal, InvalidOperation
from collections import Counter
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

.lock-button {
    background-color: #28a745;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    font-weight: bold;
}

.switch-user-button {
    background-color: #007bff;
    color: white;
    padding: 8px 16px;
    border-radius: 5px;
    border: none;
    font-weight: bold;
    margin: 5px 0;
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
    
    @staticmethod
    def get_data_timestamp(key):
        """Get data timestamp"""
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('timestamp')
        return None

class InventoryAnalyzer:
    """Enhanced inventory analysis with comprehensive reporting"""
    
    def __init__(self):
        self.debug = False
        self.persistence = self
        self.status_colors = {
            'Within Norms': '#4CAF50',    # Green
            'Excess Inventory': '#2196F3', # Blue
            'Short Inventory': '#F44336'   # Red
        }
        
    def analyze_inventory(self, pfep_data, current_inventory, tolerance=None):
        """Analyze inventory using PFEP and Inventory Dump data.
        Updated Logic:
        - Ideal Inventory = AVG CONSUMPTION/DAY * Ideal Inventory Days (Admin Set)
        - Deviation % = ((Current - Ideal) / Ideal) * 100
        - Short: Deviation < -Tolerance
        - Excess: Deviation > +Tolerance
        """
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)  # default to 30%
            
        # Get Admin defined Ideal Days
        ideal_days = st.session_state.get("ideal_inventory_days", 30)

        results = []
        # Normalize and create lookup dictionaries
        pfep_dict = {str(item['Part_No']).strip().upper(): item for item in pfep_data}
        inventory_dict = {str(item['Part_No']).strip().upper(): item for item in current_inventory}
        
        for part_no, inventory_item in inventory_dict.items():
            pfep_item = pfep_dict.get(part_no)
            if not pfep_item:
                continue  # Skip unmatched parts
            try:
                # Basic Data
                part_desc = pfep_item.get('Description', '')
                unit_price = float(pfep_item.get('unit_price', 0)) or 1.0
                
                # Inventory Data
                current_qty = float(inventory_item.get('Current_QTY', 0)) or 0.0
                current_value = current_qty * unit_price
                
                # New Calculation Logic
                avg_per_day = self.safe_float_convert(pfep_item.get('AVG CONSUMPTION/DAY', 0))
                
                # Calculate Ideal Inventory
                ideal_qty = avg_per_day * ideal_days
                ideal_value = ideal_qty * unit_price
                
                # Calculate Deviation Percentage
                # Formula: ((Current - Ideal) / Ideal) * 100
                if ideal_qty > 0:
                    deviation_percent = ((current_qty - ideal_qty) / ideal_qty) * 100
                else:
                    # If ideal is 0 but we have stock, it's 100% excess (technically infinite)
                    deviation_percent = 100.0 if current_qty > 0 else 0.0

                # Calculate Absolute Deviation Qty
                deviation_qty = current_qty - ideal_qty
                deviation_value = deviation_qty * unit_price

                # Determine Status based on Deviation % and Tolerance
                if deviation_percent < -tolerance:
                    status = 'Short Inventory'
                elif deviation_percent > tolerance:
                    status = 'Excess Inventory'
                else:
                    status = 'Within Norms'

                # Final result per part
                result = {
                    'PART NO': part_no,
                    'PART DESCRIPTION': part_desc,
                    'Vendor Name': pfep_item.get('Vendor_Name', 'Unknown'),
                    'Vendor_Code': pfep_item.get('Vendor_Code', ''),
                    'AVG CONSUMPTION/DAY': avg_per_day,
                    'Ideal Days': ideal_days,
                    'Ideal Inventory Qty': ideal_qty,
                    'Ideal Inventory Value': ideal_value,
                    'UNIT PRICE': unit_price,
                    'Current Inventory - Qty': current_qty,
                    'Current Inventory - VALUE': current_value,
                    'Deviation Percentage': deviation_percent,
                    'SHORT/EXCESS INVENTORY': deviation_qty,
                    'Stock Deviation Value': deviation_value,
                    'Status': status,
                    'INVENTORY REMARK STATUS': status
                }
                results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Error analyzing part {part_no}: {e}")
                continue
        if not results:
            st.error("❌ No analysis results generated. Please check data for mismatches or missing fields.")
        return results 
        
    def get_vendor_summary(self, processed_data):
        """Summarize inventory by vendor using actual Stock_Value field from the file."""
        from collections import defaultdict
        summary = defaultdict(lambda: {
            'total_parts': 0,
            'short_parts': 0,
            'excess_parts': 0,
            'normal_parts': 0,
            'total_value': 0.0,
            'excess_value_above_norm': 0.0,  # New field for excess value above norm
            'short_value_below_norm': 0.0    # New field for short value below norm
        })
        for item in processed_data:
            vendor = item.get('Vendor Name', 'Unknown')
            status = item.get('INVENTORY REMARK STATUS', 'Unknown')
            stock_value = item.get('Stock_Value') or item.get('Current Inventory - VALUE') or 0
            
            # Get current quantity and ideal values
            current_qty = item.get('Current Inventory - Qty', 0)
            ideal_qty = item.get('Ideal Inventory Qty', 0)
            unit_price = item.get('UNIT PRICE', 0)

            try:
                stock_value = float(stock_value)
                current_qty = float(current_qty) if current_qty else 0
                ideal_qty = float(ideal_qty) if ideal_qty else 0
                unit_price = float(unit_price)
            except (ValueError, TypeError):
                stock_value = 0.0
                current_qty = 0.0
                ideal_qty = 0.0
                unit_price = 0.0

            summary[vendor]['total_parts'] += 1
            summary[vendor]['total_value'] += stock_value
            
            if status == "Short Inventory":
                summary[vendor]['short_parts'] += 1
                # Value of shortage (Ideal - Current)
                if ideal_qty > current_qty and unit_price > 0:
                    short_value = (ideal_qty - current_qty) * unit_price
                    summary[vendor]['short_value_below_norm'] += short_value
            elif status == "Excess Inventory":
                summary[vendor]['excess_parts'] += 1
                # Value of excess (Current - Ideal)
                if current_qty > ideal_qty and unit_price > 0:
                    excess_value = (current_qty - ideal_qty) * unit_price
                    summary[vendor]['excess_value_above_norm'] += excess_value
            elif status == "Within Norms":
                summary[vendor]['normal_parts'] += 1
        return summary
        
    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, value_format='lakhs', top_n=10):
        """Show top N vendors by deviation value and count of excess/short parts."""
        # Filter by inventory status
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        vendor_totals = {}
        vendor_counts = {}
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            try:
                current_qty = float(item.get('Current Inventory - Qty', 0) or 0)
                ideal_qty = float(item.get('Ideal Inventory Qty', 0) or 0)
                unit_price = float(item.get('UNIT PRICE', 0) or 0)
                
                if status_filter == "Excess Inventory" and current_qty > ideal_qty:
                    deviation_value = (current_qty - ideal_qty) * unit_price
                    vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + deviation_value
                    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
                elif status_filter == "Short Inventory" and ideal_qty > current_qty:
                    deviation_value = (ideal_qty - current_qty) * unit_price
                    vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + deviation_value
                    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
            except (ValueError, TypeError, ZeroDivisionError):
                continue
        if not vendor_totals:
            st.info(f"No vendors found in '{status_filter}' status.")
            return
        # Combine totals and counts
        combined = [
            (vendor, vendor_totals[vendor], vendor_counts.get(vendor, 0))
            for vendor in vendor_totals
        ]
        # Sort and take top N by value
        top_vendors = sorted(combined, key=lambda x: x[1], reverse=True)[:top_n]
        vendor_names = [v[0] for v in top_vendors]
        raw_values = [v[1] for v in top_vendors]
        counts = [v[2] for v in top_vendors]
        
        # Update Title to reflect dynamic number
        dynamic_title = chart_title.replace("Top 10", f"Top {top_n}")

        # Value formatting
        if value_format == 'lakhs':
            values = [v / 100000 for v in raw_values]
            y_axis_title = "Value (₹ Lakhs)"
            hover_suffix = "L"
            tick_suffix = "L"
        elif value_format == 'millions':
            values = [v / 1000000 for v in raw_values]
            y_axis_title = "Value (Millions)"
            hover_suffix = "M"
            tick_suffix = "M"
        elif value_format == 'crores':
            values = [v / 10000000 for v in raw_values]
            y_axis_title = "Value (₹ Crores)"
            hover_suffix = "Cr"
            tick_suffix = "Cr"
        else:
            values = raw_values
            y_axis_title = "Value (₹)"
            hover_suffix = ""
            tick_suffix = ""
            
        if status_filter == "Excess Inventory":
            y_axis_title = y_axis_title.replace("Value", "Excess Value Above Ideal")
            hover_label = "Excess Value"
        elif status_filter == "Short Inventory":
            y_axis_title = y_axis_title.replace("Value", "Short Value Below Ideal")
            hover_label = "Short Value"
        else:
            hover_label = "Stock Value"
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vendor_names,
            y=values,
            marker_color=color,
            text=[f"{hover_label}: ₹{v:,.1f}{hover_suffix}<br>Parts: {c}" for v, c in zip(values, counts)],
            hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
        ))
        fig.update_layout(
            title=dynamic_title,
            xaxis_title="Vendor",
            yaxis_title=y_axis_title,
            showlegend=False,
            yaxis=dict(tickformat=".1f", ticksuffix=tick_suffix),
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

    def safe_float_convert(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

class InventoryManagementSystem:
    """Main application class"""
    
    def __init__(self):
        self.debug = True
        self.analyzer = InventoryAnalyzer()
        self.persistence = DataPersistence()
        self.initialize_session_state()
        self.status_colors = {
            'Within Norms': '#4CAF50',    # Green
            'Excess Inventory': '#2196F3', # Blue
            'Short Inventory': '#F44336'   # Red
        }
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'default_tolerance': 30,
                'chart_theme': 'plotly'
            }
        
        # Initialize persistent data keys
        self.persistent_keys = [
            'persistent_pfep_data',
            'persistent_pfep_locked',
            'persistent_inventory_data', 
            'persistent_inventory_locked',
            'persistent_analysis_results'
        ]
        
        # Initialize persistent data if not exists
        for key in self.persistent_keys:
            if key not in st.session_state:
                st.session_state[key] = None

        # Initialize Admin Setting for Ideal Inventory Days
        if 'ideal_inventory_days' not in st.session_state:
            st.session_state.ideal_inventory_days = 30 # Default to 30 days

    def safe_print(self, message):
        """Safely print to streamlit or console"""
        try:
            st.write(message)
        except NameError:
            print(message)
    
    def safe_error(self, message):
        """Safely show error in streamlit or console"""
        try:
            st.error(message)
        except NameError:
            print(f"ERROR: {message}")
    
    def safe_warning(self, message):
        """Safely show warning in streamlit or console"""
        try:
            st.warning(message)
        except NameError:
            print(f"WARNING: {message}")
    
    def safe_float_convert(self, value: Any) -> float:
        """Safely convert value to float"""
        if value is None or pd.isna(value) or value == '':
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        try:
            # Clean string values
            if isinstance(value, str):
                value = value.strip()
                # Remove currency symbols and commas
                value = re.sub(r'[₹$€£,]', '', value)
                # Handle percentage
                if '%' in value:
                    return float(value.replace('%', '')) / 100
            return float(value)
        except (ValueError, TypeError) as e:
            # Debug problematic values
            if hasattr(self, 'debug') and self.debug:
                st.warning(f"⚠️ Could not convert '{value}' to float: {e}")
            return 0.0
            
    def safe_int_convert(self, value: Any) -> int:
        """Convert value to integer using safe float conversion."""
        float_result = self.safe_float_convert(value)
        return int(float_result)
            
    def authenticate_user(self):
        """Enhanced authentication system with better UX and user switching"""
        st.sidebar.markdown("### 🔐 Authentication")
        
        if st.session_state.user_role is None:
            role = st.sidebar.selectbox(
                "Select Role", 
                ["Select Role", "Admin", "User"],
                help="Choose your role to access appropriate features"
            )
            
            if role == "Admin":
                with st.sidebar.container():
                    st.markdown("**Admin Login**")
                    password = st.text_input("Admin Password", type="password", key="admin_pass")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔑 Login", key="admin_login"):
                            if password == "Agilomatrix@123":
                                st.session_state.user_role = "Admin"
                                st.success("✅ Admin authenticated!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid password")
                    with col2:
                        if st.button("🏠 Demo", key="admin_demo"):
                            st.session_state.user_role = "Admin"
                            st.info("🎮 Demo mode activated!")
                            st.rerun()
            
            elif role == "User":
                if st.sidebar.button("👤 Enter as User", key="user_login"):
                    st.session_state.user_role = "User"
                    st.sidebar.success("✅ User access granted!")
                    st.rerun()
        else:
            # User info and controls
            st.sidebar.success(f"✅ **{st.session_state.user_role}** logged in")
            
            # Display data status
            self.display_data_status()
            
            # User switching option for Admin
            if st.session_state.user_role == "Admin":
                # ✅ Show PFEP lock status
                pfep_locked = st.session_state.get("persistent_pfep_locked", False)
                st.sidebar.markdown(f"🔒 PFEP Locked: **{pfep_locked}**")
                # ✅ Always show switch role if PFEP is locked
                if pfep_locked:
                    st.sidebar.markdown("### 🔄 Switch Role")
                    if st.sidebar.button("👤 Switch to User View", key="switch_to_user"):
                        st.session_state.user_role = "User"
                        st.sidebar.success("✅ Switched to User view!")
                        st.rerun()
                else:
                    st.sidebar.info("ℹ️ PFEP is not locked. Lock PFEP to allow switching to User.")

            
            # User preferences (for Admin only)
            if st.session_state.user_role == "Admin":
                with st.sidebar.expander("⚙️ Preferences"):
                    st.session_state.user_preferences['chart_theme'] = st.selectbox(
                        "Chart Theme", ['plotly', 'plotly_white', 'plotly_dark'],
                        key="pref_theme"
                    )
            
            # Logout button
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Logout", key="logout_btn"):
                # Only clear user session, not persistent data
                keys_to_keep = self.persistent_keys + ['user_preferences', 'ideal_inventory_days']
                session_copy = {k: v for k, v in st.session_state.items() if k in keys_to_keep}
                
                # Clear all session state
                st.session_state.clear()
                
                # Restore persistent data
                for k, v in session_copy.items():
                    st.session_state[k] = v
                
                st.rerun()
    
    def display_data_status(self):
        """Display current data loading status in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Data Status")
        
        # Check persistent PFEP data
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if pfep_data:
            pfep_count = len(pfep_data)
            lock_icon = "🔒" if pfep_locked else "🔓"
            st.sidebar.success(f"✅ PFEP Data: {pfep_count} parts {lock_icon}")
            timestamp = self.persistence.get_data_timestamp('persistent_pfep_data')
            if timestamp:
                st.sidebar.caption(f"Loaded: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.sidebar.error("❌ PFEP Data: Not loaded")
        
        # Check persistent inventory data
        inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        inventory_locked = st.session_state.get('persistent_inventory_locked', False)
        
        if inventory_data:
            inv_count = len(inventory_data)
            lock_icon = "🔒" if inventory_locked else "🔓"
            st.sidebar.success(f"✅ Inventory: {inv_count} parts {lock_icon}")
            timestamp = self.persistence.get_data_timestamp('persistent_inventory_data')
            if timestamp:
                st.sidebar.caption(f"Loaded: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.sidebar.error("❌ Inventory: Not loaded")
        
        # Analysis results status
        analysis_data = self.persistence.load_data_from_session_state('persistent_analysis_results')
        if analysis_data:
            st.sidebar.info(f"📈 Analysis: {len(analysis_data)} parts analyzed")
    
    def load_sample_pfep_data(self):
        pfep_sample = [
           ["AC0303020106", "FLAT ALUMINIUM PROFILE", 4.000, "V001", "Vendor_A", "Mumbai", "Maharashtra", 2.5],
           ["JJ1010101010", "WINDSHIELD WASHER", 25, "V002", "Vendor_B", "Delhi", "Delhi", 1.8],
           ["TEST001", "HIGH CONSUMPTION PART", 100, "V003", "Vendor_C", "Pune", "MH", 10.0],
           ["TEST002", "LOW CONSUMPTION PART", 10, "V003", "Vendor_C", "Pune", "MH", 0.5]
        ]
        pfep_data = []
        for row in pfep_sample:
            pfep_data.append({
                'Part_No': row[0],
                'Description': row[1],
                'RM_IN_QTY': self.safe_float_convert(row[2]),
                'Vendor_Code': row[3],
                'Vendor_Name': row[4],
                'City': row[5],
                'State': row[6],
                'Unit_Price': 100,            
                'RM_IN_DAYS': 7,              
                'AVG CONSUMPTION/DAY': self.safe_float_convert(row[7]) if len(row) > 7 else 0
            })
        return pfep_data
    
    def load_sample_current_inventory(self):
        """Load sample current inventory data with consistent fields"""
        current_sample = [
            ["AC0303020106", "FLAT ALUMINIUM PROFILE", 5.230, 523],
            ["JJ1010101010", "WINDSHIELD WASHER", 33, 3300],
            ["TEST001", "HIGH CONSUMPTION PART", 500, 50000], # Excess (Needs ~300)
            ["TEST002", "LOW CONSUMPTION PART", 2, 200] # Short (Needs ~15)
        ]
        return [{
            'Part_No': row[0],
            'Description': row[1],
            'Current_QTY': self.safe_float_convert(row[2]),
            'Current Inventory - VALUE': self.safe_float_convert(row[3])
        } for row in current_sample]
    
    def standardize_pfep_data(self, df):
        """Enhanced PFEP data standardization with added Unit_Price and RM_IN_DAYS support"""
        if df is None or df.empty:
            return []
        
        column_mappings = {
            'part_no': [
                'part_no', 'part_number', 'material', 'material_code', 'item_code', 
                'code', 'part no', 'partno', 'Part No', 'Part_No', 'PART_NO'
            ],
            'description': [
                'description', 'item_description', 'part_description', 'desc', 
                'part description', 'material_description', 'item desc', 'Part Description'
            ],
            'rm_qty': [
                'rm_in_qty', 'rm_qty', 'required_qty', 'norm_qty', 'target_qty', 
                'rm', 'ri_in_qty', 'rm in qty', 'RM_IN_QTY', 'RM_QTY', 'RM IN QTY'
            ],
            'unit_price': [
                'unit_price', 'price', 'unit cost', 'unit rate', 'unitprice', 'Unit Price',
                'UNIT_PRICE', 'UNIT PRICE', 'Price', 'PRICE'
            ],
            'vendor_code': [
                'vendor_code', 'vendor_id', 'supplier_code', 'supplier_id', 'vendor id', 
                'Vendor Code', 'vendor code'
            ],
            'vendor_name': [
                'vendor_name', 'vendor', 'supplier_name', 'supplier', 'Vendor Name', 
                'vendor name'
            ],
            'avg_consumption_per_day': [
                'avg consumption/day', 'average consumption/day', 'avg_per_day',
                'avg daily usage', 'AVG CONSUMPTION/DAY', 'Average Per Day',
                'avg consumption per day', 'daily consumption', 'consumption per day'
            ],
            'city': ['city', 'location', 'place', 'City'],
            'state': ['state', 'region', 'province', 'State']
        }
        
        df_columns_lookup = {}
        for col in df.columns:
            if col is not None:
                clean_col = str(col).strip()
                df_columns_lookup[clean_col.lower()] = clean_col

        mapped_columns = {}
        for key, variations in column_mappings.items():
            for variation in variations:
                variation_lower = variation.lower().strip()
                if variation_lower in df_columns_lookup:
                    mapped_columns[key] = df_columns_lookup[variation_lower]
                    break
        
        if 'part_no' not in mapped_columns:
            st.error("❌ Part Number column not found. Please ensure your file has a Part Number column.")
            return []
            
        if 'avg_consumption_per_day' not in mapped_columns:
            st.warning("⚠️ AVG CONSUMPTION/DAY column not found. This is critical for Ideal Inventory calculation.")
            
        standardized_data = []
        for idx, row in df.iterrows():
            try:
                unit_price_value = 100.0
                if 'unit_price' in mapped_columns:
                    raw_price = row[mapped_columns['unit_price']]
                    unit_price_value = self.safe_float_convert(raw_price)

                avg_consumption_value = 0.0
                if 'avg_consumption_per_day' in mapped_columns:
                    raw_consumption = row[mapped_columns['avg_consumption_per_day']]
                    if pd.notna(raw_consumption) and str(raw_consumption).strip() != '':
                        avg_consumption_value = self.safe_float_convert(raw_consumption)
                        
                item = {
                    'Part_No': str(row[mapped_columns['part_no']]).strip(),
                    'Description': str(row.get(mapped_columns.get('description', ''), '')).strip(),
                    'RM_IN_QTY': self.safe_float_convert(row.get(mapped_columns.get('rm_qty', ''), 0)),
                    'unit_price': unit_price_value,
                    'Vendor_Code': str(row.get(mapped_columns.get('vendor_code', ''), '')).strip(),
                    'Vendor_Name': str(row.get(mapped_columns.get('vendor_name', ''), 'Unknown')).strip(),
                    'AVG CONSUMPTION/DAY': avg_consumption_value,
                    'City': str(row.get(mapped_columns.get('city', ''), '')).strip(),
                    'State': str(row.get(mapped_columns.get('state', ''), '')).strip()
                }
                
                if not item['Part_No'] or item['Part_No'].lower() in ['nan', 'none', '']:
                    continue
                standardized_data.append(item)
            except Exception as e:
                continue
        
        st.success(f"✅ Processed {len(standardized_data)} PFEP records")
        return standardized_data
    
    def standardize_current_inventory(self, df):
        """Standardize current inventory data with full column mappings and debugging."""
        if df is None or df.empty:
            return []
        column_mappings = {
            'part_no': ['part_no', 'part_number', 'material', 'material_code', 'item_code', 'code', 'part no', 'Part No'],
            'description': ['description', 'item_description', 'part_description', 'desc','Part Description'],
            'current_qty': ['current_qty', 'qty', 'quantity', 'stock_qty', 'available_qty', 'Current_QTY'],
            'stock_value': ['stock_value', 'value', 'total_value', 'inventory_value', 'Stock Value', 'Stock_Value','Current Inventory - VALUE', 'Current Inventory-VALUE'],
            'uom': ['uom', 'unit', 'unit_of_measure'],
            'location': ['location', 'store', 'warehouse', 'site'],
            'vendor_code': ['vendor_code', 'vendor id', 'supplier_code', 'Vendor Code'],
            'batch': ['batch', 'batch_number', 'lot', 'lot_number']
        }
        df_columns_lower = {col.lower().strip(): col for col in df.columns if col is not None}
        mapped_columns = {}
        for key, variations in column_mappings.items():
            for variation in variations:
                if variation.lower() in df_columns_lower:
                    mapped_columns[key] = df_columns_lower[variation.lower()]
                    break

        if 'part_no' not in mapped_columns or 'current_qty' not in mapped_columns:
            st.error("❌ Required columns not found. Please ensure your file has Part Number and Current Quantity columns.")
            return []
        standardized_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                part_no = str(row[mapped_columns['part_no']]).strip()
                if part_no.lower() in ('nan', '', 'none'):
                    continue
                item = {
                    'Part_No': part_no,
                    'Current_QTY': self.safe_float_convert(row[mapped_columns['current_qty']]),
                    'Current Inventory - VALUE': self.safe_float_convert(row.get(mapped_columns.get('stock_value', ''), 0)),
                    'Description': str(row.get(mapped_columns.get('description', ''), '')).strip(),
                    'UOM': str(row.get(mapped_columns.get('uom', ''), '')).strip(),
                    'Location': str(row.get(mapped_columns.get('location', ''), '')).strip(),
                    'Vendor_Code': str(row.get(mapped_columns.get('vendor_code', ''), '')).strip(),
                    'Batch': str(row.get(mapped_columns.get('batch', ''), '')).strip()
                }
                standardized_data.append(item)
            except Exception as e:
                continue
        return standardized_data
    
    def validate_inventory_against_pfep(self, inventory_data):
        """Validate inventory data against PFEP master data."""
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        if not pfep_data:
            return {'is_valid': False, 'issues': ['No PFEP data available'], 'warnings': []}
        
        def normalize(pn): return str(pn).strip().upper()
        pfep_df = pd.DataFrame(pfep_data)
        inventory_df = pd.DataFrame(inventory_data)
        pfep_df['Part_No'] = pfep_df['Part_No'].apply(normalize)
        inventory_df['Part_No'] = inventory_df['Part_No'].apply(normalize)

        pfep_parts = set(pfep_df['Part_No'])
        inventory_parts = set(inventory_df['Part_No'])

        issues = []
        warnings = []

        missing_parts = pfep_parts - inventory_parts
        extra_parts = inventory_parts - pfep_parts
        
        zero_qty_parts = inventory_df[inventory_df['Current_QTY'] == 0]['Part_No'].tolist()
        if zero_qty_parts:
            warnings.append(f"Parts with zero quantity: {len(zero_qty_parts)} parts")
        is_valid = len(issues) == 0
        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings,
            'pfep_parts_count': len(pfep_parts),
            'inventory_parts_count': len(inventory_parts),
            'matching_parts_count': len(pfep_parts & inventory_parts),
            'missing_parts_count': len(missing_parts),
            'extra_parts_count': len(extra_parts),
            'missing_parts_list': list(missing_parts),
            'extra_parts_list': list(extra_parts),
            'zero_qty_parts_list': zero_qty_parts
        }
        
    def admin_data_management(self):
        """Admin-only PFEP data management interface"""
        st.header("🔧 Admin Dashboard - PFEP Data Management")
        
        # Check if PFEP data is locked
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if pfep_locked:
            st.warning("🔒 PFEP data is currently locked. Users are working with this data.")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info("To modify PFEP data, first unlock it. This will reset all user analysis.")
            with col2:
                if st.button("🔓 Unlock Data", type="secondary"):
                    st.session_state.persistent_pfep_locked = False
                    st.session_state.persistent_inventory_data = None
                    st.session_state.persistent_inventory_locked = False
                    st.session_state.persistent_analysis_results = None
                    st.success("✅ PFEP data unlocked. Users need to re-upload inventory data.")
                    st.rerun()
            with col3:
                if st.button("👤 Go to User View", type="primary", help="Switch to user interface"):
                    st.session_state.user_role = "User"
                    st.rerun()
            
            # Display current PFEP data if available
            pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
            if pfep_data:
                self.display_pfep_data_preview(pfep_data)
            return

        # --- ADMIN SETTINGS SECTION ---
        st.subheader("⚙️ Inventory Settings (Admin Only)")
        
        col_tol, col_days = st.columns(2)
        
        with col_tol:
            # Initialize admin_tolerance if not exists
            if "admin_tolerance" not in st.session_state:
                st.session_state.admin_tolerance = 30
        
            new_tolerance = st.selectbox(
                "Deviation Tolerance Zone (+/-)",
                options=[0, 1, 5, 10, 20, 30, 40, 50],
                index=[0, 1, 5, 10, 20, 30, 40, 50].index(st.session_state.admin_tolerance),
                format_func=lambda x: f"{x}%",
                key="tolerance_selector",
                help="If Deviation % is more than this, it is Excess. If less than negative of this, it is Short."
            )
            if new_tolerance != st.session_state.admin_tolerance:
                st.session_state.admin_tolerance = new_tolerance
                st.success(f"✅ Tolerance updated to {new_tolerance}%")
                
        with col_days:
            # Initialize ideal_inventory_days if not exists
            if "ideal_inventory_days" not in st.session_state:
                st.session_state.ideal_inventory_days = 30

            new_ideal_days = st.number_input(
                "Ideal Inventory Days",
                min_value=1,
                max_value=365,
                value=st.session_state.ideal_inventory_days,
                step=1,
                key="ideal_days_selector",
                help="Ideal Inventory = Avg Daily Consumption * Ideal Inventory Days"
            )
            if new_ideal_days != st.session_state.ideal_inventory_days:
                st.session_state.ideal_inventory_days = new_ideal_days
                st.success(f"✅ Ideal Inventory Days updated to {new_ideal_days} days")

        st.markdown(f"**Current Settings:** Ideal Days = {st.session_state.ideal_inventory_days} | Tolerance = ±{st.session_state.admin_tolerance}%")
        st.markdown("---")
        
        # PFEP Data Management Section
        st.subheader("📊 PFEP Master Data Management")
        
        tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🧪 Load Sample", "📋 Current Data"])
        
        with tab1:
            st.markdown("**Upload PFEP Excel/CSV File**")
            uploaded_file = st.file_uploader(
                "Choose PFEP file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your PFEP master data file. Must contain 'AVG CONSUMPTION/DAY'."
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded: {len(df)} rows")
                    
                    standardized_data = self.standardize_pfep_data(df)
                    
                    if standardized_data:
                        st.success(f"✅ Standardized: {len(standardized_data)} valid records")
                        
                        if st.button("💾 Save PFEP Data", type="primary"):
                            self.persistence.save_data_to_session_state(
                                'persistent_pfep_data', 
                                standardized_data
                            )
                            st.success("✅ PFEP data saved successfully!")
                            st.rerun()
                    else:
                        st.error("❌ No valid data found after standardization")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        with tab2:
            st.markdown("**Load Sample PFEP Data for Testing**")
            
            if st.button("🧪 Load Sample PFEP Data", type="secondary"):
                sample_data = self.load_sample_pfep_data()
                self.persistence.save_data_to_session_state(
                    'persistent_pfep_data', 
                    sample_data
                )
                st.success(f"✅ Sample PFEP data loaded: {len(sample_data)} parts")
                st.rerun()
        
        with tab3:
            st.markdown("**Current PFEP Data Status**")
            pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
            
            if pfep_data:
                self.display_pfep_data_preview(pfep_data)
                
                st.markdown("---")
                st.markdown("**🔒 Lock Data for Users**")
                st.info("Locking PFEP data allows users to upload inventory and perform analysis")
                
                if st.button("🔒 Lock PFEP Data", type="primary"):
                    st.session_state.persistent_pfep_locked = True
                    st.success("✅ PFEP data locked! Users can now upload inventory data.")
                    st.rerun()
            else:
                st.warning("❌ No PFEP data available. Please upload or load sample data first.")
    
    def display_pfep_data_preview(self, pfep_data):
        """Display PFEP data preview with statistics"""
        st.markdown("**📊 PFEP Data Overview**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parts", len(pfep_data))
        with col2:
            vendors = set(item.get('Vendor_Name', 'Unknown') for item in pfep_data)
            st.metric("Unique Vendors", len(vendors))
        with col3:
            total_consumption = sum(item.get('AVG CONSUMPTION/DAY', 0) for item in pfep_data)
            st.metric("Total Avg Daily Consumption", f"{total_consumption:,.0f}")
        with col4:
            avg_unit_price = sum(item.get('unit_price', 0) for item in pfep_data) / len(pfep_data) if len(pfep_data) > 0 else 0
            st.metric("Avg Unit Price", f"₹{avg_unit_price:.2f}")
        
        with st.expander("📋 Data Preview (First 10 rows)"):
            preview_df = pd.DataFrame(pfep_data[:10])
            st.dataframe(preview_df)
    
    def user_inventory_upload(self):
        """User interface for inventory upload and analysis"""
        st.header("📦 Inventory Analysis System")
        
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if not pfep_data or not pfep_locked:
            st.error("❌ PFEP master data is not available or not locked by admin.")
            st.info("Please contact admin to load and lock PFEP data first.")
            return
        
        st.success(f"✅ PFEP Master Data: {len(pfep_data)} parts available")
        
        inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        inventory_locked = st.session_state.get('persistent_inventory_locked', False)
        
        if inventory_locked and inventory_data:
            st.info("🔒 Inventory data is locked. Proceeding to analysis...")
            self.display_analysis_interface()
            return
        
        st.subheader("📊 Upload Current Inventory Data")
        
        tab1, tab2 = st.tabs(["📁 Upload File", "🧪 Load Sample"])
        
        with tab1:
            st.markdown("**Upload Current Inventory Excel/CSV File**")
            uploaded_file = st.file_uploader(
                "Choose inventory file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your current inventory data file"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded: {len(df)} rows")
                    standardized_data = self.standardize_current_inventory(df)
                    
                    if standardized_data:
                        st.success(f"✅ Standardized: {len(standardized_data)} valid records")
                        
                        validation_result = self.validate_inventory_against_pfep(standardized_data)
                        self.display_validation_results(validation_result)
                        
                        if validation_result['is_valid']:
                            if st.button("💾 Save & Lock Inventory Data", type="primary"):
                                self.persistence.save_data_to_session_state(
                                    'persistent_inventory_data', 
                                    standardized_data
                                )
                                st.session_state.persistent_inventory_locked = True
                                st.success("✅ Inventory data saved and locked!")
                                st.rerun()
                        else:
                            st.error("❌ Please fix validation issues before proceeding")
                    else:
                        st.error("❌ No valid data found after standardization")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        with tab2:
            st.markdown("**Load Sample Inventory Data for Testing**")
            
            if st.button("🧪 Load Sample Inventory Data", type="secondary"):
                sample_data = self.load_sample_current_inventory()
                
                validation_result = self.validate_inventory_against_pfep(sample_data)
                self.display_validation_results(validation_result)
                
                if validation_result['is_valid']:
                    self.persistence.save_data_to_session_state(
                        'persistent_inventory_data', 
                        sample_data
                    )
                    st.session_state.persistent_inventory_locked = True
                    st.success(f"✅ Sample inventory data loaded and locked: {len(sample_data)} parts")
                    st.rerun()

    def display_validation_results(self, validation_result):
        """Display inventory validation results"""
        st.markdown("**📋 Validation Results**")
        if validation_result['is_valid']:
            st.success("✅ Inventory data validation passed!")
        else:
            st.error("❌ Inventory data validation failed!")
            for issue in validation_result['issues']:
                st.error(f"• {issue}")
        
        if validation_result['warnings']:
            st.warning("⚠️ Validation Warnings:")
            for warning in validation_result['warnings']:
                st.warning(f"• {warning}")
    
    def display_analysis_interface(self):
        """Main analysis interface for users"""
        st.subheader("📈 Inventory Analysis Results")
        try:
            pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
            inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        except Exception as e:
            st.error("❌ Error loading PFEP or Inventory data.")
            st.code(str(e))
            return
        
        if not pfep_data or not inventory_data:
            st.error("❌ Required data not available.")
            return
        
        # Get tolerance and ideal days from settings
        tolerance = st.session_state.get('admin_tolerance', 30)
        ideal_days = st.session_state.get('ideal_inventory_days', 30)
        
        st.info(f"📐 Settings: Ideal Days = {ideal_days} | Tolerance = ±{tolerance}%")

        # Auto re-analyze if needed
        analysis_data = self.persistence.load_data_from_session_state('persistent_analysis_results')
        
        # We should check if settings changed
        last_tolerance = st.session_state.get('last_analysis_tolerance', None)
        last_ideal_days = st.session_state.get('last_analysis_ideal_days', None)

        if not analysis_data or last_tolerance != tolerance or last_ideal_days != ideal_days:
            st.info(f"🔄 Re-analyzing with new settings...")
            with st.spinner("Analyzing inventory..."):
                try:
                    analysis_results = self.analyzer.analyze_inventory(
                        pfep_data,
                        inventory_data,
                        tolerance=tolerance
                    )
                except Exception as e:
                    st.error("❌ Error during inventory analysis")
                    st.code(str(e))
                    return
            if analysis_results:
                self.persistence.save_data_to_session_state('persistent_analysis_results', analysis_results)
                st.session_state.last_analysis_tolerance = tolerance
                st.session_state.last_analysis_ideal_days = ideal_days
                st.success("✅ Analysis completed successfully!")
                st.rerun()
            else:
                st.error("❌ Analysis failed. No results generated.")
                return
        
        try:
            self.display_analysis_results()
        except Exception as e:
            st.error("❌ Unexpected error during analysis results display")
            st.code(str(e))
            return
            
    def display_comprehensive_analysis(self, analysis_results):
        """Display comprehensive analysis results"""
        st.success(f"✅ Analysis Complete: {len(analysis_results)} parts analyzed")
        
        try:
            self.display_enhanced_summary_metrics(analysis_results)
        except Exception as e:
            st.error("❌ Error in Summary Metrics")
        
        try:
            self.display_enhanced_analysis_charts(analysis_results)
        except Exception as e:
            st.error("❌ Error in Charts")
            st.code(str(e))

        try:
            self.display_enhanced_detailed_tables(analysis_results)
        except Exception as e:
            st.error("❌ Error in Detailed Tables")

    def display_enhanced_summary_metrics(self, analysis_results):
        """Enhanced summary metrics dashboard"""
        st.header("📊 Executive Summary Dashboard")
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
            text-align: center;
            color: white;
        }
        .status-normal { background: linear-gradient(135deg, #4CAF50, #45a049); }
        .status-excess { background: linear-gradient(135deg, #2196F3, #1976D2); }
        .status-short { background: linear-gradient(135deg, #F44336, #D32F2F); }
        .metric-value { font-size: 1.6rem; font-weight: bold; }
        .metric-label { font-size: 1.1rem; }
        </style>
        """, unsafe_allow_html=True)
        
        df = pd.DataFrame(analysis_results)
        value_col = 'Stock Deviation Value'
        status_col = 'Status'
        
        if not df.empty:
            short_value = df[df[status_col] == 'Short Inventory'][value_col].sum()
            excess_value = df[df[status_col] == 'Excess Inventory'][value_col].sum()
            total_stock_value = df['Current Inventory - VALUE'].sum()
        else:
            short_value, excess_value, total_stock_value = 0, 0, 0
            
        cols = st.columns(4)
        
        status_counts = df[status_col].value_counts()
        
        with cols[0]:
            st.markdown(f"""<div class="metric-card status-normal">
                <div class="metric-label">Within Norms</div>
                <div class="metric-value">{status_counts.get('Within Norms', 0)} parts</div>
            </div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""<div class="metric-card status-excess">
                <div class="metric-label">Excess Inventory</div>
                <div class="metric-value">{status_counts.get('Excess Inventory', 0)} parts</div>
                <div>₹{excess_value:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""<div class="metric-card status-short">
                <div class="metric-label">Short Inventory</div>
                <div class="metric-value">{status_counts.get('Short Inventory', 0)} parts</div>
                <div>₹{abs(short_value):,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Total Inventory</div>
                <div class="metric-value">₹{total_stock_value:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    def display_enhanced_detailed_tables(self, analysis_results):
        """Display detailed tables"""
        st.header("📊 Detailed Analysis Tables")
        df = pd.DataFrame(analysis_results)
        
        tab1, tab2, tab3 = st.tabs(["🔴 Short Inventory", "🔵 Excess Inventory", "🔍 All Items"])
        
        cols_to_show = [
            'PART NO', 'PART DESCRIPTION', 'Status', 'Deviation Percentage', 
            'Current Inventory - Qty', 'Ideal Inventory Qty', 
            'Current Inventory - VALUE', 'Stock Deviation Value'
        ]
        
        with tab1:
            short_items = df[df['Status'] == 'Short Inventory'].copy()
            if not short_items.empty:
                st.dataframe(short_items[cols_to_show].sort_values('Deviation Percentage'), use_container_width=True)
            else:
                st.success("No shortage items.")
                
        with tab2:
            excess_items = df[df['Status'] == 'Excess Inventory'].copy()
            if not excess_items.empty:
                st.dataframe(excess_items[cols_to_show].sort_values('Deviation Percentage', ascending=False), use_container_width=True)
            else:
                st.success("No excess items.")
                
        with tab3:
            st.dataframe(df[cols_to_show], use_container_width=True)

    def display_enhanced_analysis_charts(self, analysis_results):
        """Display enhanced visual summaries including Ideal Inventory Line"""
        st.subheader("📊 Enhanced Inventory Charts")
        
        # Add Unit Toggle AND Top N Slider
        col1, col2, col3 = st.columns([1, 2, 3])
        with col1:
            chart_unit = st.selectbox("Select Currency Unit:", ["Lakhs", "Millions"], key="chart_unit_selector")
        with col2:
            top_n = st.slider("Number of items to show:", min_value=5, max_value=50, value=10, step=5, key="top_n_slider")
        
        if chart_unit == "Millions":
            divisor = 1_000_000
            suffix = "M"
            unit_name = "Millions"
        else:
            divisor = 100_000
            suffix = "L"
            unit_name = "Lakhs"
            
        df = pd.DataFrame(analysis_results)
        if df.empty:
            st.warning("⚠️ No data available for charts.")
            return
            
        # ✅ 1. Top N Parts by Value with Ideal Line
        value_col = 'Current Inventory - VALUE'
        if value_col in df.columns:
            # Filter top N parts with non-zero value
            chart_data = (
                df[df[value_col] > 0]
                .sort_values(by=value_col, ascending=False)
                .head(top_n)
                .copy()
            )
            
            # Convert values
            chart_data['Value_Converted'] = chart_data[value_col] / divisor
            chart_data['Ideal_Value_Converted'] = chart_data['Ideal Inventory Value'] / divisor
            
            chart_data['Part'] = chart_data.apply(
                lambda row: f"{row['PART DESCRIPTION'][:20]}...\n({row['PART NO']})",
                axis=1
            )
            
            color_map = {
                "Excess Inventory": "#2196F3",
                "Short Inventory": "#F44336", 
                "Within Norms": "#4CAF50"
            }
            
            chart_data['Bar_Color'] = chart_data['Status'].map(color_map)
    
            # Create Combo Chart (Bar + Line)
            fig1 = go.Figure()
            
            # Bar Trace
            fig1.add_trace(go.Bar(
                x=chart_data['Part'],
                y=chart_data['Value_Converted'],
                name='Current Stock Value',
                marker_color=chart_data['Bar_Color'],
                text=[f"Dev: {row['Deviation Percentage']:.1f}%" for _, row in chart_data.iterrows()],
                textposition='auto'
            ))
            
            # Line Trace for Ideal Inventory
            fig1.add_trace(go.Scatter(
                x=chart_data['Part'],
                y=chart_data['Ideal_Value_Converted'],
                mode='lines+markers',
                name='Ideal Inventory Value',
                line=dict(color='black', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))

            fig1.update_layout(
                title=f"Top {top_n} Parts: Actual vs Ideal Value",
                xaxis_title="Parts",
                yaxis_title=f"Value ({unit_name})",
                xaxis_tickangle=-45,
                yaxis=dict(tickformat=',.1f', ticksuffix=suffix),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            st.info("ℹ️ The black dashed line represents the Ideal Inventory Value based on Average Daily Consumption.")

    def apply_advanced_filters(self, df):
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        # Filters can be added here
        return filtered_df
        
    def run(self):
        st.title("📊 Inventory Analyzer")
        st.markdown("<p style='font-size:18px; font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
        st.markdown("---")

        self.authenticate_user()

        if st.session_state.user_role == "Admin":
            self.admin_data_management()
        elif st.session_state.user_role == "User":
            self.user_inventory_upload()
        else:
            st.info("👋 Please select your role and authenticate to access the system.")

    def display_analysis_results(self):
        """Main method to display all analysis results"""
        analysis_results = self.persistence.load_data_from_session_state('persistent_analysis_results')
        if not analysis_results:
            st.error("❌ No analysis results available.")
            return

        df = pd.DataFrame(analysis_results)
        self.display_comprehensive_analysis(df.to_dict('records'))

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
