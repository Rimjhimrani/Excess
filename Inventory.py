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
    
    def safe_float_convert(self, value, default=0.0):
        try:
            if isinstance(value, str):
                value = re.sub(r'[₹$€£,]', '', value)
                if '%' in value:
                    return float(value.replace('%', '')) / 100
            return float(value)
        except (ValueError, TypeError):
            return default

    def analyze_inventory(self, pfep_data, current_inventory, tolerance=None):
        """Analyze inventory using Admin defined Ideal Days.
        
        Formulas:
        1. Ideal Inventory = Average Daily Consumption * Admin Ideal Days
        2. Deviation % = ((Current - Ideal) / Ideal) * 100
        3. Status: 
           - Positive Deviation > Tolerance = Excess
           - Negative Deviation < -Tolerance = Short
        """
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)
        
        # ✅ Get Admin Ideal Days
        admin_ideal_days = st.session_state.get("admin_ideal_days", 30)
        
        results = []
        # Normalize and create lookup dictionaries
        pfep_dict = {str(item['Part_No']).strip().upper(): item for item in pfep_data}
        inventory_dict = {str(item['Part_No']).strip().upper(): item for item in current_inventory}
        
        for part_no, inventory_item in inventory_dict.items():
            pfep_item = pfep_dict.get(part_no)
            if not pfep_item:
                continue  # Skip unmatched parts
            try:
                # From Inventory & PFEP
                current_qty = float(inventory_item.get('Current_QTY', 0)) or 0.0
                part_desc = pfep_item.get('Description', '')
                unit_price = float(pfep_item.get('unit_price', 0)) or 1.0
                avg_per_day = self.safe_float_convert(pfep_item.get('AVG CONSUMPTION/DAY', 0))
                
                # ✅ NEW CALCULATION: Ideal Inventory
                # Ideal Inventory = Avg Daily Consumption * Ideal Inventory Days (Admin defined)
                ideal_inventory_qty = avg_per_day * admin_ideal_days
                
                # ✅ NEW CALCULATION: Deviation Percentage
                # Formula: ((Current - Ideal) / Ideal) * 100
                if ideal_inventory_qty > 0:
                    deviation_pct = ((current_qty - ideal_inventory_qty) / ideal_inventory_qty) * 100
                    deviation_qty = current_qty - ideal_inventory_qty
                else:
                    # Handle cases where Ideal is 0 (no consumption)
                    if current_qty > 0:
                        deviation_pct = 100.0 # Treat as 100% excess if ideal is 0 but stock exists
                        deviation_qty = current_qty
                    else:
                        deviation_pct = 0.0
                        deviation_qty = 0.0

                # Calculate Values
                current_value = current_qty * unit_price
                ideal_value = ideal_inventory_qty * unit_price
                deviation_value = deviation_qty * unit_price
                
                # Determine Status based on Deviation % and Tolerance
                if deviation_pct > tolerance:
                    status = 'Excess Inventory'
                elif deviation_pct < -tolerance:
                    status = 'Short Inventory'
                else:
                    status = 'Within Norms'

                # Calculate bounds for reference (visuals)
                lower_bound = ideal_inventory_qty * (1 - tolerance / 100)
                upper_bound = ideal_inventory_qty * (1 + tolerance / 100)

                # Final result per part
                result = {
                    'PART NO': part_no,
                    'PART DESCRIPTION': part_desc,
                    'Vendor Name': pfep_item.get('Vendor_Name', 'Unknown'),
                    'Vendor_Code': pfep_item.get('Vendor_Code', ''),
                    'AVG CONSUMPTION/DAY': avg_per_day,
                    'RM IN DAYS': admin_ideal_days,              # Showing Admin Days here
                    'RM Norm - In Qty': ideal_inventory_qty,     # This is now the Ideal Qty
                    'Revised Norm Qty': ideal_inventory_qty,     # Main comparator
                    'Ideal Inventory Qty': ideal_inventory_qty,  # Explicit new key
                    'Lower Bound Qty': lower_bound,
                    'Upper Bound Qty': upper_bound,  
                    'UNIT PRICE': unit_price,
                    'Current Inventory - Qty': current_qty,
                    'Current Inventory - VALUE': current_value,
                    'Ideal Inventory - VALUE': ideal_value,      # ✅ Added for Line Graph
                    'SHORT/EXCESS INVENTORY': deviation_qty,
                    'Inventory Deviation %': deviation_pct,      # ✅ Calculated %
                    'Stock Deviation Value': deviation_value,
                    'Status': status,
                    'INVENTORY REMARK STATUS': status
                }
                results.append(result)
            except Exception as e:
                if self.debug:
                    st.warning(f"⚠️ Error analyzing part {part_no}: {e}")
                continue
        if not results:
            st.error("❌ No analysis results generated. Please check data for mismatches or missing fields.")
        return results 
        
    def get_vendor_summary(self, processed_data):
        """Summarize inventory by vendor."""
        summary = defaultdict(lambda: {
            'total_parts': 0,
            'short_parts': 0,
            'excess_parts': 0,
            'normal_parts': 0,
            'total_value': 0.0,
            'excess_value_above_norm': 0.0,
            'short_value_below_norm': 0.0
        })
        for item in processed_data:
            vendor = item.get('Vendor Name', 'Unknown')
            status = item.get('INVENTORY REMARK STATUS', 'Unknown')
            stock_value = item.get('Current Inventory - VALUE', 0)
            
            summary[vendor]['total_parts'] += 1
            summary[vendor]['total_value'] += stock_value
            
            deviation_val = item.get('Stock Deviation Value', 0)
            
            if status == "Short Inventory":
                summary[vendor]['short_parts'] += 1
                summary[vendor]['short_value_below_norm'] += abs(deviation_val)
            elif status == "Excess Inventory":
                summary[vendor]['excess_parts'] += 1
                summary[vendor]['excess_value_above_norm'] += deviation_val
            elif status == "Within Norms":
                summary[vendor]['normal_parts'] += 1
        return summary
        
    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, value_format='lakhs', top_n=10):
        """Show top N vendors by deviation value."""
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        vendor_totals = {}
        vendor_counts = {}
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            # Use calculated deviation value
            deviation_value = item.get('Stock Deviation Value', 0)
            
            # For charts, we want magnitude
            if status_filter == "Short Inventory":
                deviation_value = abs(deviation_value)
            
            # Only count if meaningful
            if deviation_value > 0:
                vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + deviation_value
                vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1

        if not vendor_totals:
            st.info(f"No vendors found in '{status_filter}' status.")
            return
            
        # Combine and sort
        combined = [
            (vendor, vendor_totals[vendor], vendor_counts.get(vendor, 0))
            for vendor in vendor_totals
        ]
        top_vendors = sorted(combined, key=lambda x: x[1], reverse=True)[:top_n]
        
        vendor_names = [v[0] for v in top_vendors]
        raw_values = [v[1] for v in top_vendors]
        counts = [v[2] for v in top_vendors]
        
        # Update Title
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
        else:
            values = raw_values
            y_axis_title = "Value (₹)"
            hover_suffix = ""
            tick_suffix = ""
            
        hover_label = "Excess Value" if status_filter == "Excess Inventory" else "Shortage Value"

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
        
        # ✅ NEW: Initialize Admin Ideal Inventory Days
        if 'admin_ideal_days' not in st.session_state:
            st.session_state.admin_ideal_days = 30  # Default to 30 days
        
        # Initialize persistent data keys
        self.persistent_keys = [
            'persistent_pfep_data',
            'persistent_pfep_locked',
            'persistent_inventory_data', 
            'persistent_inventory_locked',
            'persistent_analysis_results'
        ]
        
        for key in self.persistent_keys:
            if key not in st.session_state:
                st.session_state[key] = None

    def safe_float_convert(self, value: Any) -> float:
        try:
            if isinstance(value, str):
                value = re.sub(r'[₹$€£,]', '', value)
                if '%' in value:
                    return float(value.replace('%', '')) / 100
            return float(value)
        except (ValueError, TypeError):
            return 0.0
            
    def authenticate_user(self):
        """Enhanced authentication system"""
        st.sidebar.markdown("### 🔐 Authentication")
        
        if st.session_state.user_role is None:
            role = st.sidebar.selectbox("Select Role", ["Select Role", "Admin", "User"])
            
            if role == "Admin":
                with st.sidebar.container():
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
            st.sidebar.success(f"✅ **{st.session_state.user_role}** logged in")
            self.display_data_status()
            
            if st.session_state.user_role == "Admin":
                pfep_locked = st.session_state.get("persistent_pfep_locked", False)
                st.sidebar.markdown(f"🔒 PFEP Locked: **{pfep_locked}**")
                if pfep_locked:
                    st.sidebar.markdown("### 🔄 Switch Role")
                    if st.sidebar.button("👤 Switch to User View", key="switch_to_user"):
                        st.session_state.user_role = "User"
                        st.sidebar.success("✅ Switched to User view!")
                        st.rerun()
            
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Logout", key="logout_btn"):
                keys_to_keep = self.persistent_keys + ['user_preferences', 'admin_ideal_days']
                session_copy = {k: v for k, v in st.session_state.items() if k in keys_to_keep}
                st.session_state.clear()
                for k, v in session_copy.items():
                    st.session_state[k] = v
                st.rerun()
    
    def display_data_status(self):
        """Display current data loading status in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Data Status")
        
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if pfep_data:
            st.sidebar.success(f"✅ PFEP Data: {len(pfep_data)} parts {'🔒' if pfep_locked else '🔓'}")
        else:
            st.sidebar.error("❌ PFEP Data: Not loaded")
        
        inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        if inventory_data:
            st.sidebar.success(f"✅ Inventory: {len(inventory_data)} parts")
        else:
            st.sidebar.error("❌ Inventory: Not loaded")
    
    def load_sample_pfep_data(self):
        pfep_sample = [
           ["AC0303020106", "FLAT ALUMINIUM PROFILE", 4.000, "V001", "Vendor_A", "Mumbai", "Maharashtra", 2.5],
           ["JJ1010101010", "WINDSHIELD WASHER", 25, "V002", "Vendor_B", "Delhi", "Delhi", 1.8]
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
                'AVG CONSUMPTION/DAY': self.safe_float_convert(row[7]) if len(row) > 7 else ""
            })
        return pfep_data
    
    def load_sample_current_inventory(self):
        current_sample = [
            ["AC0303020106", "FLAT ALUMINIUM PROFILE", 5.230, 496],
            ["JJ1010101010", "WINDSHIELD WASHER", 33, 495]
        ]
        return [{
            'Part_No': row[0],
            'Description': row[1],
            'Current_QTY': self.safe_float_convert(row[2]),
            'Current Inventory - VALUE': self.safe_float_convert(row[3])
        } for row in current_sample]
    
    def standardize_pfep_data(self, df):
        if df is None or df.empty: return []
        
        # Simple mapping for robustness
        col_map = {}
        normalized_cols = {col.strip().lower(): col for col in df.columns}
        
        targets = {
            'part_no': ['part_no', 'part no', 'material', 'item_code'],
            'description': ['description', 'part description', 'material description'],
            'unit_price': ['unit_price', 'price', 'unit cost', 'rate'],
            'avg_consumption': ['avg consumption/day', 'avg_per_day', 'daily consumption'],
            'vendor_name': ['vendor_name', 'vendor', 'supplier']
        }
        
        for key, possible_names in targets.items():
            for name in possible_names:
                if name in normalized_cols:
                    col_map[key] = normalized_cols[name]
                    break
        
        if 'part_no' not in col_map:
            st.error("❌ Part Number column not found.")
            return []

        standardized_data = []
        for idx, row in df.iterrows():
            try:
                item = {
                    'Part_No': str(row[col_map['part_no']]).strip(),
                    'Description': str(row.get(col_map.get('description'), '')).strip(),
                    'unit_price': self.safe_float_convert(row.get(col_map.get('unit_price'), 100.0)),
                    'AVG CONSUMPTION/DAY': self.safe_float_convert(row.get(col_map.get('avg_consumption'), 0.0)),
                    'Vendor_Name': str(row.get(col_map.get('vendor_name'), 'Unknown')).strip(),
                    # Default values for fields not strictly required for this calculation
                    'RM_IN_DAYS': 7, 
                    'RM_IN_QTY': 0 
                }
                if item['Part_No'].lower() not in ['nan', 'none', '']:
                    standardized_data.append(item)
            except Exception:
                continue
        return standardized_data
    
    def standardize_current_inventory(self, df):
        if df is None or df.empty: return []
        
        col_map = {}
        normalized_cols = {col.strip().lower(): col for col in df.columns}
        
        targets = {
            'part_no': ['part_no', 'part no', 'material', 'item_code'],
            'current_qty': ['current_qty', 'qty', 'quantity', 'stock'],
            'stock_value': ['stock_value', 'value', 'amount', 'total_value']
        }
        
        for key, possible_names in targets.items():
            for name in possible_names:
                if name in normalized_cols:
                    col_map[key] = normalized_cols[name]
                    break
        
        if 'part_no' not in col_map or 'current_qty' not in col_map:
            st.error("❌ Required columns (Part No, Quantity) not found.")
            return []

        standardized_data = []
        for _, row in df.iterrows():
            try:
                item = {
                    'Part_No': str(row[col_map['part_no']]).strip(),
                    'Current_QTY': self.safe_float_convert(row[col_map['current_qty']]),
                    'Current Inventory - VALUE': self.safe_float_convert(row.get(col_map.get('stock_value'), 0)),
                    'Description': str(row.get('Description', '')).strip()
                }
                if item['Part_No'].lower() not in ['nan', 'none', '']:
                    standardized_data.append(item)
            except: continue
        return standardized_data
    
    def validate_inventory_against_pfep(self, inventory_data):
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        if not pfep_data:
            return {'is_valid': False, 'issues': ['No PFEP data available'], 'warnings': []}
        
        pfep_parts = set(str(i['Part_No']).strip().upper() for i in pfep_data)
        inv_parts = set(str(i['Part_No']).strip().upper() for i in inventory_data)
        
        return {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'pfep_parts_count': len(pfep_parts),
            'inventory_parts_count': len(inv_parts),
            'matching_parts_count': len(pfep_parts.intersection(inv_parts)),
            'missing_parts_count': len(pfep_parts - inv_parts),
            'extra_parts_count': len(inv_parts - pfep_parts),
            'missing_parts_list': list(pfep_parts - inv_parts),
            'extra_parts_list': list(inv_parts - pfep_parts)
        }
        
    def admin_data_management(self):
        """Admin-only PFEP data management interface"""
        st.header("🔧 Admin Dashboard - PFEP Data Management")
        
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if pfep_locked:
            st.warning("🔒 PFEP data is currently locked.")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔓 Unlock Data", type="secondary"):
                    st.session_state.persistent_pfep_locked = False
                    st.session_state.persistent_inventory_data = None
                    st.session_state.persistent_analysis_results = None
                    st.success("Unlocked.")
                    st.rerun()
            with col2:
                if st.button("👤 Go to User View", type="primary"):
                    st.session_state.user_role = "User"
                    st.rerun()
            
            pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
            if pfep_data:
                self.display_pfep_data_preview(pfep_data)
            return

        # ✅ NEW: Global Inventory Parameters Section
        st.subheader("📐 Set Inventory Parameters (Admin Only)")
        st.info("These settings apply to the calculation of Ideal Inventory and Deviation.")
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            if "admin_tolerance" not in st.session_state:
                st.session_state.admin_tolerance = 30
        
            new_tolerance = st.selectbox(
                "Deviation Tolerance Zone (+/-)",
                options=[0, 10, 20, 30, 40, 50],
                index=[0, 10, 20, 30, 40, 50].index(st.session_state.admin_tolerance),
                format_func=lambda x: f"{x}%",
                key="tolerance_selector"
            )
            if new_tolerance != st.session_state.admin_tolerance:
                st.session_state.admin_tolerance = new_tolerance
                # Force recalculation if results exist
                if st.session_state.get('persistent_analysis_results'):
                    st.session_state.persistent_analysis_results = None
                st.success("Tolerance Updated")

        with param_col2:
            # ✅ Input for Ideal Inventory Days
            new_days = st.number_input(
                "Ideal Inventory Days",
                min_value=1,
                max_value=365,
                value=st.session_state.admin_ideal_days,
                step=1,
                help="Ideal Inventory = Avg Daily Consumption * Ideal Days"
            )
            if new_days != st.session_state.admin_ideal_days:
                st.session_state.admin_ideal_days = new_days
                # Force recalculation if results exist
                if st.session_state.get('persistent_analysis_results'):
                    st.session_state.persistent_analysis_results = None
                st.success(f"Ideal Days Updated to {new_days}")

        st.markdown("---")
        
        # Tab interface
        tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🧪 Load Sample", "📋 Current Data"])
        
        with tab1:
            uploaded_file = st.file_uploader("Choose PFEP file", type=['xlsx', 'xls', 'csv'])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                    else: df = pd.read_excel(uploaded_file)
                    
                    standardized_data = self.standardize_pfep_data(df)
                    if standardized_data:
                        st.success(f"✅ Standardized: {len(standardized_data)} records")
                        if st.button("💾 Save PFEP Data", type="primary"):
                            self.persistence.save_data_to_session_state('persistent_pfep_data', standardized_data)
                            st.success("Saved!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with tab2:
            if st.button("🧪 Load Sample PFEP Data", type="secondary"):
                sample_data = self.load_sample_pfep_data()
                self.persistence.save_data_to_session_state('persistent_pfep_data', sample_data)
                st.success("Sample Loaded")
                st.rerun()
        
        with tab3:
            pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
            if pfep_data:
                self.display_pfep_data_preview(pfep_data)
                if st.button("🔒 Lock PFEP Data", type="primary"):
                    st.session_state.persistent_pfep_locked = True
                    st.success("Locked!")
                    st.rerun()
            else:
                st.warning("No Data")
    
    def display_pfep_data_preview(self, pfep_data):
        st.markdown("**📊 PFEP Data Overview**")
        col1, col2 = st.columns(2)
        with col1: st.metric("Total Parts", len(pfep_data))
        with col2: 
            vendors = set(item.get('Vendor_Name', 'Unknown') for item in pfep_data)
            st.metric("Unique Vendors", len(vendors))
        with st.expander("📋 Data Preview"):
            st.dataframe(pd.DataFrame(pfep_data[:10]))
    
    def user_inventory_upload(self):
        st.header("📦 Inventory Analysis System")
        
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if not pfep_data or not pfep_locked:
            st.error("❌ PFEP master data is not available or not locked by admin.")
            return
        
        st.success(f"✅ PFEP Master Data: {len(pfep_data)} parts")
        
        if st.session_state.get('persistent_analysis_results'):
            self.display_analysis_interface()
            return
        
        # Upload
        st.subheader("📊 Upload Current Inventory Data")
        tab1, tab2 = st.tabs(["📁 Upload File", "🧪 Load Sample"])
        
        with tab1:
            uploaded_file = st.file_uploader("Choose inventory file", type=['xlsx', 'xls', 'csv'])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                    else: df = pd.read_excel(uploaded_file)
                    
                    standardized_data = self.standardize_current_inventory(df)
                    if standardized_data:
                        validation = self.validate_inventory_against_pfep(standardized_data)
                        self.display_validation_results(validation)
                        
                        if validation['is_valid']:
                            if st.button("💾 Save & Analyze", type="primary"):
                                self.persistence.save_data_to_session_state('persistent_inventory_data', standardized_data)
                                st.session_state.persistent_inventory_locked = True
                                # Trigger analysis immediately
                                with st.spinner("Analyzing..."):
                                    results = self.analyzer.analyze_inventory(pfep_data, standardized_data)
                                    self.persistence.save_data_to_session_state('persistent_analysis_results', results)
                                st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        with tab2:
            if st.button("🧪 Load Sample Inventory"):
                sample = self.load_sample_current_inventory()
                self.persistence.save_data_to_session_state('persistent_inventory_data', sample)
                st.session_state.persistent_inventory_locked = True
                with st.spinner("Analyzing..."):
                    results = self.analyzer.analyze_inventory(pfep_data, sample)
                    self.persistence.save_data_to_session_state('persistent_analysis_results', results)
                st.rerun()

    def run(self):
        st.title("📊 Inventory Analyzer")
        self.authenticate_user()

        if st.session_state.user_role == "Admin":
            self.admin_data_management()
        elif st.session_state.user_role == "User":
            self.user_inventory_upload()
        else:
            st.info("👋 Please select your role.")
    
    def display_validation_results(self, res):
        st.markdown("**📋 Validation Results**")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("PFEP Parts", res['pfep_parts_count'])
        with col2: st.metric("Inventory Parts", res['inventory_parts_count'])
        with col3: st.metric("Matching", res['matching_parts_count'])
    
    def display_analysis_interface(self):
        st.subheader("📈 Analysis Dashboard")
        
        # New Analysis Button
        if st.button("🔄 Analyze New File", type="secondary"):
            st.session_state.persistent_analysis_results = None
            st.session_state.persistent_inventory_data = None
            st.rerun()

        # Load Results
        results = self.persistence.load_data_from_session_state('persistent_analysis_results')
        if not results:
            st.error("No results found.")
            return

        # Display Sections
        self.display_enhanced_summary_metrics(results)
        self.display_enhanced_analysis_charts(results)
        self.display_enhanced_detailed_tables(results)
        self.display_enhanced_export_options(results)

    def display_enhanced_summary_metrics(self, results):
        st.markdown("### 📊 Executive Summary")
        df = pd.DataFrame(results)
        
        total_val = df['Current Inventory - VALUE'].sum()
        total_parts = len(df)
        excess_val = df[df['Status'] == 'Excess Inventory']['Stock Deviation Value'].sum()
        short_val = df[df['Status'] == 'Short Inventory']['Stock Deviation Value'].sum() # Is negative
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Value", f"₹{total_val:,.0f}")
        col2.metric("Total Parts", total_parts)
        col3.metric("Excess Value", f"₹{excess_val:,.0f}")
        col4.metric("Shortage Impact", f"₹{abs(short_val):,.0f}")

    def display_enhanced_analysis_charts(self, analysis_results):
        """Display charts with Ideal Inventory Line and Percentage Hover."""
        st.markdown("---")
        st.subheader("📊 Enhanced Inventory Charts")
        
        # UI Controls
        col1, col2 = st.columns([1, 2])
        with col1:
            chart_unit = st.selectbox("Select Currency Unit:", ["Lakhs", "Millions"], key="chart_unit_selector")
        with col2:
            top_n = st.slider("Number of items to show:", min_value=5, max_value=50, value=10, step=5, key="top_n_slider")
        
        if chart_unit == "Millions":
            divisor = 1_000_000
            suffix = "M"
            unit_name = "Millions"
            format_key = "millions"
        else:
            divisor = 100_000
            suffix = "L"
            unit_name = "Lakhs"
            format_key = "lakhs"
            
        df = pd.DataFrame(analysis_results)
        if df.empty:
            st.warning("⚠️ No data available for charts.")
            return
            
        # ✅ CHART 1: Current vs Ideal Inventory (Combo Chart)
        value_col = 'Current Inventory - VALUE'
        ideal_val_col = 'Ideal Inventory - VALUE'
        
        # Ensure columns exist
        if value_col in df.columns and 'PART NO' in df.columns:
            # Sort by Current Value and take Top N
            chart_data = (
                df[df[value_col] > 0]
                .sort_values(by=value_col, ascending=False)
                .head(top_n)
                .copy()
            )
            
            # Prepare Data for Plotting
            chart_data['Current_Val_Conv'] = chart_data[value_col] / divisor
            
            # Calculate Ideal Value if not present (backward compatibility)
            if ideal_val_col not in chart_data.columns:
                chart_data[ideal_val_col] = chart_data['RM Norm - In Qty'] * chart_data['UNIT PRICE']
                
            chart_data['Ideal_Val_Conv'] = chart_data[ideal_val_col] / divisor
            
            # Create Labels
            chart_data['Part'] = chart_data.apply(
                lambda row: f"{row['PART DESCRIPTION'][:20]}... ({row['PART NO']})", axis=1
            )
            
            # Define Colors
            color_map = {
                "Excess Inventory": "#2196F3", # Blue
                "Short Inventory": "#F44336",  # Red
                "Within Norms": "#4CAF50"      # Green
            }
            chart_data['Bar_Color'] = chart_data['Status'].map(color_map)

            # ✅ HOVER TEXT: Including the Deviation Percentage
            chart_data['HOVER_TEXT'] = chart_data.apply(lambda row: (
                f"Part: {row['PART DESCRIPTION']}<br>"
                f"Current Value: ₹{row[value_col]:,.0f}<br>"
                f"Ideal Value: ₹{row[ideal_val_col]:,.0f}<br>"
                f"<b>Deviation: {row.get('Inventory Deviation %', 0):.2f}%</b><br>"
                f"Status: {row['Status']}"
            ), axis=1)

            fig1 = go.Figure()

            # 1. Bar Trace (Current Inventory)
            fig1.add_trace(go.Bar(
                x=chart_data['Part'],
                y=chart_data['Current_Val_Conv'],
                name='Current Stock',
                marker_color=chart_data['Bar_Color'],
                customdata=chart_data['HOVER_TEXT'],
                hovertemplate='%{customdata}<extra></extra>'
            ))

            # 2. ✅ Line Trace (Ideal Inventory) - The overlay requested
            fig1.add_trace(go.Scatter(
                x=chart_data['Part'],
                y=chart_data['Ideal_Val_Conv'],
                mode='lines+markers',
                name='Ideal Inventory',
                line=dict(color='black', width=3, dash='dot'),
                marker=dict(symbol='diamond', size=10, color='gold'),
                hovertemplate='Ideal: ₹%{y:.2f}' + suffix + '<extra></extra>'
            ))

            fig1.update_layout(
                title=f"Top {top_n} Parts: Current Stock vs Ideal Inventory",
                xaxis_title="Parts",
                yaxis_title=f"Value (₹ {unit_name})",
                xaxis_tickangle=-45,
                yaxis=dict(tickformat=',.1f', ticksuffix=suffix),
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
                hovermode="x unified"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        else:
            st.warning("⚠️ Required columns for chart not found.")

        # ✅ CHART 2: Vendor Analysis
        st.markdown("### Vendor Analysis")
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            self.analyzer.show_vendor_chart_by_status(
                analysis_results, "Excess Inventory", "Top Vendors - Excess", "v1", "#2196F3", 
                format_key, top_n
            )
        with col_v2:
             self.analyzer.show_vendor_chart_by_status(
                analysis_results, "Short Inventory", "Top Vendors - Short", "v2", "#F44336", 
                format_key, top_n
            )

    def display_enhanced_detailed_tables(self, results):
        st.markdown("### 📋 Detailed Data")
        df = pd.DataFrame(results)
        
        # Format for display
        cols = [
            'PART NO', 'PART DESCRIPTION', 'Status', 
            'Current Inventory - VALUE', 'Ideal Inventory - VALUE',
            'Inventory Deviation %'
        ]
        
        # Only keep existing columns
        cols = [c for c in cols if c in df.columns]
        
        st.dataframe(df[cols].style.format({
            'Current Inventory - VALUE': '₹{:,.0f}',
            'Ideal Inventory - VALUE': '₹{:,.0f}',
            'Inventory Deviation %': '{:.2f}%'
        }))

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
