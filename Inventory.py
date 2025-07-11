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
        Applies updated logic where:
        - Short Inventory: < RM_IN_QTY × (1 - Tolerance%)
        - Excess Inventory: > RM_IN_QTY × (1 + Tolerance%)
        - Within Norms: between the above thresholds
        """
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)  # default to 30%
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
                rm_days = self.safe_float_convert(pfep_item.get('RM_IN_DAYS', 0))
                rm_qty = self.safe_float_convert(pfep_item.get('RM_IN_QTY', 0))
                # Inventory value
                current_value = current_qty * unit_price
                # Norms with tolerance
                lower_bound = rm_qty * (1 - tolerance / 100)
                upper_bound = rm_qty * (1 + tolerance / 100)

                # Revised Norm shown for reference
                revised_norm_qty = upper_bound  # you can rename if needed
                # Deviation quantity and value
                deviation_qty = current_qty - revised_norm_qty
                deviation_value = deviation_qty * unit_price

                # Status based on range
                if current_qty < lower_bound:
                    status = 'Short Inventory'
                elif current_qty > upper_bound:
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
                    'RM IN DAYS': rm_days,
                    'RM Norm - In Qty': rm_qty,
                    'Revised Norm Qty': revised_norm_qty,
                    'Lower Bound Qty': lower_bound,                 # ✅ Added
                    'Upper Bound Qty': upper_bound,  
                    'UNIT PRICE': unit_price,
                    'Current Inventory - Qty': current_qty,
                    'Current Inventory - VALUE': current_value,
                    'SHORT/EXCESS INVENTORY': deviation_qty,
                    'Stock Deviation Qty w.r.t Revised Norm': deviation_qty,
                    'Stock Deviation Value': deviation_value,
                    'Status': status,
                    'INVENTORY REMARK STATUS': status
                }
                # ✅ Add these two lines
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
            # Get current quantity and norm values for calculating excess/short amounts
            current_qty = item.get('Current Inventory - QTY', 0)
            norm_qty = item.get('Norm', 0)
            unit_price = 0
            try:
                stock_value = float(stock_value)
                current_qty = float(current_qty) if current_qty else 0
                norm_qty = float(norm_qty) if norm_qty else 0
                # Calculate unit price from stock value and current quantity
                if current_qty > 0:
                    unit_price = stock_value / current_qty
            except (ValueError, TypeError):
                stock_value = 0.0
                current_qty = 0.0
                norm_qty = 0.0
                unit_price = 0.0
            summary[vendor]['total_parts'] += 1
            summary[vendor]['total_value'] += stock_value
            if status == "Short Inventory":
                summary[vendor]['short_parts'] += 1
                # Calculate value of shortage (norm - current) * unit_price
                if norm_qty > current_qty and unit_price > 0:
                    short_value = (norm_qty - current_qty) * unit_price
                    summary[vendor]['short_value_below_norm'] += short_value
            elif status == "Excess Inventory":
                summary[vendor]['excess_parts'] += 1
                # Calculate value of excess (current - norm) * unit_price
                if current_qty > norm_qty and unit_price > 0:
                    excess_value = (current_qty - norm_qty) * unit_price
                    summary[vendor]['excess_value_above_norm'] += excess_value
            elif status == "Within Norms":
                summary[vendor]['normal_parts'] += 1
        return summary
        
    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color,value_format='lakhs'):
        """Show top 10 vendors by deviation value and count of excess/short parts."""
        # Filter by inventory status
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        vendor_totals = {}
        vendor_counts = {}
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            try:
                current_qty = float(item.get('Current Inventory - Qty', 0) or 0)
                norm_qty = float(item.get('Revised Norm Qty', 0) or 0)
                stock_value = float(item.get('Current Inventory - VALUE', 0) or 0)
                unit_price = float(item.get('UNIT PRICE', 0) or 0)
                if unit_price == 0 and current_qty > 0:
                    unit_price = stock_value / current_qty
                if status_filter == "Excess Inventory" and current_qty > norm_qty:
                    deviation_value = (current_qty - norm_qty) * unit_price
                    vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + deviation_value
                    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
                elif status_filter == "Short Inventory" and norm_qty > current_qty:
                    deviation_value = (norm_qty - current_qty) * unit_price
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
        # Sort and take top 10 by value
        top_vendors = sorted(combined, key=lambda x: x[1], reverse=True)[:10]
        vendor_names = [v[0] for v in top_vendors]
        raw_values = [v[1] for v in top_vendors]
        counts = [v[2] for v in top_vendors]
        # Value formatting
        if value_format == 'lakhs':
            values = [v / 100000 for v in raw_values]
            y_axis_title = "Value (₹ Lakhs)"
            hover_suffix = "L"
            tick_suffix = "L"
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
            y_axis_title = y_axis_title.replace("Value", "Excess Value Above Norm")
            hover_label = "Excess Value"
        elif status_filter == "Short Inventory":
            y_axis_title = y_axis_title.replace("Value", "Short Value Below Norm")
            hover_label = "Short Value"
        else:
            hover_label = "Stock Value"
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vendor_names,
            y=values,
            marker_color=color,
            text=[f"{hover_label}: ₹{v:,.0f}{hover_suffix}<br>Parts: {c}" for v, c in zip(values, counts)],
            hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
        ))
        fig.update_layout(
            title=chart_title,
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
                st.session_state[key] = None  # BUG: should be None, not empty list
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
            
    def create_top_parts_chart(self, data, status_filter, bar_color, key):
        """Top 10 parts chart by status — shows EXCESS/SHORT VALUE vs NORM VALUE comparison"""
        df = pd.DataFrame(data)
        # ✅ Check required columns
        if 'PART NO' not in df.columns or 'Stock Deviation Value' not in df.columns:
            st.warning("⚠️ Required columns missing for top parts chart.")
            return
        # ✅ Filter by status
        df = df[df['INVENTORY REMARK STATUS'] == status_filter]
    
        if status_filter == "Excess Inventory":
            # For excess inventory, show only positive deviation values (excess amount)
            df = df[df['Stock Deviation Value'] > 0]
            df = df.sort_values(by='Stock Deviation Value', ascending=False).head(10)
            chart_title = "Top 10 Excess Inventory Parts - Norm vs Actual (₹ in Lakhs)"
            y_title = "Inventory Value (₹ Lakhs)"
        elif status_filter == "Short Inventory":
            # For short inventory, show absolute value of negative deviation
            df = df[df['Stock Deviation Value'] < 0]
            df['Abs_Deviation_Value'] = abs(df['Stock Deviation Value'])
            df = df.sort_values(by='Abs_Deviation_Value', ascending=False).head(10)
            chart_title = "Top 10 Short Inventory Parts - Norm vs Actual (₹ in Lakhs)"
            y_title = "Inventory Value (₹ Lakhs)"
        else:
            st.info(f"No chart available for '{status_filter}' status.")
            return
        if df.empty:
            st.info(f"No data found for '{status_filter}' parts.")
            return
        # ✅ Prepare chart data with norm vs actual comparison
        if status_filter == "Excess Inventory":
            # Calculate norm value (current stock value - excess)
            df['Current_Stock_Value'] = df['CURRENT STOCK VALUE'] if 'CURRENT STOCK VALUE' in df.columns else 0
            df['Norm_Value'] = df['Current_Stock_Value'] - df['Stock Deviation Value']
            df['Excess_Value'] = df['Stock Deviation Value']
            # Convert to lakhs
            df['Norm_Value_Lakh'] = df['Norm_Value'] / 100000
            df['Current_Value_Lakh'] = df['Current_Stock_Value'] / 100000
            df['Deviation_Value_Lakh'] = df['Excess_Value'] / 100000
        else:  # Short Inventory
            # Calculate what the stock value should be (current + shortage)
            df['Current_Stock_Value'] = df['CURRENT STOCK VALUE'] if 'CURRENT STOCK VALUE' in df.columns else 0
            df['Norm_Value'] = df['Current_Stock_Value'] + df['Abs_Deviation_Value']
            df['Shortage_Value'] = df['Abs_Deviation_Value']
        
            # Convert to lakhs
            df['Norm_Value_Lakh'] = df['Norm_Value'] / 100000
            df['Current_Value_Lakh'] = df['Current_Stock_Value'] / 100000
            df['Deviation_Value_Lakh'] = df['Shortage_Value'] / 100000
        # Prepare labels and hover text
        df['PART_DESC_NO'] = df['PART DESCRIPTION'].astype(str) + " (" + df['PART NO'].astype(str) + ")"
    
        if status_filter == "Excess Inventory":
            df['HOVER_TEXT_NORM'] = df.apply(lambda row: (
                f"Description: {row.get('PART DESCRIPTION', 'N/A')}<br>"
                f"Part No: {row.get('PART NO')}<br>"
                f"Norm Value: ₹{row['Norm_Value']:,.0f}<br>"
                f"Type: Within Norm Limit"
            ), axis=1)
        
            df['HOVER_TEXT_CURRENT'] = df.apply(lambda row: (
                f"Description: {row.get('PART DESCRIPTION', 'N/A')}<br>"
                f"Part No: {row.get('PART NO')}<br>"
                f"Current Value: ₹{row['Current_Stock_Value']:,.0f}<br>"
                f"Excess Value: ₹{row['Excess_Value']:,.0f}<br>"
                f"Type: Current Stock (Excess)"
            ), axis=1)
        else:  # Short Inventory
            df['HOVER_TEXT_NORM'] = df.apply(lambda row: (
                f"Description: {row.get('PART DESCRIPTION', 'N/A')}<br>"
                f"Part No: {row.get('PART NO')}<br>"
                f"Required Value: ₹{row['Norm_Value']:,.0f}<br>"
                f"Type: Required Norm Level"
            ), axis=1)
        
            df['HOVER_TEXT_CURRENT'] = df.apply(lambda row: (
                f"Description: {row.get('PART DESCRIPTION', 'N/A')}<br>"
                f"Part No: {row.get('PART NO')}<br>"
                f"Current Value: ₹{row['Current_Stock_Value']:,.0f}<br>"
                f"Shortage Value: ₹{row['Shortage_Value']:,.0f}<br>"
                f"Type: Current Stock (Short)"
            ), axis=1)
        fig = go.Figure()
        # Add norm value bars
        fig.add_trace(go.Bar(
            x=df['PART_DESC_NO'],
            y=df['Norm_Value_Lakh'],
            name='Norm Value' if status_filter == "Excess Inventory" else 'Required Value',
            marker_color='#4CAF50',  # Green for norm
            hovertemplate='<b>%{x}</b><br>%{customdata}<extra></extra>',
            customdata=df['HOVER_TEXT_NORM']
        ))
        # Add current value bars
        fig.add_trace(go.Bar(
            x=df['PART_DESC_NO'],
            y=df['Current_Value_Lakh'],
            name='Current Value',
            marker_color=bar_color,
            hovertemplate='<b>%{x}</b><br>%{customdata}<extra></extra>',
            customdata=df['HOVER_TEXT_CURRENT']
        ))
    
        # ✅ Update chart formatting
        fig.update_layout(
            title=chart_title,
            xaxis_title="Part Description (Part No)",
            yaxis_title=y_title,
            xaxis_tickangle=-45,
            barmode='group',  # Side-by-side bars
            yaxis=dict(
                tickformat=',.1f',
                ticksuffix='L'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, key=key)
    
        # ✅ Add summary table showing the comparison
        st.subheader(f"📊 Summary - {status_filter}")
        summary_df = df[['PART NO', 'PART DESCRIPTION', 'Current_Value_Lakh', 'Norm_Value_Lakh', 'Deviation_Value_Lakh']].copy()
        summary_df.columns = ['Part No', 'Part Description', 'Current Value (₹L)', 'Norm Value (₹L)', 
                              'Excess Value (₹L)' if status_filter == "Excess Inventory" else 'Shortage Value (₹L)']
        summary_df = summary_df.round(2)
        st.dataframe(summary_df, use_container_width=True)

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
                            if password == "Agilomatrix@123":  # BUG: wrong password
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
                    st.session_state.user_preferences['default_tolerance'] = st.selectbox(
                        "Default Tolerance", [0, 10, 20, 30, 40, 50], 
                        index=2, key="pref_tolerance"
                    )
                    st.session_state.user_preferences['chart_theme'] = st.selectbox(
                        "Chart Theme", ['plotly', 'plotly_white', 'plotly_dark'],
                        key="pref_theme"
                    )
            
            # Logout button
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Logout", key="logout_btn"):
                # Only clear user session, not persistent data
                keys_to_keep = self.persistent_keys + ['user_preferences']
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
           ["JJ1010101010", "WINDSHIELD WASHER", 25, "V002", "Vendor_B", "Delhi", "Delhi", 1.8]
           # Add more sample data with consumption values...
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
                'Unit_Price': 100,            # 🔁 you can customize this per part
                'RM_IN_DAYS': 7,              # 🔁 default or configurable
                'AVG CONSUMPTION/DAY': self.safe_float_convert(row[7]) if len(row) > 7 else ""  # ✅ Added consumption data
            })
        return pfep_data
    
    def load_sample_current_inventory(self):
        """Load sample current inventory data with consistent fields"""
        current_sample = [
            ["AC0303020106", "FLAT ALUMINIUM PROFILE", 5.230, 496],
            # ... rest of your data
            ["JJ1010101010", "WINDSHIELD WASHER", 33, 495]
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
        # Debug: Show original column names
        if self.debug:
            st.write("🔍 DEBUG: Original Excel columns:")
            for i, col in enumerate(df.columns):
                st.write(f"  {i}: '{col}' (type: {type(col)})")
        # Comprehensive column mapping with more variations
        column_mappings = {
            'part_no': [
                'part_no', 'part_number', 'material', 'material_code', 'item_code', 
                'code', 'part no', 'partno', 'Part No', 'Part_No', 'PART_NO',
                'Material', 'Material Code', 'Item Code'
            ],
            'description': [
                'description', 'item_description', 'part_description', 'desc', 
                'part description', 'material_description', 'item desc', 'Part Description',
                'Description', 'DESCRIPTION', 'Material Description'
            ],
            'rm_qty': [
                'rm_in_qty', 'rm_qty', 'required_qty', 'norm_qty', 'target_qty', 
                'rm', 'ri_in_qty', 'rm in qty', 'RM_IN_QTY', 'RM_QTY', 'RM IN QTY',
                'Required Qty', 'Norm Qty', 'Target Qty'
            ],
            'rm_days': [
                'rm_in_days', 'rm days', 'inventory days', 'rmindays', 'RM_IN_DAYS',
                'RM IN DAYS', 'RM Days', 'Inventory Days'
            ],
            'unit_price': [
                'unit_price', 'price', 'unit cost', 'unit rate', 'unitprice', 'Unit Price',
                'UNIT_PRICE', 'UNIT PRICE', 'Price', 'PRICE', 'Unit Cost', 'UNIT_COST',
                'unit_cost', 'Unit Rate', 'UNIT_RATE', 'unit_rate', 'rate', 'Rate', 'RATE',
                'cost', 'Cost', 'COST', 'Unit_Price'
            ],
            'vendor_code': [
                'vendor_code', 'vendor_id', 'supplier_code', 'supplier_id', 'vendor id', 
                'Vendor Code', 'vendor code', 'VENDOR_CODE', 'VENDOR CODE', 'Vendor_Code',
                'Supplier Code', 'SUPPLIER_CODE'
            ],
            'vendor_name': [
                'vendor_name', 'vendor', 'supplier_name', 'supplier', 'Vendor Name', 
                'vendor name', 'VENDOR_NAME', 'VENDOR NAME', 'Vendor_Name',
                'Supplier Name', 'SUPPLIER_NAME', 'Supplier'
            ],
            'avg_consumption_per_day': [
                'avg consumption/day', 'average consumption/day', 'avg_per_day',
                'avg daily usage', 'AVG CONSUMPTION/DAY', 'Average Per Day',
                'avg consumption per day', 'daily consumption', 'consumption per day',
                'AVG_CONSUMPTION_PER_DAY', 'AVERAGE_CONSUMPTION_PER_DAY'
            ],
            'city': ['city', 'location', 'place', 'City', 'CITY', 'Location', 'LOCATION'],
            'state': ['state', 'region', 'province', 'State', 'STATE', 'Region', 'REGION']
        }
        # Create case-insensitive column lookup
        df_columns_lookup = {}
        for col in df.columns:
            if col is not None:
                # Clean column name (remove extra spaces, special characters)
                clean_col = str(col).strip()
                df_columns_lookup[clean_col.lower()] = clean_col
        if self.debug:
            st.write("🔍 DEBUG: Cleaned column lookup:")
            for k, v in df_columns_lookup.items():
                st.write(f"  '{k}' -> '{v}'")
        # Map columns using case-insensitive matching
        mapped_columns = {}
        for key, variations in column_mappings.items():
            found = False
            for variation in variations:
                variation_lower = variation.lower().strip()
                if variation_lower in df_columns_lookup:
                    mapped_columns[key] = df_columns_lookup[variation_lower]
                    if self.debug:
                        st.write(f"✅ Mapped {key} -> '{mapped_columns[key]}' (from variation: '{variation}')")
                    found = True
                    break
            if not found and self.debug:
                st.write(f"❌ No mapping found for {key}")
        # Check for required columns
        if 'part_no' not in mapped_columns:
            st.error("❌ Part Number column not found. Please ensure your file has a Part Number column.")
            return []
        if 'rm_qty' not in mapped_columns:
            st.error("❌ RM Quantity column not found. Please ensure your file has an RM/Required Quantity column.")
            return []
        # Warning for missing unit price
        if 'unit_price' not in mapped_columns:
            st.warning("⚠️ Unit Price column not found. Using default value of 100 for all parts.")
            st.info("💡 Expected Unit Price column names: Unit Price, Price, Unit Cost, Rate, etc.")
            
        if 'avg_consumption_per_day' not in mapped_columns:
            st.warning("⚠️ AVG CONSUMPTION/DAY column not found. This will be left empty.")
            st.info("💡 Expected column names: AVG CONSUMPTION/DAY, Average Per Day, Daily Consumption, etc.")
            
        if self.debug:
            st.write("🔍 DEBUG: Final column mappings:")
            for k, v in mapped_columns.items():
                st.write(f"  {k} -> '{v}'")
        # Standardize data
        standardized_data = []
        for idx, row in df.iterrows():
            try:
                # Extract unit price with detailed debugging
                unit_price_value = 100.0  # Default value
                if 'unit_price' in mapped_columns:
                    raw_price = row[mapped_columns['unit_price']]
                    unit_price_value = self.safe_float_convert(raw_price)
                    if self.debug and idx < 3:  # Debug first 3 rows
                        st.write(f"🔍 Row {idx+1} Unit Price: '{raw_price}' -> {unit_price_value}")
                    # Extract AVG CONSUMPTION/DAY with proper handling
                    avg_consumption_value = ""  # Default empty string
                    if 'avg_consumption_per_day' in mapped_columns:
                        raw_consumption = row[mapped_columns['avg_consumption_per_day']]
                        # Handle different data types
                        if pd.notna(raw_consumption) and str(raw_consumption).strip() != '':
                            avg_consumption_value = self.safe_float_convert(raw_consumption)
                        if self.debug and idx < 3:  # Debug first 3 rows
                            st.write(f"🔍 Row {idx+1} AVG CONSUMPTION/DAY: '{raw_consumption}' -> {avg_consumption_value}")
                item = {
                    'Part_No': str(row[mapped_columns['part_no']]).strip(),
                    'Description': str(row.get(mapped_columns.get('description', ''), '')).strip(),
                    'RM_IN_QTY': self.safe_float_convert(row[mapped_columns['rm_qty']]),
                    'RM_IN_DAYS': self.safe_float_convert(row.get(mapped_columns.get('rm_days', ''), 7)),  # Default 7 days
                    'unit_price': unit_price_value,  # Fixed: use lowercase to match analyzer expectation
                    'Vendor_Code': str(row.get(mapped_columns.get('vendor_code', ''), '')).strip(),
                    'Vendor_Name': str(row.get(mapped_columns.get('vendor_name', ''), 'Unknown')).strip(),
                    'AVG CONSUMPTION/DAY': row.get(mapped_columns.get('avg_consumption_per_day', ''), ''),
                    'City': str(row.get(mapped_columns.get('city', ''), '')).strip(),
                    'State': str(row.get(mapped_columns.get('state', ''), '')).strip()
                }
                # Skip rows with empty part numbers
                if not item['Part_No'] or item['Part_No'].lower() in ['nan', 'none', '']:
                    continue
                standardized_data.append(item)
            except Exception as e:
                if self.debug:
                    st.warning(f"⚠️ Error processing row {idx+1}: {e}")
                continue
        # Summary of data processing
        st.success(f"✅ Processed {len(standardized_data)} PFEP records")
        # Show unit price statistics
        if standardized_data:
            prices = [item['unit_price'] for item in standardized_data if item['unit_price'] > 0]
            if prices:
                avg_price = sum(prices) / len(prices)
                st.info(f"💰 Unit Price Summary: {len(prices)} parts with prices, Average: ₹{avg_price:.2f}")
            else:
                st.warning("⚠️ No valid unit prices found in the data")
            # Show AVG CONSUMPTION/DAY statistics
            consumption_values = [
                item.get('AVG CONSUMPTION/DAY')
                for item in standardized_data
                if item.get('AVG CONSUMPTION/DAY') not in (None, '', 'nan')
            ]
            if consumption_values:
                # Convert strings to floats, skip zeros
                numeric = [self.safe_float_convert(val) for val in consumption_values if self.safe_float_convert(val) > 0]
                if numeric:
                    avg_consumption = sum(numeric) / len(numeric)
                    st.info(f"📊 AVG CONSUMPTION/DAY Summary: {len(numeric)} parts with consumption data, Average: {avg_consumption:.2f}")
                else:
                    st.warning("⚠️ All AVG CONSUMPTION/DAY values were zero or invalid.")
            else:
                st.warning("⚠️ No AVG CONSUMPTION/DAY data found in the file")
        return standardized_data
    
    def standardize_current_inventory(self, df):
        """Standardize current inventory data with full column mappings and debugging."""
        if df is None or df.empty:
            return []
        # 🔁 Add all possible column mappings
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
        # Debug: show mappings
        if self.debug:
            st.write("🔍 DEBUG: Column mappings found:")
            for key, col in mapped_columns.items():
                st.write(f"  {key} → {col}")
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
                if self.debug and i < 5:
                    st.write(f"🔍 Row {i+1}:", item)
            except Exception as e:
                if self.debug:
                    st.write(f"⚠️ Error processing row {i+1}: {e}")
                continue
        if self.debug:
            st.write(f"✅ Total standardized records: {len(standardized_data)}")
        return standardized_data
    
    def validate_inventory_against_pfep(self, inventory_data):
        """Validate inventory data against PFEP master data with normalized keys and warnings."""
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        if not pfep_data:
            return {'is_valid': False, 'issues': ['No PFEP data available'], 'warnings': []}
        # Normalize part numbers
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
        
        # Check for parts with zero quantity
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
                    # Clear related data when PFEP is unlocked
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
        # Tolerance Setting for Admin
        st.subheader("📐 Set Analysis Tolerance (Admin Only)")
        # Initialize admin_tolerance if not exists
        if "admin_tolerance" not in st.session_state:
            st.session_state.admin_tolerance = 30
    
        # Create selectbox with proper callback
        new_tolerance = st.selectbox(
            "Tolerance Zone (+/-)",
            options=[0, 10, 20, 30, 40, 50],
            index=[0, 10, 20, 30, 40, 50].index(st.session_state.admin_tolerance),
            format_func=lambda x: f"{x}%",
            key="tolerance_selector"
        )
        # Update tolerance if changed
        if new_tolerance != st.session_state.admin_tolerance:
            st.session_state.admin_tolerance = new_tolerance
            st.success(f"✅ Tolerance updated to {new_tolerance}%")
            
            # If analysis exists, refresh it with new tolerance
            if st.session_state.get('persistent_analysis_results'):
                st.info("🔄 Analysis will be refreshed with new tolerance on next run")
        
        st.markdown(f"**Current Tolerance:** {st.session_state.admin_tolerance}%")
        st.markdown("---")
        
        # PFEP Data Management Section
        st.subheader("📊 PFEP Master Data Management")
        
        # Tab interface for different data input methods
        tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🧪 Load Sample", "📋 Current Data"])
        
        with tab1:
            st.markdown("**Upload PFEP Excel/CSV File**")
            uploaded_file = st.file_uploader(
                "Choose PFEP file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your PFEP master data file"
            )
            
            if uploaded_file is not None:
                try:
                    # Read file based on extension
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded: {len(df)} rows")
                    
                    # Show preview
                    with st.expander("📋 Preview Raw Data"):
                        st.dataframe(df.head())
                    
                    # Standardize data
                    standardized_data = self.standardize_pfep_data(df)
                    
                    if standardized_data:
                        st.success(f"✅ Standardized: {len(standardized_data)} valid records")
                        
                        # Show standardized preview
                        with st.expander("📋 Preview Standardized Data"):
                            preview_df = pd.DataFrame(standardized_data[:5])
                            st.dataframe(preview_df)
                        
                        # Save button
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
            st.info("This will load pre-configured sample data for demonstration")
            
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
                
                # Lock data for users
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
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parts", len(pfep_data))
        with col2:
            vendors = set(item.get('Vendor_Name', 'Unknown') for item in pfep_data)
            st.metric("Unique Vendors", len(vendors))
        with col3:
            total_rm_qty = sum(item.get('RM_IN_QTY', 0) for item in pfep_data)
            st.metric("Total RM Qty", f"{total_rm_qty:,.0f}")
        with col4:
            avg_unit_price = sum(item.get('Unit_Price', 0) for item in pfep_data) / len(pfep_data)
            st.metric("Avg Unit Price", f"₹{avg_unit_price:.2f}")
        
        # Data preview
        with st.expander("📋 Data Preview (First 10 rows)"):
            preview_df = pd.DataFrame(pfep_data[:10])
            st.dataframe(preview_df)
        
        # Vendor summary
        with st.expander("📈 Vendor Summary"):
            vendor_summary = {}
            for item in pfep_data:
                vendor = item.get('Vendor_Name', 'Unknown')
                if vendor not in vendor_summary:
                    vendor_summary[vendor] = {'count': 0, 'total_qty': 0}
                vendor_summary[vendor]['count'] += 1
                vendor_summary[vendor]['total_qty'] += item.get('RM_IN_QTY', 0)
            
            vendor_df = pd.DataFrame([
                {'Vendor': k, 'Parts Count': v['count'], 'Total RM Qty': v['total_qty']}
                for k, v in vendor_summary.items()
            ])
            st.dataframe(vendor_df)
    
    def user_inventory_upload(self):
        """User interface for inventory upload and analysis"""
        st.header("📦 Inventory Analysis System")
        
        # Check if PFEP data is available and locked
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if not pfep_data or not pfep_locked:
            st.error("❌ PFEP master data is not available or not locked by admin.")
            st.info("Please contact admin to load and lock PFEP data first.")
            return
        
        # Display PFEP status
        st.success(f"✅ PFEP Master Data: {len(pfep_data)} parts available")
        
        # Check if inventory is already loaded and locked
        inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        inventory_locked = st.session_state.get('persistent_inventory_locked', False)
        
        if inventory_locked and inventory_data:
            st.info("🔒 Inventory data is locked. Proceeding to analysis...")
            self.display_analysis_interface()
            return
        
        # Inventory upload interface
        st.subheader("📊 Upload Current Inventory Data")
        
        # Tab interface
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
                    # Read file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded: {len(df)} rows")
                    
                    # Show preview
                    with st.expander("📋 Preview Raw Data"):
                        st.dataframe(df.head())
                    
                    # Standardize data
                    standardized_data = self.standardize_current_inventory(df)
                    
                    if standardized_data:
                        st.success(f"✅ Standardized: {len(standardized_data)} valid records")
                        
                        # Validate against PFEP
                        validation_result = self.validate_inventory_against_pfep(standardized_data)
                        
                        # Display validation results
                        self.display_validation_results(validation_result)
                        
                        if validation_result['is_valid']:
                            # Show standardized preview
                            with st.expander("📋 Preview Standardized Data"):
                                preview_df = pd.DataFrame(standardized_data[:5])
                                st.dataframe(preview_df)
                            
                            # Save and lock button
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
            st.info("This will load pre-configured sample inventory data")
            
            if st.button("🧪 Load Sample Inventory Data", type="secondary"):
                sample_data = self.load_sample_current_inventory()
                
                # Validate sample data
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
    def run(self):
        st.title("📊 Inventory Analyzer")
        st.markdown(
            "<p style='font-size:18px; font-style:italic;'>Designed and Developed by Agilomatrix</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        # Authenticate user
        self.authenticate_user()

        # Show UI based on role
        if st.session_state.user_role == "Admin":
            self.admin_data_management()
        elif st.session_state.user_role == "User":
            self.user_inventory_upload()
        else:
            st.info("👋 Please select your role and authenticate to access the system.")
    
    def display_validation_results(self, validation_result):
        """Display inventory validation results"""
        st.markdown("**📋 Validation Results**")
        if validation_result['is_valid']:
            st.success("✅ Inventory data validation passed!")
        else:
            st.error("❌ Inventory data validation failed!")
            for issue in validation_result['issues']:
                st.error(f"• {issue}")
        
        # Display warnings
        if validation_result['warnings']:
            st.warning("⚠️ Validation Warnings:")
            for warning in validation_result['warnings']:
                st.warning(f"• {warning}")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PFEP Parts", validation_result['pfep_parts_count'])
        with col2:
            st.metric("Inventory Parts", validation_result['inventory_parts_count'])
        with col3:
            st.metric("Matching Parts", validation_result['matching_parts_count'])
        
        # Detailed breakdown
        with st.expander("📊 Detailed Breakdown"):
            col1, col2 = st.columns(2)
            with col1:
                if validation_result['missing_parts_count'] > 0:
                    st.warning(f"Missing Parts ({validation_result['missing_parts_count']}):")
                    st.text("\n".join(validation_result['missing_parts_list'][:10]))
                    if len(validation_result['missing_parts_list']) > 10:
                        st.text(f"... and {len(validation_result['missing_parts_list']) - 10} more")
            
            with col2:
                if validation_result['extra_parts_count'] > 0:
                    st.info(f"Extra Parts ({validation_result['extra_parts_count']}):")
                    st.text("\n".join(validation_result['extra_parts_list'][:10]))
                    if len(validation_result['extra_parts_list']) > 10:
                        st.text(f"... and {len(validation_result['extra_parts_list']) - 10} more")
    
    def display_analysis_interface(self):
        """Main analysis interface for users"""
        st.subheader("📈 Inventory Analysis Results")
        # Get PFEP and Inventory data
        try:
            pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
            inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        except Exception as e:
            st.error("❌ Error loading PFEP or Inventory data.")
            st.code(str(e))
            return
        if not pfep_data or not inventory_data:
            st.error("❌ Required data not available. Please upload PFEP and Inventory data first.")
            return
        # Get tolerance from admin settings
        tolerance = st.session_state.get('admin_tolerance', 30)
        st.info(f"📐 Analysis Tolerance: ±{tolerance}% (Set by Admin)")

        # Check if analysis needs to be performed or updated
        analysis_data = self.persistence.load_data_from_session_state('persistent_analysis_results')
        last_tolerance = st.session_state.get('last_analysis_tolerance', None)

        # Auto re-analyze if tolerance changed or no data exists
        if not analysis_data or last_tolerance != tolerance:
            st.info(f"🔄 Re-analyzing with ±{tolerance}% tolerance...")
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
                st.success("✅ Analysis completed successfully!")
                st.rerun()
            else:
                st.error("❌ Analysis failed. No results generated.")
                return
        # ✅ Use the full dashboard method
        try:
            self.display_analysis_results()
        except Exception as e:
            st.error("❌ Unexpected error during analysis results display")
            st.code(str(e))
            return
            
    def display_comprehensive_analysis(self, analysis_results):
        """Display comprehensive analysis results with enhanced features"""
        st.success(f"✅ Analysis Complete: {len(analysis_results)} parts analyzed")
        try:
            self.display_enhanced_summary_metrics(analysis_results)
        except Exception as e:
            st.error("❌ Error in Summary Metrics")
            st.code(str(e))
        try:
            self.display_enhanced_detailed_tables(analysis_results)
        except Exception as e:
            st.error("❌ Error in Detailed Tables")
            st.code(str(e))
        try:
            self.display_enhanced_analysis_charts(analysis_results)
        except Exception as e:
            st.error("❌ Error in Charts")
            st.code(str(e))
        try:
            self.display_enhanced_export_options(analysis_results)
        except Exception as e:
            st.error("❌ Error in Export Options")
            st.code(str(e))

    def display_enhanced_export_options(self, analysis_results):
        """Allow users to export the analysis results"""
        st.subheader("📤 Export Analysis Results")
        df = pd.DataFrame(analysis_results)
        # Export to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="inventory_analysis.csv",
            mime="text/csv"
        )
        # Export to Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Analysis')
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name="inventory_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    def display_enhanced_summary_metrics(self, analysis_results):
        """Enhanced summary metrics dashboard - Fixed Width Issues"""
        st.header("📊 Executive Summary Dashboard")
        # Add CSS with better responsive design
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem; /* Increased from 1.2rem */
            border-radius: 12px; /* Slightly more rounded */
            margin: 0.5rem 0; /* Increased margin */
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3); /* Enhanced shadow */
            align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            max-width: 100%;
            box-sizing: border-box;
            min-height: 140px; /* Increased from 120px */
            transition: transform 0.2s ease; /* Added hover effect */
        }
        .metric-card:hover {
            transform: translateY(-2px); /* Subtle lift on hover */
        }
        /* Status-specific styling remains the same */
        .status-normal { background: linear-gradient(135deg, #4CAF50, #45a049); }
        .status-excess { background: linear-gradient(135deg, #2196F3, #1976D2); }
        .status-short { background: linear-gradient(135deg, #F44336, #D32F2F); }
        .status-total { background: linear-gradient(135deg, #FF9800, #F57C00); }
        .metric-value {
            color: white;
            font-weight: bold;
            font-size: 1.6rem; /* Increased from 1.4rem */
            margin-bottom: 0.4rem; /* Increased spacing */
            word-wrap: break-word;
            line-height: 1.2;
        }
        .metric-label {
            color: #f0f0f0;
            font-size: 1.1rem; /* Increased from 1.0rem */
            margin-bottom: 0.5rem; /* Increased spacing */
            word-wrap: break-word;
            font-weight: 500; /* Added font weight */
        }
        .metric-delta {
            color: #e0e0e0;
            font-size: 0.9rem; /* Increased from 0.85rem */
            word-wrap: break-word;
            font-weight: 400;
        }
        .highlight-box {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem; /* Increased padding */
            border-radius: 12px; /* Consistent with cards */
            color: white;
            margin: 1rem 0; /* Increased margin */
            max-width: 85%;
            box-sizing: border-box;
        }
        .dashboard-container {
            max-width: 85%; /* Increased from 80% for more space */
            overflow-x: auto;
            margin: 0 auto; /* Center the container */
        }
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .metric-card {
                min-height: 100px;
                padding: 1rem;
            }
            .metric-value {
                font-size: 1.3rem;
            }
            .metric-label {
                font-size: 0.9rem;
            }
            .dashboard-container {
                max-width: 95%;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        # Wrap everything in a container with 70% max width
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        # DataFrame prep
        df = pd.DataFrame(analysis_results)
        # Identify value and status columns
        value_col = 'Stock Deviation Value'
        status_col = 'Status' if 'Status' in df.columns else 'INVENTORY REMARK STATUS'
        # Compute KPI values safely
        if not df.empty and value_col in df.columns and status_col in df.columns:
            short_value = df[df[status_col] == 'Short Inventory'][value_col].sum()
            excess_value = df[df[status_col] == 'Excess Inventory'][value_col].sum()
        else:
            short_value = 0
            excess_value = 0
        # Total values
        total_parts = len(df)
        inventory_value_col = next((col for col in [
            'Current Inventory - VALUE', 'Stock_Value', 'VALUE'
        ] if col in df.columns), None)
        total_stock_value = df[inventory_value_col].sum() if inventory_value_col else 0
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 🎯 Key Inventory KPIs
        - **Total Parts Analyzed**: {total_parts:,}
        - **Total Inventory Value**: ₹{total_stock_value:,.0f}
        - **Short Inventory Impact**: ₹{abs(short_value):,.0f}
        - **Excess Inventory Impact**: ₹{excess_value:,.0f}
        - **Net Financial Impact**: ₹{abs(short_value) - excess_value:,.0f}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        # Status breakdown
        status_values = {}
        for label in ['Within Norms', 'Excess Inventory', 'Short Inventory']:
            filtered = df[df[status_col] == label]
            status_values[label] = {
                'count': len(filtered),
                'value': filtered[inventory_value_col].sum() if inventory_value_col in filtered.columns else 0
            }
        # Display 4 columns
        cols = st.columns([1, 1, 1, 1])
        with cols[0]:
            norm = status_values.get('Within Norms', {'count': 0, 'value': 0})
            st.markdown(f"""
            <div class="metric-card status-normal">
                <div class="metric-label">🟢 Within Norms</div>
                <div class="metric-value">{norm['count']} parts</div>
                <div class="metric-delta">₹{norm['value']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            excess = status_values.get('Excess Inventory', {'count': 0, 'value': 0})
            st.markdown(f"""
            <div class="metric-card status-excess">
                <div class="metric-label">🔵 Excess Inventory</div>
                <div class="metric-value">{excess['count']} parts</div>
                <div class="metric-delta">₹{excess['value']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            short = status_values.get('Short Inventory', {'count': 0, 'value': 0})
            st.markdown(f"""
            <div class="metric-card status-short">
                <div class="metric-label">🔴 Short Inventory</div>
                <div class="metric-value">{short['count']} parts</div>
                <div class="metric-delta">₹{short['value']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""
            <div class="metric-card status-total">
                <div class="metric-label">📊 Total Inventory</div>
                <div class="metric-value">{total_parts} parts</div>
                <div class="metric-delta">₹{total_stock_value:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        # Close the container
        st.markdown('</div>', unsafe_allow_html=True)
            
    def display_enhanced_vendor_summary(self, analysis_results):
        """Enhanced vendor summary with better analytics"""
        st.header("🏢 Vendor Performance Analysis")
        df = pd.DataFrame(analysis_results)
        # Define vendor column only if one exist
        vendor_col = None
        if 'Vendor' in df.columns:
            vendor_col = 'Vendor'
        elif 'Vendor Name' in df.columns:
            vendor_col = 'Vendor Name'
        elif 'VENDOR' in df.columns:
            vendor_col = 'VENDOR'
        if vendor_col is None:
            st.warning("Vendor information not available in analysis data.")
            return  # Early exit if no vendor column found
        value_col = 'Current Inventory - VALUE'
        vendor_summary = {}
        for vendor in df[vendor_col].dropna().unique():
            vendor_data = df[df[vendor_col] == vendor]
            vendor_summary[vendor] = {
                'total_parts': len(vendor_data),
                'total_value': vendor_data[value_col].sum() if value_col in vendor_data.columns else 0,
                'short_parts': len(vendor_data[vendor_data['Status'] == 'Short Inventory']),
                'excess_parts': len(vendor_data[vendor_data['Status'] == 'Excess Inventory']),
                'normal_parts': len(vendor_data[vendor_data['Status'] == 'Within Norms']),
                'short_value': vendor_data[vendor_data['Status'] == 'Short Inventory'][value_col].sum() if value_col in vendor_data.columns else 0,
                'excess_value': vendor_data[vendor_data['Status'] == 'Excess Inventory'][value_col].sum() if value_col in vendor_data.columns else 0,
            }
            # Create vendor DataFrame OUTSIDE the loop
            vendor_df = pd.DataFrame([
                {
                    'Vendor': vendor,
                    'Total Parts': data['total_parts'],
                    'Short Inventory': data['short_parts'],
                    'Excess Inventory': data['excess_parts'],
                    'Within Norms': data['normal_parts'],
                    'Total Value (₹)': f"₹{data['total_value']:,.0f}",
                    'Performance Score': round((data['normal_parts'] / data['total_parts']) * 100, 1)
                    if data['total_parts'] > 0 else 0
                }
                for vendor, data in vendor_summary.items()
            ])
            if vendor_df.empty:
                st.warning("No vendor data available for analysis.")
                return
            def color_performance(val):
                if isinstance(val, (int, float)):
                    if val >= 80:
                        return 'background-color: #4CAF50; color: white'
                    elif val >= 60:
                        return 'background-color: #FF9800; color: white'
                    else:
                        return 'background-color: #F44336; color: white'
                return ''
            st.dataframe(
                vendor_df.style.map(color_performance, subset=['Performance Score']),
                use_container_width=True,
                hide_index=True
            )
            fig = px.bar(
                vendor_df.head(10),
                x='Vendor',
                y='Performance Score',
                title="Top 10 Vendor Performance Scores",
                color='Performance Score',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

    def display_enhanced_detailed_tables(self, analysis_results):
        """Display enhanced detailed tables with proper formatting"""
        st.header("📊 Detailed Analysis Tables")
        df = pd.DataFrame(analysis_results)
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 All Items", "🔴 Short Inventory", "🔵 Excess Inventory", "🟢 Within Norms"])
        with tab1:
            st.subheader("All Inventory Items")
            # Add search functionality
            search_term = st.text_input("🔍 Search parts (Part No, Description, Vendor):", key="search_all")
            display_df = df.copy()
            if search_term:
                search_mask = (
                    df['PART NO'].astype(str).str.contains(search_term, case=False, na=False) |
                    df.get('PART DESCRIPTION', pd.Series(dtype=str)).fillna('').str.contains(search_term, case=False, na=False) |
                    df.get('VENDOR', pd.Series(dtype=str)).fillna('').str.contains(search_term, case=False, na=False)
                )
                display_df = df[search_mask]
            # Select key columns for display
            display_columns = self._get_key_display_columns(display_df)
            if display_columns:
                styled_df = display_df[display_columns].style.format(self._get_column_formatters())
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            else:
                st.dataframe(display_df, use_container_width=True, height=400)
            st.info(f"📊 Showing {len(display_df)} of {len(df)} total items")
        with tab2:
            st.subheader("🔴 Short Inventory Items")
            short_items = df[df['Status'] == 'Short Inventory']
            if not short_items.empty:
                # Sort by impact/value
                value_col = self._get_value_column(short_items)
                if value_col:
                    short_items = short_items.sort_values(value_col, ascending=False)
                st.error(f"⚠️ {len(short_items)} items are short on inventory")
                # Add urgency classification
                if value_col:
                    short_items['Urgency'] = pd.cut(
                        short_items[value_col], 
                        bins=[0, 10000, 50000, float('inf')], 
                        labels=['Low', 'Medium', 'High'],
                        include_lowest=True
                    )
                display_columns = self._get_key_display_columns(short_items)
                if display_columns:
                    styled_df = short_items[display_columns].style.format(self._get_column_formatters())
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )
                else:
                    st.dataframe(short_items, use_container_width=True, height=400)
            else:
                st.success("✅ No items are currently short on inventory!")
        with tab3:
            st.subheader("🔵 Excess Inventory Items")
            excess_items = df[df['Status'] == 'Excess Inventory']
            if not excess_items.empty:
                # Sort by value
                value_col = self._get_value_column(excess_items)
                if value_col:
                    excess_items = excess_items.sort_values(value_col, ascending=False)
                st.warning(f"📦 {len(excess_items)} items have excess inventory")
                # Calculate potential savings
                if value_col:
                    total_excess_value = excess_items[value_col].sum()
                    st.metric("Total Excess Value", f"₹{total_excess_value:,.0f}")
                display_columns = self._get_key_display_columns(excess_items)
                if display_columns:
                    styled_df = excess_items[display_columns].style.format(self._get_column_formatters())
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )
                else:
                    st.dataframe(excess_items, use_container_width=True, height=400)
            else:
                st.success("✅ No items have excess inventory!")
        with tab4:
            st.subheader("🟢 Items Within Norms")
            normal_items = df[df['Status'] == 'Within Norms']
            if not normal_items.empty:
                st.success(f"✅ {len(normal_items)} items are within normal inventory levels")
                display_columns = self._get_key_display_columns(normal_items)
                if display_columns:
                    styled_df = normal_items[display_columns].style.format(self._get_column_formatters())
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )
                else:
                    st.dataframe(normal_items, use_container_width=True, height=400)
            else:
                st.warning("⚠️ No items are currently within normal inventory levels!")
        # Return the display dataframe for further use if needed
        display_columns = self._get_key_display_columns(df)
        return df[display_columns] if display_columns else df
        
    def _get_value_column(self, df):
        """Helper method to identify the main value column"""
        value_columns = ['Stock_Value', 'Current Inventory - VALUE', 'Current Inventory-VALUE']
        for col in value_columns:
            if col in df.columns:
                return col
        return None
        
    def _get_key_display_columns(self, df):
        """Helper method to select key columns for display"""
        # Define priority columns to show
        priority_columns = [
             # Part identification
            "PART NO",
            "PART DESCRIPTION",
            "Vendor Name",
            "Vendor_Code",
            "AVG CONSUMPTION/DAY",
            "RM IN DAYS",
            "RM Norm - In Qty",
            "Revised Norm Qty",
            "Lower Bound Qty",                # ✅ Added
            "Upper Bound Qty", 
            "UNIT PRICE",
            "Current Inventory - Qty",
            "Current Inventory - VALUE",
            "SHORT/EXCESS INVENTORY",
            "Stock Deviation Value",
            "INVENTORY REMARK STATUS"
        ]
        # Select columns that exist in the dataframe
        available_columns = []
        for col in priority_columns:
            if col in df.columns and col not in available_columns:
                available_columns.append(col)
        # Add any remaining important columns not in priority list
        for col in df.columns:
            if col not in available_columns and len(available_columns) < 10:
                available_columns.append(col)
        return available_columns # Limit to 10 c
        
    def _get_column_formatters(self,df=None):
        """Get column formatters for styling dataframes"""
        formatters = {}
        if df is not None:
            for col in df.columns:
                if 'VALUE' in col.upper() or 'PRICE' in col.upper() or 'COST' in col.upper():
                    formatters[col] = lambda x: f"₹{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
                elif 'QTY' in col.upper() or 'QUANTITY' in col.upper():
                    formatters[col] = lambda x: f"{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
                elif 'PERCENTAGE' in col.upper() or 'SCORE' in col.upper():
                    formatters[col] = lambda x: f"{x:.1f}%" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
        else:
            # Default formatters when no dataframe is provided
            formatters = {
                'VALUE(Unit Price* Short/Excess Inventory)': lambda x: f"₹{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x),
                'Current Inventory - VALUE': lambda x: f"₹{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x),
                'Current Inventory - Qty': lambda x: f"{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x),
                'UNIT PRICE': lambda x: f"₹{x:,.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x),
            }
        return formatters
    
    def display_overview_metrics(self, analysis_results):
        """Display key overview metrics"""
        st.header("📊 Inventory Overview")
        df = pd.DataFrame(analysis_results)
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_parts = len(df)
            st.metric("Total Parts", f"{total_parts:,}")
        with col2:
            value_col = self._get_value_column(df)
            if value_col:
                total_value = df[value_col].sum()
                st.metric("Total Value", f"₹{total_value:,.0f}")
            else:
                st.metric("Total Value", "N/A")
        with col3:
            if 'Status' in df.columns:
                within_norms = (df['Status'] == 'Within Norms').sum()
                efficiency = (within_norms / total_parts * 100) if total_parts > 0 else 0
                st.metric("Efficiency", f"{efficiency:.1f}%", delta=f"{within_norms} parts")
            else:
                st.metric("Efficiency", "N/A")
        with col4:
            if 'Status' in df.columns:
                issues = len(df[df['Status'] != 'Within Norms'])
                st.metric("Issues", f"{issues:,}", delta="Needs attention" if issues > 0 else "All good")
            else:
                st.metric("Issues", "N/A")
                
    def display_top_parts_analysis(self, analysis_results):
        """Display top parts analysis by different criteria"""
        st.subheader("🏆 Top Parts Analysis")
        df = pd.DataFrame(analysis_results)
        value_col = self._get_value_column(df)
        if not value_col:
            st.warning("⚠️ No value column found for top parts analysis.")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💰 Highest Value Parts")
            top_value = df.nlargest(10, value_col)
            if not top_value.empty:
                display_cols = ['PART NO', 'PART DESCRIPTION', value_col, 'Status']
                available_cols = [col for col in display_cols if col in top_value.columns]
                st.dataframe(
                    top_value[available_cols].style.format(self._get_column_formatters(top_value)),
                    use_container_width=True
                )
        with col2:
            st.subheader("⚠️ Most Critical Issues")
            if 'Status' in df.columns:
                critical_issues = df[df['Status'] != 'Within Norms'].nlargest(10, value_col)
                if not critical_issues.empty:
                    display_cols = ['PART NO', 'PART DESCRIPTION', value_col, 'Status']
                    available_cols = [col for col in display_cols if col in critical_issues.columns]
                    st.dataframe(
                        critical_issues[available_cols].style.format(self._get_column_formatters(critical_issues)),
                        use_container_width=True
                    )
                else:
                    st.success("✅ No critical issues found!")
            else:
                st.info("ℹ️ Status information not available.")

    def create_enhanced_top_parts_chart(self, processed_data, status_filter, color, key, top_n=10):
        """Enhanced top parts chart with better visualization"""
        filtered_data = [
            item for item in processed_data 
            if item.get('Status') == status_filter or item.get('INVENTORY REMARK STATUS') == status_filter
        ]
        if not filtered_data:
            st.info(f"No {status_filter} parts found.")
            return

        top_parts = sorted(
            filtered_data,
            key=lambda x: x.get('Current Inventory - VALUE', 0),
            reverse=True
        )[:top_n]

        labels = [f"{item['PART NO']}<br>{item.get('PART DESCRIPTION', '')[:30]}..." for item in top_parts]
        values = [item.get('Current Inventory - VALUE', 0) for item in top_parts]
        variance_values = [item.get('VALUE(Unit Price* Short/Excess Inventory)', 0) for item in top_parts]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Stock Value',
            x=labels,
            y=values,
            marker_color=color,
            text=[f"₹{v:,.0f}" for v in values],
            textposition='auto',
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    def apply_advanced_filters(self, df):
        """Apply advanced filters to dataframe"""
        filtered_df = df.copy()
        # Apply value filted
        if hasattr(st.session_state, 'value_filter') and 'Current Inventory - VALUE' in df.columns:
            min_val, max_val = st.session_state.value_filter
            filtered_df = filtered_df[
                (filtered_df['Current Inventory - VALUE'] >= min_val) & 
                (filtered_df['Current Inventory - VALUE'] <= max_val)
            ]
        # Apply quantity filter
        if hasattr(st.session_state, 'qty_filter') and 'Current Inventory - Qty' in df.columns:
            min_qty, max_qty = st.session_state.qty_filter
            filtered_df = filtered_df[
                (filtered_df['Current Inventory - Qty'] >= min_qty) & 
                (filtered_df['Current Inventory - Qty'] <= max_qty)
            ]
        # Apply vendor filter (FIXED: properly handle vendor column)
        if hasattr(st.session_state, 'vendor_filter'):
            vendor_col = None
            if 'Vendor' in df.columns:
                vendor_col = 'Vendor'
            elif 'Vendor Name' in df.columns:
                vendor_col = 'Vendor Name'
            if vendor_col and vendor_col in df.columns:
                filtered_df = filtered_df[filtered_df[vendor_col].isin(st.session_state.vendor_filter)]
        return filtered_df
            
    def display_trend_analysis(self, analysis_results):
        """Display trend analysis and forecasting"""
        st.header("📈 Trend Analysis & Forecasting")
        df = pd.DataFrame(analysis_results)
        # Create trend analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Status Trends", "💹 Value Trends", "🔮 Forecasting"])
        with tab1:
            # Status distribution over time (if timestamp data available)
            status_counts = df['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Current Inventory Status Distribution",
                color=status_counts.index, 
                color_discrete_map={
                    'Within Norms': '#4CAF50',
                    'Excess Inventory': '#2196F3',
                    'Short Inventory': '#F44336'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            # Status by category if available
            if 'Category' in df.columns or 'PART CATEGORY' in df.columns:
                category_col = 'Category' if 'Category' in df.columns else 'PART CATEGORY'
                status_category = df.groupby([category_col, 'Status']).size().unstack(fill_value=0)
                fig = px.bar(
                    status_category,
                    title="Status Distribution by Category",
                    color_discrete_map={
                        'Within Norms': '#4CAF50',
                        'Excess Inventory': '#2196F3',
                        'Short Inventory': '#F44336'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                df['VALUE_LAKH'] = df['Current Inventory - VALUE'] / 100000
                # ✅ 1️⃣ Value Distribution Analysis (Bar Chart)
                value_ranges = pd.cut(df['VALUE_LAKH'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                value_status = pd.crosstab(value_ranges, df['Status'])
                fig1 = px.bar(
                    value_status,
                    title="Status Distribution by Value Range (in ₹ Lakhs)",
                    color_discrete_map={
                        'Within Norms': '#4CAF50',
                        'Excess Inventory': '#2196F3',
                        'Short Inventory': '#F44336'
                    },
                    barmode='stack'
                )
                fig1.update_layout(yaxis_title="Number of Parts")
                st.plotly_chart(fig1, use_container_width=True)
                # ✅ 2️⃣ Top Value Contributors (Scatter Plot)
                top_value_parts = df.nlargest(20, 'VALUE_LAKH')
                fig2 = px.scatter(
                    top_value_parts,
                    x='Current Inventory - Qty',
                    y='VALUE_LAKH',
                    color='Status',
                    size='VALUE_LAKH',
                    hover_data=['PART NO', 'PART DESCRIPTION'],
                    title="Top 20 Parts by Value - Quantity vs Value Analysis",
                    color_discrete_map={
                        'Within Norms': '#4CAF50',
                        'Excess Inventory': '#2196F3',
                        'Short Inventory': '#F44336'
                    }
                )
                fig2.update_layout(
                    xaxis_title="Inventory Quantity",
                    yaxis_title="Inventory Value (in ₹ Lakhs)",
                    yaxis=dict(tickformat=',.0f', ticksuffix='L')
                )
                st.plotly_chart(fig2, use_container_width=True)

            with tab3:
                st.subheader("🔮 Predictive Insights")
                # Create two columns for better layout
                col1, col2 = st.columns(2)
                with col1:
                    # Calculate reorder predictions
                    reorder_candidates = df[
                        (df['Status'] == 'Within Norms') & 
                        (df['Current Inventory - Qty'] <= df['Current Inventory - VALUE'] * 1.2)
                    ]
                    if not reorder_candidates.empty:
                        st.warning(f"📋 **Reorder Alert**: {len(reorder_candidates)} parts may need reordering soon")
                        # Display reorder table
                        reorder_display = reorder_candidates[['PART NO', 'PART DESCRIPTION', 'Current Inventory - Qty', 
                                                              'Current Inventory - VALUE']].copy()
                        reorder_display['Days to Reorder'] = np.random.randint(5, 30, len(reorder_display))  # Simulated
                        reorder_display['Priority'] = np.where(
                            reorder_display['Current Inventory - VALUE'] > reorder_display['Current Inventory - VALUE'].median(),
                            'High', 'Medium'
                        )
                        st.dataframe(reorder_display, use_container_width=True)
                    else:
                        st.info("✅ No immediate reorder candidates identified")
                with col2:
                    # Excess Inventory Reorder Point Analysis
                    excess_inventory = df[df['Status'] == 'Excess Inventory'].copy()
                    if not excess_inventory.empty:
                        st.error(f"🚨 **Excess Inventory Alert**: {len(excess_inventory)} parts have excess stock")
                        # Calculate excess reorder metrics
                        excess_inventory['Excess_Qty'] = excess_inventory['Current Inventory - Qty'] - (
                            excess_inventory['Current Inventory - Qty'] * 0.7  # Assuming 70% as optimal level
                        )
                        excess_inventory['Excess_Value_Lakh'] = (excess_inventory['Excess_Qty'] * 
                                                                 excess_inventory['Current Inventory - VALUE'] / 
                                                                 excess_inventory['Current Inventory - Qty']) / 100000
                        excess_inventory['Reorder_Days'] = np.random.randint(60, 180, len(excess_inventory))  # Days until next reorder needed
                        excess_inventory['Action_Required'] = np.where(
                            excess_inventory['Excess_Value_Lakh'] > 1, 'Urgent', 'Monitor'
                        )
                
                        # Display excess reorder days table
                        excess_display = excess_inventory[['PART NO', 'PART DESCRIPTION', 'Current Inventory - Qty', 
                                                           'Excess_Qty', 'Excess_Value_Lakh', 'Reorder_Days', 
                                                           'Action_Required']].copy()
                        excess_display.columns = ['Part No', 'Description', 'Current Qty', 'Excess Qty', 
                                                  'Excess Value (₹L)', 'Days Until Reorder', 'Action']
                        st.dataframe(excess_display, use_container_width=True)
                
                        # Summary metrics for excess inventory
                        total_excess_value = excess_inventory['Excess_Value_Lakh'].sum()
                        urgent_actions = len(excess_inventory[excess_inventory['Action_Required'] == 'Urgent'])
                    else:
                        st.success("✅ No excess inventory detected")
                # Combined Forecasting Table
                st.subheader("📊 Comprehensive Forecasting Analysis")
        
                # Create comprehensive forecasting dataframe
                forecasting_df = df.copy()
                forecasting_df['Predicted_Demand'] = np.random.randint(10, 500, len(forecasting_df))  # Simulated
                forecasting_df['Lead_Time_Days'] = np.random.randint(7, 45, len(forecasting_df))  # Simulated
                forecasting_df['Safety_Stock'] = forecasting_df['Current Inventory - Qty'] * 0.2  # 20% safety stock
                forecasting_df['Optimal_Reorder_Days'] = (
                    forecasting_df['Current Inventory - Qty'] / forecasting_df['Predicted_Demand'] * 30
                ).round(0)  # Days until reorder needed based on consumption rate
                forecasting_df['Reorder_Recommendation'] = np.where(
                    forecasting_df['Optimal_Reorder_Days'] <= forecasting_df['Lead_Time_Days'],
                    'Reorder Now',
                    np.where(
                        forecasting_df['Optimal_Reorder_Days'] <= forecasting_df['Lead_Time_Days'] * 1.5,
                        'Monitor Closely',
                        'Sufficient Stock'
                    )
                )
                # Display comprehensive table
                comprehensive_display = forecasting_df[[
                    'PART NO', 'PART DESCRIPTION', 'Status', 'Current Inventory - Qty',
                    'Predicted_Demand', 'Optimal_Reorder_Days', 'Lead_Time_Days', 'Reorder_Recommendation'
                ]].copy()
                comprehensive_display.columns = [
                    'Part No', 'Description', 'Current Status', 'Current Qty',
                    'Predicted Monthly Demand', 'Days Until Reorder', 'Lead Time (Days)', 'Recommendation'
                ]
                # Add color coding for recommendations
                def color_recommendation(val):
                    if val == 'Reorder Now':
                        return 'background-color: #ffebee; color: #c62828'
                    elif val == 'Monitor Closely':
                        return 'background-color: #fff3e0; color: #ef6c00'
                    else:
                        return 'background-color: #e8f5e8; color: #2e7d32'
                styled_df = comprehensive_display.style.applymap(color_recommendation, subset=['Recommendation'])
                st.dataframe(styled_df, use_container_width=True)
        
                # Seasonal analysis placeholder
                st.info("📊 **Seasonal Analysis**: Historical data integration required for advanced forecasting")
        
                # Key insights summary
                st.subheader("🔍 Key Insights")
                insights_col1, insights_col2, insights_col3 = st.columns(3)
                with insights_col1:
                    reorder_now_count = len(forecasting_df[forecasting_df['Reorder_Recommendation'] == 'Reorder Now'])
                    st.metric("Parts Needing Reorder", reorder_now_count)
                with insights_col2:
                    monitor_count = len(forecasting_df[forecasting_df['Reorder_Recommendation'] == 'Monitor Closely'])
                    st.metric("Parts to Monitor", monitor_count)
                with insights_col3:
                    sufficient_count = len(forecasting_df[forecasting_df['Reorder_Recommendation'] == 'Sufficient Stock'])
                    st.metric("Parts with Sufficient Stock", sufficient_count)
                    
    def display_export_options(self, analysis_results):
        """Enhanced export options"""
        st.header("📥 Export & Reporting Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Export Full Analysis", type="primary"):
                self.export_comprehensive_report(analysis_results)
        with col2:
            if st.button("🚨 Export Critical Items Only"):
                self.export_critical_items(analysis_results)
        with col3:
            if st.button("📈 Export Executive Summary"):
                self.export_executive_summary(analysis_results)
        # Export format options
        st.subheader("Export Format Options")
        export_format = st.selectbox(
            "Select Export Format",
            ["Excel (.xlsx)", "CSV (.csv)", "PDF Report", "PowerPoint Summary"]
        )
        if st.button("🎯 Custom Export"):
            self.export_custom_format(analysis_results, export_format)
            
    def export_comprehensive_report(self, analysis_results):
        """Export comprehensive analysis report"""
        try:
            df = pd.DataFrame(analysis_results)
            # Create Excel writer
            from io import BytesIO
            output = BytesIO()
        
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Main analysis sheet
                df.to_excel(writer, sheet_name='Full Analysis', index=False)
                # Summary sheet
                summary_data = {
                    'Status': df['Status'].value_counts().index.tolist(),
                    'Count': df['Status'].value_counts().values.tolist(),
                    'Total Value': [df[df['Status'] == status]['Current Inventory - VALUE'].sum() 
                                    for status in df['Status'].value_counts().index]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                # Critical items sheet
                critical_items = df[df['Current Inventory - VALUE'] > 100000]
                critical_items.to_excel(writer, sheet_name='Critical Items', index=False)
                # Download button
                st.download_button(
                    label="📥 Download Comprehensive Report",
                    data=output.getvalue(),
                    file_name=f"inventory_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✅ Comprehensive report prepared for download!")
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            
    def export_critical_items(self, analysis_results):
        """Export only critical items"""
        try:
            df = pd.DataFrame(analysis_results)
            # Filter critical items
            critical_items = df[
                (df['Status'] != 'Within Norms') & 
                (df['Current Inventory - VALUE'] > st.session_state.get('critical_threshold', 100000))
            ]
            if critical_items.empty:
                st.warning("No critical items found based on current criteria.")
                return
            # Create CSV
            csv = critical_items.to_csv(index=False)
            st.download_button(
                label="📥 Download Critical Items Report",
                data=csv,
                file_name=f"critical_items_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.success(f"✅ Critical items report prepared! ({len(critical_items)} items)")
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            
    def export_executive_summary(self, analysis_results):
        """Export executive summary"""
        try:
            df = pd.DataFrame(analysis_results)
            # Create executive summary data
            summary = {
                'Metric': [
                    'Total Parts Analyzed',
                    'Total Inventory Value (₹)',
                    'Parts Within Norms',
                    'Excess Inventory Parts',
                    'Short Inventory Parts',
                    'Inventory Efficiency (%)',
                    'Excess Value (₹)',
                    'Shortage Impact (₹)'
                ],
                'Value': [
                    len(df),
                    f"₹{df['Current Inventory - VALUE'].sum():,.0f}",
                    (df['Status'] == 'Within Norms').sum(),
                    (df['Status'] == 'Excess Inventory').sum(),
                    (df['Status'] == 'Short Inventory').sum(),
                    f"{(df['Status'] == 'Within Norms').mean() * 100:.1f}%",
                    f"₹{df[df['Status'] == 'Excess Inventory']['Current Inventory - VALUE'].sum():,.0f}",
                    f"₹{abs(df[df['Status'] == 'Short Inventory']['Stock Deviation Value'].sum()):,.0f}"
                ]
            }
            summary_df = pd.DataFrame(summary)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Executive Summary",
                data=csv,
                file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.success("✅ Executive summary prepared for download!")
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            
    def export_custom_format(self, analysis_results, format_type):
        """Export in custom format"""
        try:
            df = pd.DataFrame(analysis_results)
            if format_type == "Excel (.xlsx)":
                self.export_comprehensive_report(analysis_results)
            elif format_type == "CSV (.csv)":
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"inventory_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif format_type == "PDF Report":
                st.info("📄 PDF export functionality requires additional setup. Using CSV format instead.")
                self.export_custom_format(analysis_results, "CSV (.csv)")
            elif format_type == "PowerPoint Summary":
                st.info("📊 PowerPoint export functionality requires additional setup. Using Excel format instead.")
                self.export_comprehensive_report(analysis_results)
            st.success(f"✅ Export completed in {format_type} format!")
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            
    def display_help_and_documentation(self):
        """Display help and documentation"""
        st.header("❓ Help & Documentation")
        with st.expander("📖 Understanding Analysis Results"):
            st.markdown("""
            ### Status Categories:
            - **🟢 Within Norms**: Inventory levels are optimal
            - **🔵 Excess Inventory**: Stock levels exceed requirements
            - **🔴 Short Inventory**: Stock levels are below minimum requirements
            
            ### Key Metrics:
            - **Stock Value**: Total monetary value of current inventory
            - **Variance Impact**: Financial impact of excess/shortage
            - **Performance Score**: Percentage of parts within norms
            """)
        with st.expander("🔧 Advanced Features"):
            st.markdown("""
            ### Filtering Options:
            - Use sidebar filters to focus on specific value ranges
            - Filter by vendor performance
            - Set critical value thresholds
            ### Export Options:
            - **Full Analysis**: Complete detailed report
            - **Critical Items**: High-value problem items only
            - **Executive Summary**: Key metrics for management
            """)
        with st.expander("💡 Best Practices"):
            st.markdown("""
            ### Optimization Tips:
            1. **Regular Monitoring**: Run analysis weekly/monthly
            2. **Vendor Performance**: Track vendor consistency
            3. **Critical Thresholds**: Adjust based on business needs
            4. **Action Items**: Follow up on recommendations
            5. **Trend Analysis**: Monitor patterns over time
            """)
        
    def display_actionable_insights(self, analysis_results):
        """Display actionable insights and recommendations"""
        st.header("💡 Actionable Insights & Recommendations")
        df = pd.DataFrame(analysis_results)
        # Create insights tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🚨 Immediate Actions", "💰 Cost Optimization", "📊 Performance", "🔄 Process Improvement"])
        
        with tab1:
            st.subheader("🚨 Immediate Action Required")
            # Critical shortage
            critical_shortages = df[
                (df['Status'] == 'Short Inventory') & 
                (df['Current Inventory - VALUE'] > 50000)
            ].sort_values('Current Inventory - VALUE', ascending=False)
            if not critical_shortages.empty:
                st.error(f"🔴 **URGENT**: {len(critical_shortages)} high-value parts critically short!")
                for idx, part in critical_shortages.head(5).iterrows():
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                        with col1:
                            st.write(f"**{part['PART NO']}** - {part['PART DESCRIPTION'][:50]}...")
                        with col2:
                            st.write(f"Value: ₹{part['Current Inventory - VALUE']:,.0f}")
                        with col3:
                            # Calculate shortage in rupees
                            # Assuming you have a unit price column or can calculate it
                            if 'Current Inventory - Qty' in df.columns and part['Current Inventory - Qty'] > 0:
                                unit_price = part['Current Inventory - VALUE'] / part['Current Inventory - Qty']
                                # If you have a target/required quantity column, use it. Otherwise, estimate shortage
                                if 'Required Qty' in df.columns:
                                    shortage_qty = part['Required Qty'] - part['Current Inventory - Qty']
                                else:
                                    # Estimate shortage as a percentage of current inventory
                                    shortage_qty = part['Current Inventory - Qty'] * 0.3  # 30% more needed
                                shortage_value = shortage_qty * unit_price
                                st.write(f"Need: ₹{shortage_value:,.0f}")
                            else:
                                st.write("Need: Data unavailable")
                        with col4:
                            if 'Current Inventory - Qty' in df.columns:
                                if 'Required Qty' in df.columns:
                                    shortage_qty = part['Required Qty'] - part['Current Inventory - Qty']
                                else:
                                    shortage_qty = part['Current Inventory - Qty'] * 0.3
                                st.write(f"Qty: {shortage_qty:,.0f}")
                            else:
                                st.write("Qty: N/A")
                                
            # Excess inventory actions
            excess_items = df[
                (df['Status'] == 'Excess Inventory') & 
                (df['Current Inventory - VALUE'] > 100000)
            ].sort_values('Current Inventory - VALUE', ascending=False)
        
            if not excess_items.empty:
                st.warning(f"🟡 **Consider**: {len(excess_items)} high-value excess items for reallocation")
                
        with tab2:
            st.subheader("💰 Cost Optimization Opportunities")
            # Required columns
            required_cols = [
                'Status', 'Current Inventory - VALUE', 'Current Inventory - Qty',
                'Stock Deviation Value', 'UNIT PRICE', 'RM Norm - In Qty'
            ]
            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if not missing_cols:
                # 1️⃣ Filter excess inventory with error handling
                excess_df = df[df['Status'] == 'Excess Inventory'].copy()
                if len(excess_df) == 0:
                    st.info("No excess inventory found in the current dataset.")
                else:
                    # 2️⃣ Handle potential NaN values and calculate excess value
                    excess_df['Stock Deviation Value'] = pd.to_numeric(
                        excess_df['Stock Deviation Value'], errors='coerce'
                    ).fillna(0)
                    excess_value = excess_df['Stock Deviation Value'].sum()
                    # 3️⃣ Calculate savings and capital with configurable rates
                    CARRYING_COST_RATE = 0.10  # 10% carrying cost
                    RECOVERY_RATE = 0.70       # 70% capital recovery rate
                    potential_savings = excess_value * CARRYING_COST_RATE
                    freed_capital = excess_value * RECOVERY_RATE
                    # 4️⃣ Display KPIs with better formatting
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Excess Value",
                            f"₹{excess_value:,.0f}",
                            help="Total value of excess inventory"
                        )
                    with col2:
                        st.metric(
                            "Potential Annual Savings",
                            f"₹{potential_savings:,.0f}",
                            help=f"Based on {CARRYING_COST_RATE*100}% carrying cost reduction"
                        )
                    with col3:
                        st.metric(
                            "Capital That Could Be Freed",
                            f"₹{freed_capital:,.0f}",
                            help=f"Estimated {RECOVERY_RATE*100}% recoverable capital from excess stock"
                        )
                    # 5️⃣ Top optimization candidates with enhanced calculations
                    st.subheader("🎯 Top Optimization Candidates")
                    top_excess = excess_df.copy()
                    # Ensure numeric columns are properly converted
                    numeric_cols = ['Current Inventory - Qty', 'RM Norm - In Qty', 'UNIT PRICE']
                    for col in numeric_cols:
                        if col in top_excess.columns:
                            top_excess[col] = pd.to_numeric(top_excess[col], errors='coerce').fillna(0)
                    # Calculate excess quantity and optimization potential
                    top_excess['Excess Qty'] = top_excess['Current Inventory - Qty'] - top_excess['RM Norm - In Qty']
                    top_excess['Optimization Potential'] = top_excess['Excess Qty'] * top_excess['UNIT PRICE']
                    # Filter out items with zero or negative optimization potential
                    top_excess = top_excess[top_excess['Optimization Potential'] > 0]
                    # Sort and get top candidates
                    top_excess = top_excess.sort_values(by='Optimization Potential', ascending=False).head(10)
                    if len(top_excess) > 0:
                        # Format the display dataframe
                        display_df = top_excess[[
                            'PART NO', 'PART DESCRIPTION', 'Current Inventory - Qty', 'RM Norm - In Qty',
                            'Excess Qty', 'UNIT PRICE', 'Optimization Potential'
                        ]].copy()
                        # Format monetary columns
                        display_df['UNIT PRICE'] = display_df['UNIT PRICE'].apply(lambda x: f"₹{x:,.2f}")
                        display_df['Optimization Potential'] = display_df['Optimization Potential'].apply(lambda x: f"₹{x:,.0f}")
                        # Rename columns for better displa
                        display_df.columns = [
                            'Part Number', 'Description', 'Current Qty', 'Required Qty',
                            'Excess Qty', 'Unit Price', 'Optimization Value'
                        ]
                        st.dataframe(display_df, use_container_width=True)
            
                        # 6️⃣ Additional insights
                        st.subheader("📊 Optimization Insights")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Top Categories by Excess Value:**")
                            if 'CATEGORY' in excess_df.columns:
                                category_analysis = excess_df.groupby('CATEGORY')['Optimization Potential'].sum().sort_values(ascending=False).head(5)
                                for category, value in category_analysis.items():
                                    st.write(f"• {category}: ₹{value:,.0f}")
                            else:
                                st.write("Category information not available")
                        with col2:
                            st.write("**Optimization Summary:**")
                            st.write(f"• Total excess items: {len(excess_df):,}")
                            st.write(f"• Avg excess per item: ₹{excess_value/len(excess_df):,.0f}")
                            st.write(f"• Top 10 items represent: {top_excess['Optimization Potential'].sum()/excess_value*100:.1f}% of total")
                    else:
                        st.info("No items with positive optimization potential found.")
            else:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.write("**Available columns:**")
                st.write(list(df.columns))
                st.write("\n**Required columns:**")
                for col in required_cols:
                    status = "✅" if col in df.columns else "❌"
                    st.write(f"{status} {col}")
                
        with tab3:
            st.subheader("📊 Performance Analysis")
            # Vendor performance insights
            if 'VENDOR' in df.columns:
                vendor_performance = df.groupby('VENDOR').agg({
                    'Status': lambda x: (x == 'Within Norms').mean() * 100,
                    'Current Inventory - VALUE': 'sum',
                    'PART NO': 'count'
                }).round(2)
                vendor_performance.columns = ['Performance %', 'Total Value', 'Part Count']
                vendor_performance = vendor_performance.sort_values('Performance %', ascending=False)
                st.subheader("🏆 Vendor Performance Ranking")
                st.dataframe(vendor_performance, use_container_width=True)
            # Category performance
            if 'Category' in df.columns or 'PART CATEGORY' in df.columns:
                category_col = 'Category' if 'Category' in df.columns else 'PART CATEGORY'
                category_performance = df.groupby(category_col).agg({
                    'Status': lambda x: (x == 'Within Norms').mean() * 100,
                    'Current Inventory - VALUE': 'sum'
                }).round(2)
                category_performance.columns = ['Performance %', 'Total Value']
                fig = px.scatter(
                    category_performance.reset_index(),
                    x='Performance %',
                    y='Total Value',
                    size='Total Value',
                    hover_data=[category_col],
                    title="Category Performance vs Value Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
        with tab4:
            st.subheader("🔄 Process Improvement Recommendations")
            improvement_score = (df['Status'] == 'Within Norms').mean() * 100
            recommendations = []
            if improvement_score < 60:
                recommendations.extend([
                    "🔴 **Critical**: Implement ABC analysis for better inventory classification",
                    "🔴 **Critical**: Review and update MIN/MAX thresholds quarterly",
                    "🔴 **Critical**: Establish vendor performance monitoring system"
                ])
            elif improvement_score < 80:
                recommendations.extend([
                    "🟡 **Important**: Consider implementing JIT for fast-moving items",
                    "🟡 **Important**: Set up automated reorder alerts",
                    "🟡 **Important**: Review slow-moving inventory monthly"
                ])
            else:
                recommendations.extend([
                    "🟢 **Good**: Maintain current practices with minor optimizations",
                    "🟢 **Good**: Consider advanced forecasting methods",
                    "🟢 **Good**: Implement continuous improvement processes"
                ])
                
    def display_advanced_filtering_options(self, analysis_results):
        """Display advanced filtering options in sidebar"""
        st.sidebar.header("🔍 Advanced Filters")
        df = pd.DataFrame(analysis_results)
        # Value range filter with proper handling of edge cases
        if 'Current Inventory - VALUE' in df.columns and not df['Current Inventory - VALUE'].empty:
            min_value = float(df['Current Inventory - VALUE'].min())
            max_value = float(df['Current Inventory - VALUE'].max())
        # Handle case where min and max are the same
            if min_value == max_value:
                if min_value == 0:
                    # If all values are 0, create a reasonable range
                    min_value = 0.0
                    max_value = 100000.0
                    st.sidebar.info("ℹ️ All stock values are 0. Using default range for filtering.")
                else:
                    # If all values are the same non-zero value, create a small range around i
                    range_buffer = max_value * 0.1 if max_value > 0 else 1000
                    min_value = max_value - range_buffer
                    max_value = max_value + range_buffer
                    st.sidebar.info(f"ℹ️ All stock values are {df['Current Inventory - VALUE'].iloc[0]:,.0f}. Adjusted range for filtering.")
            # Ensure min_value is always less than max_value
            if min_value >= max_value:
                max_value = min_value + 1000
            value_range = st.sidebar.slider(
                "Stock Value Range (₹)",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value),
                format="₹%.0f"
            )
        else:
            st.sidebar.warning("⚠️ Stock Value column not found or empty. Skipping value filter.")
            value_range = (0, 100000)  # Default range
        # Status filter
        if 'Status' in df.columns:
            status_options = df['Status'].unique().tolist()
            selected_statuses = st.sidebar.multiselect(
                "Filter by Status",
                options=status_options,
                default=status_options
            )
        else:
            selected_statuses = []
            st.sidebar.warning("⚠️ Status column not found.")
        # Category filter (if available)
        category_col = None
        if 'Category' in df.columns:
            category_col = 'Category'
        elif 'PART CATEGORY' in df.columns:
            category_col = 'PART CATEGORY'
        selected_categories = None
        if category_col and category_col in df.columns:
            categories = df[category_col].dropna().unique().tolist()
            if categories:
                selected_categories = st.sidebar.multiselect(
                    f"Filter by {category_col}",
                    options=categories,
                    default=categories
                )
            else:
                st.sidebar.info(f"ℹ️ No valid {category_col} values found.")
        # Vendor filter (if availablee)
        vendor_col = None
        if 'Vendor' in df.columns:
            vendor_col = 'Vendor'
        elif 'Vendor Name' in df.columns:
            vendor_col = 'Vendor Name'
        elif 'VENDOR' in df.columns:
            vendor_col = 'VENDOR'
        selected_vendors = None
        if vendor_col and vendor_col in df.columns:
            vendors = df[vendor_col].dropna().unique().tolist()
            if vendors:
                selected_vendors = st.sidebar.multiselect(
                    f"Filter by {vendor_col}",
                    options=vendors,
                    default=vendors
                )
            else:
                st.sidebar.info(f"ℹ️ No valid {vendor_col} values found.")
        # Critical threshold setting
        critical_threshold = st.sidebar.number_input(
            "Critical Value Threshold (₹)",
            min_value=0,
            value=100000,
            step=10000,
            help="Parts above this value are considered critical"
        )
        # Store filter values in session state
        st.session_state.filter_value_range = value_range
        st.session_state.filter_statuses = selected_statuses
        st.session_state.filter_categories = selected_categories
        st.session_state.filter_vendors = selected_vendors
        st.session_state.critical_threshold = critical_threshold
        # Display current filter summary
        with st.sidebar.expander("📋 Current Filters Summary"):
            st.write(f"**Value Range:** ₹{value_range[0]:,.0f} - ₹{value_range[1]:,.0f}")
            st.write(f"**Statuses:** {len(selected_statuses) if selected_statuses else 0} selected")
            if selected_categories:
                st.write(f"**Categories:** {len(selected_categories)} selected")
            if selected_vendors:
                st.write(f"**Vendors:** {len(selected_vendors)} selected")
            st.write(f"**Critical Threshold:** ₹{critical_threshold:,.0f}")

    def apply_advanced_filters(self, df):
        """Apply advanced filters to the dataframe with improved error handling"""
        filtered_df = df.copy()
        try:
            # Apply value range filter
            if (hasattr(st.session_state, 'filter_value_range') and 
                'Current Inventory - VALUE' in filtered_df.columns):
                    min_val, max_val = st.session_state.filter_value_range
                    filtered_df = filtered_df[
                        (filtered_df['Current Inventory - VALUE'] >= min_val) & 
                        (filtered_df['Current Inventory - VALUE'] <= max_val)
                    ]
            # Apply status filter
            if (hasattr(st.session_state, 'filter_statuses') and 
                st.session_state.filter_statuses and 
                'Status' in filtered_df.columns):
                    filtered_df = filtered_df[filtered_df['Status'].isin(st.session_state.filter_statuses)]
            # Apply category filter
            if (hasattr(st.session_state, 'filter_categories') and 
                st.session_state.filter_categories):
                    category_col = None
                    if 'Category' in filtered_df.columns:
                        category_col = 'Category'
                    elif 'PART CATEGORY' in filtered_df.columns:
                        category_col = 'PART CATEGORY'
                    if category_col:
                        filtered_df = filtered_df[filtered_df[category_col].isin(st.session_state.filter_categories)]
            # Apply vendor filter
            if (hasattr(st.session_state, 'filter_vendors') and 
                st.session_state.filter_vendors):
                    vendor_col = None
                    if 'Vendor' in filtered_df.columns:
                        vendor_col = 'Vendor'
                    elif 'Vendor Name' in filtered_df.columns:
                        vendor_col = 'Vendor Name'
                    elif 'VENDOR' in filtered_df.columns:
                        vendor_col = 'VENDOR'
                    if vendor_col:
                        filtered_df = filtered_df[filtered_df[vendor_col].isin(st.session_state.filter_vendors)]
            # Show filtering results
            original_count = len(df)
            filtered_count = len(filtered_df)
            if original_count != filtered_count:
                st.sidebar.success(f"✅ Filtered: {filtered_count:,} of {original_count:,} items")
            return filtered_df
        except Exception as e:
            st.sidebar.error(f"❌ Filter error: {str(e)}")
            st.sidebar.info("ℹ️ Returning unfiltered data")
            return df

    def generate_analysis_summary(self, analysis_results):
        """Generate a comprehensive analysis summary"""
        df = pd.DataFrame(analysis_results)
        summary = {
            'total_parts': len(df),
            'total_value': df['Current Inventory - VALUE'].sum(),
            'within_norms': (df['Status'] == 'Within Norms').sum(),
            'excess_inventory': (df['Status'] == 'Excess Inventory').sum(),
            'short_inventory': (df['Status'] == 'Short Inventory').sum(),
            'efficiency_rate': (df['Status'] == 'Within Norms').mean() * 100,
            'excess_value': df[df['Status'] == 'Excess Inventory']['Current Inventory - VALUE'].sum(),
            'shortage_impact': abs(df[df['Status'] == 'Short Inventory']['VALUE(Unit Price* Short/Excess Inventory)'].sum()),
            'critical_items': len(df[df['Current Inventory - VALUE'] > st.session_state.get('critical_threshold', 100000)]),
            'avg_stock_value': df['Current Inventory - VALUE'].mean()
        }
        return summary
        
    def main(self):
        """Main execution method for the inventory analyzer"""
        st.set_page_config(
            page_title="Advanced Inventory Analysis Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        # Initialize session state
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        # Main application logic
        if not st.session_state.analysis_complete:
            # Show file upload and analysis interface
            self.display_file_upload_interface()
        else:
            # Show analysis results
            self.display_analysis_results()
        
        # Option to analyze new file
        if st.sidebar.button("🔄 Analyze New File", type="secondary"):
            st.session_state.analysis_complete = False
            st.rerun()
            
    def display_analysis_results(self):
        """Main method to display all analysis results"""
        analysis_results = self.persistence.load_data_from_session_state('persistent_analysis_results')
        if not analysis_results:
            st.error("❌ No analysis results available.")
            return
        # Debug: show sample keys to catch missing columns
        st.code(f"Sample columns: {list(analysis_results[0].keys())}" if analysis_results else "No data structure found.")

        # Display advanced filtering options
        self.display_advanced_filtering_options(analysis_results)
    
        # Apply filters to data
        df = pd.DataFrame(analysis_results)
        if 'Current Inventory - VALUE' not in df.columns:
            st.warning("⚠️ 'Current Inventory - VALUE' column missing from results. Some features may not work.")

        filtered_df = self.apply_advanced_filters(df)
        filtered_results = filtered_df.to_dict('records')
    
        # Display main dashboard
        self.display_comprehensive_analysis(filtered_results)
    
        # Additional analysis sections
        st.markdown("---")
        self.display_trend_analysis(filtered_results)
    
        st.markdown("---")
        self.display_actionable_insights(filtered_results)
    
        st.markdown("---")
        self.display_export_options(filtered_results)
    
        st.markdown("---")
        self.display_help_and_documentation()
        
    def display_enhanced_analysis_charts(self, analysis_results):
        """Display enhanced visual summaries with better error handling"""
        st.subheader("📊 Enhanced Inventory Charts")
        df = pd.DataFrame(analysis_results)
        if df.empty:
            st.warning("⚠️ No data available for charts.")
            return
            
        # ✅ 1. Top 10 Parts by Value - Check for multiple possible value columns
        value_col = None
        for col in ['Current Inventory - VALUE', 'Stock_Value', 'Current Inventory-VALUE']:
            if col in df.columns:
                value_col = col
                break
        if value_col and 'PART NO' in df.columns and 'PART DESCRIPTION' in df.columns:
            # Filter top 10 parts with non-zero value
            chart_data = (
                df[df[value_col] > 0]
                .sort_values(by=value_col, ascending=False)
                .head(10)
                .copy()
            )
            # Convert to lakhs
            chart_data['Value_Lakh'] = chart_data[value_col] / 100_000
            # Combine description and part no into a single label
            chart_data['Part'] = chart_data.apply(
                lambda row: f"{row['PART DESCRIPTION']}\n({row['PART NO']})",
                axis=1
            )
            # Use the Status column from analyze_inventory results
            # If Status column exists, use it; otherwise, determine based on bounds
            if 'Status' in chart_data.columns:
                chart_data['Inventory_Status'] = chart_data['Status']
            elif 'INVENTORY REMARK STATUS' in chart_data.columns:
                chart_data['Inventory_Status'] = chart_data['INVENTORY REMARK STATUS']
            else:
                # Fallback: calculate status if bounds are available
                def determine_status_from_bounds(row):
                    current_qty = row.get('Current Inventory - Qty', 0)
                    lower_bound = row.get('Lower Bound Qty', 0)
                    upper_bound = row.get('Upper Bound Qty', 0)
                    if lower_bound > 0 and upper_bound > 0:  # Valid bounds exist
                        if current_qty < lower_bound:
                            return 'Short Inventory'
                        elif current_qty > upper_bound:
                            return 'Excess Inventory'
                        else:
                            return 'Within Norms'
                    else:
                        return 'Within Norms'  # Default if no bounds
                chart_data['Inventory_Status'] = chart_data.apply(determine_status_from_bounds, axis=1)
            # Create color mapping
            color_map = {
                "Excess Inventory": "#2196F3",
                "Short Inventory": "#F44336", 
                "Within Norms": "#4CAF50"
            }
            # Enhanced hover text
            chart_data['HOVER_TEXT'] = chart_data.apply(lambda row: (
                f"Description: {row['PART DESCRIPTION']}<br>"
                f"Part No: {row['PART NO']}<br>"
                f"Current Qty: {row.get('Current Inventory - Qty', 'N/A')}<br>"
                f"RM Norm Qty: {row.get('RM Norm - In Qty', 'N/A')}<br>"
                f"Lower Bound: {row.get('Lower Bound Qty', 'N/A')}<br>"
                f"Upper Bound: {row.get('Upper Bound Qty', 'N/A')}<br>"
                f"Value: ₹{row[value_col]:,.0f}<br>"
                f"Status: {row['Inventory_Status']}"
            ), axis=1)
            # Create a list of colors for each bar based on status
            chart_data['Bar_Color'] = chart_data['Inventory_Status'].map(color_map)
    
            # Create bar chart with individual colors (not grouped by status)
            fig1 = go.Figure()
    
            # Add bars one by one to maintain value-based sorting
            for i, row in chart_data.iterrows():
                fig1.add_trace(go.Bar(
                    x=[row['Part']],
                    y=[row['Value_Lakh']],
                    name=row['Inventory_Status'],
                    marker_color=row['Bar_Color'],
                    customdata=[row['HOVER_TEXT']],
                    hovertemplate='<b>%{x}</b><br>%{customdata}<extra></extra>',
                    showlegend=False  # We'll add legend manually
                ))
            # Add legend traces (invisible bars just for legend)
            for status, color in color_map.items():
                fig1.add_trace(go.Bar(
                    x=[None],
                    y=[None],
                    name=status,
                    marker_color=color,
                    showlegend=True
                ))
            fig1.update_layout(
                title="Top 10 Parts by Stock Value (Color-coded by Inventory Status)",
                xaxis_title="Parts",
                yaxis_title="Stock Value (in ₹ Lakhs)",
                xaxis_tickangle=-45,
                yaxis=dict(
                    tickformat=',.0f',
                    ticksuffix='L'
                ),
                xaxis=dict(
                    tickfont=dict(size=10)
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("⚠️ Required columns for parts value chart not found.")

              
        # ✅ 3. Vendor vs Value (Fixed vendor_col definition)
        vendor_col = next((col for col in ['Vendor', 'Vendor Name', 'VENDOR'] if col in df.columns), None)
        if vendor_col and value_col and vendor_col in df.columns:
            # Get vendor data with status information
            vendor_data = []
            # Group by vendor and calculate aggregated status
            for vendor_name, vendor_group in df[df[value_col] > 0].groupby(vendor_col):
                total_value = vendor_group[value_col].sum()
                total_qty = vendor_group['Current Inventory - Qty'].sum() if 'Current Inventory - Qty' in vendor_group.columns else 0
                part_count = len(vendor_group)
                # Determine vendor status based on majority of parts
                if 'Status' in vendor_group.columns:
                    status_counts = vendor_group['Status'].value_counts()
                elif 'INVENTORY REMARK STATUS' in vendor_group.columns:
                    status_counts = vendor_group['INVENTORY REMARK STATUS'].value_counts()
                else:
                    # Fallback calculation
                    def calc_status(row):
                        current_qty = row.get('Current Inventory - Qty', 0)
                        lower_bound = row.get('Lower Bound Qty', 0)
                        upper_bound = row.get('Upper Bound Qty', 0)
                        if lower_bound > 0 and upper_bound > 0:
                            if current_qty < lower_bound:
                                return 'Short Inventory'
                            elif current_qty > upper_bound:
                                return 'Excess Inventory'
                            else:
                                return 'Within Norms'
                        return 'Within Norms'
                    vendor_group_status = vendor_group.apply(calc_status, axis=1)
                    status_counts = vendor_group_status.value_counts()
                # Assign vendor status based on majority
                vendor_status = status_counts.index[0] if not status_counts.empty else 'Within Norms'
                vendor_data.append({
                    vendor_col: vendor_name,
                    value_col: total_value,
                    'Total_Parts': part_count,
                    'Total_Qty': total_qty,
                    'Vendor_Status': vendor_status
                })
            # Convert to DataFrame and get top 10
            vendor_df = pd.DataFrame(vendor_data).sort_values(by=value_col, ascending=False).head(10)
            if not vendor_df.empty:
                vendor_df['Value_Lakh'] = vendor_df[value_col] / 100000  # Convert to ₹ Lakhs
                # Create color mapping
                color_map = {
                    "Excess Inventory": "#2196F3",
                    "Short Inventory": "#F44336", 
                    "Within Norms": "#4CAF50"
                }
                # Enhanced hover text
                vendor_df['HOVER_TEXT'] = vendor_df.apply(lambda row: (
                    f"Vendor: {row[vendor_col]}<br>"
                    f"Inventory Value: ₹{row[value_col]:,.0f}<br>"
                    f"Total Parts: {row['Total_Parts']}<br>"
                    f"Total Qty: {row['Total_Qty']:,.0f}<br>"
                    f"Status: {row['Vendor_Status']}"
                ), axis=1)
                # Create color list based on status
                vendor_df['Bar_Color'] = vendor_df['Vendor_Status'].map(color_map)
        
                # Create bar chart with individual colors
                fig3 = go.Figure()
        
                # Add bars one by one to maintain value-based sorting
                for i, row in vendor_df.iterrows():
                    fig3.add_trace(go.Bar(
                        x=[row[vendor_col]],
                        y=[row['Value_Lakh']],
                        name=row['Vendor_Status'],
                        marker_color=row['Bar_Color'],
                        customdata=[row['HOVER_TEXT']],
                        hovertemplate='<b>%{x}</b><br>%{customdata}<extra></extra>',
                        showlegend=False  # We'll add legend manually
                    ))
                # Add legend traces (invisible bars just for legend)
                for status, color in color_map.items():
                    fig3.add_trace(go.Bar(
                        x=[None],
                        y=[None],
                        name=status,
                        marker_color=color,
                        showlegend=True
                    ))
                fig3.update_layout(
                    title='Top 10 Vendors by Stock Value (Color-coded by Inventory Status)',
                    xaxis_title="Vendors",
                    yaxis_title="Inventory Value (in ₹ Lakhs)",
                    xaxis_tickangle=-45,
                    yaxis=dict(
                        tickformat=',.0f',
                        ticksuffix='L'
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("ℹ️ No valid vendor data found (all values are 0).")
        else:
            missing_cols = []
            if not vendor_col:
                missing_cols.append("vendor column")
            if not value_col:
                missing_cols.append("value column")
            st.warning(f"⚠️ Vendor analysis chart cannot be displayed. Missing: {', '.join(missing_cols)}")
                
        # ✅ 4. Top 10 Parts by Inventory Status
        try:
            st.markdown("## 🧩 Top 10 Parts by Inventory Status")
            # Check if required columns exist
            if 'PART NO' not in df.columns or 'Stock Deviation Value' not in df.columns:
                st.warning("⚠️ Required columns missing for top parts chart.")
                return
            # Define status colors
            status_colors = {
                "Excess Inventory": "#2196F3",
                "Short Inventory": "#F44336"
            }
            for status, label, color in [
                ("Excess Inventory", "🔵 Top 10 Excess Inventory Parts", status_colors["Excess Inventory"]),
                ("Short Inventory", "🔴 Top 10 Short Inventory Parts", status_colors["Short Inventory"]),
            ]:
                st.subheader(label)
                if status == "Excess Inventory":
                    st.markdown(f'<div class="graph-description">Top 10 parts with highest excess inventory value (₹ above allowed norm).</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="graph-description">Top 10 parts with highest shortage value (₹ below required norm).</div>', unsafe_allow_html=True)
                # Filter by status
                status_df = df[df['INVENTORY REMARK STATUS'] == status]
                if status == "Excess Inventory":
                    # For excess inventory, show only positive deviation values (excess amount)
                    status_df = status_df[status_df['Stock Deviation Value'] > 0]
                    status_df = status_df.sort_values(by='Stock Deviation Value', ascending=False).head(10)
                    chart_title = "Top 10 Excess Inventory Parts (₹ Excess Value in Lakhs)"
                    y_title = "Excess Inventory Value (₹ Lakhs)"
                elif status == "Short Inventory":
                    # For short inventory, show absolute value of negative deviation
                    status_df = status_df[status_df['Stock Deviation Value'] < 0]
                    status_df['Abs_Deviation_Value'] = abs(status_df['Stock Deviation Value'])
                    status_df = status_df.sort_values(by='Abs_Deviation_Value', ascending=False).head(10)
                    chart_title = "Top 10 Short Inventory Parts (₹ Shortage Value in Lakhs)"
                    y_title = "Shortage Value (₹ Lakhs)"
                if status_df.empty:
                    st.info(f"No data found for '{status}' parts.")
                    continue
                # Prepare chart data
                if status == "Excess Inventory":
                    status_df['Value_Lakh'] = status_df['Stock Deviation Value'] / 100000  # Excess value in lakhs
                    hover_value = status_df['Stock Deviation Value']
                else:  # Short Inventory
                    status_df['Value_Lakh'] = status_df['Abs_Deviation_Value'] / 100000  # Shortage value in lakhs
                    hover_value = status_df['Abs_Deviation_Value']
                status_df['PART_DESC_NO'] = status_df['PART DESCRIPTION'].astype(str) + " (" + status_df['PART NO'].astype(str) + ")"
                status_df['HOVER_TEXT'] = status_df.apply(lambda row: (
                    f"Description: {row.get('PART DESCRIPTION', 'N/A')}<br>"
                    f"Part No: {row.get('PART NO')}<br>"
                    f"{'Excess' if status == 'Excess Inventory' else 'Shortage'} Value: ₹{hover_value.loc[row.name]:,.0f}"
                ), axis=1)
                # Build chart
                fig = px.bar(
                    status_df,
                    x='PART_DESC_NO',
                    y='Value_Lakh',
                    color_discrete_sequence=[color],
                    title=chart_title
                )
                # Update chart formatting
                fig.update_traces(
                    hovertemplate='<b>%{x}</b><br>%{customdata}<extra></extra>',
                    customdata=status_df['HOVER_TEXT']
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    yaxis_title=y_title,
                    yaxis=dict(
                        tickformat=',.1f',
                        ticksuffix='L'
                    )
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{status.lower().replace(' ', '_')}_parts")
        except Exception as e:
            st.error("❌ Error displaying Top Parts by Status")
            st.code(str(e))
          
        # ✅ 5. Top 10 Vendors by Inventory Status (₹ Lakhs)
        try:
            st.markdown("## 🏢 Top Vendors by Inventory Status")
            # Define chart config: status → chart title, key, color
            chart_configs = [
                ("Excess Inventory", "Top 10 Vendors - Excess Value Above Norm", "excess_vendors", self.status_colors["Excess Inventory"]),
                ("Short Inventory", "Top 10 Vendors - Short Value Below Norm", "short_vendors", self.status_colors["Short Inventory"]),
            ]
            # ✅ Loop through both excess and short charts
            for status, title, key, color in chart_configs:
                # ✅ FIXED: Call method from analyzer, not self
                self.analyzer.show_vendor_chart_by_status(
                    processed_data=analysis_results,
                    status_filter=status,
                    chart_title=title,
                    chart_key=key,
                    color=color,
                    value_format="lakhs"  # can change to "crores" if needed
                )
        except Exception as e:
            st.error("❌ Error displaying Top Vendors by Status")
            st.code(str(e))

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()  # This runs the full dashboard
